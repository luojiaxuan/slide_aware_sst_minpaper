import copy
import json
from pathlib import Path

import pytest

from scripts.build_mcif_beyond_ocr_reliability_workspace import build_bundle
from scripts.build_mcif_visual_token_controls import file_sha256, load_jsonl
from scripts.mcif_beyond_ocr_reliability import (
    apply_adjudications,
    append_annotation_event,
    blank_annotation,
    build_pre_adjudication_report,
    create_hmac_key,
    freeze_annotations,
    initialize_events,
    load_config,
    load_hmac_key,
    prepare_adjudication_release,
    release_target_validator_stage2,
    release_visual_stage,
    validate_event_log,
)
from test_build_mcif_beyond_ocr_reliability_workspace import build_kwargs


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "mcif_beyond_ocr_reliability_v2.json"


def workspace(tmp_path: Path):
    kwargs = build_kwargs(tmp_path)
    root = tmp_path / "v2"
    build_bundle(root, **kwargs)
    config = load_config(CONFIG_PATH)
    key_path = tmp_path / "private" / "annotation.key"
    create_hmac_key(key_path)
    return root, config, load_hmac_key(key_path)


def complete_events(rows, annotator_id, key, config, annotation_factory):
    events = initialize_events(
        rows, annotator_id=annotator_id, expected_items=len(rows), key=key
    )
    for row in rows:
        events = append_annotation_event(
            events,
            rows,
            item_id=row["item_id"],
            expected_event_index=0,
            annotation_status="completed",
            annotation=annotation_factory(row),
            submitted_at_utc="2026-08-01T20:00:00Z",
            annotator_id=annotator_id,
            expected_items=len(rows),
            key=key,
            config=config,
        )
    return events


def freeze(tmp_path, name, rows, annotator_id, events, key, config):
    root = tmp_path / name
    freeze_annotations(
        root,
        input_rows=rows,
        events=events,
        annotator_id=annotator_id,
        expected_items=len(rows),
        locked_at_utc="2026-08-01T21:00:00Z",
        config_sha256=file_sha256(CONFIG_PATH),
        key=key,
        config=config,
    )
    return load_jsonl(root / "frozen_annotations.jsonl")


def visual_annotation(row, value="yes"):
    annotation = blank_annotation(row["role"], row["stage"])
    annotation[next(key for key in annotation if key.endswith("support") or key == "descriptor_fidelity")] = value
    if value != "yes":
        annotation["reason_codes"] = ["ambiguous_visual"]
    return annotation


def test_event_chain_rejects_stale_tampered_and_completed_overwrite(tmp_path):
    root, config, key = workspace(tmp_path)
    rows = load_jsonl(root / "visual_a_r0_view" / "items.jsonl")
    events = initialize_events(rows, annotator_id="Visual A", expected_items=2, key=key)
    first = rows[0]
    events = append_annotation_event(
        events,
        rows,
        item_id=first["item_id"],
        expected_event_index=0,
        annotation_status="completed",
        annotation=visual_annotation(first),
        submitted_at_utc="2026-08-01T20:00:00Z",
        annotator_id="Visual A",
        expected_items=2,
        key=key,
        config=config,
    )
    with pytest.raises(ValueError, match="completed annotation is immutable"):
        append_annotation_event(
            events,
            rows,
            item_id=first["item_id"],
            expected_event_index=1,
            annotation_status="completed",
            annotation=visual_annotation(first, "no"),
            submitted_at_utc="2026-08-01T20:01:00Z",
            annotator_id="Visual A",
            expected_items=2,
            key=key,
            config=config,
        )
    with pytest.raises(ValueError, match="stale event version"):
        append_annotation_event(
            events,
            rows,
            item_id=rows[1]["item_id"],
            expected_event_index=7,
            annotation_status="completed",
            annotation=visual_annotation(rows[1]),
            submitted_at_utc="2026-08-01T20:01:00Z",
            annotator_id="Visual A",
            expected_items=2,
            key=key,
            config=config,
        )
    tampered = copy.deepcopy(events)
    tampered[-1]["r0_support"] = "no"
    with pytest.raises(ValueError, match="event signature differs"):
        validate_event_log(
            tampered,
            rows,
            annotator_id="Visual A",
            expected_items=2,
            key=key,
            config=config,
        )


def test_next_visual_stage_requires_both_complete_disjoint_full_cohorts(tmp_path):
    root, config, key = workspace(tmp_path)
    rows_a = load_jsonl(root / "visual_a_r0_view" / "items.jsonl")
    rows_b = load_jsonl(root / "visual_b_r0_view" / "items.jsonl")
    events_a = complete_events(rows_a, "Visual A", key, config, visual_annotation)
    events_b = complete_events(rows_b, "Visual B", key, config, visual_annotation)
    frozen_a = freeze(tmp_path, "freeze-a", rows_a, "Visual A", events_a, key, config)
    frozen_b = freeze(tmp_path, "freeze-b", rows_b, "Visual B", events_b, key, config)
    output = tmp_path / "r1-release"
    report = release_visual_stage(
        output,
        workspace_root=root,
        private_visual_rows=load_jsonl(root / "scorer_private" / "visual_material.jsonl"),
        mapping_rows=load_jsonl(root / "scorer_private" / "item_mapping.jsonl"),
        prior_input_a=rows_a,
        prior_input_b=rows_b,
        prior_frozen_a=frozen_a,
        prior_frozen_b=frozen_b,
        next_stage="r1",
        expected_items=2,
        key=key,
        config=config,
    )
    released_a = load_jsonl(output / "visual_a_r1_view" / "items.jsonl")
    released_b = load_jsonl(output / "visual_b_r1_view" / "items.jsonl")
    assert report["items_per_cohort"] == 2
    assert len(released_a) == len(released_b) == 2
    assert all("r1_blocks" in row and "current_slide" not in row for row in released_a + released_b)
    assert all(row["locked_judgments"] == {"r0_support": "yes"} for row in released_a + released_b)
    assert all(row["prior_cohort_lock_sha256"] == report["prior_cohort_lock_sha256"] for row in released_a + released_b)

    incomplete_b = frozen_b[:-1]
    with pytest.raises(ValueError, match="frozen item count differs"):
        release_visual_stage(
            tmp_path / "blocked",
            workspace_root=root,
            private_visual_rows=load_jsonl(root / "scorer_private" / "visual_material.jsonl"),
            mapping_rows=load_jsonl(root / "scorer_private" / "item_mapping.jsonl"),
            prior_input_a=rows_a,
            prior_input_b=rows_b,
            prior_frozen_a=frozen_a,
            prior_frozen_b=incomplete_b,
            next_stage="r1",
            expected_items=2,
            key=key,
            config=config,
        )
    conflict_b = complete_events(rows_b, " visual a ", key, config, visual_annotation)
    frozen_conflict = freeze(
        tmp_path, "freeze-conflict", rows_b, " visual a ", conflict_b, key, config
    )
    with pytest.raises(ValueError, match="roles must be disjoint"):
        release_visual_stage(
            tmp_path / "identity-blocked",
            workspace_root=root,
            private_visual_rows=load_jsonl(root / "scorer_private" / "visual_material.jsonl"),
            mapping_rows=load_jsonl(root / "scorer_private" / "item_mapping.jsonl"),
            prior_input_a=rows_a,
            prior_input_b=rows_b,
            prior_frozen_a=frozen_a,
            prior_frozen_b=frozen_conflict,
            next_stage="r1",
            expected_items=2,
            key=key,
            config=config,
        )


def author_annotation(_row):
    return {
        "candidate_eligibility": "yes",
        "canonical_source_event_en": "candidate event",
        "acceptable_target_realizations_zh": ["候选事件"],
        "forbidden_target_realizations_zh": [],
        "target_reference_alignment": "explicit",
        "reason_codes": [],
        "annotation_note": "",
    }


def validator_stage1_annotation(_row):
    return {
        "candidate_eligibility": "yes",
        "target_reference_alignment": "explicit",
        "reason_codes": [],
        "annotation_note": "",
    }


def test_target_author_text_is_released_only_after_independent_freezes(tmp_path):
    root, config, key = workspace(tmp_path)
    author_rows = load_jsonl(root / "target_author_view" / "items.jsonl")
    validator_rows = load_jsonl(root / "target_validator_stage1_view" / "items.jsonl")
    assert all("canonical_source_event_en" not in row for row in validator_rows)
    author_events = complete_events(
        author_rows, "Target Author", key, config, author_annotation
    )
    validator_events = complete_events(
        validator_rows, "Target Validator", key, config, validator_stage1_annotation
    )
    author_frozen = freeze(
        tmp_path,
        "author-freeze",
        author_rows,
        "Target Author",
        author_events,
        key,
        config,
    )
    validator_frozen = freeze(
        tmp_path,
        "validator-freeze",
        validator_rows,
        "Target Validator",
        validator_events,
        key,
        config,
    )
    output = tmp_path / "target-stage2"
    report = release_target_validator_stage2(
        output,
        private_target_rows=load_jsonl(root / "scorer_private" / "target_material.jsonl"),
        mapping_rows=load_jsonl(root / "scorer_private" / "item_mapping.jsonl"),
        author_input_rows=author_rows,
        author_frozen_rows=author_frozen,
        validator_input_rows=validator_rows,
        validator_frozen_rows=validator_frozen,
        expected_items=2,
        key=key,
        config=config,
    )
    released = load_jsonl(output / "items.jsonl")
    assert report["items"] == 2
    assert all(row["author_canonical_source_event_en"] == "candidate event" for row in released)
    assert all(row["locked_candidate_eligibility"] == "yes" for row in released)
    assert all(row["author_identity_exposed"] is False for row in released)
    assert all("author_id" not in row for row in released)


def test_hmac_key_is_create_once_private_and_not_json(tmp_path):
    path = tmp_path / "keys" / "annotation.key"
    create_hmac_key(path)
    assert len(path.read_bytes()) == 32
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    with pytest.raises(FileExistsError, match="must not already exist"):
        create_hmac_key(path)
    with pytest.raises((UnicodeDecodeError, json.JSONDecodeError)):
        json.loads(path.read_text())


def full_visual_chain(tmp_path, root, config, key, *, disagreement=False):
    inputs = {"r0": {}}
    frozen = {"r0": {}}
    for role, annotator in (("visual_a", "Visual A"), ("visual_b", "Visual B")):
        rows = load_jsonl(root / f"{role}_r0_view" / "items.jsonl")
        inputs["r0"][role] = rows
        disagreement_id = rows[0]["item_id"] if disagreement and role == "visual_b" else None
        events = complete_events(
            rows,
            annotator,
            key,
            config,
            lambda row: visual_annotation(
                row, "uncertain" if row["item_id"] == disagreement_id else "no"
            ),
        )
        frozen["r0"][role] = freeze(
            tmp_path, f"{role}-r0-freeze", rows, annotator, events, key, config
        )
    for stage in ("r1", "pixels", "descriptor"):
        previous = {"r1": "r0", "pixels": "r1", "descriptor": "pixels"}[stage]
        release_root = tmp_path / f"{stage}-release"
        release_visual_stage(
            release_root,
            workspace_root=root,
            private_visual_rows=load_jsonl(root / "scorer_private" / "visual_material.jsonl"),
            mapping_rows=load_jsonl(root / "scorer_private" / "item_mapping.jsonl"),
            prior_input_a=inputs[previous]["visual_a"],
            prior_input_b=inputs[previous]["visual_b"],
            prior_frozen_a=frozen[previous]["visual_a"],
            prior_frozen_b=frozen[previous]["visual_b"],
            next_stage=stage,
            expected_items=2,
            key=key,
            config=config,
        )
        inputs[stage] = {}
        frozen[stage] = {}
        for role, annotator in (("visual_a", "Visual A"), ("visual_b", "Visual B")):
            rows = load_jsonl(release_root / f"{role}_{stage}_view" / "items.jsonl")
            inputs[stage][role] = rows
            events = complete_events(rows, annotator, key, config, visual_annotation)
            frozen[stage][role] = freeze(
                tmp_path,
                f"{role}-{stage}-freeze",
                rows,
                annotator,
                events,
                key,
                config,
            )
    return inputs, frozen


def full_target_chain(tmp_path, root, config, key):
    author_inputs = load_jsonl(root / "target_author_view" / "items.jsonl")
    validator1_inputs = load_jsonl(root / "target_validator_stage1_view" / "items.jsonl")
    author_events = complete_events(
        author_inputs, "Target Author", key, config, author_annotation
    )
    validator1_events = complete_events(
        validator1_inputs,
        "Target Validator",
        key,
        config,
        validator_stage1_annotation,
    )
    author_frozen = freeze(
        tmp_path,
        "report-author-freeze",
        author_inputs,
        "Target Author",
        author_events,
        key,
        config,
    )
    validator1_frozen = freeze(
        tmp_path,
        "report-validator1-freeze",
        validator1_inputs,
        "Target Validator",
        validator1_events,
        key,
        config,
    )
    stage2_root = tmp_path / "report-target-stage2"
    release_target_validator_stage2(
        stage2_root,
        private_target_rows=load_jsonl(root / "scorer_private" / "target_material.jsonl"),
        mapping_rows=load_jsonl(root / "scorer_private" / "item_mapping.jsonl"),
        author_input_rows=author_inputs,
        author_frozen_rows=author_frozen,
        validator_input_rows=validator1_inputs,
        validator_frozen_rows=validator1_frozen,
        expected_items=2,
        key=key,
        config=config,
    )
    validator2_inputs = load_jsonl(stage2_root / "items.jsonl")

    def accept(_row):
        return {
            "review_decision": "accept",
            "edited_canonical_source_event_en": "",
            "edited_acceptable_target_realizations_zh": [],
            "edited_forbidden_target_realizations_zh": [],
            "reason_codes": [],
            "annotation_note": "",
        }

    validator2_events = complete_events(
        validator2_inputs, "Target Validator", key, config, accept
    )
    validator2_frozen = freeze(
        tmp_path,
        "report-validator2-freeze",
        validator2_inputs,
        "Target Validator",
        validator2_events,
        key,
        config,
    )
    return {
        "target_author_inputs": author_inputs,
        "target_author_frozen": author_frozen,
        "target_validator_stage1_inputs": validator1_inputs,
        "target_validator_stage1_frozen": validator1_frozen,
        "target_validator_stage2_inputs": validator2_inputs,
        "target_validator_stage2_frozen": validator2_frozen,
    }


def test_pre_adjudication_report_uses_raw_freezes_and_passes_perfect_instrument(tmp_path):
    root, config, key = workspace(tmp_path)
    config = copy.deepcopy(config)
    config["reliability"]["bootstrap_samples"] = 50
    visual_inputs, visual_frozen = full_visual_chain(tmp_path, root, config, key)
    target = full_target_chain(tmp_path, root, config, key)
    rows, summary = build_pre_adjudication_report(
        visual_inputs=visual_inputs,
        visual_frozen=visual_frozen,
        mapping_rows=load_jsonl(root / "scorer_private" / "item_mapping.jsonl"),
        expected_items=2,
        key=key,
        config=config,
        **target,
    )
    assert summary["instrument_gate_passed"] is True
    assert summary["status"] == "PASS_ADJUDICATION_MAY_BEGIN"
    assert summary["adjudication_rate"] == 0.0
    assert summary["pre_adjudication_composite_exact_agreement"] == 1.0
    assert all(metric["exact_agreement"]["value"] == 1.0 for metric in summary["metrics"].values())
    assert all(row["adjudication_applied"] is False for row in rows)
    assert all(row["final_candidate_status"] is None for row in rows)
    json.dumps(summary)


def test_report_rejects_rehashed_but_wrong_cross_cohort_predecessor_lock(tmp_path):
    root, config, key = workspace(tmp_path)
    config = copy.deepcopy(config)
    config["reliability"]["bootstrap_samples"] = 10
    visual_inputs, visual_frozen = full_visual_chain(tmp_path, root, config, key)
    target = full_target_chain(tmp_path, root, config, key)
    changed = copy.deepcopy(visual_inputs)
    changed["r1"]["visual_a"][0]["prior_cohort_lock_sha256"] = "f" * 64
    from scripts.build_mcif_visual_token_controls import canonical_sha256

    changed["r1"]["visual_a"][0]["row_sha256"] = canonical_sha256(
        {
            key_name: value
            for key_name, value in changed["r1"]["visual_a"][0].items()
            if key_name != "row_sha256"
        }
    )
    with pytest.raises(ValueError, match="binding/signature differs|predecessor lock differs"):
        build_pre_adjudication_report(
            visual_inputs=changed,
            visual_frozen=visual_frozen,
            mapping_rows=load_jsonl(root / "scorer_private" / "item_mapping.jsonl"),
            expected_items=2,
            key=key,
            config=config,
            **target,
        )


def test_adjudication_is_blocked_before_gate_and_does_not_recompute_raw_metrics(tmp_path):
    root, config, key = workspace(tmp_path)
    config = copy.deepcopy(config)
    config["reliability"]["bootstrap_samples"] = 20
    visual_inputs, visual_frozen = full_visual_chain(
        tmp_path, root, config, key, disagreement=True
    )
    target = full_target_chain(tmp_path, root, config, key)
    rows, failed = build_pre_adjudication_report(
        visual_inputs=visual_inputs,
        visual_frozen=visual_frozen,
        mapping_rows=load_jsonl(root / "scorer_private" / "item_mapping.jsonl"),
        expected_items=2,
        key=key,
        config=config,
        **target,
    )
    assert failed["instrument_gate_passed"] is False
    with pytest.raises(ValueError, match="failed instrument cannot enter adjudication"):
        prepare_adjudication_release(
            tmp_path / "blocked-adjudication",
            pre_adjudication_rows=rows,
            pre_adjudication_summary=failed,
            private_visual_rows=load_jsonl(root / "scorer_private" / "visual_material.jsonl"),
            mapping_rows=load_jsonl(root / "scorer_private" / "item_mapping.jsonl"),
            target_validator_stage2_inputs=target["target_validator_stage2_inputs"],
            workspace_root=root,
            visual_adjudicator_id="Visual Adjudicator",
            target_adjudicator_id="Target Adjudicator",
            expected_items=2,
        )

    permissive = copy.deepcopy(config)
    permissive["reliability"]["minimum_exact_agreement"] = 0.0
    permissive["reliability"]["minimum_gwet_ac1"] = -1.0
    permissive["reliability"]["maximum_adjudication_rate"] = 1.0
    rows, passed = build_pre_adjudication_report(
        visual_inputs=visual_inputs,
        visual_frozen=visual_frozen,
        mapping_rows=load_jsonl(root / "scorer_private" / "item_mapping.jsonl"),
        expected_items=2,
        key=key,
        config=permissive,
        **target,
    )
    assert passed["instrument_gate_passed"] is True
    adjudication_root = tmp_path / "adjudication"
    release = prepare_adjudication_release(
        adjudication_root,
        pre_adjudication_rows=rows,
        pre_adjudication_summary=passed,
        private_visual_rows=load_jsonl(root / "scorer_private" / "visual_material.jsonl"),
        mapping_rows=load_jsonl(root / "scorer_private" / "item_mapping.jsonl"),
        target_validator_stage2_inputs=target["target_validator_stage2_inputs"],
        workspace_root=root,
        visual_adjudicator_id="Visual Adjudicator",
        target_adjudicator_id="Target Adjudicator",
        expected_items=2,
    )
    assert release["visual_items"] == 1
    assert release["target_items"] == 0
    visual_items = load_jsonl(adjudication_root / "visual_adjudicator_view" / "items.jsonl")

    def adjudicate(_row):
        return {
            "adjudicated_judgment": "no",
            "reason_codes": ["ambiguous_visual"],
            "annotation_note": "resolved from R0 only",
        }

    adjudication_events = complete_events(
        visual_items,
        "Visual Adjudicator",
        key,
        permissive,
        adjudicate,
    )
    visual_adjudication_frozen = freeze(
        tmp_path,
        "visual-adjudication-freeze",
        visual_items,
        "Visual Adjudicator",
        adjudication_events,
        key,
        permissive,
    )
    applied, applied_summary = apply_adjudications(
        pre_adjudication_rows=rows,
        pre_adjudication_summary=passed,
        visual_adjudication_inputs=visual_items,
        visual_adjudication_frozen=visual_adjudication_frozen,
        target_adjudication_inputs=[],
        target_adjudication_frozen=[],
        key=key,
        config=permissive,
    )
    assert applied_summary["metrics"] == passed["metrics"]
    assert applied_summary["raw_metrics_recomputed"] is False
    assert all(row["adjudication_applied"] is True for row in applied)
    assert all(row["raw_row_sha256"] in {raw["row_sha256"] for raw in rows} for row in applied)
