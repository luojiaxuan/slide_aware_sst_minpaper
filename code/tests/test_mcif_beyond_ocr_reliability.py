import copy
import json
from pathlib import Path

import pytest
from scripts.build_mcif_beyond_ocr_reliability_workspace import build_bundle
from scripts.build_mcif_visual_token_controls import canonical_sha256, load_jsonl
from scripts.mcif_beyond_ocr_reliability import (
    append_annotation_event,
    append_event_log,
    apply_adjudications,
    blank_annotation,
    build_event_head_ledger,
    build_pre_adjudication_report,
    create_hmac_key,
    freeze_annotations,
    initialize_event_log,
    initialize_events,
    load_config,
    prepare_adjudication_release,
    release_target_validator_stage2,
    release_visual_stage,
    resolve_private_media,
    signed_payload,
    validate_event_log,
    validate_identity_registry,
    validate_input_rows,
)
from test_build_mcif_beyond_ocr_reliability_workspace import (
    build_kwargs,
    identity_registry_fixture,
)

CONFIG_PATH = (
    Path(__file__).parents[1] / "configs" / "mcif_beyond_ocr_reliability_v2.json"
)


def identity_registry():
    return identity_registry_fixture()


def workspace(tmp_path: Path, *, config_override=None):
    kwargs = build_kwargs(tmp_path)
    if config_override is not None:
        kwargs["config"] = config_override
    root = tmp_path / "v2"
    build_bundle(root, **kwargs)
    config = kwargs["config"]
    return root, config, kwargs["release_key"]


def run_contract(root: Path):
    return json.loads(
        (root / "scorer_private" / "run_contract.json").read_text(encoding="utf-8")
    )


def complete_events(rows, annotator_id, key, config, annotation_factory):
    events = initialize_events(
        rows,
        annotator_id=annotator_id,
        expected_items=len(rows),
        key=key,
        run_contract_sha256=rows[0]["run_contract_sha256"],
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
            run_contract_sha256=rows[0]["run_contract_sha256"],
        )
    return events


def freeze(tmp_path, name, rows, annotator_id, events, key, config, contract):
    root = tmp_path / name
    head_checkpoints = build_event_head_ledger(
        events,
        rows,
        annotator_id=annotator_id,
        expected_items=len(rows),
        key=key,
        config=config,
        run_contract_sha256=rows[0]["run_contract_sha256"],
    )
    freeze_annotations(
        root,
        input_rows=rows,
        events=events,
        head_checkpoints=head_checkpoints,
        annotator_id=annotator_id,
        expected_items=len(rows),
        locked_at_utc="2026-08-01T21:00:00Z",
        key=key,
        config=config,
        identity_registry=identity_registry(),
        run_contract=contract,
    )
    return load_jsonl(root / "frozen_annotations.jsonl")


def visual_annotation(row, value="yes"):
    annotation = blank_annotation(row["role"], row["stage"])
    annotation[
        next(
            key
            for key in annotation
            if key.endswith("support") or key == "descriptor_fidelity"
        )
    ] = value
    if value != "yes":
        annotation["reason_codes"] = ["ambiguous_visual"]
    return annotation


def test_event_chain_rejects_stale_tampered_and_completed_overwrite(tmp_path):
    root, config, key = workspace(tmp_path)
    rows = load_jsonl(root / "visual_a_r0_view" / "items.jsonl")
    events = initialize_events(
        rows,
        annotator_id="Visual A",
        expected_items=2,
        key=key,
        run_contract_sha256=rows[0]["run_contract_sha256"],
    )
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
        run_contract_sha256=rows[0]["run_contract_sha256"],
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
            run_contract_sha256=rows[0]["run_contract_sha256"],
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
            run_contract_sha256=rows[0]["run_contract_sha256"],
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
            run_contract_sha256=rows[0]["run_contract_sha256"],
        )


def test_event_log_initialization_is_exclusive_and_append_uses_disk_cas(tmp_path):
    root, config, key = workspace(tmp_path)
    rows = load_jsonl(root / "visual_a_r0_view" / "items.jsonl")
    event_log = tmp_path / "events.jsonl"
    head_ledger = tmp_path / "events.heads.jsonl"
    initialize_event_log(
        event_log,
        head_ledger,
        rows,
        annotator_id="Visual A",
        expected_items=2,
        key=key,
        config=config,
        run_contract_sha256=rows[0]["run_contract_sha256"],
    )
    append_event_log(
        event_log,
        head_ledger,
        rows,
        item_id=rows[0]["item_id"],
        expected_event_index=0,
        annotation_status="completed",
        annotation=visual_annotation(rows[0]),
        submitted_at_utc="2026-08-01T20:00:00Z",
        annotator_id="Visual A",
        expected_items=2,
        key=key,
        config=config,
        run_contract_sha256=rows[0]["run_contract_sha256"],
    )
    completed_bytes = event_log.read_bytes()
    with pytest.raises(FileExistsError, match="must not already exist"):
        initialize_event_log(
            event_log,
            head_ledger,
            rows,
            annotator_id="Visual A",
            expected_items=2,
            key=key,
            config=config,
            run_contract_sha256=rows[0]["run_contract_sha256"],
        )
    assert event_log.read_bytes() == completed_bytes
    with pytest.raises(ValueError, match="stale event version"):
        append_event_log(
            event_log,
            head_ledger,
            rows,
            item_id=rows[0]["item_id"],
            expected_event_index=0,
            annotation_status="draft",
            annotation=visual_annotation(rows[0]),
            submitted_at_utc="2026-08-01T20:01:00Z",
            annotator_id="Visual A",
            expected_items=2,
            key=key,
            config=config,
            run_contract_sha256=rows[0]["run_contract_sha256"],
        )


def test_event_head_ledger_rejects_signed_prefix_rollback(tmp_path):
    root, config, key = workspace(tmp_path)
    rows = load_jsonl(root / "visual_a_r0_view" / "items.jsonl")
    event_log = tmp_path / "rollback.events.jsonl"
    head_ledger = tmp_path / "rollback.event-heads.jsonl"
    initialize_event_log(
        event_log,
        head_ledger,
        rows,
        annotator_id="Visual A",
        expected_items=2,
        key=key,
        config=config,
        run_contract_sha256=rows[0]["run_contract_sha256"],
    )
    signed_initial_prefix = event_log.read_bytes()
    append_event_log(
        event_log,
        head_ledger,
        rows,
        item_id=rows[0]["item_id"],
        expected_event_index=0,
        annotation_status="completed",
        annotation=visual_annotation(rows[0]),
        submitted_at_utc="2026-08-01T20:00:00Z",
        annotator_id="Visual A",
        expected_items=2,
        key=key,
        config=config,
        run_contract_sha256=rows[0]["run_contract_sha256"],
    )
    assert len(load_jsonl(head_ledger)) == 2

    event_log.write_bytes(signed_initial_prefix)
    with pytest.raises(ValueError, match="event-head checkpoint differs"):
        append_event_log(
            event_log,
            head_ledger,
            rows,
            item_id=rows[0]["item_id"],
            expected_event_index=0,
            annotation_status="completed",
            annotation=visual_annotation(rows[0], "no"),
            submitted_at_utc="2026-08-01T20:01:00Z",
            annotator_id="Visual A",
            expected_items=2,
            key=key,
            config=config,
            run_contract_sha256=rows[0]["run_contract_sha256"],
        )
    with pytest.raises(ValueError, match="event-head checkpoint differs"):
        freeze_annotations(
            tmp_path / "rollback-freeze",
            input_rows=rows,
            events=load_jsonl(event_log),
            head_checkpoints=load_jsonl(head_ledger),
            annotator_id="Visual A",
            expected_items=2,
            locked_at_utc="2026-08-01T21:00:00Z",
            key=key,
            config=config,
            identity_registry=identity_registry(),
            run_contract=run_contract(root),
        )


def test_release_signature_and_semantic_firewall_reject_rehashed_stimulus_or_label(
    tmp_path,
):
    root, _, key = workspace(tmp_path)
    rows = load_jsonl(root / "target_validator_stage1_view" / "items.jsonl")
    prefilled = copy.deepcopy(rows)
    prefilled[0]["candidate_eligibility"] = "yes"
    prefilled[0]["row_sha256"] = canonical_sha256(
        {
            name: value
            for name, value in prefilled[0].items()
            if name not in {"row_sha256", "release_hmac_sha256"}
        }
    )
    with pytest.raises(ValueError, match="response fields are not blank"):
        validate_input_rows(
            prefilled,
            2,
            key=key,
            run_contract_sha256=rows[0]["run_contract_sha256"],
        )

    tampered = copy.deepcopy(rows)
    tampered[0]["target_reference_zh"] = "篡改参考。"
    tampered[0]["row_sha256"] = canonical_sha256(
        {
            name: value
            for name, value in tampered[0].items()
            if name not in {"row_sha256", "release_hmac_sha256"}
        }
    )
    with pytest.raises(ValueError, match="release signature differs"):
        validate_input_rows(
            tampered,
            2,
            key=key,
            run_contract_sha256=rows[0]["run_contract_sha256"],
        )


def test_next_visual_stage_requires_both_complete_disjoint_full_cohorts(tmp_path):
    root, config, key = workspace(tmp_path)
    contract = run_contract(root)
    rows_a = load_jsonl(root / "visual_a_r0_view" / "items.jsonl")
    rows_b = load_jsonl(root / "visual_b_r0_view" / "items.jsonl")
    events_a = complete_events(rows_a, "Visual A", key, config, visual_annotation)
    events_b = complete_events(rows_b, "Visual B", key, config, visual_annotation)
    frozen_a = freeze(
        tmp_path, "freeze-a", rows_a, "Visual A", events_a, key, config, contract
    )
    frozen_b = freeze(
        tmp_path, "freeze-b", rows_b, "Visual B", events_b, key, config, contract
    )
    output = tmp_path / "r1-release"
    report = release_visual_stage(
        output,
        workspace_root=root,
        private_visual_rows=load_jsonl(
            root / "scorer_private" / "visual_material.jsonl"
        ),
        mapping_rows=load_jsonl(root / "scorer_private" / "item_mapping.jsonl"),
        prior_input_a=rows_a,
        prior_input_b=rows_b,
        prior_frozen_a=frozen_a,
        prior_frozen_b=frozen_b,
        next_stage="r1",
        expected_items=2,
        key=key,
        config=config,
        run_contract=contract,
    )
    released_a = load_jsonl(output / "visual_a_r1_view" / "items.jsonl")
    released_b = load_jsonl(output / "visual_b_r1_view" / "items.jsonl")
    assert report["items_per_cohort"] == 2
    assert len(released_a) == len(released_b) == 2
    assert all(
        "r1_blocks" in row and "current_slide" not in row
        for row in released_a + released_b
    )
    assert all(
        row["locked_judgments"] == {"r0_support": "yes"}
        for row in released_a + released_b
    )
    assert all(
        row["prior_cohort_lock_sha256"] == report["prior_cohort_lock_sha256"]
        for row in released_a + released_b
    )

    private_visual = load_jsonl(root / "scorer_private" / "visual_material.jsonl")
    smuggled = copy.deepcopy(private_visual)
    smuggled[0]["r1_blocks"][0]["target_reference_zh"] = "不得泄漏"
    smuggled[0]["row_sha256"] = canonical_sha256(
        {name: value for name, value in smuggled[0].items() if name != "row_sha256"}
    )
    with pytest.raises(ValueError, match="private visual material contract differs"):
        release_visual_stage(
            tmp_path / "smuggled-r1-blocked",
            workspace_root=root,
            private_visual_rows=smuggled,
            mapping_rows=load_jsonl(root / "scorer_private" / "item_mapping.jsonl"),
            prior_input_a=rows_a,
            prior_input_b=rows_b,
            prior_frozen_a=frozen_a,
            prior_frozen_b=frozen_b,
            next_stage="r1",
            expected_items=2,
            key=key,
            config=config,
            run_contract=contract,
        )

    traversal = copy.deepcopy(private_visual[0]["private_media"])
    traversal["private_path"] = "media/../../target_reference.json"
    with pytest.raises(ValueError, match="private media path differs"):
        resolve_private_media(root, traversal)

    incomplete_b = frozen_b[:-1]
    with pytest.raises(ValueError, match="frozen item count differs"):
        release_visual_stage(
            tmp_path / "blocked",
            workspace_root=root,
            private_visual_rows=load_jsonl(
                root / "scorer_private" / "visual_material.jsonl"
            ),
            mapping_rows=load_jsonl(root / "scorer_private" / "item_mapping.jsonl"),
            prior_input_a=rows_a,
            prior_input_b=rows_b,
            prior_frozen_a=frozen_a,
            prior_frozen_b=incomplete_b,
            next_stage="r1",
            expected_items=2,
            key=key,
            config=config,
            run_contract=contract,
        )
    conflict_b = complete_events(
        rows_b, "visual-a@example.test", key, config, visual_annotation
    )
    with pytest.raises(ValueError, match="annotator differs from identity registry"):
        freeze(
            tmp_path,
            "freeze-conflict",
            rows_b,
            "visual-a@example.test",
            conflict_b,
            key,
            config,
            contract,
        )

    mixed_a = copy.deepcopy(frozen_a)
    mixed_payload = {
        key_name: value
        for key_name, value in mixed_a[0].items()
        if key_name != "freeze_hmac_sha256"
    }
    mixed_payload["annotator_id"] = "Visual A Substitute"
    mixed_a[0] = signed_payload(mixed_payload, key, "freeze_hmac_sha256")
    with pytest.raises(ValueError, match="visual A prior stage must use one annotator"):
        release_visual_stage(
            tmp_path / "mixed-identity-blocked",
            workspace_root=root,
            private_visual_rows=load_jsonl(
                root / "scorer_private" / "visual_material.jsonl"
            ),
            mapping_rows=load_jsonl(root / "scorer_private" / "item_mapping.jsonl"),
            prior_input_a=rows_a,
            prior_input_b=rows_b,
            prior_frozen_a=mixed_a,
            prior_frozen_b=frozen_b,
            next_stage="r1",
            expected_items=2,
            key=key,
            config=config,
            run_contract=contract,
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
    contract = run_contract(root)
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
        contract,
    )
    validator_frozen = freeze(
        tmp_path,
        "validator-freeze",
        validator_rows,
        "Target Validator",
        validator_events,
        key,
        config,
        contract,
    )
    output = tmp_path / "target-stage2"
    report = release_target_validator_stage2(
        output,
        private_target_rows=load_jsonl(
            root / "scorer_private" / "target_material.jsonl"
        ),
        mapping_rows=load_jsonl(root / "scorer_private" / "item_mapping.jsonl"),
        author_input_rows=author_rows,
        author_frozen_rows=author_frozen,
        validator_input_rows=validator_rows,
        validator_frozen_rows=validator_frozen,
        expected_items=2,
        key=key,
        config=config,
        run_contract=contract,
    )
    released = load_jsonl(output / "items.jsonl")
    assert report["items"] == 2
    assert all(
        row["author_canonical_source_event_en"] == "candidate event" for row in released
    )
    assert all(row["locked_candidate_eligibility"] == "yes" for row in released)
    assert all(row["author_identity_exposed"] is False for row in released)
    assert all("author_id" not in row for row in released)

    mixed_author = copy.deepcopy(author_frozen)
    mixed_payload = {
        key_name: value
        for key_name, value in mixed_author[0].items()
        if key_name != "freeze_hmac_sha256"
    }
    mixed_payload["annotator_id"] = "Target Author Substitute"
    mixed_author[0] = signed_payload(mixed_payload, key, "freeze_hmac_sha256")
    with pytest.raises(ValueError, match="target author must use one annotator"):
        release_target_validator_stage2(
            tmp_path / "target-mixed-identity-blocked",
            private_target_rows=load_jsonl(
                root / "scorer_private" / "target_material.jsonl"
            ),
            mapping_rows=load_jsonl(root / "scorer_private" / "item_mapping.jsonl"),
            author_input_rows=author_rows,
            author_frozen_rows=mixed_author,
            validator_input_rows=validator_rows,
            validator_frozen_rows=validator_frozen,
            expected_items=2,
            key=key,
            config=config,
            run_contract=contract,
        )


def test_hmac_key_is_create_once_private_and_not_json(tmp_path):
    path = tmp_path / "keys" / "annotation.key"
    create_hmac_key(path)
    assert len(path.read_bytes()) == 32
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    with pytest.raises(FileExistsError, match="must not already exist"):
        create_hmac_key(path)
    with pytest.raises((UnicodeDecodeError, json.JSONDecodeError)):
        json.loads(path.read_text())


def test_identity_registry_rejects_alias_and_role_identity_collisions(tmp_path):
    _, config, _ = workspace(tmp_path)
    registry = identity_registry()
    assert validate_identity_registry(registry, config) == registry["role_assignments"]

    alias_collision = copy.deepcopy(registry)
    alias_collision["people"][1]["aliases"] = [
        alias_collision["people"][0]["aliases"][0]
    ]
    alias_collision["registry_sha256"] = canonical_sha256(
        {
            name: value
            for name, value in alias_collision.items()
            if name != "registry_sha256"
        }
    )
    with pytest.raises(ValueError, match="alias maps to multiple people"):
        validate_identity_registry(alias_collision, config)

    role_collision = copy.deepcopy(registry)
    role_collision["role_assignments"]["visual_b"] = role_collision["role_assignments"][
        "visual_a"
    ]
    role_collision["registry_sha256"] = canonical_sha256(
        {
            name: value
            for name, value in role_collision.items()
            if name != "registry_sha256"
        }
    )
    with pytest.raises(ValueError, match="roles must be disjoint"):
        validate_identity_registry(role_collision, config)


def test_run_contract_rejects_post_event_identity_registry_swap(tmp_path):
    root, config, key = workspace(tmp_path)
    contract = run_contract(root)
    rows = load_jsonl(root / "visual_a_r0_view" / "items.jsonl")
    events = complete_events(rows, "Visual A", key, config, visual_annotation)
    head_checkpoints = build_event_head_ledger(
        events,
        rows,
        annotator_id="Visual A",
        expected_items=2,
        key=key,
        config=config,
        run_contract_sha256=rows[0]["run_contract_sha256"],
    )
    swapped = copy.deepcopy(identity_registry())
    swapped["people"][0]["aliases"].append("new-alias@example.test")
    swapped["registry_sha256"] = canonical_sha256(
        {name: value for name, value in swapped.items() if name != "registry_sha256"}
    )
    with pytest.raises(ValueError, match="run contract registry differs"):
        freeze_annotations(
            tmp_path / "registry-swap-blocked",
            input_rows=rows,
            events=events,
            head_checkpoints=head_checkpoints,
            annotator_id="Visual A",
            expected_items=2,
            locked_at_utc="2026-08-01T21:00:00Z",
            key=key,
            config=config,
            identity_registry=swapped,
            run_contract=contract,
        )


def full_visual_chain(tmp_path, root, config, key, *, disagreement=False):
    contract = run_contract(root)
    inputs = {"r0": {}}
    frozen = {"r0": {}}
    for role, annotator in (("visual_a", "Visual A"), ("visual_b", "Visual B")):
        rows = load_jsonl(root / f"{role}_r0_view" / "items.jsonl")
        inputs["r0"][role] = rows
        disagreement_id = (
            rows[0]["item_id"] if disagreement and role == "visual_b" else None
        )
        events = complete_events(
            rows,
            annotator,
            key,
            config,
            lambda row, disagreement_id=disagreement_id: visual_annotation(
                row, "uncertain" if row["item_id"] == disagreement_id else "no"
            ),
        )
        frozen["r0"][role] = freeze(
            tmp_path,
            f"{role}-r0-freeze",
            rows,
            annotator,
            events,
            key,
            config,
            contract,
        )
    for stage in ("r1", "pixels", "descriptor"):
        previous = {"r1": "r0", "pixels": "r1", "descriptor": "pixels"}[stage]
        release_root = tmp_path / f"{stage}-release"
        release_visual_stage(
            release_root,
            workspace_root=root,
            private_visual_rows=load_jsonl(
                root / "scorer_private" / "visual_material.jsonl"
            ),
            mapping_rows=load_jsonl(root / "scorer_private" / "item_mapping.jsonl"),
            prior_input_a=inputs[previous]["visual_a"],
            prior_input_b=inputs[previous]["visual_b"],
            prior_frozen_a=frozen[previous]["visual_a"],
            prior_frozen_b=frozen[previous]["visual_b"],
            next_stage=stage,
            expected_items=2,
            key=key,
            config=config,
            run_contract=contract,
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
                contract,
            )
    return inputs, frozen


def full_target_chain(tmp_path, root, config, key):
    contract = run_contract(root)
    author_inputs = load_jsonl(root / "target_author_view" / "items.jsonl")
    validator1_inputs = load_jsonl(
        root / "target_validator_stage1_view" / "items.jsonl"
    )
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
        contract,
    )
    validator1_frozen = freeze(
        tmp_path,
        "report-validator1-freeze",
        validator1_inputs,
        "Target Validator",
        validator1_events,
        key,
        config,
        contract,
    )
    stage2_root = tmp_path / "report-target-stage2"
    release_target_validator_stage2(
        stage2_root,
        private_target_rows=load_jsonl(
            root / "scorer_private" / "target_material.jsonl"
        ),
        mapping_rows=load_jsonl(root / "scorer_private" / "item_mapping.jsonl"),
        author_input_rows=author_inputs,
        author_frozen_rows=author_frozen,
        validator_input_rows=validator1_inputs,
        validator_frozen_rows=validator1_frozen,
        expected_items=2,
        key=key,
        config=config,
        run_contract=contract,
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
        contract,
    )
    return {
        "target_author_inputs": author_inputs,
        "target_author_frozen": author_frozen,
        "target_validator_stage1_inputs": validator1_inputs,
        "target_validator_stage1_frozen": validator1_frozen,
        "target_validator_stage2_inputs": validator2_inputs,
        "target_validator_stage2_frozen": validator2_frozen,
    }


def test_pre_adjudication_report_uses_raw_freezes_and_passes_perfect_instrument(
    tmp_path,
):
    config = copy.deepcopy(load_config(CONFIG_PATH))
    config["reliability"]["bootstrap_samples"] = 50
    root, config, key = workspace(tmp_path, config_override=config)
    contract = run_contract(root)
    visual_inputs, visual_frozen = full_visual_chain(tmp_path, root, config, key)
    target = full_target_chain(tmp_path, root, config, key)
    rows, summary = build_pre_adjudication_report(
        visual_inputs=visual_inputs,
        visual_frozen=visual_frozen,
        mapping_rows=load_jsonl(root / "scorer_private" / "item_mapping.jsonl"),
        expected_items=2,
        key=key,
        config=config,
        run_contract=contract,
        **target,
    )
    assert summary["instrument_gate_passed"] is True
    assert summary["status"] == "PASS_ADJUDICATION_MAY_BEGIN"
    assert summary["adjudication_rate"] == 0.0
    assert summary["pre_adjudication_composite_exact_agreement"] == 1.0
    assert all(
        metric["exact_agreement"]["value"] == 1.0
        for metric in summary["metrics"].values()
    )
    assert all(row["adjudication_applied"] is False for row in rows)
    assert all(row["final_candidate_status"] is None for row in rows)
    json.dumps(summary)


def test_report_rejects_rehashed_but_wrong_cross_cohort_predecessor_lock(tmp_path):
    config = copy.deepcopy(load_config(CONFIG_PATH))
    config["reliability"]["bootstrap_samples"] = 10
    root, config, key = workspace(tmp_path, config_override=config)
    contract = run_contract(root)
    visual_inputs, visual_frozen = full_visual_chain(tmp_path, root, config, key)
    target = full_target_chain(tmp_path, root, config, key)
    changed = copy.deepcopy(visual_inputs)
    changed["r1"]["visual_a"][0]["prior_cohort_lock_sha256"] = "f" * 64
    changed["r1"]["visual_a"][0]["row_sha256"] = canonical_sha256(
        {
            key_name: value
            for key_name, value in changed["r1"]["visual_a"][0].items()
            if key_name not in {"row_sha256", "release_hmac_sha256"}
        }
    )
    with pytest.raises(
        ValueError,
        match="release signature differs|binding/signature differs|predecessor lock differs",
    ):
        build_pre_adjudication_report(
            visual_inputs=changed,
            visual_frozen=visual_frozen,
            mapping_rows=load_jsonl(root / "scorer_private" / "item_mapping.jsonl"),
            expected_items=2,
            key=key,
            config=config,
            run_contract=contract,
            **target,
        )


def test_adjudication_is_blocked_before_gate_and_does_not_recompute_raw_metrics(
    tmp_path,
):
    config = copy.deepcopy(load_config(CONFIG_PATH))
    config["reliability"]["bootstrap_samples"] = 20
    root, config, key = workspace(tmp_path, config_override=config)
    contract = run_contract(root)
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
        run_contract=contract,
        **target,
    )
    assert failed["instrument_gate_passed"] is False
    forged = copy.deepcopy(failed)
    forged["instrument_gate_passed"] = True
    forged["status"] = "PASS_ADJUDICATION_MAY_BEGIN"
    with pytest.raises(ValueError, match="report binding differs"):
        prepare_adjudication_release(
            tmp_path / "forged-gate-blocked",
            pre_adjudication_rows=rows,
            pre_adjudication_summary=forged,
            private_visual_rows=load_jsonl(
                root / "scorer_private" / "visual_material.jsonl"
            ),
            mapping_rows=load_jsonl(root / "scorer_private" / "item_mapping.jsonl"),
            target_validator_stage2_inputs=target["target_validator_stage2_inputs"],
            workspace_root=root,
            visual_adjudicator_id="Visual Adjudicator",
            target_adjudicator_id="Target Adjudicator",
            expected_items=2,
            key=key,
            config=config,
            identity_registry=identity_registry(),
            run_contract=contract,
        )
    with pytest.raises(ValueError, match="failed instrument cannot enter adjudication"):
        prepare_adjudication_release(
            tmp_path / "blocked-adjudication",
            pre_adjudication_rows=rows,
            pre_adjudication_summary=failed,
            private_visual_rows=load_jsonl(
                root / "scorer_private" / "visual_material.jsonl"
            ),
            mapping_rows=load_jsonl(root / "scorer_private" / "item_mapping.jsonl"),
            target_validator_stage2_inputs=target["target_validator_stage2_inputs"],
            workspace_root=root,
            visual_adjudicator_id="Visual Adjudicator",
            target_adjudicator_id="Target Adjudicator",
            expected_items=2,
            key=key,
            config=config,
            identity_registry=identity_registry(),
            run_contract=contract,
        )

    permissive = copy.deepcopy(config)
    permissive["reliability"]["minimum_exact_agreement"] = 0.0
    permissive["reliability"]["minimum_gwet_ac1"] = -1.0
    permissive["reliability"]["maximum_adjudication_rate"] = 1.0
    with pytest.raises(ValueError, match="run contract binding differs"):
        build_pre_adjudication_report(
            visual_inputs=visual_inputs,
            visual_frozen=visual_frozen,
            mapping_rows=load_jsonl(root / "scorer_private" / "item_mapping.jsonl"),
            expected_items=2,
            key=key,
            config=permissive,
            run_contract=contract,
            **target,
        )

    permissive_workspace = tmp_path / "permissive-workspace"
    permissive_root, permissive, permissive_key = workspace(
        permissive_workspace, config_override=permissive
    )
    permissive_contract = run_contract(permissive_root)
    permissive_run = tmp_path / "permissive-run"
    permissive_visual_inputs, permissive_visual_frozen = full_visual_chain(
        permissive_run,
        permissive_root,
        permissive,
        permissive_key,
        disagreement=True,
    )
    permissive_target = full_target_chain(
        permissive_run, permissive_root, permissive, permissive_key
    )
    rows, passed = build_pre_adjudication_report(
        visual_inputs=permissive_visual_inputs,
        visual_frozen=permissive_visual_frozen,
        mapping_rows=load_jsonl(
            permissive_root / "scorer_private" / "item_mapping.jsonl"
        ),
        expected_items=2,
        key=permissive_key,
        config=permissive,
        run_contract=permissive_contract,
        **permissive_target,
    )
    assert passed["instrument_gate_passed"] is True
    adjudication_root = tmp_path / "adjudication"
    release = prepare_adjudication_release(
        adjudication_root,
        pre_adjudication_rows=rows,
        pre_adjudication_summary=passed,
        private_visual_rows=load_jsonl(
            permissive_root / "scorer_private" / "visual_material.jsonl"
        ),
        mapping_rows=load_jsonl(
            permissive_root / "scorer_private" / "item_mapping.jsonl"
        ),
        target_validator_stage2_inputs=permissive_target[
            "target_validator_stage2_inputs"
        ],
        workspace_root=permissive_root,
        visual_adjudicator_id="Visual Adjudicator",
        target_adjudicator_id="Target Adjudicator",
        expected_items=2,
        key=permissive_key,
        config=permissive,
        identity_registry=identity_registry(),
        run_contract=permissive_contract,
    )
    assert release["visual_items"] == 1
    assert release["target_items"] == 0
    visual_items = load_jsonl(
        adjudication_root / "visual_adjudicator_view" / "items.jsonl"
    )

    def adjudicate(_row):
        return {
            "adjudicated_judgment": "no",
            "reason_codes": ["ambiguous_visual"],
            "annotation_note": "resolved from R0 only",
        }

    adjudication_events = complete_events(
        visual_items,
        "Visual Adjudicator",
        permissive_key,
        permissive,
        adjudicate,
    )
    visual_adjudication_frozen = freeze(
        tmp_path,
        "visual-adjudication-freeze",
        visual_items,
        "Visual Adjudicator",
        adjudication_events,
        permissive_key,
        permissive,
        permissive_contract,
    )
    rebound_payload = {
        name: value
        for name, value in release.items()
        if name != "release_report_hmac_sha256"
    }
    rebound_payload["mapping_rows_sha256"] = "f" * 64
    rebound_release = signed_payload(
        rebound_payload, permissive_key, "release_report_hmac_sha256"
    )
    with pytest.raises(ValueError, match="adjudication release report differs"):
        apply_adjudications(
            pre_adjudication_rows=rows,
            pre_adjudication_summary=passed,
            adjudication_release_report=rebound_release,
            visual_adjudication_inputs=visual_items,
            visual_adjudication_frozen=visual_adjudication_frozen,
            target_adjudication_inputs=[],
            target_adjudication_frozen=[],
            key=permissive_key,
            config=permissive,
            identity_registry=identity_registry(),
            run_contract=permissive_contract,
        )
    applied, applied_summary = apply_adjudications(
        pre_adjudication_rows=rows,
        pre_adjudication_summary=passed,
        adjudication_release_report=release,
        visual_adjudication_inputs=visual_items,
        visual_adjudication_frozen=visual_adjudication_frozen,
        target_adjudication_inputs=[],
        target_adjudication_frozen=[],
        key=permissive_key,
        config=permissive,
        identity_registry=identity_registry(),
        run_contract=permissive_contract,
    )
    assert applied_summary["metrics"] == passed["metrics"]
    assert applied_summary["raw_metrics_recomputed"] is False
    assert all(row["adjudication_applied"] is True for row in applied)
    assert all(
        row["raw_row_sha256"] in {raw["row_sha256"] for raw in rows} for row in applied
    )
