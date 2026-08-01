import hashlib
import json
import os
import socket
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.score_event_trajectories import git_head, validate_config, validate_target_is_protected

from slidesst.eval.event_timing import (
    CausalAudioBrokerAudit,
    CausalAudioReleaseLog,
    CausalAudioSchedule,
    ControlPairSpec,
    EvidencePacketPayload,
    EvidencePacketSpec,
    EventScoringConfig,
    EventTrajectory,
    InferenceContract,
    InferenceEnvironmentAudit,
    InferenceResultAttestation,
    InferenceScientificConfig,
    SourceEventTiming,
    TargetEventSpec,
    TrajectoryObservation,
    apply_development_gate,
    causal_observation_sha256,
    canonical_json_sha256,
    directory_tree_sha256,
    joint_talk_cluster_bootstrap,
    realization_present,
    render_evidence_packet,
    score_trajectory,
    summarize_contrast,
    summarize_acoustic_interaction,
    validate_complete_matrix,
    validate_causal_audio_provenance,
    validate_control_pairs,
    validate_evidence_packets,
    validate_inference_provenance,
    worker_process_identity_tree_sha256,
    text_sha256,
)
from slidesst.eval.inference_audit import capture_inference_environment
from slidesst.eval.inference_contract import (
    build_inference_contract,
    load_frozen_scientific_config,
    wait_for_inference_contract_ready,
)


CONTRACT_SHA = "1" * 64
TOKENIZER_MODEL = "fixture/tokenizer"
TOKENIZER_REVISION = "f" * 40
TOKENIZER_ARTIFACT_SHA256 = "e" * 64
EXTRACTOR_REVISION = "a" * 40
MODEL_ARTIFACT_TREE_SHA256 = "c" * 64


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def scientific_config(model_artifact_tree_sha256=MODEL_ARTIFACT_TREE_SHA256):
    return InferenceScientificConfig(
        schema_version="acl6060_event_inference_scientific_config_v1",
        model_id="fixture/model",
        model_revision="d" * 40,
        model_artifact_tree_sha256=model_artifact_tree_sha256,
        execution_backend="in_process_transformers",
        source_language="en",
        target_language="zh",
        chunk_policy="external_causal_audio_prefix_broker_v1",
        prompt_template="Translate the source speech using context:\n{evidence}",
        expected_conditions=list(EventScoringConfig.model_validate(minimal_config_dict()).expected_conditions),
        decoding={"max_new_tokens": 96, "do_sample": False, "num_beams": 1},
    )


def packet_payload(condition):
    context_kind = {
        "audio_only": "none",
        "document_only": "document",
        "empty": "empty",
        "ocr": "ocr",
        "matched_wrong_ocr": "ocr",
        "correct_semantic": "semantic",
        "matched_wrong_semantic": "semantic",
        "correct_relation": "relation",
        "matched_wrong_relation": "relation",
    }[condition]
    text_items = {
        "audio_only": [],
        "document_only": ["docxx"],
        "empty": [],
        "ocr": ["alpha"],
        "matched_wrong_ocr": ["bravo"],
        "correct_semantic": ["alpha"],
        "matched_wrong_semantic": ["bravo"],
        "correct_relation": ["alpha"],
        "matched_wrong_relation": ["bravo"],
    }[condition]
    context_items = []
    for index, value in enumerate(text_items):
        context_items.append(
            {
                "text": value,
                "artifact_path": f"artifacts/{condition}.json",
                "artifact_sha256": hashlib.sha256(source_artifact_bytes(condition)).hexdigest(),
                "item_index": index,
            }
        )
    return {
        "schema_version": "acl6060_source_evidence_packet_v1",
        "context_kind": context_kind,
        "context_items": context_items,
    }


def source_media_bytes(condition):
    return f"source-media:{condition}\n".encode()


def source_artifact_dict(condition):
    payload = packet_payload_without_references(condition)
    return {
        "schema_version": "acl6060_source_evidence_artifact_v1",
        "event_id": "e1",
        "context_kind": payload["context_kind"],
        "source_media_kind": (
            "source_document" if payload["context_kind"] == "document" else "slide_image"
        ),
        "source_media_path": f"media/{condition}.{'txt' if payload['context_kind'] == 'document' else 'png'}",
        "source_media_sha256": hashlib.sha256(source_media_bytes(condition)).hexdigest(),
        "extractor": "fixture-extractor",
        "extractor_revision": EXTRACTOR_REVISION,
        "items": payload["context_items"],
    }


def source_artifact_bytes(condition):
    return (
        json.dumps(
            source_artifact_dict(condition),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def packet_payload_without_references(condition):
    context_kind = {
        "audio_only": "none",
        "document_only": "document",
        "empty": "empty",
        "ocr": "ocr",
        "matched_wrong_ocr": "ocr",
        "correct_semantic": "semantic",
        "matched_wrong_semantic": "semantic",
        "correct_relation": "relation",
        "matched_wrong_relation": "relation",
    }[condition]
    context_items = {
        "audio_only": [],
        "document_only": ["docxx"],
        "empty": [],
        "ocr": ["alpha"],
        "matched_wrong_ocr": ["bravo"],
        "correct_semantic": ["alpha"],
        "matched_wrong_semantic": ["bravo"],
        "correct_relation": ["alpha"],
        "matched_wrong_relation": ["bravo"],
    }[condition]
    return {"context_kind": context_kind, "context_items": context_items}


def materialize_source_artifacts(root, conditions):
    (root / "artifacts").mkdir(parents=True)
    (root / "media").mkdir(parents=True)
    for condition in conditions:
        values = packet_payload_without_references(condition)["context_items"]
        if not values:
            continue
        (root / "artifacts" / f"{condition}.json").write_bytes(
            source_artifact_bytes(condition)
        )
        suffix = "txt" if packet_payload_without_references(condition)["context_kind"] == "document" else "png"
        (root / "media" / f"{condition}.{suffix}").write_bytes(source_media_bytes(condition))


def fake_tokenize(value):
    return list(value.encode("utf-8"))


def rendered_packet(condition):
    return render_evidence_packet(EvidencePacketPayload.model_validate(packet_payload(condition)))


def packet_hash(condition):
    if condition not in {
        "audio_only",
        "document_only",
        "empty",
        "ocr",
        "matched_wrong_ocr",
        "correct_semantic",
        "matched_wrong_semantic",
        "correct_relation",
        "matched_wrong_relation",
    }:
        return digest(f"packet-{condition}")
    return canonical_json_sha256(packet_payload(condition))


def minimal_config_dict():
    path = Path(__file__).parents[1] / "configs" / "acl6060_event_trajectory_scoring_v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def control_pairs_for_config(config, tokenize=fake_tokenize):
    rows = []
    for contrast in config.contrasts:
        if not contrast.requires_matched_control:
            continue
        rows.append(
            ControlPairSpec(
                event_id="e1",
                contrast_id=contrast.id,
                control_pair_id=f"pair-{contrast.id}",
                evidence_type=contrast.evidence_type,
                first_condition=contrast.first,
                second_condition=contrast.second,
                first_packet_id=f"packet-{contrast.first}",
                first_packet_sha256=packet_hash(contrast.first),
                second_packet_id=f"packet-{contrast.second}",
                second_packet_sha256=packet_hash(contrast.second),
                first_available_sec=0.0,
                second_available_sec=0.0,
                first_token_count=len(tokenize(rendered_packet(contrast.first))),
                second_token_count=len(tokenize(rendered_packet(contrast.second))),
            )
        )
    return rows


def evidence_packets_for_config(
    config,
    tokenize=fake_tokenize,
    tokenizer_model=TOKENIZER_MODEL,
    tokenizer_revision=TOKENIZER_REVISION,
    tokenizer_artifact_sha256=TOKENIZER_ARTIFACT_SHA256,
):
    def build(condition):
        payload = packet_payload(condition)
        rendered = rendered_packet(condition)
        token_ids = tokenize(rendered)
        return EvidencePacketSpec(
            event_id="e1",
            condition=condition,
            packet_id=f"packet-{condition}",
            packet_sha256=canonical_json_sha256(payload),
            evidence_type={
                "audio_only": "none",
                "document_only": "document",
                "empty": "empty",
                "ocr": "ocr",
                "matched_wrong_ocr": "ocr",
                "correct_semantic": "semantic",
                "matched_wrong_semantic": "semantic",
                "correct_relation": "relation",
                "matched_wrong_relation": "relation",
            }[condition],
            evidence_role=(
                "matched_wrong"
                if condition.startswith("matched_wrong")
                else "correct"
                if condition in {"ocr", "correct_semantic", "correct_relation"}
                else "baseline"
            ),
            available_sec=0.0,
            tokenizer_model=tokenizer_model,
            tokenizer_revision=tokenizer_revision,
            tokenizer_artifact_sha256=tokenizer_artifact_sha256,
            token_ids=token_ids,
            token_ids_sha256=canonical_json_sha256(token_ids),
            rendered_text_sha256=text_sha256(rendered),
            packet_payload=payload,
        )

    return [
        build(condition)
        for condition in config.expected_conditions
    ]


def source(event_id="e1", talk_id="t1", endpoint=6.0, boundary=3.0):
    source_conditions = [
        "document_only",
        "ocr",
        "matched_wrong_ocr",
        "correct_semantic",
        "matched_wrong_semantic",
        "correct_relation",
        "matched_wrong_relation",
    ]
    return SourceEventTiming(
        event_id=event_id,
        talk_id=talk_id,
        primary_eligible=True,
        evidence_available_sec=0.0,
        audio_insufficient_until_sec=boundary,
        audio_endpoint_sec=endpoint,
        expected_evidence_sources=[
            {
                "condition": condition,
                "context_kind": source_artifact_dict(condition)["context_kind"],
                "source_media_kind": source_artifact_dict(condition)["source_media_kind"],
                "source_media_path": source_artifact_dict(condition)["source_media_path"],
                "source_media_sha256": source_artifact_dict(condition)["source_media_sha256"],
                "extractor": source_artifact_dict(condition)["extractor"],
                "extractor_revision": source_artifact_dict(condition)["extractor_revision"],
            }
            for condition in source_conditions
        ],
    )


def target(event_id="e1"):
    return TargetEventSpec(
        event_id=event_id,
        acceptable_realizations=["图神经网络", "graph neural network"],
        forbidden_realizations=["卷积神经网络"],
    )


def trajectory(hypotheses, condition="correct", event_id="e1", talk_id="t1", acoustic="native"):
    return EventTrajectory(
        event_id=event_id,
        talk_id=talk_id,
        condition=condition,
        acoustic_condition=acoustic,
        inference_run_id="run-1",
        inference_contract_sha256=CONTRACT_SHA,
        evidence_packet_id=f"packet-{condition}",
        evidence_packet_sha256=packet_hash(condition),
        observations=[
            {
                "audio_time_sec": index + 1.0,
                "causal_audio_prefix_id": f"prefix:{event_id}:{acoustic}:{index}",
                "causal_audio_prefix_sha256": digest(
                    f"prefix:{event_id}:{acoustic}:{index}"
                ),
                "hypothesis": hypothesis,
            }
            for index, hypothesis in enumerate(hypotheses)
        ],
    )


def test_realization_matching_respects_words_and_cjk_spacing():
    assert realization_present("这是图 神经网络。", "图神经网络")
    assert realization_present("a graph neural network model", "graph neural network")
    assert not realization_present("a graphical neural network model", "graph neural network")
    assert realization_present("GPT-4 and U.S. labs", "GPT4")
    assert realization_present("GPT-4 and U.S. labs", "US")
    assert realization_present("C++ compiler", "C++")
    assert not realization_present("C compiler", "C++")
    assert not realization_present("C++ compiler", "C")
    assert not realization_present("图，神经网络", "图神经网络")
    assert not realization_present("3.14 percent", "314 percent")
    assert not realization_present("-5 dB", "5 dB")
    assert not realization_present("−5 dB", "5 dB")
    assert not realization_present(".5 probability", "5 probability")
    assert not realization_present(".NET runtime", "NET runtime")
    assert not realization_present("5%", "5")
    assert realization_present("version 1.2", "1.2")


def test_score_uses_stable_tail_and_conservative_audio_boundary():
    score = score_trajectory(
        source(endpoint=4.0),
        target(),
        trajectory(["图神经网络", "卷积神经网络", "图神经网络", "图神经网络"]),
    )
    assert score.first_correct_sec == 1.0
    assert score.first_stable_correct_sec == 3.0
    assert score.stable_correct_before_audio_sufficient is True
    assert score.correctness_retractions == 1
    assert score.final_correct is True
    assert score.ever_forbidden is True


def test_stable_time_is_start_of_full_final_correct_run():
    score = score_trajectory(
        source(endpoint=4.0),
        target(),
        trajectory(["", "图神经网络", "图神经网络", "图神经网络"]),
    )
    assert score.first_stable_correct_sec == 2.0


def test_isolated_final_correct_is_right_censored_not_stable():
    score = score_trajectory(source(endpoint=3.0), target(), trajectory(["", "", "图神经网络"]))
    assert score.final_correct is True
    assert score.first_stable_correct_sec is None
    assert score.stable_right_censored is True
    assert score.stable_correct_before_audio_sufficient is False


def test_forbidden_realization_blocks_correctness_and_marks_overcommit():
    score = score_trajectory(
        source(endpoint=3.0),
        target(),
        trajectory(["图神经网络", "图神经网络", "图神经网络和卷积神经网络"]),
    )
    assert score.final_correct is False
    assert score.final_forbidden is True
    assert score.overcommit is True


def test_trajectory_cannot_exceed_causal_endpoint():
    row = EventTrajectory(
        event_id="e1",
        talk_id="t1",
        condition="correct",
        acoustic_condition="native",
        inference_run_id="run-1",
        inference_contract_sha256=CONTRACT_SHA,
        evidence_packet_id="packet-correct",
        evidence_packet_sha256=packet_hash("correct"),
        observations=[
            {
                "audio_time_sec": 7.0,
                "causal_audio_prefix_id": "prefix:e1:native:0",
                "causal_audio_prefix_sha256": digest("prefix:e1:native:0"),
                "hypothesis": "图神经网络",
            }
        ],
    )
    with pytest.raises(ValueError, match="causal endpoint"):
        score_trajectory(source(), target(), row)


def test_trajectory_cannot_stop_before_causal_endpoint():
    with pytest.raises(ValueError, match="must end"):
        score_trajectory(source(), target(), trajectory(["图神经网络", "图神经网络"]))


def test_source_evidence_must_precede_audio_insufficient_boundary():
    row = source().model_dump()
    row["evidence_available_sec"] = 4.0
    with pytest.raises(ValueError, match="evidence becomes available after"):
        SourceEventTiming.model_validate(row)


def test_time_fields_reject_non_finite_values():
    row = source().model_dump()
    row["audio_endpoint_sec"] = float("inf")
    with pytest.raises(ValueError):
        SourceEventTiming.model_validate(row)
    observation = {"audio_time_sec": float("nan"), "hypothesis": ""}
    with pytest.raises(ValueError):
        EventTrajectory.model_validate(
            {
                **trajectory([""] * 6).model_dump(exclude={"observations"}),
                "observations": [observation],
            }
        )


def test_complete_matrix_rejects_missing_condition():
    with pytest.raises(ValueError, match="incomplete trajectory matrix"):
        validate_complete_matrix(
            [source()],
            [target()],
            [trajectory(["", ""], condition="correct")],
            expected_conditions=["correct", "control"],
            expected_acoustic_conditions=["native"],
        )


def test_complete_matrix_rejects_condition_specific_audio_grid():
    left = trajectory(["", "", "", "", "", ""], condition="correct")
    right = EventTrajectory(
        event_id="e1",
        talk_id="t1",
        condition="control",
        acoustic_condition="native",
        inference_run_id="run-1",
        inference_contract_sha256=CONTRACT_SHA,
        evidence_packet_id="packet-control",
        evidence_packet_sha256=digest("packet-control"),
        observations=[
            {
                "audio_time_sec": value,
                "causal_audio_prefix_id": f"prefix:e1:native:{index}",
                "causal_audio_prefix_sha256": digest(f"prefix:e1:native:{index}"),
                "hypothesis": "",
            }
            for index, value in enumerate([0.5, 2.0, 3.0, 4.0, 5.0, 6.0])
        ],
    )
    with pytest.raises(ValueError, match="different audio-time grids"):
        validate_complete_matrix(
            [source()],
            [target()],
            [left, right],
            expected_conditions=["correct", "control"],
            expected_acoustic_conditions=["native"],
        )


def test_complete_matrix_rejects_acoustic_specific_audio_grid():
    rows = [
        trajectory([""] * 6, condition=condition, acoustic=acoustic)
        for acoustic in ["native", "noisy"]
        for condition in ["correct", "control"]
    ]
    for row in rows:
        if row.acoustic_condition == "noisy":
            row.observations[0].audio_time_sec = 0.5
    with pytest.raises(ValueError, match="acoustic conditions"):
        validate_complete_matrix(
            [source()],
            [target()],
            rows,
            expected_conditions=["correct", "control"],
            expected_acoustic_conditions=["native", "noisy"],
        )


def test_contrast_reports_talk_equal_result_and_commit_advance():
    rows = []
    specs = [
        ("e1", "t1", ["图神经网络"] * 4, ["", "", "图神经网络", "图神经网络"]),
        ("e2", "t1", ["", "", "图神经网络", "图神经网络"], ["", "", "图神经网络", "图神经网络"]),
        ("e3", "t2", ["图神经网络"] * 4, ["", "", "图神经网络", "图神经网络"]),
    ]
    for event_id, talk_id, correct_hypotheses, control_hypotheses in specs:
        src = source(event_id, talk_id, endpoint=4.0)
        src.audio_insufficient_until_sec = 2.0
        tgt = target(event_id)
        rows.append(score_trajectory(src, tgt, trajectory(correct_hypotheses, "correct", event_id, talk_id)))
        rows.append(score_trajectory(src, tgt, trajectory(control_hypotheses, "control", event_id, talk_id)))
    summary = summarize_contrast(
        rows,
        first="correct",
        second="control",
        acoustic_group="native",
        acoustic_conditions=["native"],
    )
    assert summary["pooled_early_risk_difference"] == pytest.approx(2 / 3)
    assert summary["talk_equal_early_risk_difference"] == pytest.approx(0.75)
    assert summary["mean_commit_advance_sec"] == pytest.approx(4 / 3)


def test_acoustic_group_averages_replicates_without_inflating_event_count():
    rows = []
    src = source(endpoint=2.0, boundary=2.0)
    for acoustic, correct_hypotheses in [
        ("seed0", ["图神经网络", "图神经网络"]),
        ("seed1", ["图神经网络", "图神经网络"]),
        ("seed2", ["", ""]),
    ]:
        rows.append(
            score_trajectory(
                src,
                target(),
                trajectory(correct_hypotheses, "correct", acoustic=acoustic),
            )
        )
        rows.append(
            score_trajectory(
                src,
                target(),
                trajectory(["", ""], "control", acoustic=acoustic),
            )
        )
    summary = summarize_contrast(
        rows,
        first="correct",
        second="control",
        acoustic_group="three_seeds",
        acoustic_conditions=["seed0", "seed1", "seed2"],
    )
    assert summary["event_count"] == 1
    assert summary["paired_trajectory_count"] == 3
    assert summary["replicates_per_event"] == 3
    assert summary["talk_equal_early_risk_difference"] == pytest.approx(2 / 3)


def test_acoustic_interaction_is_difference_of_content_specific_differences():
    native = {
        "first": "correct",
        "second": "control",
        "acoustic_group": "native",
        "event_count": 10,
        "talk_count": 2,
        "per_talk_early_risk_difference": {"t1": 0.0, "t2": 0.2},
        "talk_equal_early_risk_difference": 0.1,
        "pooled_early_risk_difference": 0.2,
    }
    noisy = {
        "first": "correct",
        "second": "control",
        "acoustic_group": "babble_5db",
        "event_count": 10,
        "talk_count": 2,
        "per_talk_early_risk_difference": {"t1": 0.2, "t2": 0.6},
        "talk_equal_early_risk_difference": 0.4,
        "pooled_early_risk_difference": 0.5,
    }
    interaction = summarize_acoustic_interaction(native, noisy)
    assert interaction["talk_equal_early_risk_difference_interaction"] == pytest.approx(0.3)
    assert interaction["pooled_early_risk_difference_interaction"] == pytest.approx(0.3)
    assert interaction["directionally_positive_talk_count"] == 2


def test_joint_talk_bootstrap_preserves_paired_noise_structure():
    by_group = {
        group: {
            "per_talk_early_risk_difference": {
                "t1": level * 0.1,
                "t2": level * 0.2,
            }
        }
        for level, group in enumerate(
            ["native", "babble_10db", "babble_5db", "babble_0db", "babble_minus5db"]
        )
    }
    result = joint_talk_cluster_bootstrap(
        by_group,
        native_group="native",
        severity_order=list(by_group),
        samples=100,
        seed=7,
    )
    assert result["interaction_ci95_by_acoustic_group"]["babble_5db"][0] > 0
    assert result["severity_monotonic_bootstrap_probability"] == 1.0
    assert result["severity_correlation_ci95"] == pytest.approx([1.0, 1.0])


def test_joint_talk_bootstrap_does_not_condition_interval_on_defined_draws():
    order = ["native", "babble_10db", "babble_5db", "babble_0db", "babble_minus5db"]
    by_group = {
        group: {
            "per_talk_early_risk_difference": {
                "constant": 0.0,
                "increasing": level * 0.1,
            }
        }
        for level, group in enumerate(order)
    }
    result = joint_talk_cluster_bootstrap(
        by_group,
        native_group="native",
        severity_order=order,
        samples=100,
        seed=7,
    )
    assert 0 < result["severity_correlation_undefined_samples"] < 100
    assert result["severity_correlation_ci95"] is None
    assert result["severity_correlation_interval_status"] == "UNDEFINED_DRAWS_PRESENT"


def test_development_gate_labels_exploratory_point_estimate_safety():
    summary = {
        "per_talk_early_risk_difference": {f"t{i}": 0.1 for i in range(5)},
        "talk_equal_early_risk_difference": 0.1,
        "talk_equal_final_correct_risk_difference": 0.0,
        "talk_equal_forbidden_adoption_risk_difference": 0.2,
        "talk_equal_overcommit_risk_difference": 0.0,
        "talk_count": 5,
    }
    signal = EventScoringConfig.model_validate(minimal_config_dict()).development_signal
    apply_development_gate(summary, signal)
    assert summary["development_gate"]["forbidden_adoption_point_estimate_pass"] is False
    assert summary["development_gate"]["all_components_pass"] is False


def test_scoring_config_requires_acoustic_groups_to_be_a_partition():
    config = minimal_config_dict()
    config["acoustic_groups"][0]["members"].append("babble_p10_s0")
    with pytest.raises(ValueError, match="exactly one group"):
        validate_config(config)


def test_scoring_config_rejects_acoustic_group_reordering():
    config = minimal_config_dict()
    config["acoustic_groups"][0], config["acoustic_groups"][1] = (
        config["acoustic_groups"][1],
        config["acoustic_groups"][0],
    )
    with pytest.raises(ValueError, match="group order"):
        validate_config(config)


def test_strict_schemas_reject_leakage_and_config_drift(tmp_path):
    row = trajectory([""] * 6).model_dump()
    row["target_reference"] = "forbidden future target"
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        EventTrajectory.model_validate(row)

    config = minimal_config_dict()
    config["min_stability_observations"] = 2.9
    with pytest.raises(ValueError):
        validate_config(config)

    for field, value in [
        ("min_stability_observations", 3),
        ("bootstrap_samples", 1001),
        ("bootstrap_seed", 7),
    ]:
        config = minimal_config_dict()
        config[field] = value
        with pytest.raises(ValueError, match="v1"):
            validate_config(config)

    config = minimal_config_dict()
    config["babble_severity_order"] = ["native"] * 5
    with pytest.raises(ValueError, match="v1 babble severity order"):
        validate_config(config)

    config = EventScoringConfig.model_validate(minimal_config_dict())
    packet = evidence_packets_for_config(config)[0].model_dump()
    packet["packet_payload"]["target_reference"] = "leaked"
    packet["packet_sha256"] = canonical_json_sha256(packet["packet_payload"])
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        EvidencePacketSpec.model_validate(packet)

    packet = evidence_packets_for_config(config)[0].model_dump()
    packet["token_ids"].append(999)
    with pytest.raises(ValueError, match="token-id hash mismatch"):
        EvidencePacketSpec.model_validate(packet)

    packet = evidence_packets_for_config(config)[0].model_dump()
    packet["token_ids"] = [99]
    packet["token_ids_sha256"] = canonical_json_sha256(packet["token_ids"])
    forged = EvidencePacketSpec.model_validate(packet)
    source_artifact_root = tmp_path / "source-artifacts"
    materialize_source_artifacts(source_artifact_root, config.expected_conditions)
    with pytest.raises(ValueError, match="tokenizer replay"):
        validate_evidence_packets(
            [forged, *evidence_packets_for_config(config)[1:]],
            source_by_id={"e1": source()},
            trajectories=[
                trajectory([""] * 6, condition=condition, acoustic=acoustic)
                for acoustic in config.expected_acoustic_conditions
                for condition in config.expected_conditions
            ],
            config=config,
            expected_tokenizer_model=TOKENIZER_MODEL,
            expected_tokenizer_revision=TOKENIZER_REVISION,
            expected_tokenizer_artifact_sha256=TOKENIZER_ARTIFACT_SHA256,
            expected_source_artifact_tree_sha256=directory_tree_sha256(source_artifact_root),
            source_artifact_root=source_artifact_root,
            tokenize=fake_tokenize,
        )

    packets = evidence_packets_for_config(config)
    leaked = packets[5].model_dump()
    leaked["packet_payload"]["context_items"][0]["text"] = "gold target"
    leaked_payload = EvidencePacketPayload.model_validate(leaked["packet_payload"])
    leaked_rendered = render_evidence_packet(leaked_payload)
    leaked["packet_sha256"] = canonical_json_sha256(leaked_payload)
    leaked["rendered_text_sha256"] = text_sha256(leaked_rendered)
    leaked["token_ids"] = fake_tokenize(leaked_rendered)
    leaked["token_ids_sha256"] = canonical_json_sha256(leaked["token_ids"])
    packets[5] = EvidencePacketSpec.model_validate(leaked)
    with pytest.raises(ValueError, match="differs from frozen artifact"):
        validate_evidence_packets(
            packets,
            source_by_id={"e1": source()},
            trajectories=[
                trajectory([""] * 6, condition=condition, acoustic=acoustic)
                for acoustic in config.expected_acoustic_conditions
                for condition in config.expected_conditions
            ],
            config=config,
            expected_tokenizer_model=TOKENIZER_MODEL,
            expected_tokenizer_revision=TOKENIZER_REVISION,
            expected_tokenizer_artifact_sha256=TOKENIZER_ARTIFACT_SHA256,
            expected_source_artifact_tree_sha256=directory_tree_sha256(source_artifact_root),
            source_artifact_root=source_artifact_root,
            tokenize=fake_tokenize,
        )


def test_control_pair_manifest_binds_type_time_budget_and_trajectory_packets(tmp_path):
    config = EventScoringConfig.model_validate(minimal_config_dict())
    rows = [
        trajectory([""] * 6, condition=condition, acoustic=acoustic)
        for acoustic in config.expected_acoustic_conditions
        for condition in config.expected_conditions
    ]
    source_artifact_root = tmp_path / "source-artifacts"
    materialize_source_artifacts(source_artifact_root, config.expected_conditions)
    packet_by_key = validate_evidence_packets(
        evidence_packets_for_config(config),
        source_by_id={"e1": source()},
        trajectories=rows,
        config=config,
        expected_tokenizer_model=TOKENIZER_MODEL,
        expected_tokenizer_revision=TOKENIZER_REVISION,
        expected_tokenizer_artifact_sha256=TOKENIZER_ARTIFACT_SHA256,
        expected_source_artifact_tree_sha256=directory_tree_sha256(source_artifact_root),
        source_artifact_root=source_artifact_root,
        tokenize=fake_tokenize,
    )
    pairs = control_pairs_for_config(config)
    pair = pairs[0]
    validate_control_pairs(
        pairs,
        source_by_id={"e1": source()},
        trajectories=rows,
        config=config,
        evidence_packet_by_key=packet_by_key,
    )
    bad = pair.model_dump()
    bad["second_token_count"] = 31
    with pytest.raises(ValueError, match="token counts differ"):
        ControlPairSpec.model_validate(bad)
    bad = pair.model_dump()
    bad["first_available_sec"] = 1.0
    bad["second_available_sec"] = 1.0
    bad_pairs = [ControlPairSpec.model_validate(bad), *pairs[1:]]
    with pytest.raises(ValueError, match="differs from source event"):
        validate_control_pairs(
            bad_pairs,
            source_by_id={"e1": source()},
            trajectories=rows,
            config=config,
            evidence_packet_by_key=packet_by_key,
        )


def test_source_event_rejects_reindexed_media_or_extractor_drift(tmp_path):
    config = EventScoringConfig.model_validate(minimal_config_dict())
    source_artifact_root = tmp_path / "source-artifacts"
    materialize_source_artifacts(source_artifact_root, config.expected_conditions)
    rows = [
        trajectory([""] * 6, condition=condition, acoustic=acoustic)
        for acoustic in config.expected_acoustic_conditions
        for condition in config.expected_conditions
    ]
    source_row = source().model_dump()
    ocr_source = next(
        value
        for value in source_row["expected_evidence_sources"]
        if value["condition"] == "ocr"
    )
    ocr_source["source_media_path"] = "media/future-slide.png"
    ocr_source["extractor_revision"] = "b" * 40
    with pytest.raises(ValueError, match="differs from event source identity"):
        validate_evidence_packets(
            evidence_packets_for_config(config),
            source_by_id={"e1": SourceEventTiming.model_validate(source_row)},
            trajectories=rows,
            config=config,
            expected_tokenizer_model=TOKENIZER_MODEL,
            expected_tokenizer_revision=TOKENIZER_REVISION,
            expected_tokenizer_artifact_sha256=TOKENIZER_ARTIFACT_SHA256,
            expected_source_artifact_tree_sha256=directory_tree_sha256(source_artifact_root),
            source_artifact_root=source_artifact_root,
            tokenize=fake_tokenize,
        )
def inference_audit(phase="workers_start"):
    workers = [
        {
            "pid": 101,
            "parent_pid": 1,
            "process_start_time_ticks": 456,
            "command": (
                "python inference.py --run-id run-1 "
                "--inference-contract /run/inference_contract.json "
                "--inference-contract-ready-file /run/inference_contract.ready.json "
                "--scientific-config /run/scientific_config.json "
                "--model-artifact-root /models/frozen "
                "--tokenizer-artifact-root /tokenizers/frozen "
                f"--model-id fixture/model --model-revision {'d' * 40}"
            ),
            "marker_process": True,
            "executable_path": "/usr/bin/python3",
            "executable_sha256": "9" * 64,
            "working_directory": "/data/repo",
            "entrypoint_path": "/data/repo/inference.py",
            "entrypoint_sha256": "4" * 64,
            "environment_sha256": "5" * 64,
        }
    ]
    return InferenceEnvironmentAudit(
        schema_version="acl6060_event_inference_environment_audit_v5",
        run_id="run-1",
        container_name="test-container",
        container_id="b" * 64,
        container_image_id="8" * 64,
        container_read_only_rootfs=True,
        container_network_mode="none",
        capture_host="hyper00",
        captured_at_utc="2026-08-01T12:00:00Z",
        capture_command="docker_inspect_proc_tree_worker_discovery_git_v5",
        capture_phase=phase,
        worker_command_match="--run-id run-1",
        worker_processes=workers,
        docker_inspect_sha256="c" * 64,
        process_listing_sha256="7" * 64,
        proc_open_files_sha256="d" * 64,
        process_identity_tree_sha256=worker_process_identity_tree_sha256(workers),
        inference_repo_path="/data/repo",
        inference_git_commit="2" * 40,
        inference_git_status_sha256=hashlib.sha256(b"").hexdigest(),
        forbidden_container_artifact_roots=["/private/targets", "/private/references"],
        forbidden_host_mount_source_roots=["/host/private", "/audio/source", "/scoring"],
        observed_mounts=[
            {"source": "/host/public", "destination": "/data/public", "read_only": True},
            {
                "source": "/host/scientific_config.json",
                "destination": "/run/scientific_config.json",
                "read_only": True,
            },
            {"source": "/host/model", "destination": "/models/frozen", "read_only": True},
            {
                "source": "/host/tokenizer",
                "destination": "/tokenizers/frozen",
                "read_only": True,
            },
        ],
        process_open_file_paths=["/data/public/audio.wav"],
        forbidden_artifact_exposure_detected=False,
    )


def inference_contract(config):
    return InferenceContract(
        schema_version="acl6060_event_inference_contract_v1",
        run_id="run-1",
        created_at_utc="2026-08-01T11:59:00Z",
        git_commit="2" * 40,
        scientific_config_sha256="3" * 64,
        scoring_config_sha256="7" * 64,
        model_id="fixture/model",
        model_revision="d" * 40,
        model_artifact_tree_sha256=MODEL_ARTIFACT_TREE_SHA256,
        tokenizer_model=TOKENIZER_MODEL,
        tokenizer_revision=TOKENIZER_REVISION,
        tokenizer_artifact_sha256=TOKENIZER_ARTIFACT_SHA256,
        source_artifact_tree_sha256="6" * 64,
        source_events_sha256="4" * 64,
        evidence_packets_sha256="8" * 64,
        control_pairs_sha256="5" * 64,
        target_scores_sha256="0" * 64,
        outcome_commitment_sha256="a" * 64,
        outcome_artifact_tree_sha256="b" * 64,
        causal_audio_schedule_sha256="b" * 64,
        causal_audio_broker_audit_sha256="d" * 64,
        causal_audio_protocol="external_talk_synchronized_prefix_broker_v2",
        causal_audio_broker_entrypoint_sha256="6" * 64,
        expected_conditions=config.expected_conditions,
        expected_acoustic_conditions=config.expected_acoustic_conditions,
        target_artifact_mounted=False,
        reference_artifact_mounted=False,
        future_audio_access=False,
        forbidden_container_artifact_roots=["/private/targets", "/private/references"],
        forbidden_host_mount_source_roots=["/host/private", "/audio/source", "/scoring"],
        scoring_protected_artifact_roots=["/scoring/targets", "/scoring/references"],
        worker_command_match="--run-id run-1",
        worker_inference_contract_path="/run/inference_contract.json",
        worker_contract_ready_file_path="/run/inference_contract.ready.json",
        scientific_config_host_path="/host/scientific_config.json",
        worker_scientific_config_path="/run/scientific_config.json",
        model_artifact_host_root_path="/host/model",
        worker_model_artifact_root_path="/models/frozen",
        tokenizer_artifact_host_root_path="/host/tokenizer",
        worker_tokenizer_artifact_root_path="/tokenizers/frozen",
        expected_worker_count=1,
        inference_repo_path="/data/repo",
        container_image_id="8" * 64,
        environment_start_audit_sha256="a" * 64,
        worker_process_identity_tree_sha256=inference_audit().process_identity_tree_sha256,
    )


def result_attestation():
    return InferenceResultAttestation(
        schema_version="acl6060_event_inference_result_attestation_v1",
        run_id="run-1",
        created_at_utc="2026-08-01T13:00:00Z",
        inference_contract_sha256=CONTRACT_SHA,
        trajectories_sha256="f" * 64,
        causal_audio_release_log_sha256="c" * 64,
        environment_start_audit_sha256="a" * 64,
        environment_end_audit_sha256="e" * 64,
    )


def test_inference_contract_and_result_attestation_bind_outputs_and_inputs():
    config = EventScoringConfig.model_validate(minimal_config_dict())
    start_audit = inference_audit("workers_start")
    end_audit = inference_audit("workers_end")
    contract = inference_contract(config)
    attestation = result_attestation()
    kwargs = {
        "contract_sha256": CONTRACT_SHA,
        "trajectories_sha256": "f" * 64,
        "source_events_sha256": "4" * 64,
        "evidence_packets_sha256": "8" * 64,
        "control_pairs_sha256": "5" * 64,
        "scientific_config_sha256": "3" * 64,
        "scoring_config_sha256": "7" * 64,
        "target_scores_sha256": "0" * 64,
        "environment_start_audit_sha256": "a" * 64,
        "environment_end_audit_sha256": "e" * 64,
        "config": config,
        "scientific_config": scientific_config(),
        "model_artifact_tree_sha256": MODEL_ARTIFACT_TREE_SHA256,
        "trajectories": [trajectory([""] * 6)],
        "expected_git_commit": "2" * 40,
    }
    validate_inference_provenance(
        contract, attestation, start_audit, end_audit, **kwargs
    )
    with pytest.raises(ValueError, match="source-events hash mismatch"):
        validate_inference_provenance(
            contract,
            attestation,
            start_audit,
            end_audit,
            **{**kwargs, "source_events_sha256": "6" * 64},
        )
    with pytest.raises(ValueError, match="does not bind trajectories"):
        validate_inference_provenance(
            contract,
            attestation,
            start_audit,
            end_audit,
            **{**kwargs, "trajectories_sha256": "9" * 64},
        )
    bad_contract = contract.model_dump()
    bad_contract["target_artifact_mounted"] = True
    with pytest.raises(ValueError):
        InferenceContract.model_validate(bad_contract)
    bad_audit = start_audit.model_dump()
    bad_audit["process_open_file_paths"] = ["/private/targets/gold.jsonl"]
    with pytest.raises(ValueError, match="forbidden artifact root"):
        InferenceEnvironmentAudit.model_validate(bad_audit)
    bad_audit = start_audit.model_dump()
    bad_audit["observed_mounts"] = [
        {"source": "/host/data", "destination": "/private", "read_only": True}
    ]
    with pytest.raises(ValueError, match="mount exposes"):
        InferenceEnvironmentAudit.model_validate(bad_audit)
    bad_audit = start_audit.model_dump()
    bad_audit["inference_git_status_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="dirty checkout"):
        InferenceEnvironmentAudit.model_validate(bad_audit)
    bad_audit = start_audit.model_dump()
    bad_audit["container_network_mode"] = "bridge"
    with pytest.raises(ValueError):
        InferenceEnvironmentAudit.model_validate(bad_audit)
    bad_mount_payload = start_audit.model_dump()
    bad_mount_payload["observed_mounts"][-2]["read_only"] = False
    bad_start_audit = InferenceEnvironmentAudit.model_validate(bad_mount_payload)
    bad_mount_payload["capture_phase"] = "workers_end"
    bad_end_audit = InferenceEnvironmentAudit.model_validate(bad_mount_payload)
    with pytest.raises(ValueError, match="model artifact tree.*read-only"):
        validate_inference_provenance(
            contract,
            attestation,
            bad_start_audit,
            bad_end_audit,
            **kwargs,
        )
    bad_tokenizer_mount_payload = start_audit.model_dump()
    bad_tokenizer_mount_payload["observed_mounts"][-1]["read_only"] = False
    bad_tokenizer_start_audit = InferenceEnvironmentAudit.model_validate(
        bad_tokenizer_mount_payload
    )
    bad_tokenizer_mount_payload["capture_phase"] = "workers_end"
    bad_tokenizer_end_audit = InferenceEnvironmentAudit.model_validate(
        bad_tokenizer_mount_payload
    )
    with pytest.raises(ValueError, match="tokenizer artifact tree.*read-only"):
        validate_inference_provenance(
            contract,
            attestation,
            bad_tokenizer_start_audit,
            bad_tokenizer_end_audit,
            **kwargs,
        )
    replaced_end_payload = end_audit.model_dump()
    replaced_end_payload["container_id"] = "6" * 64
    with pytest.raises(ValueError, match="container or mount topology changed"):
        validate_inference_provenance(
            contract,
            attestation,
            start_audit,
            InferenceEnvironmentAudit.model_validate(replaced_end_payload),
            **kwargs,
        )

    bad_scientific_config = scientific_config().model_dump()
    bad_scientific_config["prompt_template"] = "Use {evidence} and {gold_target}"
    with pytest.raises(ValueError, match="only one.*evidence"):
        InferenceScientificConfig.model_validate(bad_scientific_config)

    config = minimal_config_dict()
    config["scope"] = "confirmatory"
    with pytest.raises(ValueError):
        validate_config(config)


def test_live_environment_capture_binds_docker_and_worker_proc_outputs():
    inspect_stdout = json.dumps(
        [
            {
                "Id": "b" * 64,
                "Image": "sha256:" + "8" * 64,
                "State": {"Running": True},
                "HostConfig": {"ReadonlyRootfs": True, "NetworkMode": "none"},
                "Mounts": [
                    {
                        "Source": "/data/jaxan",
                        "Destination": "/data",
                        "RW": True,
                    }
                ],
            }
        ]
    )

    def fake_run(command, **kwargs):
        assert kwargs == {"check": True, "capture_output": True, "text": True}
        if command[:2] == ["docker", "inspect"]:
            return subprocess.CompletedProcess(command, 0, inspect_stdout, "")
        payload = command[3:]
        if payload[:2] == ["ps", "-eo"]:
            stdout = (
                "101 1 python inference.py --run-id run-1\n"
                "102 1 unrelated.py\n"
                "103 101 python helper.py\n"
                "104 1 python inference.py --run-id run-10\n"
            )
        elif payload[0] == "cat" and payload[-1].endswith("/stat"):
            pid = int(payload[-1].split("/")[2])
            stat_fields = ["S"] + ["0"] * 18 + [str(pid + 1000), "0"]
            stdout = f"{pid} (python worker) {' '.join(stat_fields)}\n"
        elif payload[0] == "realpath":
            stdout = payload[-1] + "\n"
        elif payload[0] == "readlink":
            if payload[1] == "-f":
                stdout = payload[-1] + "\n"
            elif payload[1].endswith("/cwd"):
                stdout = "/data/repo\n"
            else:
                stdout = "/usr/bin/python3\n"
        elif payload[0] == "sha256sum":
            value = "4" * 64 if payload[-1].endswith("inference.py") else "9" * 64
            stdout = f"{value}  {payload[-1]}\n"
        elif payload[0] == "sh":
            stdout = (
                "A=1\0"
                if "environ" in payload[2]
                else "/data/audio/101.wav\nsocket:[123]\n"
            )
        elif "rev-parse" in payload:
            stdout = f"{'2' * 40}\n"
        elif "status" in payload:
            stdout = ""
        else:
            raise AssertionError(command)
        return subprocess.CompletedProcess(command, 0, stdout, "")

    audit = capture_inference_environment(
        run_id="run-1",
        container_name="test-container",
        worker_command_match="--run-id run-1",
        inference_repo_path="/data/repo",
        forbidden_container_artifact_roots=["/private/targets", "/private/references"],
        forbidden_host_mount_source_roots=["/host/private"],
        capture_phase="workers_start",
        run_command=fake_run,
        capture_host="hyper00",
        captured_at_utc="2026-08-01T12:00:00Z",
    )
    assert audit.container_id == "b" * 64
    assert [process.pid for process in audit.worker_processes] == [101, 103]
    assert [process.pid for process in audit.worker_processes if process.marker_process] == [101]
    assert [process.process_start_time_ticks for process in audit.worker_processes] == [1101, 1103]
    assert audit.process_open_file_paths == ["/data/audio/101.wav"]
    assert audit.docker_inspect_sha256 == hashlib.sha256(inspect_stdout.encode()).hexdigest()
    assert audit.inference_git_commit == "2" * 40


def test_causal_audio_chain_rejects_prefix_drift_and_unprotected_source_root():
    config = EventScoringConfig.model_validate(minimal_config_dict())
    row = trajectory(["", ""], condition="audio_only", acoustic="native")
    schedule = CausalAudioSchedule.model_validate(
        {
            "schema_version": "acl6060_causal_audio_schedule_v3",
            "run_id": "run-1",
            "expected_conditions": config.expected_conditions,
            "source_audio_roots": ["/audio/source"],
            "sources": [
                {
                    "source_id": "source:t1:native",
                    "talk_id": "t1",
                    "acoustic_condition": "native",
                    "source_pcm_path": "/audio/source/t1-native.f32le",
                    "source_pcm_sha256": "1" * 64,
                    "pcm_format": "float32le_mono",
                    "sample_rate": 16_000,
                    "total_sample_count": 32_000,
                    "materialization_kind": "native",
                    "upstream_audio_sha256": "5" * 64,
                    "materializer_git_commit": "2" * 40,
                    "materializer_entrypoint_sha256": "3" * 64,
                    "source_provenance_path": "/audio/source/provenance.jsonl",
                    "source_provenance_sha256": "4" * 64,
                }
            ],
            "prefixes": [
                {
                    "source_id": "source:t1:native",
                    "event_id": "e1",
                    "acoustic_condition": "native",
                    "sequence_index": index,
                    "audio_time_sec": float(index + 1),
                    "prefix_id": f"prefix:e1:native:{index}",
                    "prefix_pcm_sha256": digest(f"prefix:e1:native:{index}"),
                    "sample_rate": 16_000,
                    "sample_count": (index + 1) * 16_000,
                }
                for index in range(2)
            ],
        }
    )
    broker_audit = CausalAudioBrokerAudit.model_validate(
        {
            "schema_version": "acl6060_causal_audio_broker_audit_v2",
            "run_id": "run-1",
            "schedule_sha256": "b" * 64,
            "broker_git_commit": "2" * 40,
            "broker_repo_path": "/data/repo",
            "broker_entrypoint_path": "/data/repo/broker.py",
            "broker_entrypoint_sha256": "6" * 64,
            "broker_command": "python broker.py --run-id run-1",
            "broker_pid": 123,
            "socket_path": "/run/user/1000/acl6060.sock",
            "release_events_path": "/audio/logs/releases.jsonl",
            "source_audio_roots": ["/audio/source"],
            "delivery_protocol": "length_prefixed_unix_socket_v1",
            "captured_at_utc": "2026-08-01T12:00:00Z",
        }
    )
    interaction_rows = []
    previous = hashlib.sha256(b"").hexdigest()
    for index in range(2):
        release = {
            "record_type": "prefix_release",
            "source_id": "source:t1:native",
            "session_id": "session-1",
            "server_ordinal": len(interaction_rows),
            "event_id": "e1",
            "condition": "audio_only",
            "acoustic_condition": "native",
            "sequence_index": index,
            "audio_time_sec": float(index + 1),
            "prefix_id": f"prefix:e1:native:{index}",
            "prefix_pcm_sha256": digest(f"prefix:e1:native:{index}"),
            "sample_count": (index + 1) * 16_000,
            "request_id": f"request:{index}",
            "granted_monotonic_ns": len(interaction_rows) + 1,
            "granted_at_utc": f"2026-08-01T12:00:0{index}Z",
            "previous_record_sha256": previous,
        }
        release["record_sha256"] = canonical_json_sha256(release)
        previous = release["record_sha256"]
        interaction_rows.append(release)
        commit = {
            "record_type": "observation_commit",
            "source_id": "source:t1:native",
            "session_id": "session-1",
            "server_ordinal": len(interaction_rows),
            "event_id": "e1",
            "condition": "audio_only",
            "acoustic_condition": "native",
            "sequence_index": index,
            "prefix_id": f"prefix:e1:native:{index}",
            "prefix_pcm_sha256": digest(f"prefix:e1:native:{index}"),
            "observation_sha256": causal_observation_sha256(
                run_id=row.inference_run_id,
                inference_contract_sha256=row.inference_contract_sha256,
                event_id=row.event_id,
                condition=row.condition,
                acoustic_condition=row.acoustic_condition,
                sequence_index=index,
                observation=row.observations[index],
            ),
            "request_id": f"commit:{index}",
            "committed_monotonic_ns": len(interaction_rows) + 1,
            "committed_at_utc": f"2026-08-01T12:00:0{index}Z",
            "previous_record_sha256": previous,
        }
        commit["record_sha256"] = canonical_json_sha256(commit)
        previous = commit["record_sha256"]
        interaction_rows.append(commit)
    release_log = CausalAudioReleaseLog.model_validate(
        {
            "schema_version": "acl6060_causal_audio_release_log_v3",
            "run_id": "run-1",
            "schedule_sha256": "b" * 64,
            "broker_audit_sha256": "d" * 64,
            "interactions": interaction_rows,
        }
    )
    contract = inference_contract(config)
    attestation = result_attestation()
    kwargs = {
        "schedule_sha256": "b" * 64,
        "release_log_sha256": "c" * 64,
        "broker_audit_sha256": "d" * 64,
        "expected_broker_git_commit": "2" * 40,
        "observed_broker_entrypoint_sha256": "6" * 64,
        "config": config,
        "source_by_id": {"e1": source()},
    }
    validate_causal_audio_provenance(
        contract,
        attestation,
        schedule,
        release_log,
        broker_audit,
        trajectories=[row],
        **kwargs,
    )

    corrupt = row.model_dump()
    corrupt["observations"][0]["causal_audio_prefix_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="prefix identity mismatch"):
        validate_causal_audio_provenance(
            contract,
            attestation,
            schedule,
            release_log,
            broker_audit,
            trajectories=[EventTrajectory.model_validate(corrupt)],
            **kwargs,
        )
    backfilled = row.model_dump()
    backfilled["observations"][0]["hypothesis"] = "generated after full audio"
    with pytest.raises(ValueError, match="observation commit hash mismatch"):
        validate_causal_audio_provenance(
            contract,
            attestation,
            schedule,
            release_log,
            broker_audit,
            trajectories=[EventTrajectory.model_validate(backfilled)],
            **kwargs,
        )
    unprotected = contract.model_copy(
        update={"forbidden_host_mount_source_roots": ["/host/private", "/scoring"]}
    )
    with pytest.raises(ValueError, match="not forbidden"):
        validate_causal_audio_provenance(
            unprotected,
            attestation,
            schedule,
            release_log,
            broker_audit,
            trajectories=[row],
            **kwargs,
        )


def test_git_head_rejects_dirty_inference_checkout(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    (repo / "tracked.txt").write_text("frozen\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "freeze"], cwd=repo, check=True)
    assert len(git_head(repo)) == 40
    (repo / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dirty"):
        git_head(repo)


def test_directory_tree_hash_pins_local_tokenizer_bytes_and_rejects_symlinks(tmp_path):
    root = tmp_path / "tokenizer"
    root.mkdir()
    config_path = root / "tokenizer_config.json"
    config_path.write_text('{"version":1}\n', encoding="utf-8")
    first = directory_tree_sha256(root)
    config_path.write_text('{"version":2}\n', encoding="utf-8")
    assert directory_tree_sha256(root) != first
    (root / "mutable-link").symlink_to(config_path)
    with pytest.raises(ValueError, match="mutable symlink"):
        directory_tree_sha256(root)
    root_link = tmp_path / "tokenizer-link"
    root_link.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="root is a mutable symlink"):
        directory_tree_sha256(root_link)


def test_target_artifact_must_be_inside_manifest_forbidden_root(tmp_path):
    target = tmp_path / "targets" / "target.jsonl"
    target.parent.mkdir()
    target.write_text("{}\n", encoding="utf-8")
    validate_target_is_protected(target, [str(target.parent)])
    with pytest.raises(ValueError, match="outside manifest forbidden roots"):
        validate_target_is_protected(target, [str(tmp_path / "decoy")])


def test_scoring_cli_writes_hashed_complete_analysis(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    tokenizer_dir = tmp_path / "tokenizer"
    backend = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.save_pretrained(tokenizer_dir)
    tokenizer_artifact_sha256 = directory_tree_sha256(tokenizer_dir)

    def tokenize(value):
        return tokenizer.encode(value, add_special_tokens=False)

    inference_repo = tmp_path / "inference_repo"
    inference_repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=inference_repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=inference_repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=inference_repo, check=True
    )
    (inference_repo / "frozen.txt").write_text("frozen\n", encoding="utf-8")
    (inference_repo / "inference.py").write_text("# frozen worker\n", encoding="utf-8")
    (inference_repo / "broker.py").write_text("# frozen broker\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=inference_repo, check=True)
    subprocess.run(["git", "commit", "-qm", "freeze"], cwd=inference_repo, check=True)
    inference_git_commit = git_head(inference_repo)

    source_path = tmp_path / "source.jsonl"
    evidence_packet_path = tmp_path / "evidence_packets.jsonl"
    control_path = tmp_path / "control_pairs.jsonl"
    scientific_config_path = tmp_path / "scientific_config.json"
    environment_start_audit_path = tmp_path / "inference_environment_start_audit.json"
    environment_end_audit_path = tmp_path / "inference_environment_end_audit.json"
    causal_audio_schedule_path = tmp_path / "causal_audio_schedule.json"
    causal_audio_release_log_path = tmp_path / "causal_audio_release_log.json"
    causal_audio_broker_audit_path = tmp_path / "causal_audio_broker_audit.json"
    inference_contract_path = tmp_path / "inference_contract.json"
    inference_contract_ready_path = tmp_path / "inference_contract.ready.json"
    inference_result_attestation_path = tmp_path / "inference_result_attestation.json"
    target_path = tmp_path / "target.jsonl"
    outcome_artifact_root = tmp_path / "outcome-artifacts"
    outcome_commitment_path = tmp_path / "outcome-commitment.json"
    audio_root = tmp_path / "causal-audio"
    trajectory_path = tmp_path / "trajectories.jsonl"
    config_path = tmp_path / "config.json"
    output_root = tmp_path / "output"
    source_path.write_text(
        source(endpoint=2.0, boundary=1.5).model_dump_json() + "\n",
        encoding="utf-8",
    )
    config = minimal_config_dict()
    parsed_config = EventScoringConfig.model_validate(config)
    source_artifact_root = tmp_path / "source-artifacts"
    materialize_source_artifacts(source_artifact_root, parsed_config.expected_conditions)
    source_artifact_tree_sha256 = directory_tree_sha256(source_artifact_root)
    evidence_packet_path.write_text(
        "".join(
            json.dumps(packet.model_dump()) + "\n"
            for packet in evidence_packets_for_config(
                parsed_config,
                tokenize=tokenize,
                tokenizer_model=str(tokenizer_dir),
                tokenizer_artifact_sha256=tokenizer_artifact_sha256,
            )
        ),
        encoding="utf-8",
    )
    control_path.write_text(
        "".join(
            json.dumps(pair.model_dump()) + "\n"
            for pair in control_pairs_for_config(parsed_config, tokenize=tokenize)
        ),
        encoding="utf-8",
    )
    scientific_config_path.write_text(
        scientific_config(tokenizer_artifact_sha256).model_dump_json() + "\n",
        encoding="utf-8",
    )
    def audit_payload(phase):
        workers = [
            {
                "pid": 101,
                "parent_pid": 1,
                "process_start_time_ticks": 456,
                "command": (
                        f"python inference.py --run-id run-1 "
                        f"--inference-contract {inference_contract_path} "
                        f"--inference-contract-ready-file {inference_contract_ready_path} "
                        f"--scientific-config {scientific_config_path} "
                        f"--model-artifact-root {tokenizer_dir} "
                        f"--tokenizer-artifact-root {tokenizer_dir} "
                        f"--model-id fixture/model --model-revision {'d' * 40}"
                ),
                "marker_process": True,
                "executable_path": "/usr/bin/python3",
                "executable_sha256": "9" * 64,
                "working_directory": str(inference_repo),
                "entrypoint_path": str(inference_repo / "inference.py"),
                "entrypoint_sha256": "4" * 64,
                "environment_sha256": "5" * 64,
            }
        ]
        return {
            "schema_version": "acl6060_event_inference_environment_audit_v5",
            "run_id": "run-1",
            "container_name": "test-container",
            "container_id": "b" * 64,
            "container_image_id": "8" * 64,
            "container_read_only_rootfs": True,
            "container_network_mode": "none",
            "capture_host": "hyper00",
            "captured_at_utc": "2026-08-01T12:00:00Z",
            "capture_command": "docker_inspect_proc_tree_worker_discovery_git_v5",
            "capture_phase": phase,
            "worker_command_match": "--run-id run-1",
            "worker_processes": workers,
            "docker_inspect_sha256": "c" * 64,
            "process_listing_sha256": "7" * 64,
            "proc_open_files_sha256": "d" * 64,
            "process_identity_tree_sha256": worker_process_identity_tree_sha256(workers),
            "inference_repo_path": str(inference_repo),
            "inference_git_commit": inference_git_commit,
            "inference_git_status_sha256": hashlib.sha256(b"").hexdigest(),
            "forbidden_container_artifact_roots": ["/private/targets"],
            "forbidden_host_mount_source_roots": [
                str(target_path),
                str(outcome_commitment_path),
                str(outcome_artifact_root),
                str(audio_root),
            ],
            "observed_mounts": [
                {"source": "/host/public", "destination": "/data/public", "read_only": True},
                {
                    "source": str(scientific_config_path),
                    "destination": str(scientific_config_path),
                    "read_only": True,
                },
                {
                    "source": str(tokenizer_dir),
                    "destination": str(tokenizer_dir),
                    "read_only": True,
                },
            ],
            "process_open_file_paths": ["/data/public/audio.wav"],
            "forbidden_artifact_exposure_detected": False,
        }

    environment_start_audit_path.write_text(
        json.dumps(audit_payload("workers_start")), encoding="utf-8"
    )
    environment_end_audit_path.write_text(
        json.dumps(audit_payload("workers_end")), encoding="utf-8"
    )
    target_path.write_text(
        json.dumps(
            {
                "event_id": "e1",
                "acceptable_realizations": ["alpha"],
                "forbidden_realizations": ["beta"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config_path.write_text(json.dumps(config), encoding="utf-8")
    outcome_artifact_root.mkdir()
    outcome_roles = [
        "source_annotation_report",
        "source_adjudication",
        "target_annotation_report",
        "target_adjudication",
    ]
    outcome_artifacts = []
    for role in outcome_roles:
        path = outcome_artifact_root / f"{role}.json"
        path.write_text(json.dumps({"role": role}) + "\n", encoding="utf-8")
        outcome_artifacts.append(
            {
                "role": role,
                "relative_path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    outcome_commitment_path.write_text(
        json.dumps(
            {
                "schema_version": "acl6060_outcome_commitment_v1",
                "created_at_utc": "2026-08-01T11:58:00Z",
                "source_events_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                "target_scores_sha256": hashlib.sha256(target_path.read_bytes()).hexdigest(),
                "artifacts": outcome_artifacts,
            }
        ),
        encoding="utf-8",
    )
    audio_root.mkdir()
    provenance_path = audio_root / "provenance.json"
    provenance_path.write_text('{"fixture":true}\n', encoding="utf-8")
    pcm_by_acoustic = {}
    for acoustic_index, acoustic_condition in enumerate(config["expected_acoustic_conditions"]):
        pcm = struct.pack("<f", acoustic_index / 32.0) * 32_000
        pcm_path = audio_root / f"{acoustic_condition}.f32le"
        pcm_path.write_bytes(pcm)
        pcm_by_acoustic[acoustic_condition] = (pcm_path, pcm)
    causal_audio_schedule = {
        "schema_version": "acl6060_causal_audio_schedule_v3",
        "run_id": "run-1",
        "expected_conditions": parsed_config.expected_conditions,
        "source_audio_roots": [str(audio_root)],
        "sources": [
            {
                "source_id": f"source:t1:{acoustic_condition}",
                "talk_id": "t1",
                "acoustic_condition": acoustic_condition,
                "source_pcm_path": str(pcm_by_acoustic[acoustic_condition][0]),
                "source_pcm_sha256": hashlib.sha256(
                    pcm_by_acoustic[acoustic_condition][1]
                ).hexdigest(),
                "pcm_format": "float32le_mono",
                "sample_rate": 16_000,
                "total_sample_count": 32_000,
                "materialization_kind": (
                    "native"
                    if acoustic_condition == "native"
                    else "rir"
                    if acoustic_condition == "rir_medium_near"
                    else "generic_noise"
                    if acoustic_condition == "generic_noise_0_s0"
                    else "music"
                    if acoustic_condition == "music_0_s0"
                    else "babble"
                ),
                "upstream_audio_sha256": "5" * 64,
                "materializer_git_commit": inference_git_commit,
                "materializer_entrypoint_sha256": "3" * 64,
                "source_provenance_path": str(provenance_path),
                "source_provenance_sha256": hashlib.sha256(
                    provenance_path.read_bytes()
                ).hexdigest(),
            }
            for acoustic_condition in config["expected_acoustic_conditions"]
        ],
        "prefixes": [
            {
                "source_id": f"source:t1:{acoustic_condition}",
                "event_id": "e1",
                "acoustic_condition": acoustic_condition,
                "sequence_index": sequence_index,
                "audio_time_sec": float(sequence_index + 1),
                "prefix_id": f"prefix:e1:{acoustic_condition}:{sequence_index}",
                "prefix_pcm_sha256": hashlib.sha256(
                    pcm_by_acoustic[acoustic_condition][1][
                        : (sequence_index + 1) * 16_000 * 4
                    ]
                ).hexdigest(),
                "sample_rate": 16_000,
                "sample_count": (sequence_index + 1) * 16_000,
            }
            for acoustic_condition in config["expected_acoustic_conditions"]
            for sequence_index in range(2)
        ],
    }
    causal_audio_schedule_path.write_text(json.dumps(causal_audio_schedule), encoding="utf-8")
    causal_audio_schedule_sha256 = hashlib.sha256(
        causal_audio_schedule_path.read_bytes()
    ).hexdigest()
    broker_entrypoint_sha256 = hashlib.sha256(
        (inference_repo / "broker.py").read_bytes()
    ).hexdigest()
    broker_socket_path = Path("/tmp") / (
        f"slidesst-contract-{os.getpid()}-{tmp_path.name[-6:]}.sock"
    )
    broker_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    broker_socket.bind(str(broker_socket_path))
    causal_audio_broker_audit = {
        "schema_version": "acl6060_causal_audio_broker_audit_v2",
        "run_id": "run-1",
        "schedule_sha256": causal_audio_schedule_sha256,
        "broker_git_commit": inference_git_commit,
        "broker_repo_path": str(inference_repo),
        "broker_entrypoint_path": str(inference_repo / "broker.py"),
        "broker_entrypoint_sha256": broker_entrypoint_sha256,
        "broker_command": "python broker.py --run-id run-1",
        "broker_pid": os.getpid(),
        "socket_path": str(broker_socket_path),
        "release_events_path": str(tmp_path / "release-events.jsonl"),
        "source_audio_roots": [str(audio_root)],
        "delivery_protocol": "length_prefixed_unix_socket_v1",
        "captured_at_utc": "2026-08-01T12:00:00Z",
    }
    causal_audio_broker_audit_path.write_text(
        json.dumps(causal_audio_broker_audit), encoding="utf-8"
    )
    causal_audio_broker_audit_sha256 = hashlib.sha256(
        causal_audio_broker_audit_path.read_bytes()
    ).hexdigest()
    build_inference_contract(
        run_id="run-1",
        source_events=source_path,
        source_artifact_root=source_artifact_root,
        evidence_packets=evidence_packet_path,
        control_pairs=control_path,
        scientific_config=scientific_config_path,
        scoring_config=config_path,
        model_artifact_root=tokenizer_dir,
        target_scores=target_path,
        outcome_commitment=outcome_commitment_path,
        outcome_artifact_root=outcome_artifact_root,
        causal_audio_schedule=causal_audio_schedule_path,
        causal_audio_broker_audit=causal_audio_broker_audit_path,
        tokenizer_artifact_root=tokenizer_dir,
        tokenizer_model=str(tokenizer_dir),
        tokenizer_revision=TOKENIZER_REVISION,
        environment_start_audit=environment_start_audit_path,
        worker_inference_contract_path=str(inference_contract_path),
        worker_contract_ready_file_path=str(inference_contract_ready_path),
        worker_scientific_config_path=str(scientific_config_path),
        worker_model_artifact_root_path=str(tokenizer_dir),
        worker_tokenizer_artifact_root_path=str(tokenizer_dir),
        scoring_protected_artifact_roots=[
            str(target_path),
            str(outcome_commitment_path),
            str(outcome_artifact_root),
        ],
        code_repo=inference_repo,
        output=inference_contract_path,
        ready_file=inference_contract_ready_path,
    )
    ready_payload = json.loads(inference_contract_ready_path.read_text(encoding="utf-8"))
    inference_contract_sha256 = hashlib.sha256(inference_contract_path.read_bytes()).hexdigest()
    assert ready_payload["inference_contract_sha256"] == inference_contract_sha256
    frozen_contract = wait_for_inference_contract_ready(
        run_id="run-1",
        inference_contract=inference_contract_path,
        ready_file=inference_contract_ready_path,
        timeout_sec=0.1,
    )
    assert frozen_contract.run_id == "run-1"
    assert load_frozen_scientific_config(
        frozen_contract,
        scientific_config_path,
    ).model_id == "fixture/model"
    broker_socket.close()
    broker_socket_path.unlink()
    interaction_rows = []
    previous_interaction_sha256 = hashlib.sha256(b"").hexdigest()
    for sequence_index in range(2):
        for acoustic_condition in config["expected_acoustic_conditions"]:
            for condition in config["expected_conditions"]:
                hypothesis = "alpha" if condition in {
                    "ocr",
                    "correct_semantic",
                    "correct_relation",
                } else ""
                prefix_sha256 = hashlib.sha256(
                    pcm_by_acoustic[acoustic_condition][1][
                        : (sequence_index + 1) * 16_000 * 4
                    ]
                ).hexdigest()
                release = {
                    "record_type": "prefix_release",
                    "source_id": f"source:t1:{acoustic_condition}",
                    "session_id": f"session:e1:{condition}:{acoustic_condition}",
                    "server_ordinal": len(interaction_rows),
                    "event_id": "e1",
                    "condition": condition,
                    "acoustic_condition": acoustic_condition,
                    "sequence_index": sequence_index,
                    "audio_time_sec": float(sequence_index + 1),
                    "prefix_id": f"prefix:e1:{acoustic_condition}:{sequence_index}",
                    "prefix_pcm_sha256": prefix_sha256,
                    "sample_count": (sequence_index + 1) * 16_000,
                    "request_id": (
                        f"release:e1:{condition}:{acoustic_condition}:{sequence_index}"
                    ),
                    "granted_monotonic_ns": len(interaction_rows) + 1,
                    "granted_at_utc": "2026-08-01T12:00:00Z",
                    "previous_record_sha256": previous_interaction_sha256,
                }
                release["record_sha256"] = canonical_json_sha256(release)
                previous_interaction_sha256 = release["record_sha256"]
                interaction_rows.append(release)
                observation = {
                    "audio_time_sec": float(sequence_index + 1),
                    "causal_audio_prefix_id": f"prefix:e1:{acoustic_condition}:{sequence_index}",
                    "causal_audio_prefix_sha256": prefix_sha256,
                    "hypothesis": hypothesis,
                }
                commit = {
                    "record_type": "observation_commit",
                    "source_id": f"source:t1:{acoustic_condition}",
                    "session_id": f"session:e1:{condition}:{acoustic_condition}",
                    "server_ordinal": len(interaction_rows),
                    "event_id": "e1",
                    "condition": condition,
                    "acoustic_condition": acoustic_condition,
                    "sequence_index": sequence_index,
                    "prefix_id": f"prefix:e1:{acoustic_condition}:{sequence_index}",
                    "prefix_pcm_sha256": prefix_sha256,
                    "observation_sha256": causal_observation_sha256(
                        run_id="run-1",
                        inference_contract_sha256=inference_contract_sha256,
                        event_id="e1",
                        condition=condition,
                        acoustic_condition=acoustic_condition,
                        sequence_index=sequence_index,
                        observation=TrajectoryObservation.model_validate(observation),
                    ),
                    "request_id": (
                        f"commit:e1:{condition}:{acoustic_condition}:{sequence_index}"
                    ),
                    "committed_monotonic_ns": len(interaction_rows) + 1,
                    "committed_at_utc": "2026-08-01T12:00:00Z",
                    "previous_record_sha256": previous_interaction_sha256,
                }
                commit["record_sha256"] = canonical_json_sha256(commit)
                previous_interaction_sha256 = commit["record_sha256"]
                interaction_rows.append(commit)
    causal_audio_release_log = {
        "schema_version": "acl6060_causal_audio_release_log_v3",
        "run_id": "run-1",
        "schedule_sha256": causal_audio_schedule_sha256,
        "broker_audit_sha256": causal_audio_broker_audit_sha256,
        "interactions": interaction_rows,
    }
    causal_audio_release_log_path.write_text(
        json.dumps(causal_audio_release_log), encoding="utf-8"
    )
    causal_audio_release_log_sha256 = hashlib.sha256(
        causal_audio_release_log_path.read_bytes()
    ).hexdigest()
    rows = []
    for acoustic_condition in config["expected_acoustic_conditions"]:
        for condition in config["expected_conditions"]:
            hypotheses = ["alpha", "alpha"] if condition in {
                "ocr",
                "correct_semantic",
                "correct_relation",
            } else ["", ""]
            rows.append(
                {
                    "event_id": "e1",
                    "talk_id": "t1",
                    "condition": condition,
                    "acoustic_condition": acoustic_condition,
                    "inference_run_id": "run-1",
                    "inference_contract_sha256": inference_contract_sha256,
                    "evidence_packet_id": f"packet-{condition}",
                    "evidence_packet_sha256": packet_hash(condition),
                        "observations": [
                            {
                                "audio_time_sec": float(index + 1),
                                "causal_audio_prefix_id": f"prefix:e1:{acoustic_condition}:{index}",
                                "causal_audio_prefix_sha256": hashlib.sha256(
                                    pcm_by_acoustic[acoustic_condition][1][
                                        : (index + 1) * 16_000 * 4
                                    ]
                                ).hexdigest(),
                                "hypothesis": hypothesis,
                            }
                            for index, hypothesis in enumerate(hypotheses)
                        ],
                }
            )
    trajectory_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    inference_result_attestation_path.write_text(
        json.dumps(
            {
                "schema_version": "acl6060_event_inference_result_attestation_v1",
                "run_id": "run-1",
                "created_at_utc": "2026-08-01T13:00:00Z",
                "inference_contract_sha256": inference_contract_sha256,
                "trajectories_sha256": hashlib.sha256(trajectory_path.read_bytes()).hexdigest(),
                "causal_audio_release_log_sha256": causal_audio_release_log_sha256,
                "environment_start_audit_sha256": hashlib.sha256(
                    environment_start_audit_path.read_bytes()
                ).hexdigest(),
                "environment_end_audit_sha256": hashlib.sha256(
                    environment_end_audit_path.read_bytes()
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    code_root = Path(__file__).parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(code_root / "scripts" / "score_event_trajectories.py"),
            "--source-events",
            str(source_path),
            "--evidence-packets",
            str(evidence_packet_path),
            "--control-pairs",
            str(control_path),
            "--scientific-config",
            str(scientific_config_path),
            "--inference-contract",
            str(inference_contract_path),
            "--inference-result-attestation",
            str(inference_result_attestation_path),
            "--inference-environment-start-audit",
            str(environment_start_audit_path),
            "--inference-environment-end-audit",
            str(environment_end_audit_path),
            "--causal-audio-schedule",
            str(causal_audio_schedule_path),
            "--causal-audio-release-log",
            str(causal_audio_release_log_path),
            "--causal-audio-broker-audit",
            str(causal_audio_broker_audit_path),
            "--inference-repo",
            str(inference_repo),
            "--model-artifact-root",
            str(tokenizer_dir),
            "--tokenizer-artifact-root",
            str(tokenizer_dir),
            "--source-artifact-root",
            str(source_artifact_root),
            "--outcome-commitment",
            str(outcome_commitment_path),
            "--outcome-artifact-root",
            str(outcome_artifact_root),
            "--target-scores",
            str(target_path),
            "--trajectories",
            str(trajectory_path),
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
        ],
        cwd=code_root,
        env={**os.environ, "PYTHONPATH": "src"},
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads((output_root / "event_timing_summary.json").read_text())
    assert summary["event_count"] == 1
    assert summary["trajectory_count"] == 144
    assert len(summary["contrasts"]) == 32
    assert len(summary["noise_interactions"]) == 28
    assert len(summary["babble_severity_curves"]) == 4
    assert summary["contrasts"][0]["talk_cluster_bootstrap_samples"] == 10000
    assert len(summary["input_sha256"]) == 19
    assert {
        "causal_audio_schedule",
        "causal_audio_release_log",
        "causal_audio_broker_audit",
    } <= set(summary["input_sha256"])
    assert summary["event_scores_sha256"]
