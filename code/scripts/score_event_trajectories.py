#!/usr/bin/env python3
"""Score frozen SimulST event trajectories without exposing targets to inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from slidesst.eval.causal_audio import verify_causal_audio_source_bytes
from slidesst.eval.event_timing import (
    ControlPairSpec,
    CausalAudioBrokerAudit,
    CausalAudioReleaseLog,
    CausalAudioSchedule,
    EvidencePacketSpec,
    EventScoringConfig,
    EventTrajectory,
    InferenceEnvironmentAudit,
    InferenceContract,
    InferenceResultAttestation,
    InferenceScientificConfig,
    OutcomeCommitment,
    SourceEventTiming,
    TargetEventSpec,
    apply_development_gate,
    directory_tree_sha256,
    joint_talk_cluster_bootstrap,
    score_trajectory,
    summarize_acoustic_interaction,
    summarize_babble_severity_curve,
    summarize_contrast,
    validate_complete_matrix,
    validate_causal_audio_provenance,
    validate_control_pairs,
    validate_evidence_packets,
    validate_inference_provenance,
    validate_outcome_commitment,
)


def load_jsonl(path: Path, model_type):
    return [
        model_type.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_jsonl_bytes(payload: bytes, model_type):
    return [
        model_type.model_validate_json(line)
        for line in payload.splitlines()
        if line.strip()
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(path: Path) -> str:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError("inference repository checkout is dirty or has untracked files")
    return head


def load_tokenize(tokenizer_artifact_root: Path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_artifact_root,
        local_files_only=True,
        trust_remote_code=False,
    )

    def tokenize(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    return tokenize


def validate_config(config: dict) -> EventScoringConfig:
    return EventScoringConfig.model_validate(config)


def validate_target_is_protected(path: Path, forbidden_roots: list[str]) -> None:
    target = path.resolve()
    protected = False
    for value in forbidden_roots:
        root = Path(value).resolve()
        if target == root or root in target.parents:
            protected = True
            break
    if not protected:
        raise ValueError("target scoring artifact is outside manifest forbidden roots")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-events", type=Path, required=True)
    parser.add_argument("--evidence-packets", type=Path, required=True)
    parser.add_argument("--control-pairs", type=Path, required=True)
    parser.add_argument("--scientific-config", type=Path, required=True)
    parser.add_argument("--inference-contract", type=Path, required=True)
    parser.add_argument("--inference-result-attestation", type=Path, required=True)
    parser.add_argument("--inference-environment-start-audit", type=Path, required=True)
    parser.add_argument("--inference-environment-end-audit", type=Path, required=True)
    parser.add_argument("--causal-audio-schedule", type=Path, required=True)
    parser.add_argument("--causal-audio-release-log", type=Path, required=True)
    parser.add_argument("--causal-audio-broker-audit", type=Path, required=True)
    parser.add_argument("--inference-repo", type=Path, required=True)
    parser.add_argument("--model-artifact-root", type=Path, required=True)
    parser.add_argument("--tokenizer-artifact-root", type=Path, required=True)
    parser.add_argument("--source-artifact-root", type=Path, required=True)
    parser.add_argument("--outcome-commitment", type=Path, required=True)
    parser.add_argument("--outcome-artifact-root", type=Path, required=True)
    parser.add_argument("--target-scores", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"scoring output root already exists: {args.output_root}")

    input_paths = {
        "source_events": args.source_events,
        "evidence_packets": args.evidence_packets,
        "control_pairs": args.control_pairs,
        "scientific_config": args.scientific_config,
        "inference_contract": args.inference_contract,
        "inference_result_attestation": args.inference_result_attestation,
        "inference_environment_start_audit": args.inference_environment_start_audit,
        "inference_environment_end_audit": args.inference_environment_end_audit,
        "causal_audio_schedule": args.causal_audio_schedule,
        "causal_audio_release_log": args.causal_audio_release_log,
        "causal_audio_broker_audit": args.causal_audio_broker_audit,
        "outcome_commitment": args.outcome_commitment,
        "target_scores": args.target_scores,
        "trajectories": args.trajectories,
        "config": args.config,
    }
    input_bytes = {name: path.read_bytes() for name, path in input_paths.items()}
    input_sha256 = {
        name: hashlib.sha256(payload).hexdigest() for name, payload in input_bytes.items()
    }

    config = validate_config(json.loads(input_bytes["config"]))
    scientific_config = InferenceScientificConfig.model_validate_json(
        input_bytes["scientific_config"]
    )
    sources = load_jsonl_bytes(input_bytes["source_events"], SourceEventTiming)
    evidence_packets = load_jsonl_bytes(input_bytes["evidence_packets"], EvidencePacketSpec)
    control_pairs = load_jsonl_bytes(input_bytes["control_pairs"], ControlPairSpec)
    inference_contract = InferenceContract.model_validate_json(
        input_bytes["inference_contract"]
    )
    inference_result_attestation = InferenceResultAttestation.model_validate_json(
        input_bytes["inference_result_attestation"]
    )
    inference_environment_start_audit = InferenceEnvironmentAudit.model_validate_json(
        input_bytes["inference_environment_start_audit"]
    )
    inference_environment_end_audit = InferenceEnvironmentAudit.model_validate_json(
        input_bytes["inference_environment_end_audit"]
    )
    causal_audio_schedule = CausalAudioSchedule.model_validate_json(
        input_bytes["causal_audio_schedule"]
    )
    causal_audio_release_log = CausalAudioReleaseLog.model_validate_json(
        input_bytes["causal_audio_release_log"]
    )
    causal_audio_broker_audit = CausalAudioBrokerAudit.model_validate_json(
        input_bytes["causal_audio_broker_audit"]
    )
    outcome_commitment = OutcomeCommitment.model_validate_json(
        input_bytes["outcome_commitment"]
    )
    validate_target_is_protected(
        args.target_scores,
        inference_contract.scoring_protected_artifact_roots,
    )
    validate_target_is_protected(
        args.outcome_artifact_root,
        inference_contract.scoring_protected_artifact_roots,
    )
    validate_target_is_protected(
        args.outcome_commitment,
        inference_contract.scoring_protected_artifact_roots,
    )
    targets = load_jsonl_bytes(input_bytes["target_scores"], TargetEventSpec)
    trajectories = load_jsonl_bytes(input_bytes["trajectories"], EventTrajectory)
    tokenizer_artifact_sha256 = directory_tree_sha256(args.tokenizer_artifact_root)
    if tokenizer_artifact_sha256 != inference_contract.tokenizer_artifact_sha256:
        raise ValueError("inference contract tokenizer artifact hash mismatch")
    tokenize = load_tokenize(args.tokenizer_artifact_root)
    model_artifact_tree_sha256 = directory_tree_sha256(args.model_artifact_root)
    source_by_id, target_by_id, trajectories = validate_complete_matrix(
        sources,
        targets,
        trajectories,
        expected_conditions=config.expected_conditions,
        expected_acoustic_conditions=config.expected_acoustic_conditions,
    )
    inference_contract_sha256 = input_sha256["inference_contract"]
    validate_inference_provenance(
        inference_contract,
        inference_result_attestation,
        inference_environment_start_audit,
        inference_environment_end_audit,
        contract_sha256=inference_contract_sha256,
        trajectories_sha256=input_sha256["trajectories"],
        source_events_sha256=input_sha256["source_events"],
        evidence_packets_sha256=input_sha256["evidence_packets"],
        control_pairs_sha256=input_sha256["control_pairs"],
        scientific_config_sha256=input_sha256["scientific_config"],
        scoring_config_sha256=input_sha256["config"],
        target_scores_sha256=input_sha256["target_scores"],
        environment_start_audit_sha256=input_sha256[
            "inference_environment_start_audit"
        ],
        environment_end_audit_sha256=input_sha256["inference_environment_end_audit"],
        config=config,
        scientific_config=scientific_config,
        model_artifact_tree_sha256=model_artifact_tree_sha256,
        trajectories=trajectories,
        expected_git_commit=git_head(args.inference_repo),
    )
    validate_outcome_commitment(
        outcome_commitment,
        artifact_root=args.outcome_artifact_root,
        commitment_sha256=input_sha256["outcome_commitment"],
        artifact_tree_sha256=directory_tree_sha256(args.outcome_artifact_root),
        contract=inference_contract,
    )
    validate_causal_audio_provenance(
        inference_contract,
        inference_result_attestation,
        causal_audio_schedule,
        causal_audio_release_log,
        causal_audio_broker_audit,
        schedule_sha256=input_sha256["causal_audio_schedule"],
        release_log_sha256=input_sha256["causal_audio_release_log"],
        broker_audit_sha256=input_sha256["causal_audio_broker_audit"],
        expected_broker_git_commit=git_head(Path(causal_audio_broker_audit.broker_repo_path)),
        observed_broker_entrypoint_sha256=sha256_file(
            Path(causal_audio_broker_audit.broker_entrypoint_path)
        ),
        config=config,
        source_by_id=source_by_id,
        trajectories=trajectories,
    )
    verify_causal_audio_source_bytes(causal_audio_schedule)
    evidence_packet_by_key = validate_evidence_packets(
        evidence_packets,
        source_by_id=source_by_id,
        trajectories=trajectories,
        config=config,
        expected_tokenizer_model=inference_contract.tokenizer_model,
        expected_tokenizer_revision=inference_contract.tokenizer_revision,
        expected_tokenizer_artifact_sha256=tokenizer_artifact_sha256,
        expected_source_artifact_tree_sha256=inference_contract.source_artifact_tree_sha256,
        source_artifact_root=args.source_artifact_root,
        tokenize=tokenize,
    )
    if directory_tree_sha256(args.tokenizer_artifact_root) != tokenizer_artifact_sha256:
        raise ValueError("tokenizer artifact tree changed during validation")
    if directory_tree_sha256(args.model_artifact_root) != model_artifact_tree_sha256:
        raise ValueError("model artifact tree changed during validation")
    validate_control_pairs(
        control_pairs,
        source_by_id=source_by_id,
        trajectories=trajectories,
        config=config,
        evidence_packet_by_key=evidence_packet_by_key,
    )
    min_stability = config.min_stability_observations
    scores = [
        score_trajectory(
            source_by_id[trajectory.event_id],
            target_by_id[trajectory.event_id],
            trajectory,
            min_stability_observations=min_stability,
        )
        for trajectory in trajectories
    ]

    contrasts = []
    interactions = []
    babble_curves = []
    native = config.native_acoustic_group
    for contrast_index, contrast_spec in enumerate(config.contrasts):
        by_group = {}
        for group_index, acoustic_group in enumerate(config.acoustic_groups):
            summary = summarize_contrast(
                scores,
                first=contrast_spec.first,
                second=contrast_spec.second,
                acoustic_group=acoustic_group.id,
                acoustic_conditions=acoustic_group.members,
                bootstrap_samples=config.bootstrap_samples,
                bootstrap_seed=config.bootstrap_seed + contrast_index * 100 + group_index * 2,
            )
            apply_development_gate(summary, config.development_signal)
            summary["contrast_id"] = contrast_spec.id
            contrasts.append(summary)
            by_group[acoustic_group.id] = summary
        joint_bootstrap = joint_talk_cluster_bootstrap(
            by_group,
            native_group=native,
            severity_order=config.babble_severity_order,
            samples=config.bootstrap_samples,
            seed=config.bootstrap_seed + contrast_index * 1000 + 500,
        )
        for acoustic_group_id, summary in by_group.items():
            if acoustic_group_id == native:
                continue
            interaction = summarize_acoustic_interaction(by_group[native], summary)
            interaction["talk_cluster_bootstrap_ci95"] = joint_bootstrap[
                "interaction_ci95_by_acoustic_group"
            ][acoustic_group_id]
            interaction["talk_cluster_bootstrap_samples"] = joint_bootstrap["samples"]
            interaction["talk_cluster_bootstrap_seed"] = joint_bootstrap["seed"]
            interaction["contrast_id"] = contrast_spec.id
            interactions.append(interaction)
        curve = summarize_babble_severity_curve(by_group, config.babble_severity_order)
        curve["talk_cluster_correlation_ci95"] = joint_bootstrap[
            "severity_correlation_ci95"
        ]
        curve["talk_cluster_correlation_defined_samples"] = joint_bootstrap[
            "severity_correlation_defined_samples"
        ]
        curve["talk_cluster_correlation_undefined_samples"] = joint_bootstrap[
            "severity_correlation_undefined_samples"
        ]
        curve["talk_cluster_correlation_interval_status"] = joint_bootstrap[
            "severity_correlation_interval_status"
        ]
        curve["talk_cluster_monotonic_bootstrap_probability"] = joint_bootstrap[
            "severity_monotonic_bootstrap_probability"
        ]
        curve["talk_cluster_bootstrap_samples"] = joint_bootstrap["samples"]
        curve["talk_cluster_bootstrap_seed"] = joint_bootstrap["seed"]
        curve["contrast_id"] = contrast_spec.id
        babble_curves.append(curve)

    args.output_root.mkdir(parents=True)
    score_path = args.output_root / "event_timing_scores.jsonl"
    with score_path.open("x", encoding="utf-8") as output:
        output.write(
            "".join(
                json.dumps(score.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
                for score in scores
            )
        )
    summary = {
        "schema_version": "acl6060_event_trajectory_analysis_v1",
        "scope": "exploratory_acl_dev_not_confirmatory",
        "event_count": len(source_by_id),
        "trajectory_count": len(trajectories),
        "primary_development_estimand": config.primary_development_estimand,
        "min_stability_observations": min_stability,
        "contrasts": contrasts,
        "noise_interactions": interactions,
        "babble_severity_curves": babble_curves,
        "input_sha256": {
            "source_events": input_sha256["source_events"],
            "source_artifact_tree": inference_contract.source_artifact_tree_sha256,
            "model_artifact_tree": model_artifact_tree_sha256,
            "outcome_commitment": input_sha256["outcome_commitment"],
            "outcome_artifact_tree": directory_tree_sha256(args.outcome_artifact_root),
            "evidence_packets": input_sha256["evidence_packets"],
            "control_pairs": input_sha256["control_pairs"],
            "scientific_config": input_sha256["scientific_config"],
            "inference_contract": inference_contract_sha256,
            "inference_result_attestation": input_sha256["inference_result_attestation"],
            "inference_environment_start_audit": input_sha256[
                "inference_environment_start_audit"
            ],
            "inference_environment_end_audit": input_sha256[
                "inference_environment_end_audit"
            ],
            "causal_audio_schedule": input_sha256["causal_audio_schedule"],
            "causal_audio_release_log": input_sha256["causal_audio_release_log"],
            "causal_audio_broker_audit": input_sha256["causal_audio_broker_audit"],
            "target_scores": input_sha256["target_scores"],
            "trajectories": input_sha256["trajectories"],
            "config": input_sha256["config"],
            "tokenizer_artifact_tree": tokenizer_artifact_sha256,
        },
        "event_scores_sha256": sha256_file(score_path),
    }
    with (args.output_root / "event_timing_summary.json").open("x", encoding="utf-8") as output:
        output.write(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event_count": len(source_by_id), "trajectory_count": len(trajectories)}))


if __name__ == "__main__":
    main()
