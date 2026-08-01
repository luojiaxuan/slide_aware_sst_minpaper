from __future__ import annotations

import hashlib
import json
import os
import shlex
import stat
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from slidesst.eval.causal_audio import git_head_clean, sha256_file
from slidesst.eval.event_timing import (
    CausalAudioBrokerAudit,
    CausalAudioSchedule,
    ControlPairSpec,
    EventScoringConfig,
    EvidencePacketSpec,
    InferenceContract,
    InferenceEnvironmentAudit,
    InferenceScientificConfig,
    OutcomeCommitment,
    SourceEventTiming,
    TargetEventSpec,
    command_contains_exact_marker,
    directory_tree_sha256,
    path_is_within,
    require_read_only_mount,
)


ModelT = TypeVar("ModelT", bound=BaseModel)


def _load_json(payload: bytes, path: Path) -> dict:
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _validate_jsonl(payload: bytes, path: Path, model_type: type[ModelT]) -> None:
    rows = [
        model_type.model_validate_json(line)
        for line in payload.splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"cannot freeze an empty JSONL input: {path}")


def _require_protected(path: Path, protected_roots: list[str]) -> None:
    resolved = path.resolve(strict=True).as_posix()
    if not any(path_is_within(resolved, root) for root in protected_roots):
        raise ValueError(f"scorer-private artifact is outside protected roots: {path}")


def _validate_outcome_artifacts(commitment: OutcomeCommitment, root: Path) -> None:
    resolved_root = root.resolve(strict=True)
    for artifact in commitment.artifacts:
        path = resolved_root / artifact.relative_path
        if path.is_symlink() or path.resolve(strict=True) != path:
            raise ValueError(f"outcome artifact is symlinked or non-canonical: {path}")
        if sha256_file(path) != artifact.sha256:
            raise ValueError(f"outcome artifact hash mismatch: {artifact.role}")


def _write_exclusive(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def build_inference_contract(
    *,
    run_id: str,
    source_events: Path,
    source_artifact_root: Path,
    evidence_packets: Path,
    control_pairs: Path,
    scientific_config: Path,
    scoring_config: Path,
    model_artifact_root: Path,
    target_scores: Path,
    outcome_commitment: Path,
    outcome_artifact_root: Path,
    causal_audio_schedule: Path,
    causal_audio_broker_audit: Path,
    tokenizer_artifact_root: Path,
    tokenizer_model: str,
    tokenizer_revision: str,
    environment_start_audit: Path,
    worker_inference_contract_path: str,
    worker_contract_ready_file_path: str,
    worker_scientific_config_path: str,
    worker_model_artifact_root_path: str,
    scoring_protected_artifact_roots: list[str],
    code_repo: Path,
    output: Path,
    ready_file: Path,
) -> InferenceContract:
    if output.exists() or ready_file.exists():
        raise FileExistsError("inference contract and ready marker must be created exactly once")
    if output.resolve() == ready_file.resolve():
        raise ValueError("inference contract and ready marker paths must differ")

    input_paths = {
        "source_events": source_events,
        "evidence_packets": evidence_packets,
        "control_pairs": control_pairs,
        "scientific_config": scientific_config,
        "scoring_config": scoring_config,
        "target_scores": target_scores,
        "outcome_commitment": outcome_commitment,
        "causal_audio_schedule": causal_audio_schedule,
        "causal_audio_broker_audit": causal_audio_broker_audit,
        "environment_start_audit": environment_start_audit,
    }
    input_bytes = {name: path.read_bytes() for name, path in input_paths.items()}
    input_sha256 = {
        name: hashlib.sha256(payload).hexdigest() for name, payload in input_bytes.items()
    }
    _validate_jsonl(input_bytes["source_events"], source_events, SourceEventTiming)
    _validate_jsonl(input_bytes["evidence_packets"], evidence_packets, EvidencePacketSpec)
    _validate_jsonl(input_bytes["control_pairs"], control_pairs, ControlPairSpec)
    _validate_jsonl(input_bytes["target_scores"], target_scores, TargetEventSpec)
    scientific = InferenceScientificConfig.model_validate_json(
        input_bytes["scientific_config"]
    )
    config = EventScoringConfig.model_validate_json(input_bytes["scoring_config"])
    if scientific.expected_conditions != config.expected_conditions:
        raise ValueError("scientific and scoring configs use different condition matrices")
    model_artifact_tree_sha256 = directory_tree_sha256(model_artifact_root)
    if scientific.model_artifact_tree_sha256 != model_artifact_tree_sha256:
        raise ValueError("scientific config model artifact tree hash mismatch")
    schedule = CausalAudioSchedule.model_validate_json(
        input_bytes["causal_audio_schedule"]
    )
    broker_audit = CausalAudioBrokerAudit.model_validate_json(
        input_bytes["causal_audio_broker_audit"]
    )
    commitment = OutcomeCommitment.model_validate_json(
        input_bytes["outcome_commitment"]
    )
    start_audit = InferenceEnvironmentAudit.model_validate_json(
        input_bytes["environment_start_audit"]
    )

    if schedule.run_id != run_id or broker_audit.run_id != run_id or start_audit.run_id != run_id:
        raise ValueError("pre-run artifacts do not share the requested run id")
    if start_audit.capture_phase != "workers_start":
        raise ValueError("contract builder requires a workers_start environment audit")
    if broker_audit.schedule_sha256 != input_sha256["causal_audio_schedule"]:
        raise ValueError("broker audit does not bind the causal audio schedule")
    if broker_audit.source_audio_roots != schedule.source_audio_roots:
        raise ValueError("broker and schedule expose different causal audio roots")
    if sha256_file(Path(broker_audit.broker_entrypoint_path)) != broker_audit.broker_entrypoint_sha256:
        raise ValueError("broker entrypoint changed after its audit")
    broker_socket = Path(broker_audit.socket_path)
    if not broker_socket.exists() or not stat.S_ISSOCK(broker_socket.stat().st_mode):
        raise ValueError("causal audio broker audit was published before its socket was ready")
    try:
        os.kill(broker_audit.broker_pid, 0)
    except ProcessLookupError as exc:
        raise ValueError("causal audio broker process is no longer running") from exc

    git_commit = git_head_clean(code_repo.resolve(strict=True))
    if git_commit != start_audit.inference_git_commit:
        raise ValueError("contract checkout differs from inference start audit")
    if git_commit != broker_audit.broker_git_commit:
        raise ValueError("contract checkout differs from causal audio broker audit")

    source_events_sha256 = input_sha256["source_events"]
    target_scores_sha256 = input_sha256["target_scores"]
    if commitment.source_events_sha256 != source_events_sha256:
        raise ValueError("outcome commitment source-events hash mismatch")
    if commitment.target_scores_sha256 != target_scores_sha256:
        raise ValueError("outcome commitment target-scores hash mismatch")
    _validate_outcome_artifacts(commitment, outcome_artifact_root)

    normalized_scoring_roots = [Path(root).resolve(strict=True).as_posix() for root in scoring_protected_artifact_roots]
    scientific_config_host_path = scientific_config.resolve(strict=True).as_posix()
    model_artifact_host_root_path = model_artifact_root.resolve(strict=True).as_posix()
    _require_protected(target_scores, normalized_scoring_roots)
    _require_protected(outcome_commitment, normalized_scoring_roots)
    _require_protected(outcome_artifact_root, normalized_scoring_roots)
    require_read_only_mount(
        start_audit,
        source=scientific_config_host_path,
        destination=worker_scientific_config_path,
        label="scientific config",
    )
    require_read_only_mount(
        start_audit,
        source=model_artifact_host_root_path,
        destination=worker_model_artifact_root_path,
        label="model artifact tree",
    )

    marker_processes = [
        process for process in start_audit.worker_processes if process.marker_process
    ]
    required_worker_arguments = (
        ("--inference-contract", worker_inference_contract_path),
        ("--inference-contract-ready-file", worker_contract_ready_file_path),
        ("--scientific-config", worker_scientific_config_path),
        ("--model-artifact-root", worker_model_artifact_root_path),
        ("--model-id", scientific.model_id),
        ("--model-revision", scientific.model_revision),
    )
    if any(
        not all(
            command_contains_exact_marker(process.command, shlex.join(argument_pair))
            for argument_pair in required_worker_arguments
        )
        for process in marker_processes
    ):
        raise ValueError("inference workers are not waiting on the frozen contract barrier")
    contract = InferenceContract(
        schema_version="acl6060_event_inference_contract_v1",
        run_id=run_id,
        created_at_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        git_commit=git_commit,
        scientific_config_sha256=input_sha256["scientific_config"],
        scoring_config_sha256=input_sha256["scoring_config"],
        model_id=scientific.model_id,
        model_revision=scientific.model_revision,
        model_artifact_tree_sha256=model_artifact_tree_sha256,
        tokenizer_model=tokenizer_model,
        tokenizer_revision=tokenizer_revision,
        tokenizer_artifact_sha256=directory_tree_sha256(tokenizer_artifact_root),
        source_artifact_tree_sha256=directory_tree_sha256(source_artifact_root),
        source_events_sha256=source_events_sha256,
        evidence_packets_sha256=input_sha256["evidence_packets"],
        control_pairs_sha256=input_sha256["control_pairs"],
        target_scores_sha256=target_scores_sha256,
        outcome_commitment_sha256=input_sha256["outcome_commitment"],
        outcome_artifact_tree_sha256=directory_tree_sha256(outcome_artifact_root),
        causal_audio_schedule_sha256=input_sha256["causal_audio_schedule"],
        causal_audio_broker_audit_sha256=input_sha256["causal_audio_broker_audit"],
        causal_audio_protocol="external_talk_synchronized_prefix_broker_v2",
        causal_audio_broker_entrypoint_sha256=broker_audit.broker_entrypoint_sha256,
        expected_conditions=config.expected_conditions,
        expected_acoustic_conditions=config.expected_acoustic_conditions,
        target_artifact_mounted=False,
        reference_artifact_mounted=False,
        future_audio_access=False,
        forbidden_container_artifact_roots=start_audit.forbidden_container_artifact_roots,
        forbidden_host_mount_source_roots=start_audit.forbidden_host_mount_source_roots,
        scoring_protected_artifact_roots=normalized_scoring_roots,
        worker_command_match=start_audit.worker_command_match,
        worker_inference_contract_path=worker_inference_contract_path,
        worker_contract_ready_file_path=worker_contract_ready_file_path,
        scientific_config_host_path=scientific_config_host_path,
        worker_scientific_config_path=worker_scientific_config_path,
        model_artifact_host_root_path=model_artifact_host_root_path,
        worker_model_artifact_root_path=worker_model_artifact_root_path,
        expected_worker_count=len(marker_processes),
        inference_repo_path=start_audit.inference_repo_path,
        container_image_id=start_audit.container_image_id,
        environment_start_audit_sha256=input_sha256["environment_start_audit"],
        worker_process_identity_tree_sha256=start_audit.process_identity_tree_sha256,
    )
    contract_payload = contract.model_dump_json(indent=2) + "\n"
    contract_bytes = contract_payload.encode("utf-8")
    contract_sha256 = hashlib.sha256(contract_bytes).hexdigest()
    _write_exclusive(output, contract_payload)
    if output.read_bytes() != contract_bytes:
        raise ValueError("inference contract changed before ready-marker publication")
    ready_payload = {
        "schema_version": "acl6060_event_inference_contract_ready_v1",
        "run_id": run_id,
        "inference_contract_sha256": contract_sha256,
    }
    _write_exclusive(ready_file, json.dumps(ready_payload, indent=2) + "\n")
    return contract


def load_frozen_scientific_config(
    contract: InferenceContract,
    scientific_config: Path,
) -> InferenceScientificConfig:
    if scientific_config.as_posix() != contract.worker_scientific_config_path:
        raise ValueError("worker opened a different scientific config path")
    payload = scientific_config.read_bytes()
    if hashlib.sha256(payload).hexdigest() != contract.scientific_config_sha256:
        raise ValueError("worker scientific config hash mismatch")
    config = InferenceScientificConfig.model_validate_json(payload)
    if (
        config.model_id,
        config.model_revision,
        config.model_artifact_tree_sha256,
    ) != (
        contract.model_id,
        contract.model_revision,
        contract.model_artifact_tree_sha256,
    ):
        raise ValueError("worker scientific config model identity differs from contract")
    return config


def wait_for_inference_contract_ready(
    *,
    run_id: str,
    inference_contract: Path,
    ready_file: Path,
    timeout_sec: float,
    poll_interval_sec: float = 0.1,
) -> InferenceContract:
    if timeout_sec <= 0 or poll_interval_sec <= 0:
        raise ValueError("contract barrier timeout and polling interval must be positive")
    deadline = time.monotonic() + timeout_sec
    while not ready_file.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for inference contract ready marker")
        time.sleep(poll_interval_sec)
    if ready_file.is_symlink() or inference_contract.is_symlink():
        raise ValueError("inference contract barrier artifacts cannot be symlinks")
    ready_bytes = ready_file.read_bytes()
    contract_bytes = inference_contract.read_bytes()
    ready_payload = _load_json(ready_bytes, ready_file)
    expected_keys = {
        "schema_version",
        "run_id",
        "inference_contract_sha256",
    }
    if set(ready_payload) != expected_keys:
        raise ValueError("inference contract ready marker schema drift")
    if ready_payload["schema_version"] != "acl6060_event_inference_contract_ready_v1":
        raise ValueError("unsupported inference contract ready marker")
    if ready_payload["run_id"] != run_id:
        raise ValueError("inference contract ready marker run id mismatch")
    if ready_payload["inference_contract_sha256"] != hashlib.sha256(contract_bytes).hexdigest():
        raise ValueError("inference contract ready marker hash mismatch")
    contract = InferenceContract.model_validate_json(contract_bytes)
    if contract.run_id != run_id:
        raise ValueError("inference contract run id mismatch")
    if contract.worker_inference_contract_path != inference_contract.as_posix():
        raise ValueError("worker opened a different inference contract path")
    if contract.worker_contract_ready_file_path != ready_file.as_posix():
        raise ValueError("worker opened a different contract ready-file path")
    return contract
