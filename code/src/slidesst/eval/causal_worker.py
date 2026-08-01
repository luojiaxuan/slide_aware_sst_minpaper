from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal

import numpy as np
from pydantic import Field, TypeAdapter, model_validator

from slidesst.eval.causal_audio import (
    CausalAudioObservationCommitRequest,
    CausalAudioRequest,
    commit_causal_audio_observation,
    request_causal_audio_prefix,
)
from slidesst.eval.event_timing import (
    CausalAudioBrokerAudit,
    CausalAudioPrefixSpec,
    CausalAudioSchedule,
    EventTrajectory,
    EvidencePacketSpec,
    InferenceContract,
    InferenceEnvironmentAudit,
    StrictModel,
    TrajectoryObservation,
    causal_observation_sha256,
    canonical_absolute_posix_path,
    command_contains_exact_marker,
    render_evidence_packet,
)


@dataclass(frozen=True)
class CausalGenerationInput:
    event_id: str
    talk_id: str
    condition: str
    acoustic_condition: str
    sequence_index: int
    sample_rate: int
    audio: np.ndarray
    evidence_text: str
    prompt_text: str


@dataclass(frozen=True)
class _ScheduledGeneration:
    talk_id: str
    condition: str
    prefix: CausalAudioPrefixSpec
    packet: EvidencePacketSpec


GenerateBatch = Callable[[list[CausalGenerationInput]], list[str]]


class CausalWorkerDone(StrictModel):
    schema_version: Literal["acl6060_causal_worker_done_v1"]
    run_id: str = Field(min_length=1)
    worker_id: str = Field(min_length=1)
    worker_index: int = Field(ge=0)
    worker_count: int = Field(gt=0)
    pid: int = Field(gt=0)
    process_start_time_ticks: int = Field(gt=0)
    worker_process_identity_tree_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    inference_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    causal_audio_schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_packets_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_path: str = Field(min_length=1)
    talk_ids: list[str] = Field(min_length=1)
    trajectory_count: int = Field(gt=0)
    trajectories_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_worker_identity(self) -> "CausalWorkerDone":
        if self.worker_index >= self.worker_count:
            raise ValueError("worker done index must be smaller than worker count")
        if self.worker_id != f"worker-{self.worker_index:02d}-of-{self.worker_count:02d}":
            raise ValueError("worker done id differs from worker index/count")
        if not canonical_absolute_posix_path(self.output_path):
            raise ValueError("worker done output path must be canonical and absolute")
        return self


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def process_start_time_ticks(pid: int) -> int:
    payload = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    fields_after_command = payload[payload.rfind(")") + 2 :].split()
    if len(fields_after_command) <= 19:
        raise ValueError(f"cannot read process start time from /proc/{pid}/stat")
    value = int(fields_after_command[19])
    if value <= 0:
        raise ValueError("process start time ticks must be positive")
    return value


def load_worker_inputs(
    *,
    contract: InferenceContract,
    schedule_path: Path,
    broker_audit_path: Path,
    evidence_packets_path: Path,
) -> tuple[CausalAudioSchedule, CausalAudioBrokerAudit, list[EvidencePacketSpec]]:
    schedule_bytes = schedule_path.read_bytes()
    broker_audit_bytes = broker_audit_path.read_bytes()
    evidence_bytes = evidence_packets_path.read_bytes()
    if sha256_bytes(schedule_bytes) != contract.causal_audio_schedule_sha256:
        raise ValueError("worker causal-audio schedule hash differs from contract")
    if sha256_bytes(broker_audit_bytes) != contract.causal_audio_broker_audit_sha256:
        raise ValueError("worker broker-audit hash differs from contract")
    if sha256_bytes(evidence_bytes) != contract.evidence_packets_sha256:
        raise ValueError("worker evidence-packet hash differs from contract")
    schedule = CausalAudioSchedule.model_validate_json(schedule_bytes)
    broker_audit = CausalAudioBrokerAudit.model_validate_json(broker_audit_bytes)
    packet_adapter = TypeAdapter(EvidencePacketSpec)
    packets = [
        packet_adapter.validate_json(line)
        for line in evidence_bytes.splitlines()
        if line.strip()
    ]
    if not packets:
        raise ValueError("worker received no evidence packets")
    if schedule.run_id != contract.run_id or broker_audit.run_id != contract.run_id:
        raise ValueError("worker causal-audio inputs use a different run id")
    if schedule.expected_conditions != contract.expected_conditions:
        raise ValueError("worker schedule condition matrix differs from contract")
    if broker_audit.schedule_sha256 != contract.causal_audio_schedule_sha256:
        raise ValueError("worker broker audit does not bind the schedule")
    return schedule, broker_audit, packets


def validate_worker_matrix(
    schedule: CausalAudioSchedule,
    packets: Iterable[EvidencePacketSpec],
    *,
    tokenizer_model: str | None = None,
    tokenizer_revision: str | None = None,
    tokenizer_artifact_sha256: str | None = None,
) -> dict[tuple[str, str], EvidencePacketSpec]:
    event_ids = {prefix.event_id for prefix in schedule.prefixes}
    packet_by_key: dict[tuple[str, str], EvidencePacketSpec] = {}
    for packet in packets:
        key = (packet.event_id, packet.condition)
        if key in packet_by_key:
            raise ValueError(f"duplicate worker evidence packet: {key}")
        packet_by_key[key] = packet
        if tokenizer_model is not None and packet.tokenizer_model != tokenizer_model:
            raise ValueError(f"worker packet tokenizer model mismatch: {key}")
        if tokenizer_revision is not None and packet.tokenizer_revision != tokenizer_revision:
            raise ValueError(f"worker packet tokenizer revision mismatch: {key}")
        if (
            tokenizer_artifact_sha256 is not None
            and packet.tokenizer_artifact_sha256 != tokenizer_artifact_sha256
        ):
            raise ValueError(f"worker packet tokenizer artifact mismatch: {key}")
    expected = {
        (event_id, condition)
        for event_id in event_ids
        for condition in schedule.expected_conditions
    }
    if set(packet_by_key) != expected:
        missing = sorted(expected - set(packet_by_key))
        extra = sorted(set(packet_by_key) - expected)
        raise ValueError(f"worker evidence matrix mismatch: missing={missing} extra={extra}")
    return packet_by_key


def verify_evidence_packet_tokenization(
    packets: Iterable[EvidencePacketSpec],
    tokenize: Callable[[str], list[int]],
) -> None:
    for packet in packets:
        rendered = render_evidence_packet(packet.packet_payload)
        if tokenize(rendered) != packet.token_ids:
            raise ValueError(
                "loaded model processor tokenizer differs from frozen evidence packet: "
                f"{packet.event_id}/{packet.condition}"
            )


def worker_talk_partition(
    schedule: CausalAudioSchedule,
    *,
    worker_index: int,
    worker_count: int,
) -> list[str]:
    if worker_count < 1 or not 0 <= worker_index < worker_count:
        raise ValueError("invalid worker index/count")
    talk_ids = sorted({source.talk_id for source in schedule.sources})
    selected = talk_ids[worker_index::worker_count]
    if not selected:
        raise ValueError("worker partition contains no talks")
    return selected


def _scheduled_by_talk(
    schedule: CausalAudioSchedule,
    packet_by_key: dict[tuple[str, str], EvidencePacketSpec],
    selected_talk_ids: set[str],
) -> dict[str, deque[_ScheduledGeneration]]:
    talk_by_source = {source.source_id: source.talk_id for source in schedule.sources}
    condition_rank = {
        condition: index for index, condition in enumerate(schedule.expected_conditions)
    }
    rows: dict[str, list[_ScheduledGeneration]] = defaultdict(list)
    for prefix in schedule.prefixes:
        talk_id = talk_by_source[prefix.source_id]
        if talk_id not in selected_talk_ids:
            continue
        for condition in schedule.expected_conditions:
            packet = packet_by_key[(prefix.event_id, condition)]
            if packet.available_sec > prefix.audio_time_sec + 1e-9:
                raise ValueError(
                    f"worker packet is unavailable at prefix release: {prefix.event_id}/{condition}"
                )
            rows[talk_id].append(
                _ScheduledGeneration(
                    talk_id=talk_id,
                    condition=condition,
                    prefix=prefix,
                    packet=packet,
                )
            )
    ordered = {}
    for talk_id, tasks in rows.items():
        tasks.sort(
            key=lambda task: (
                task.prefix.audio_time_sec,
                task.prefix.event_id,
                task.prefix.acoustic_condition,
                task.prefix.sequence_index,
                condition_rank[task.condition],
            )
        )
        ordered[talk_id] = deque(tasks)
    if set(ordered) != selected_talk_ids:
        raise ValueError("worker selected a talk with no scheduled prefixes")
    return ordered


def _session_id(worker_id: str, task: _ScheduledGeneration) -> str:
    payload = "\0".join(
        (
            worker_id,
            task.prefix.event_id,
            task.condition,
            task.prefix.acoustic_condition,
        )
    ).encode("utf-8")
    return f"session:{worker_id}:{hashlib.sha256(payload).hexdigest()[:24]}"


def run_causal_event_worker(
    *,
    run_id: str,
    worker_id: str,
    inference_contract_sha256: str,
    schedule: CausalAudioSchedule,
    packets: Iterable[EvidencePacketSpec],
    selected_talk_ids: Iterable[str],
    broker_socket: Path,
    prompt_template: str,
    generate_batch: GenerateBatch,
    tokenizer_model: str | None = None,
    tokenizer_revision: str | None = None,
    tokenizer_artifact_sha256: str | None = None,
) -> list[EventTrajectory]:
    if schedule.run_id != run_id:
        raise ValueError("worker run id differs from schedule")
    selected = set(selected_talk_ids)
    if not selected:
        raise ValueError("worker requires at least one talk")
    packet_by_key = validate_worker_matrix(
        schedule,
        packets,
        tokenizer_model=tokenizer_model,
        tokenizer_revision=tokenizer_revision,
        tokenizer_artifact_sha256=tokenizer_artifact_sha256,
    )
    queues = _scheduled_by_talk(schedule, packet_by_key, selected)
    observations: dict[tuple[str, str, str], list[TrajectoryObservation]] = defaultdict(list)
    packet_by_stream: dict[tuple[str, str, str], EvidencePacketSpec] = {}

    while any(queues.values()):
        released = []
        generation_inputs = []
        for talk_id in sorted(queues):
            if not queues[talk_id]:
                continue
            task = queues[talk_id][0]
            session_id = _session_id(worker_id, task)
            request = CausalAudioRequest(
                run_id=run_id,
                session_id=session_id,
                request_id=f"release:{worker_id}:{uuid.uuid4().hex}",
                event_id=task.prefix.event_id,
                condition=task.condition,
                acoustic_condition=task.prefix.acoustic_condition,
                sequence_index=task.prefix.sequence_index,
            )
            header, pcm = request_causal_audio_prefix(broker_socket, request)
            if (
                header["prefix_id"],
                header["prefix_pcm_sha256"],
                header["sample_rate"],
                header["sample_count"],
            ) != (
                task.prefix.prefix_id,
                task.prefix.prefix_pcm_sha256,
                task.prefix.sample_rate,
                task.prefix.sample_count,
            ):
                raise ValueError("broker response differs from worker schedule")
            evidence_text = render_evidence_packet(task.packet.packet_payload)
            generation_inputs.append(
                CausalGenerationInput(
                    event_id=task.prefix.event_id,
                    talk_id=talk_id,
                    condition=task.condition,
                    acoustic_condition=task.prefix.acoustic_condition,
                    sequence_index=task.prefix.sequence_index,
                    sample_rate=task.prefix.sample_rate,
                    audio=np.frombuffer(pcm, dtype="<f4").copy(),
                    evidence_text=evidence_text,
                    prompt_text=prompt_template.format(evidence=evidence_text),
                )
            )
            released.append((task, session_id))
        hypotheses = generate_batch(generation_inputs)
        if len(hypotheses) != len(released):
            raise ValueError("generation batch returned the wrong number of hypotheses")
        for (task, session_id), hypothesis in zip(released, hypotheses, strict=True):
            if not isinstance(hypothesis, str):
                raise TypeError("generation hypothesis must be text")
            observation = TrajectoryObservation(
                audio_time_sec=task.prefix.audio_time_sec,
                causal_audio_prefix_id=task.prefix.prefix_id,
                causal_audio_prefix_sha256=task.prefix.prefix_pcm_sha256,
                hypothesis=hypothesis,
            )
            observation_sha256 = causal_observation_sha256(
                run_id=run_id,
                inference_contract_sha256=inference_contract_sha256,
                event_id=task.prefix.event_id,
                condition=task.condition,
                acoustic_condition=task.prefix.acoustic_condition,
                sequence_index=task.prefix.sequence_index,
                observation=observation,
            )
            commit_causal_audio_observation(
                broker_socket,
                CausalAudioObservationCommitRequest(
                    run_id=run_id,
                    session_id=session_id,
                    request_id=f"commit:{worker_id}:{uuid.uuid4().hex}",
                    event_id=task.prefix.event_id,
                    condition=task.condition,
                    acoustic_condition=task.prefix.acoustic_condition,
                    sequence_index=task.prefix.sequence_index,
                    observation_sha256=observation_sha256,
                ),
            )
            stream_key = (
                task.prefix.event_id,
                task.condition,
                task.prefix.acoustic_condition,
            )
            observations[stream_key].append(observation)
            packet_by_stream[stream_key] = task.packet
            queues[task.talk_id].popleft()

    talk_by_source = {source.source_id: source.talk_id for source in schedule.sources}
    talk_by_event = {}
    for prefix in schedule.prefixes:
        talk_id = talk_by_source[prefix.source_id]
        previous = talk_by_event.setdefault(prefix.event_id, talk_id)
        if previous != talk_id:
            raise ValueError("worker event crosses talks")
    trajectories = []
    for stream_key in sorted(observations):
        event_id, condition, acoustic_condition = stream_key
        packet = packet_by_stream[stream_key]
        trajectories.append(
            EventTrajectory(
                event_id=event_id,
                talk_id=talk_by_event[event_id],
                condition=condition,
                acoustic_condition=acoustic_condition,
                inference_run_id=run_id,
                inference_contract_sha256=inference_contract_sha256,
                evidence_packet_id=packet.packet_id,
                evidence_packet_sha256=packet.packet_sha256,
                observations=observations[stream_key],
            )
        )
    return trajectories


def write_trajectory_shard(
    output_path: Path,
    done_path: Path,
    *,
    run_id: str,
    worker_id: str,
    worker_index: int,
    worker_count: int,
    pid: int,
    process_start_time_ticks: int,
    worker_process_identity_tree_sha256: str,
    inference_contract_sha256: str,
    causal_audio_schedule_sha256: str,
    evidence_packets_sha256: str,
    talk_ids: list[str],
    trajectories: list[EventTrajectory],
) -> dict:
    if not canonical_absolute_posix_path(output_path.as_posix()):
        raise ValueError("trajectory shard output path must be canonical and absolute")
    if not canonical_absolute_posix_path(done_path.as_posix()):
        raise ValueError("worker done-marker path must be canonical and absolute")
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"trajectory shard already exists: {output_path}")
    if done_path.exists() or done_path.is_symlink():
        raise FileExistsError(f"worker done marker already exists: {done_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as output:
        for trajectory in trajectories:
            output.write(trajectory.model_dump_json() + "\n")
        output.flush()
        os.fsync(output.fileno())
    output_sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
    done = CausalWorkerDone(
        schema_version="acl6060_causal_worker_done_v1",
        run_id=run_id,
        worker_id=worker_id,
        worker_index=worker_index,
        worker_count=worker_count,
        pid=pid,
        process_start_time_ticks=process_start_time_ticks,
        worker_process_identity_tree_sha256=worker_process_identity_tree_sha256,
        inference_contract_sha256=inference_contract_sha256,
        causal_audio_schedule_sha256=causal_audio_schedule_sha256,
        evidence_packets_sha256=evidence_packets_sha256,
        output_path=output_path.as_posix(),
        talk_ids=talk_ids,
        trajectory_count=len(trajectories),
        trajectories_sha256=output_sha256,
    )
    with done_path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(done.model_dump(), indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())
    return done.model_dump()


def merge_trajectory_shards(
    *,
    contract: InferenceContract,
    inference_contract_sha256: str,
    schedule: CausalAudioSchedule,
    packets: Iterable[EvidencePacketSpec],
    environment_start_audit: InferenceEnvironmentAudit,
    shard_paths: list[Path],
    done_paths: list[Path],
    output_path: Path,
) -> list[EventTrajectory]:
    if len(shard_paths) != contract.expected_worker_count:
        raise ValueError("trajectory shard count differs from contract worker count")
    if len(done_paths) != len(shard_paths):
        raise ValueError("trajectory shard and done-marker counts differ")
    packet_by_key = validate_worker_matrix(
        schedule,
        packets,
        tokenizer_model=contract.tokenizer_model,
        tokenizer_revision=contract.tokenizer_revision,
        tokenizer_artifact_sha256=contract.tokenizer_artifact_sha256,
    )
    trajectory_adapter = TypeAdapter(EventTrajectory)
    if environment_start_audit.capture_phase != "workers_start":
        raise ValueError("merge requires a workers_start environment audit")
    if environment_start_audit.run_id != contract.run_id:
        raise ValueError("merge start-audit run id differs from contract")
    if (
        environment_start_audit.process_identity_tree_sha256
        != contract.worker_process_identity_tree_sha256
    ):
        raise ValueError("merge start-audit process tree differs from contract")
    marker_process_by_pid = {
        process.pid: process
        for process in environment_start_audit.worker_processes
        if process.marker_process
    }
    if len(marker_process_by_pid) != contract.expected_worker_count:
        raise ValueError("merge start-audit worker count differs from contract")
    workers = set()
    worker_indices = set()
    worker_pids = set()
    covered_talks = set()
    rows = []
    for shard_path, done_path in zip(shard_paths, done_paths, strict=True):
        shard_bytes = shard_path.read_bytes()
        done = CausalWorkerDone.model_validate_json(done_path.read_bytes())
        if done.run_id != contract.run_id:
            raise ValueError("worker done marker run id differs from contract")
        if done.worker_count != contract.expected_worker_count:
            raise ValueError("worker done count differs from contract")
        if done.inference_contract_sha256 != inference_contract_sha256:
            raise ValueError("worker done marker contract hash mismatch")
        if done.causal_audio_schedule_sha256 != contract.causal_audio_schedule_sha256:
            raise ValueError("worker done marker schedule hash mismatch")
        if done.evidence_packets_sha256 != contract.evidence_packets_sha256:
            raise ValueError("worker done marker evidence hash mismatch")
        if (
            done.worker_process_identity_tree_sha256
            != contract.worker_process_identity_tree_sha256
        ):
            raise ValueError("worker done marker process tree mismatch")
        if done.output_path != shard_path.as_posix():
            raise ValueError("worker done marker output path mismatch")
        if done.worker_id in workers:
            raise ValueError("duplicate causal worker id")
        if done.worker_index in worker_indices:
            raise ValueError("duplicate causal worker index")
        if done.pid in worker_pids:
            raise ValueError("duplicate causal worker pid")
        process = marker_process_by_pid.get(done.pid)
        if process is None:
            raise ValueError("worker done pid is absent from the start audit")
        if process.process_start_time_ticks != done.process_start_time_ticks:
            raise ValueError("worker done process start time differs from start audit")
        expected_talk_partition = worker_talk_partition(
            schedule,
            worker_index=done.worker_index,
            worker_count=done.worker_count,
        )
        if done.talk_ids != expected_talk_partition:
            raise ValueError("worker done talk partition is not deterministic")
        required_worker_arguments = (
            ("--run-id", contract.run_id),
            ("--worker-index", str(done.worker_index)),
            ("--worker-count", str(done.worker_count)),
            ("--inference-contract", contract.worker_inference_contract_path),
            (
                "--inference-contract-ready-file",
                contract.worker_contract_ready_file_path,
            ),
            ("--scientific-config", contract.worker_scientific_config_path),
            ("--model-artifact-root", contract.worker_model_artifact_root_path),
            (
                "--tokenizer-artifact-root",
                contract.worker_tokenizer_artifact_root_path,
            ),
            ("--model-id", contract.model_id),
            ("--model-revision", contract.model_revision),
            ("--output", done.output_path),
            ("--done-file", done_path.as_posix()),
        )
        if not all(
            command_contains_exact_marker(process.command, shlex.join(argument_pair))
            for argument_pair in required_worker_arguments
        ):
            raise ValueError("worker done marker does not match its audited command")
        if len(done.talk_ids) != len(set(done.talk_ids)):
            raise ValueError("worker done marker contains duplicate talks")
        if covered_talks & set(done.talk_ids):
            raise ValueError("causal worker talk partitions overlap")
        if sha256_bytes(shard_bytes) != done.trajectories_sha256:
            raise ValueError("worker trajectory shard hash mismatch")
        shard_rows = [
            trajectory_adapter.validate_json(line)
            for line in shard_bytes.splitlines()
            if line.strip()
        ]
        if len(shard_rows) != done.trajectory_count:
            raise ValueError("worker trajectory count differs from done marker")
        if any(row.talk_id not in done.talk_ids for row in shard_rows):
            raise ValueError("worker shard contains a talk outside its partition")
        workers.add(done.worker_id)
        worker_indices.add(done.worker_index)
        worker_pids.add(done.pid)
        covered_talks.update(done.talk_ids)
        rows.extend(shard_rows)

    if worker_indices != set(range(contract.expected_worker_count)):
        raise ValueError("causal worker indices do not cover the frozen allocation")
    if worker_pids != set(marker_process_by_pid):
        raise ValueError("causal worker shards do not cover the audited worker processes")

    talk_by_source = {source.source_id: source.talk_id for source in schedule.sources}
    expected_talks = set(talk_by_source.values())
    if covered_talks != expected_talks:
        raise ValueError("causal worker talk partitions do not cover the schedule")
    prefixes_by_stream: dict[tuple[str, str], list[CausalAudioPrefixSpec]] = defaultdict(list)
    talk_by_event = {}
    for prefix in schedule.prefixes:
        stream = (prefix.event_id, prefix.acoustic_condition)
        prefixes_by_stream[stream].append(prefix)
        talk_id = talk_by_source[prefix.source_id]
        previous = talk_by_event.setdefault(prefix.event_id, talk_id)
        if previous != talk_id:
            raise ValueError("scheduled event crosses talks")
    expected_streams = {
        (event_id, condition, acoustic_condition)
        for event_id, acoustic_condition in prefixes_by_stream
        for condition in schedule.expected_conditions
    }
    row_by_stream = {}
    for row in rows:
        stream = (row.event_id, row.condition, row.acoustic_condition)
        if stream in row_by_stream:
            raise ValueError(f"duplicate worker trajectory stream: {stream}")
        if row.inference_run_id != contract.run_id:
            raise ValueError(f"worker trajectory run id mismatch: {stream}")
        if row.inference_contract_sha256 != inference_contract_sha256:
            raise ValueError(f"worker trajectory contract hash mismatch: {stream}")
        if row.talk_id != talk_by_event.get(row.event_id):
            raise ValueError(f"worker trajectory talk mismatch: {stream}")
        packet = packet_by_key.get((row.event_id, row.condition))
        if packet is None or (
            row.evidence_packet_id,
            row.evidence_packet_sha256,
        ) != (
            packet.packet_id,
            packet.packet_sha256,
        ):
            raise ValueError(f"worker trajectory evidence packet mismatch: {stream}")
        expected_prefixes = sorted(
            prefixes_by_stream.get((row.event_id, row.acoustic_condition), []),
            key=lambda prefix: prefix.sequence_index,
        )
        observed = [
            (
                observation.audio_time_sec,
                observation.causal_audio_prefix_id,
                observation.causal_audio_prefix_sha256,
            )
            for observation in row.observations
        ]
        expected = [
            (
                prefix.audio_time_sec,
                prefix.prefix_id,
                prefix.prefix_pcm_sha256,
            )
            for prefix in expected_prefixes
        ]
        if observed != expected:
            raise ValueError(f"worker trajectory prefix sequence mismatch: {stream}")
        row_by_stream[stream] = row
    if set(row_by_stream) != expected_streams:
        missing = sorted(expected_streams - set(row_by_stream))
        extra = sorted(set(row_by_stream) - expected_streams)
        raise ValueError(f"merged worker trajectory matrix mismatch: missing={missing} extra={extra}")
    merged = [row_by_stream[key] for key in sorted(row_by_stream)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as output:
        for trajectory in merged:
            output.write(trajectory.model_dump_json() + "\n")
        output.flush()
        os.fsync(output.fileno())
    return merged


def wait_for_shutdown_file(
    shutdown_file: Path,
    *,
    timeout_sec: float,
    poll_interval_sec: float = 0.2,
) -> None:
    if timeout_sec <= 0 or poll_interval_sec <= 0:
        raise ValueError("shutdown wait parameters must be positive")
    deadline = time.monotonic() + timeout_sec
    while not shutdown_file.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for post-audit shutdown marker")
        time.sleep(poll_interval_sec)
    if shutdown_file.is_symlink() or not shutdown_file.is_file():
        raise ValueError("shutdown marker must be a regular non-symlink file")


def clean_generation(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    for token in ("<|im_end|>", "<|endoftext|>", "<think>", "</think>"):
        text = text.replace(token, "")
    text = text.strip().split("\n")[0].strip().strip('"')
    for prefix in ("Translation:", "Answer:", "译文：", "译文:"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    return " ".join(text.split())


class Qwen3OmniBatchGenerator:
    def __init__(
        self,
        *,
        model_artifact_root: Path,
        max_new_tokens: int,
        device_map: str,
        attention_implementation: str,
    ) -> None:
        import torch
        from transformers import (
            AutoConfig,
            Qwen3OmniMoeProcessor,
            Qwen3OmniMoeThinkerForConditionalGeneration,
        )

        root = str(model_artifact_root)
        self.torch = torch
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(
            root,
            local_files_only=True,
            trust_remote_code=True,
        )
        self.tokenizer = self.processor.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        config = AutoConfig.from_pretrained(
            root,
            local_files_only=True,
            trust_remote_code=True,
        )
        self.model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            root,
            config=config.thinker_config,
            local_files_only=True,
            trust_remote_code=True,
            dtype="auto",
            device_map=device_map,
            attn_implementation=attention_implementation,
        ).eval()
        self.max_new_tokens = max_new_tokens

    def __call__(self, batch: list[CausalGenerationInput]) -> list[str]:
        if not batch:
            return []
        sample_rates = {item.sample_rate for item in batch}
        if len(sample_rates) != 1:
            raise ValueError("Qwen3-Omni generation batch mixes sample rates")
        texts = []
        for item in batch:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": item.prompt_text},
                        {"type": "audio", "audio": item.audio},
                    ],
                }
            ]
            texts.append(
                self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            )
        inputs = self.processor(
            text=texts,
            audio=[item.audio for item in batch],
            images=None,
            videos=None,
            return_tensors="pt",
            padding=True,
            sampling_rate=next(iter(sample_rates)),
            use_audio_in_video=False,
        )
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        for key, value in list(inputs.items()):
            if hasattr(value, "to"):
                inputs[key] = (
                    value.to(device=device, dtype=dtype)
                    if key == "input_features"
                    else value.to(device=device)
                )
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        prompt_width = inputs["input_ids"].shape[1]
        if generated.shape[0] != len(batch):
            raise RuntimeError("Qwen3-Omni returned an unexpected batch size")
        return [
            clean_generation(
                self.tokenizer.decode(row[prompt_width:], skip_special_tokens=True)
            )
            for row in generated
        ]

    def tokenize_evidence(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)
