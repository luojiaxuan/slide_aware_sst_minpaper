import hashlib
import json
import os
import struct
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from slidesst.eval.causal_audio import CausalAudioBroker, ThreadedUnixAudioServer
from slidesst.eval.causal_worker import (
    merge_trajectory_shards,
    run_causal_event_worker,
    verify_evidence_packet_tokenization,
    validate_worker_matrix,
    worker_talk_partition,
    write_trajectory_shard,
)
from slidesst.eval.event_timing import (
    CausalAudioSchedule,
    EvidencePacketPayload,
    EvidencePacketSpec,
    canonical_json_sha256,
    render_evidence_packet,
    text_sha256,
)


PROCESS_TREE_SHA256 = "7" * 64
CONTRACT_SHA256 = "6" * 64
SCHEDULE_SHA256 = "8" * 64
EVIDENCE_SHA256 = "9" * 64


def _done_kwargs(worker_index: int, worker_count: int, *, pid: int) -> dict:
    return {
        "worker_index": worker_index,
        "worker_count": worker_count,
        "pid": pid,
        "process_start_time_ticks": 456 + worker_index,
        "worker_process_identity_tree_sha256": PROCESS_TREE_SHA256,
        "inference_contract_sha256": CONTRACT_SHA256,
        "causal_audio_schedule_sha256": SCHEDULE_SHA256,
        "evidence_packets_sha256": EVIDENCE_SHA256,
    }


def _schedule(tmp_path: Path) -> CausalAudioSchedule:
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    sources = []
    prefixes = []
    for talk_index, talk_id in enumerate(("talk-1", "talk-2")):
        pcm = struct.pack(
            "<8f",
            *[float(talk_index + sample_index) / 16.0 for sample_index in range(8)],
        )
        pcm_path = audio_root / f"{talk_id}.f32le"
        pcm_path.write_bytes(pcm)
        provenance_path = audio_root / f"{talk_id}.json"
        provenance_path.write_text(
            json.dumps({"talk_id": talk_id}) + "\n",
            encoding="utf-8",
        )
        source_id = f"source:{talk_id}:native"
        sources.append(
            {
                "source_id": source_id,
                "talk_id": talk_id,
                "acoustic_condition": "native",
                "source_pcm_path": str(pcm_path),
                "source_pcm_sha256": hashlib.sha256(pcm).hexdigest(),
                "pcm_format": "float32le_mono",
                "sample_rate": 4,
                "total_sample_count": 8,
                "materialization_kind": "native",
                "upstream_audio_sha256": "3" * 64,
                "materializer_git_commit": "1" * 40,
                "materializer_entrypoint_sha256": "2" * 64,
                "source_provenance_path": str(provenance_path),
                "source_provenance_sha256": hashlib.sha256(
                    provenance_path.read_bytes()
                ).hexdigest(),
            }
        )
        for sequence_index in range(2):
            sample_count = (sequence_index + 1) * 4
            prefixes.append(
                {
                    "source_id": source_id,
                    "event_id": f"event-{talk_index + 1}",
                    "acoustic_condition": "native",
                    "sequence_index": sequence_index,
                    "audio_time_sec": float(sequence_index + 1),
                    "prefix_id": f"prefix:{talk_id}:{sequence_index}",
                    "prefix_pcm_sha256": hashlib.sha256(
                        pcm[: sample_count * 4]
                    ).hexdigest(),
                    "sample_rate": 4,
                    "sample_count": sample_count,
                }
            )
    return CausalAudioSchedule.model_validate(
        {
            "schema_version": "acl6060_causal_audio_schedule_v3",
            "run_id": "run-worker-test",
            "expected_conditions": ["audio_only", "empty"],
            "source_audio_roots": [str(audio_root)],
            "sources": sources,
            "prefixes": prefixes,
        }
    )


def _packet(event_id: str, condition: str) -> EvidencePacketSpec:
    context_kind = "none" if condition == "audio_only" else "empty"
    payload = EvidencePacketPayload(
        schema_version="acl6060_source_evidence_packet_v1",
        context_kind=context_kind,
        context_items=[],
    )
    token_ids = [] if context_kind == "none" else [1, 2, 3]
    return EvidencePacketSpec(
        event_id=event_id,
        condition=condition,
        packet_id=f"packet:{event_id}:{condition}",
        packet_sha256=canonical_json_sha256(payload),
        evidence_type="none" if condition == "audio_only" else "empty",
        evidence_role="baseline",
        available_sec=0.0,
        tokenizer_model="fixture/tokenizer",
        tokenizer_revision="4" * 40,
        tokenizer_artifact_sha256="5" * 64,
        token_ids=token_ids,
        token_ids_sha256=canonical_json_sha256(token_ids),
        rendered_text_sha256=text_sha256(render_evidence_packet(payload)),
        packet_payload=payload,
    )


def _packets() -> list[EvidencePacketSpec]:
    return [
        _packet(event_id, condition)
        for event_id in ("event-1", "event-2")
        for condition in ("audio_only", "empty")
    ]


def _same_time_multi_acoustic_schedule(tmp_path: Path) -> CausalAudioSchedule:
    audio_root = tmp_path / "same-time-audio"
    audio_root.mkdir()
    provenance_path = audio_root / "talk-1.json"
    provenance_path.write_text('{"talk_id":"talk-1"}\n', encoding="utf-8")
    sources = []
    prefixes = []
    for acoustic_index, acoustic_condition in enumerate(("native", "noisy")):
        pcm = struct.pack(
            "<8f",
            *[float(acoustic_index + sample_index) / 16.0 for sample_index in range(8)],
        )
        pcm_path = audio_root / f"{acoustic_condition}.f32le"
        pcm_path.write_bytes(pcm)
        source_id = f"source:talk-1:{acoustic_condition}"
        sources.append(
            {
                "source_id": source_id,
                "talk_id": "talk-1",
                "acoustic_condition": acoustic_condition,
                "source_pcm_path": str(pcm_path),
                "source_pcm_sha256": hashlib.sha256(pcm).hexdigest(),
                "pcm_format": "float32le_mono",
                "sample_rate": 4,
                "total_sample_count": 8,
                "materialization_kind": (
                    "native" if acoustic_condition == "native" else "generic_noise"
                ),
                "upstream_audio_sha256": "3" * 64,
                "materializer_git_commit": "1" * 40,
                "materializer_entrypoint_sha256": "2" * 64,
                "source_provenance_path": str(provenance_path),
                "source_provenance_sha256": hashlib.sha256(
                    provenance_path.read_bytes()
                ).hexdigest(),
            }
        )
        for event_id in ("event-a", "event-b"):
            for sequence_index in range(2):
                sample_count = (sequence_index + 1) * 4
                prefixes.append(
                    {
                        "source_id": source_id,
                        "event_id": event_id,
                        "acoustic_condition": acoustic_condition,
                        "sequence_index": sequence_index,
                        "audio_time_sec": float(sequence_index + 1),
                        "prefix_id": (
                            f"prefix:{event_id}:{acoustic_condition}:{sequence_index}"
                        ),
                        "prefix_pcm_sha256": hashlib.sha256(
                            pcm[: sample_count * 4]
                        ).hexdigest(),
                        "sample_rate": 4,
                        "sample_count": sample_count,
                    }
                )
    return CausalAudioSchedule.model_validate(
        {
            "schema_version": "acl6060_causal_audio_schedule_v3",
            "run_id": "run-same-time-test",
            "expected_conditions": ["audio_only", "empty"],
            "source_audio_roots": [str(audio_root)],
            "sources": sources,
            "prefixes": prefixes,
        }
    )


def test_causal_worker_batches_across_talks_and_keeps_stream_sessions_isolated(tmp_path):
    schedule = _schedule(tmp_path)
    release_events_path = tmp_path / "release-events.jsonl"
    socket_path = Path("/tmp") / f"slidesst-worker-{os.getpid()}-{tmp_path.name[-6:]}.sock"
    broker = CausalAudioBroker(schedule, release_events_path)
    server = ThreadedUnixAudioServer(str(socket_path), broker)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    batches = []

    def generate(batch):
        batches.append(batch)
        assert len({item.talk_id for item in batch}) == len(batch)
        return [
            f"{item.talk_id}/{item.condition}/{item.sequence_index}/{len(item.audio)}"
            for item in batch
        ]

    try:
        trajectories = run_causal_event_worker(
            run_id=schedule.run_id,
            worker_id="worker-00-of-01",
            inference_contract_sha256="6" * 64,
            schedule=schedule,
            packets=_packets(),
            selected_talk_ids=["talk-1", "talk-2"],
            broker_socket=socket_path,
            prompt_template="Evidence follows.\n{evidence}\nTranslate only the speech.",
            generate_batch=generate,
            tokenizer_model="fixture/tokenizer",
            tokenizer_revision="4" * 40,
            tokenizer_artifact_sha256="5" * 64,
        )
    finally:
        server.shutdown()
        server.server_close()
        broker.close()
        thread.join(timeout=2)
        socket_path.unlink(missing_ok=True)

    assert len(trajectories) == 4
    assert all(len(trajectory.observations) == 2 for trajectory in trajectories)
    assert len(batches[0]) == 2
    assert {item.talk_id for item in batches[0]} == {"talk-1", "talk-2"}
    interactions = [
        json.loads(line)
        for line in release_events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(interactions) == 16
    sessions_by_stream = {}
    for row in interactions:
        stream = (row["event_id"], row["condition"], row["acoustic_condition"])
        sessions_by_stream.setdefault(stream, set()).add(row["session_id"])
    assert all(len(sessions) == 1 for sessions in sessions_by_stream.values())
    assert len({next(iter(sessions)) for sessions in sessions_by_stream.values()}) == 4


def test_causal_worker_serializes_same_time_events_across_acoustic_conditions(tmp_path):
    schedule = _same_time_multi_acoustic_schedule(tmp_path)
    packets = [
        _packet(event_id, condition)
        for event_id in ("event-a", "event-b")
        for condition in schedule.expected_conditions
    ]
    release_events_path = tmp_path / "same-time-release-events.jsonl"
    socket_path = Path("/tmp") / (
        f"slidesst-same-time-{os.getpid()}-{tmp_path.name[-6:]}.sock"
    )
    broker = CausalAudioBroker(schedule, release_events_path)
    server = ThreadedUnixAudioServer(str(socket_path), broker)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        trajectories = run_causal_event_worker(
            run_id=schedule.run_id,
            worker_id="worker-00-of-01",
            inference_contract_sha256=CONTRACT_SHA256,
            schedule=schedule,
            packets=packets,
            selected_talk_ids=["talk-1"],
            broker_socket=socket_path,
            prompt_template="{evidence}\nTranslate only the speech.",
            generate_batch=lambda batch: [
                f"{item.event_id}/{item.acoustic_condition}/{item.sequence_index}"
                for item in batch
            ],
            tokenizer_model="fixture/tokenizer",
            tokenizer_revision="4" * 40,
            tokenizer_artifact_sha256="5" * 64,
        )
    finally:
        server.shutdown()
        server.server_close()
        broker.close()
        thread.join(timeout=2)
        socket_path.unlink(missing_ok=True)

    assert len(trajectories) == 8
    assert all(len(trajectory.observations) == 2 for trajectory in trajectories)
    interactions = [
        json.loads(line)
        for line in release_events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(interactions) == 32
    releases = [row for row in interactions if row["record_type"] == "prefix_release"]
    assert [row["audio_time_sec"] for row in releases] == [1.0] * 8 + [2.0] * 8
    sessions = {row["session_id"] for row in releases}
    assert len(sessions) == 8


def test_worker_rejects_incomplete_evidence_matrix(tmp_path):
    schedule = _schedule(tmp_path)
    with pytest.raises(ValueError, match="evidence matrix mismatch"):
        validate_worker_matrix(schedule, _packets()[:-1])


def test_worker_replays_frozen_evidence_with_loaded_tokenizer():
    packets = _packets()
    verify_evidence_packet_tokenization(
        packets,
        lambda text: [] if not text else [1, 2, 3],
    )
    with pytest.raises(ValueError, match="processor tokenizer differs"):
        verify_evidence_packet_tokenization(packets, lambda text: [99])


def test_worker_partition_is_deterministic_and_rejects_empty_shards(tmp_path):
    schedule = _schedule(tmp_path)
    assert worker_talk_partition(schedule, worker_index=0, worker_count=2) == ["talk-1"]
    assert worker_talk_partition(schedule, worker_index=1, worker_count=2) == ["talk-2"]
    with pytest.raises(ValueError, match="contains no talks"):
        worker_talk_partition(schedule, worker_index=2, worker_count=3)


def test_worker_writes_hash_bound_shard_and_done_marker(tmp_path):
    output = tmp_path / "worker.jsonl"
    done = tmp_path / "worker.done.json"
    schedule = _schedule(tmp_path)
    packet = _packet("event-1", "audio_only")
    trajectory = {
        "event_id": "event-1",
        "talk_id": "talk-1",
        "condition": "audio_only",
        "acoustic_condition": "native",
        "inference_run_id": schedule.run_id,
        "inference_contract_sha256": "6" * 64,
        "evidence_packet_id": packet.packet_id,
        "evidence_packet_sha256": packet.packet_sha256,
        "observations": [
            {
                "audio_time_sec": 1.0,
                "causal_audio_prefix_id": "prefix:talk-1:0",
                "causal_audio_prefix_sha256": schedule.prefixes[0].prefix_pcm_sha256,
                "hypothesis": "translation",
            }
        ],
    }
    from slidesst.eval.event_timing import EventTrajectory

    record = write_trajectory_shard(
        output,
        done,
        run_id=schedule.run_id,
        worker_id="worker-00-of-01",
        **_done_kwargs(0, 1, pid=101),
        talk_ids=["talk-1"],
        trajectories=[EventTrajectory.model_validate(trajectory)],
    )
    assert record["trajectories_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert json.loads(done.read_text())["trajectory_count"] == 1
    with pytest.raises(FileExistsError):
        write_trajectory_shard(
            output,
            done,
            run_id=schedule.run_id,
            worker_id="worker-00-of-01",
            **_done_kwargs(0, 1, pid=101),
            talk_ids=["talk-1"],
            trajectories=[],
        )

    other_output = tmp_path / "other.jsonl"
    with pytest.raises(FileExistsError, match="done marker already exists"):
        write_trajectory_shard(
            other_output,
            done,
            run_id=schedule.run_id,
            worker_id="worker-00-of-01",
            **_done_kwargs(0, 1, pid=101),
            talk_ids=["talk-1"],
            trajectories=[EventTrajectory.model_validate(trajectory)],
        )
    assert not other_output.exists()


def test_merge_worker_shards_requires_complete_disjoint_talk_matrix(tmp_path):
    schedule = _schedule(tmp_path)
    release_events_path = tmp_path / "merge-release-events.jsonl"
    socket_path = Path("/tmp") / f"slidesst-merge-{os.getpid()}-{tmp_path.name[-6:]}.sock"
    broker = CausalAudioBroker(schedule, release_events_path)
    server = ThreadedUnixAudioServer(str(socket_path), broker)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    shard_paths = []
    done_paths = []
    contract = SimpleNamespace(
        expected_worker_count=2,
        tokenizer_model="fixture/tokenizer",
        tokenizer_revision="4" * 40,
        tokenizer_artifact_sha256="5" * 64,
        run_id=schedule.run_id,
        causal_audio_schedule_sha256=SCHEDULE_SHA256,
        evidence_packets_sha256=EVIDENCE_SHA256,
        worker_process_identity_tree_sha256=PROCESS_TREE_SHA256,
        worker_inference_contract_path="/run/inference_contract.json",
        worker_contract_ready_file_path="/run/inference_contract.ready.json",
        worker_scientific_config_path="/run/scientific_config.json",
        worker_model_artifact_root_path="/models/frozen",
        worker_tokenizer_artifact_root_path="/tokenizers/frozen",
        model_id="fixture/model",
        model_revision="d" * 40,
    )
    try:
        for worker_index, talk_id in enumerate(("talk-1", "talk-2")):
            worker_id = f"worker-{worker_index:02d}-of-02"
            trajectories = run_causal_event_worker(
                run_id=schedule.run_id,
                worker_id=worker_id,
                inference_contract_sha256="6" * 64,
                schedule=schedule,
                packets=_packets(),
                selected_talk_ids=[talk_id],
                broker_socket=socket_path,
                prompt_template="{evidence}\nTranslate only the speech.",
                generate_batch=lambda batch: [item.condition for item in batch],
                tokenizer_model="fixture/tokenizer",
                tokenizer_revision="4" * 40,
                tokenizer_artifact_sha256="5" * 64,
            )
            shard_path = tmp_path / f"worker-{worker_index}.jsonl"
            done_path = tmp_path / f"worker-{worker_index}.done.json"
            write_trajectory_shard(
                shard_path,
                done_path,
                run_id=schedule.run_id,
                worker_id=worker_id,
                **_done_kwargs(worker_index, 2, pid=101 + worker_index),
                talk_ids=[talk_id],
                trajectories=trajectories,
            )
            shard_paths.append(shard_path)
            done_paths.append(done_path)
    finally:
        server.shutdown()
        server.server_close()
        broker.close()
        thread.join(timeout=2)
        socket_path.unlink(missing_ok=True)

    processes = []
    for worker_index, (shard_path, done_path) in enumerate(zip(shard_paths, done_paths)):
        processes.append(
            SimpleNamespace(
                pid=101 + worker_index,
                process_start_time_ticks=456 + worker_index,
                marker_process=True,
                command=(
                    f"python worker.py --run-id {schedule.run_id} "
                    f"--worker-index {worker_index} --worker-count 2 "
                    f"--inference-contract {contract.worker_inference_contract_path} "
                    "--inference-contract-ready-file "
                    f"{contract.worker_contract_ready_file_path} "
                    f"--scientific-config {contract.worker_scientific_config_path} "
                    f"--model-artifact-root {contract.worker_model_artifact_root_path} "
                    "--tokenizer-artifact-root "
                    f"{contract.worker_tokenizer_artifact_root_path} "
                    f"--model-id {contract.model_id} "
                    f"--model-revision {contract.model_revision} "
                    f"--output {shard_path} --done-file {done_path}"
                ),
            )
        )
    start_audit = SimpleNamespace(
        capture_phase="workers_start",
        run_id=schedule.run_id,
        process_identity_tree_sha256=PROCESS_TREE_SHA256,
        worker_processes=processes,
    )
    output = tmp_path / "trajectories.jsonl"
    merged = merge_trajectory_shards(
        contract=contract,
        inference_contract_sha256="6" * 64,
        schedule=schedule,
        packets=_packets(),
        environment_start_audit=start_audit,
        shard_paths=shard_paths,
        done_paths=done_paths,
        output_path=output,
    )
    assert len(merged) == 4
    assert len(output.read_text(encoding="utf-8").splitlines()) == 4
    wrong_pid_done = tmp_path / "wrong-pid.done.json"
    wrong_pid_payload = json.loads(done_paths[1].read_text(encoding="utf-8"))
    wrong_pid_payload["pid"] = 999
    wrong_pid_done.write_text(json.dumps(wrong_pid_payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="pid is absent from the start audit"):
        merge_trajectory_shards(
            contract=contract,
            inference_contract_sha256=CONTRACT_SHA256,
            schedule=schedule,
            packets=_packets(),
            environment_start_audit=start_audit,
            shard_paths=shard_paths,
            done_paths=[done_paths[0], wrong_pid_done],
            output_path=tmp_path / "wrong-pid.jsonl",
        )
    overlap_done = tmp_path / "overlap.done.json"
    overlap_payload = json.loads(done_paths[1].read_text(encoding="utf-8"))
    overlap_payload["talk_ids"] = ["talk-1"]
    overlap_done.write_text(json.dumps(overlap_payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="talk partition is not deterministic"):
        merge_trajectory_shards(
            contract=contract,
            inference_contract_sha256="6" * 64,
            schedule=schedule,
            packets=_packets(),
            environment_start_audit=start_audit,
            shard_paths=shard_paths,
            done_paths=[done_paths[0], overlap_done],
            output_path=tmp_path / "invalid.jsonl",
        )
