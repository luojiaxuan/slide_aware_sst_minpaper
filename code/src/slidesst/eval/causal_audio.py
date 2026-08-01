from __future__ import annotations

import hashlib
import json
import math
import os
import socket
import socketserver
import struct
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Callable, Literal

from pydantic import Field, TypeAdapter

from slidesst.eval.event_timing import (
    CausalAudioBrokerAudit,
    CausalAudioInteractionRecord,
    CausalAudioObservationCommitRecord,
    CausalAudioReleaseLog,
    CausalAudioReleaseRecord,
    CausalAudioSchedule,
    EMPTY_SHA256,
    StrictModel,
    canonical_json_sha256,
)


FLOAT32_BYTES = 4
MAX_JSON_FRAME_BYTES = 1024 * 1024


class CausalAudioRequest(StrictModel):
    action: Literal["release_prefix"] = "release_prefix"
    run_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    condition: str = Field(min_length=1)
    acoustic_condition: str = Field(min_length=1)
    sequence_index: int = Field(ge=0)


class CausalAudioObservationCommitRequest(StrictModel):
    action: Literal["commit_observation"] = "commit_observation"
    run_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    condition: str = Field(min_length=1)
    acoustic_condition: str = Field(min_length=1)
    sequence_index: int = Field(ge=0)
    observation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_regular_file(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_symlink() or path.resolve(strict=True) != path:
        raise ValueError(f"causal audio source path is symlinked or non-canonical: {path}")
    if not path.is_file():
        raise ValueError(f"causal audio source is not a regular file: {path}")
    return path


def verify_causal_audio_source_bytes(schedule: CausalAudioSchedule) -> None:
    prefixes_by_source = {}
    for prefix in schedule.prefixes:
        prefixes_by_source.setdefault(prefix.source_id, []).append(prefix)
    for source in schedule.sources:
        path = _verified_regular_file(source.source_pcm_path)
        provenance_path = _verified_regular_file(source.source_provenance_path)
        if sha256_file(provenance_path) != source.source_provenance_sha256:
            raise ValueError(f"causal audio provenance hash mismatch: {source.source_id}")
        expected_size = source.total_sample_count * FLOAT32_BYTES
        if path.stat().st_size != expected_size:
            raise ValueError(f"causal audio source byte length mismatch: {source.source_id}")
        if sha256_file(path) != source.source_pcm_sha256:
            raise ValueError(f"causal audio source hash mismatch: {source.source_id}")
        ordered = sorted(
            prefixes_by_source[source.source_id], key=lambda value: value.sample_count
        )
        digest = hashlib.sha256()
        consumed = 0
        with path.open("rb") as pcm:
            for prefix in ordered:
                target = prefix.sample_count * FLOAT32_BYTES
                remaining = target - consumed
                while remaining:
                    chunk = pcm.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError(
                            f"causal audio source ended before prefix: {prefix.prefix_id}"
                        )
                    digest.update(chunk)
                    consumed += len(chunk)
                    remaining -= len(chunk)
                if digest.copy().hexdigest() != prefix.prefix_pcm_sha256:
                    raise ValueError(f"causal audio prefix hash mismatch: {prefix.prefix_id}")


def _recv_exact(stream: BinaryIO, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            raise EOFError("socket closed during framed payload")
        chunks.extend(chunk)
    return bytes(chunks)


def receive_json_frame(stream: BinaryIO) -> dict | None:
    size_bytes = stream.read(4)
    if not size_bytes:
        return None
    if len(size_bytes) != 4:
        raise EOFError("socket closed during frame length")
    size = struct.unpack("!I", size_bytes)[0]
    if size == 0 or size > MAX_JSON_FRAME_BYTES:
        raise ValueError("invalid causal audio JSON frame length")
    payload = json.loads(_recv_exact(stream, size))
    if not isinstance(payload, dict):
        raise ValueError("causal audio request frame must contain a JSON object")
    return payload


def send_json_frame(stream: BinaryIO, payload: dict) -> None:
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    if not encoded or len(encoded) > MAX_JSON_FRAME_BYTES:
        raise ValueError("invalid causal audio JSON frame length")
    stream.write(struct.pack("!I", len(encoded)))
    stream.write(encoded)
    stream.flush()


def send_prefix_frame(stream: BinaryIO, header: dict, pcm: bytes = b"") -> None:
    payload = json.dumps(header, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    stream.write(struct.pack("!I", len(payload)))
    stream.write(payload)
    if pcm:
        stream.write(pcm)
    stream.flush()


def request_causal_audio_prefix(
    socket_path: Path,
    request: CausalAudioRequest,
) -> tuple[dict, bytes]:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(str(socket_path))
        with client.makefile("rwb", buffering=0) as stream:
            send_json_frame(stream, request.model_dump())
            header = receive_json_frame(stream)
            if header is None:
                raise EOFError("causal audio broker closed without a response")
            if header.get("status") != "ok":
                raise RuntimeError(
                    f"causal audio broker rejected request: {header.get('message', 'unknown error')}"
                )
            byte_count = header.get("pcm_byte_count")
            if not isinstance(byte_count, int) or byte_count <= 0:
                raise ValueError("causal audio broker returned an invalid PCM byte count")
            pcm = _recv_exact(stream, byte_count)
            if hashlib.sha256(pcm).hexdigest() != header.get("prefix_pcm_sha256"):
                raise ValueError("causal audio broker response hash mismatch")
            return header, pcm


def commit_causal_audio_observation(
    socket_path: Path,
    request: CausalAudioObservationCommitRequest,
) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(str(socket_path))
        with client.makefile("rwb", buffering=0) as stream:
            send_json_frame(stream, request.model_dump())
            response = receive_json_frame(stream)
            if response is None:
                raise EOFError("causal audio broker closed without a commit response")
            if response.get("status") != "ok":
                raise RuntimeError(
                    "causal audio broker rejected observation commit: "
                    f"{response.get('message', 'unknown error')}"
                )
            return response


class CausalAudioBroker:
    def __init__(self, schedule: CausalAudioSchedule, release_events_path: Path) -> None:
        verify_causal_audio_source_bytes(schedule)
        self.schedule = schedule
        self.release_events_path = release_events_path
        self._prefix_by_key = {
            (prefix.event_id, prefix.acoustic_condition, prefix.sequence_index): prefix
            for prefix in schedule.prefixes
        }
        self._source_by_id = {source.source_id: source for source in schedule.sources}
        self._talk_by_source_id = {
            source.source_id: source.talk_id for source in schedule.sources
        }
        self._frontier_times_by_talk: dict[str, list[float]] = {}
        self._expected_keys_by_talk_time: dict[
            tuple[str, float], set[tuple[str, str, str, int]]
        ] = {}
        for prefix in schedule.prefixes:
            talk_id = self._talk_by_source_id[prefix.source_id]
            frontier_key = (talk_id, prefix.audio_time_sec)
            expected = self._expected_keys_by_talk_time.setdefault(frontier_key, set())
            for condition in schedule.expected_conditions:
                expected.add(
                    (
                        prefix.event_id,
                        condition,
                        prefix.acoustic_condition,
                        prefix.sequence_index,
                    )
                )
        for talk_id in {source.talk_id for source in schedule.sources}:
            self._frontier_times_by_talk[talk_id] = sorted(
                time_sec
                for observed_talk_id, time_sec in self._expected_keys_by_talk_time
                if observed_talk_id == talk_id
            )
        self._frontier_index_by_talk = {
            talk_id: 0 for talk_id in self._frontier_times_by_talk
        }
        self._committed_keys_by_talk_time: dict[
            tuple[str, float], set[tuple[str, str, str, int]]
        ] = {}
        self._session_by_observation_key: dict[tuple[str, str, str, int], str] = {}
        self._stream_by_session: dict[str, tuple[str, str, str]] = {}
        self._inflight_by_talk: dict[str, tuple[str, str, str, int]] = {}
        self._last_sequence: dict[tuple[str, str, str, str], int] = {}
        self._last_committed_sequence: dict[tuple[str, str, str, str], int] = {}
        self._request_ids: set[str] = set()
        self._server_ordinal = 0
        self._record_tail_sha256 = EMPTY_SHA256
        self._lock = threading.Lock()
        release_events_path.parent.mkdir(parents=True, exist_ok=True)
        self._release_stream = release_events_path.open("x", encoding="utf-8")

    def close(self) -> None:
        self._release_stream.close()

    def _append_interaction(self, payload: dict, model_type):
        payload["server_ordinal"] = self._server_ordinal
        payload["previous_record_sha256"] = self._record_tail_sha256
        payload["record_sha256"] = canonical_json_sha256(payload)
        interaction = model_type(**payload)
        self._release_stream.write(interaction.model_dump_json() + "\n")
        self._release_stream.flush()
        os.fsync(self._release_stream.fileno())
        self._record_tail_sha256 = interaction.record_sha256
        self._server_ordinal += 1
        return interaction

    def release(self, request: CausalAudioRequest) -> tuple[dict, bytes]:
        if request.run_id != self.schedule.run_id:
            raise ValueError("causal audio request uses the wrong run id")
        key = (request.event_id, request.acoustic_condition, request.sequence_index)
        prefix = self._prefix_by_key.get(key)
        if prefix is None:
            raise ValueError("causal audio request is outside the frozen schedule")
        stream_key = (
            request.session_id,
            request.event_id,
            request.condition,
            request.acoustic_condition,
        )
        with self._lock:
            if request.request_id in self._request_ids:
                raise ValueError("causal audio request id was already used")
            expected_sequence = self._last_sequence.get(stream_key, -1) + 1
            if request.sequence_index != expected_sequence:
                raise ValueError("causal audio request is not the next monotonic prefix")
            if self._last_committed_sequence.get(stream_key, -1) != expected_sequence - 1:
                raise ValueError("prior hypothesis must be committed before the next prefix")
            talk_id = self._talk_by_source_id[prefix.source_id]
            frontier_index = self._frontier_index_by_talk[talk_id]
            frontier_times = self._frontier_times_by_talk[talk_id]
            if frontier_index >= len(frontier_times) or not math.isclose(
                prefix.audio_time_sec,
                frontier_times[frontier_index],
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    "causal audio request is ahead of the synchronized talk frontier"
                )
            observation_key = (
                request.event_id,
                request.condition,
                request.acoustic_condition,
                request.sequence_index,
            )
            if observation_key not in self._expected_keys_by_talk_time[
                (talk_id, frontier_times[frontier_index])
            ]:
                raise ValueError("causal audio request condition is outside the frozen schedule")
            if observation_key in self._session_by_observation_key:
                raise ValueError("causal audio observation was already released")
            stream_identity = (
                request.event_id,
                request.condition,
                request.acoustic_condition,
            )
            previous_stream = self._stream_by_session.setdefault(
                request.session_id,
                stream_identity,
            )
            if previous_stream != stream_identity:
                raise ValueError("causal audio session cannot cross inference streams")
            if talk_id in self._inflight_by_talk:
                raise ValueError(
                    "prior same-time talk observation must commit before another release"
                )
            source = self._source_by_id[prefix.source_id]
            byte_count = prefix.sample_count * FLOAT32_BYTES
            with Path(source.source_pcm_path).open("rb") as pcm_stream:
                pcm = pcm_stream.read(byte_count)
            if len(pcm) != byte_count:
                raise ValueError("causal audio source ended before requested prefix")
            if hashlib.sha256(pcm).hexdigest() != prefix.prefix_pcm_sha256:
                raise ValueError("causal audio source changed after broker startup")
            release_payload = {
                "record_type": "prefix_release",
                "source_id": prefix.source_id,
                "session_id": request.session_id,
                "event_id": request.event_id,
                "condition": request.condition,
                "acoustic_condition": request.acoustic_condition,
                "sequence_index": request.sequence_index,
                "audio_time_sec": prefix.audio_time_sec,
                "prefix_id": prefix.prefix_id,
                "prefix_pcm_sha256": prefix.prefix_pcm_sha256,
                "sample_count": prefix.sample_count,
                "request_id": request.request_id,
                "granted_monotonic_ns": time.monotonic_ns(),
                "granted_at_utc": datetime.now(timezone.utc).isoformat().replace(
                    "+00:00", "Z"
                ),
            }
            self._append_interaction(release_payload, CausalAudioReleaseRecord)
            self._request_ids.add(request.request_id)
            self._last_sequence[stream_key] = request.sequence_index
            self._session_by_observation_key[observation_key] = request.session_id
            self._inflight_by_talk[talk_id] = observation_key
        header = {
            "status": "ok",
            "source_id": prefix.source_id,
            "prefix_id": prefix.prefix_id,
            "prefix_pcm_sha256": prefix.prefix_pcm_sha256,
            "sample_rate": prefix.sample_rate,
            "sample_count": prefix.sample_count,
            "pcm_format": source.pcm_format,
            "pcm_byte_count": len(pcm),
        }
        return header, pcm

    def commit_observation(self, request: CausalAudioObservationCommitRequest) -> dict:
        if request.run_id != self.schedule.run_id:
            raise ValueError("causal observation commit uses the wrong run id")
        key = (request.event_id, request.acoustic_condition, request.sequence_index)
        prefix = self._prefix_by_key.get(key)
        if prefix is None:
            raise ValueError("causal observation commit is outside the frozen schedule")
        stream_key = (
            request.session_id,
            request.event_id,
            request.condition,
            request.acoustic_condition,
        )
        with self._lock:
            if request.request_id in self._request_ids:
                raise ValueError("causal audio request id was already used")
            expected_sequence = self._last_committed_sequence.get(stream_key, -1) + 1
            if request.sequence_index != expected_sequence:
                raise ValueError("causal observation commit is not the next commit")
            if self._last_sequence.get(stream_key, -1) != request.sequence_index:
                raise ValueError("causal observation commit has no matching released prefix")
            observation_key = (
                request.event_id,
                request.condition,
                request.acoustic_condition,
                request.sequence_index,
            )
            if self._session_by_observation_key.get(observation_key) != request.session_id:
                raise ValueError("causal observation commit uses the wrong release session")
            talk_id = self._talk_by_source_id[prefix.source_id]
            if self._inflight_by_talk.get(talk_id) != observation_key:
                raise ValueError("causal observation commit is not the in-flight talk observation")
            commit_payload = {
                "record_type": "observation_commit",
                "source_id": prefix.source_id,
                "session_id": request.session_id,
                "event_id": request.event_id,
                "condition": request.condition,
                "acoustic_condition": request.acoustic_condition,
                "sequence_index": request.sequence_index,
                "prefix_id": prefix.prefix_id,
                "prefix_pcm_sha256": prefix.prefix_pcm_sha256,
                "observation_sha256": request.observation_sha256,
                "request_id": request.request_id,
                "committed_monotonic_ns": time.monotonic_ns(),
                "committed_at_utc": datetime.now(timezone.utc).isoformat().replace(
                    "+00:00", "Z"
                ),
            }
            commit = self._append_interaction(
                commit_payload,
                CausalAudioObservationCommitRecord,
            )
            self._request_ids.add(request.request_id)
            self._last_committed_sequence[stream_key] = request.sequence_index
            del self._inflight_by_talk[talk_id]
            frontier_index = self._frontier_index_by_talk[talk_id]
            frontier_time = self._frontier_times_by_talk[talk_id][frontier_index]
            frontier_key = (talk_id, frontier_time)
            committed = self._committed_keys_by_talk_time.setdefault(frontier_key, set())
            committed.add(observation_key)
            if committed == self._expected_keys_by_talk_time[frontier_key]:
                self._frontier_index_by_talk[talk_id] += 1
        return {
            "status": "ok",
            "record_type": commit.record_type,
            "server_ordinal": commit.server_ordinal,
            "observation_sha256": commit.observation_sha256,
        }


class _BrokerRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        while True:
            try:
                payload = receive_json_frame(self.rfile)
                if payload is None:
                    return
                if payload.get("action") == "release_prefix":
                    request = CausalAudioRequest.model_validate(payload)
                    header, pcm = self.server.broker.release(request)
                    send_prefix_frame(self.wfile, header, pcm)
                elif payload.get("action") == "commit_observation":
                    request = CausalAudioObservationCommitRequest.model_validate(payload)
                    send_json_frame(self.wfile, self.server.broker.commit_observation(request))
                else:
                    raise ValueError("unknown causal audio broker action")
            except Exception as exc:
                send_prefix_frame(
                    self.wfile,
                    {"status": "error", "error_type": type(exc).__name__, "message": str(exc)},
                )
                return


class ThreadedUnixAudioServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, socket_path: str, broker: CausalAudioBroker) -> None:
        self.broker = broker
        super().__init__(socket_path, _BrokerRequestHandler)


def serve_causal_audio_broker(
    schedule: CausalAudioSchedule,
    *,
    socket_path: Path,
    release_events_path: Path,
    ready_callback: Callable[[], None] | None = None,
) -> None:
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.parent.chmod(0o700)
    if socket_path.exists() or socket_path.is_symlink():
        raise FileExistsError(f"causal audio socket path already exists: {socket_path}")
    broker = CausalAudioBroker(schedule, release_events_path)
    try:
        with ThreadedUnixAudioServer(str(socket_path), broker) as server:
            socket_path.chmod(0o600)
            if ready_callback is not None:
                ready_callback()
            server.serve_forever(poll_interval=0.25)
    finally:
        broker.close()
        try:
            socket_path.unlink()
        except FileNotFoundError:
            pass


def git_head_clean(repo_path: Path) -> str:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repo_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError("causal audio broker repository is dirty")
    return head


def build_broker_audit(
    schedule: CausalAudioSchedule,
    *,
    schedule_sha256: str,
    repo_path: Path,
    entrypoint_path: Path,
    broker_command: str,
    socket_path: Path,
    release_events_path: Path,
) -> CausalAudioBrokerAudit:
    resolved_repo = repo_path.resolve(strict=True)
    resolved_entrypoint = entrypoint_path.resolve(strict=True)
    if resolved_repo not in resolved_entrypoint.parents:
        raise ValueError("causal audio broker entrypoint is outside its Git repository")
    return CausalAudioBrokerAudit(
        schema_version="acl6060_causal_audio_broker_audit_v2",
        run_id=schedule.run_id,
        schedule_sha256=schedule_sha256,
        broker_git_commit=git_head_clean(resolved_repo),
        broker_repo_path=str(resolved_repo),
        broker_entrypoint_path=str(resolved_entrypoint),
        broker_entrypoint_sha256=sha256_file(resolved_entrypoint),
        broker_command=broker_command,
        broker_pid=os.getpid(),
        socket_path=str(socket_path.parent.resolve(strict=True) / socket_path.name),
        release_events_path=str(
            release_events_path.parent.resolve(strict=True) / release_events_path.name
        ),
        source_audio_roots=schedule.source_audio_roots,
        delivery_protocol="length_prefixed_unix_socket_v1",
        captured_at_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )


def finalize_release_log(
    *,
    schedule_path: Path,
    broker_audit_path: Path,
    release_events_path: Path,
    output_path: Path,
) -> CausalAudioReleaseLog:
    schedule_bytes = schedule_path.read_bytes()
    schedule = CausalAudioSchedule.model_validate_json(schedule_bytes)
    broker_audit_bytes = broker_audit_path.read_bytes()
    broker_audit = CausalAudioBrokerAudit.model_validate_json(broker_audit_bytes)
    schedule_sha256 = hashlib.sha256(schedule_bytes).hexdigest()
    broker_audit_sha256 = hashlib.sha256(broker_audit_bytes).hexdigest()
    if broker_audit.run_id != schedule.run_id:
        raise ValueError("causal audio broker audit and schedule use different run ids")
    if broker_audit.schedule_sha256 != schedule_sha256:
        raise ValueError("causal audio broker audit does not bind the schedule bytes")
    interaction_adapter = TypeAdapter(CausalAudioInteractionRecord)
    interactions = [
        interaction_adapter.validate_json(line)
        for line in release_events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    release_log = CausalAudioReleaseLog(
        schema_version="acl6060_causal_audio_release_log_v3",
        run_id=schedule.run_id,
        schedule_sha256=schedule_sha256,
        broker_audit_sha256=broker_audit_sha256,
        interactions=interactions,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as output:
        output.write(release_log.model_dump_json(indent=2) + "\n")
    return release_log
