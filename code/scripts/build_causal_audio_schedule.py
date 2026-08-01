#!/usr/bin/env python3
"""Materialize canonical float32 PCM sources and a causal prefix schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

from slidesst.eval.causal_audio import sha256_file
from slidesst.eval.event_timing import CausalAudioSchedule, EventScoringConfig, SourceEventTiming


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def git_head_clean(repo: Path) -> str:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError("causal schedule builder repository is dirty")
    return head


def read_pcm16_wav(path: Path, expected_sha256: str) -> tuple[np.ndarray, int]:
    if path.is_symlink() or path.resolve(strict=True) != path:
        raise ValueError(f"input WAV path is symlinked or non-canonical: {path}")
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"input WAV hash mismatch: {path}")
    with wave.open(str(path), "rb") as audio:
        if audio.getsampwidth() != 2 or audio.getcomptype() != "NONE":
            raise ValueError(f"expected uncompressed PCM16 WAV: {path}")
        channels = audio.getnchannels()
        sample_rate = audio.getframerate()
        values = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
    samples = values.reshape(-1, channels).astype(np.float32).mean(axis=1) / 32768.0
    return samples.astype("<f4", copy=False), sample_rate


def prefix_times(event: SourceEventTiming, step_sec: float) -> list[float]:
    first = event.evidence_available_sec
    if first <= 0:
        first = min(step_sec, event.audio_endpoint_sec)
    times = []
    current = first
    while current < event.audio_endpoint_sec - 1e-9:
        times.append(current)
        current += step_sec
    if not times or abs(times[-1] - event.audio_endpoint_sec) > 1e-9:
        times.append(event.audio_endpoint_sec)
    return times


def prefix_hashes(path: Path, sample_counts: list[int]) -> dict[int, str]:
    result = {}
    digest = hashlib.sha256()
    consumed = 0
    with path.open("rb") as pcm:
        for sample_count in sorted(set(sample_counts)):
            target = sample_count * 4
            remaining = target - consumed
            while remaining:
                chunk = pcm.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise ValueError("canonical PCM ended before a scheduled prefix")
                digest.update(chunk)
                consumed += len(chunk)
                remaining -= len(chunk)
            result[sample_count] = digest.copy().hexdigest()
    return result


def native_audio_identity(row: dict) -> tuple[Path, str]:
    audio = row.get("audio")
    if isinstance(audio, dict):
        return Path(audio["path"]), str(audio["sha256"])
    return Path(row["audio_path"]), str(row["audio_sha256"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-events", type=Path, required=True)
    parser.add_argument("--native-inference-manifest", type=Path, required=True)
    parser.add_argument("--corruption-manifest", type=Path, required=True)
    parser.add_argument("--scoring-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--schedule-output", type=Path, required=True)
    parser.add_argument("--prefix-step-sec", type=float, default=0.96)
    args = parser.parse_args()
    if args.prefix_step_sec <= 0:
        raise ValueError("prefix step must be positive")
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"causal audio output root already exists: {output_root}")
    output_root.mkdir(parents=True)

    events = [
        SourceEventTiming.model_validate_json(line)
        for line in args.source_events.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    events = [event for event in events if event.primary_eligible]
    if not events:
        raise ValueError("causal audio schedule has no primary-eligible source events")
    config = EventScoringConfig.model_validate_json(args.scoring_config.read_text(encoding="utf-8"))
    events_by_talk = {}
    for event in events:
        events_by_talk.setdefault(event.talk_id, []).append(event)
    native_by_talk = {row["talk_id"]: row for row in load_jsonl(args.native_inference_manifest)}
    corruption_by_key = {
        (row["talk_id"], row["condition_id"]): row
        for row in load_jsonl(args.corruption_manifest)
    }
    if any(talk_id not in native_by_talk for talk_id in events_by_talk):
        raise ValueError("native manifest does not cover all event talks")

    provenance_root = output_root / "provenance"
    provenance_root.mkdir()
    native_provenance = provenance_root / "native_inference_manifest.jsonl"
    corruption_provenance = provenance_root / "corruptions.jsonl"
    shutil.copyfile(args.native_inference_manifest, native_provenance)
    shutil.copyfile(args.corruption_manifest, corruption_provenance)
    provenance_by_condition = {
        "native": (native_provenance, sha256_file(native_provenance))
    }
    corruption_sha256 = sha256_file(corruption_provenance)
    for condition in config.expected_acoustic_conditions:
        if condition != "native":
            provenance_by_condition[condition] = (corruption_provenance, corruption_sha256)

    repo = Path(__file__).resolve().parents[2]
    git_commit = git_head_clean(repo)
    entrypoint_sha256 = sha256_file(Path(__file__).resolve())
    source_rows = []
    prefix_rows = []
    for talk_id, talk_events in sorted(events_by_talk.items()):
        event_times = {
            event.event_id: prefix_times(event, args.prefix_step_sec) for event in talk_events
        }
        for acoustic_condition in config.expected_acoustic_conditions:
            if acoustic_condition == "native":
                input_path, input_sha256 = native_audio_identity(native_by_talk[talk_id])
                materialization_kind = "native"
            else:
                row = corruption_by_key.get((talk_id, acoustic_condition))
                if row is None:
                    raise ValueError(f"missing corrupted audio: {talk_id}/{acoustic_condition}")
                input_path = Path(row["output_audio_path"])
                input_sha256 = str(row["output_audio_sha256"])
                materialization_kind = str(row["kind"])
            if not input_path.is_absolute():
                raise ValueError(f"input WAV path must be absolute: {input_path}")
            samples, sample_rate = read_pcm16_wav(input_path, input_sha256)
            sample_counts = [
                int(round(time_sec * sample_rate))
                for times in event_times.values()
                for time_sec in times
            ]
            if max(sample_counts) > len(samples):
                raise ValueError(f"event endpoint exceeds source audio: {talk_id}/{acoustic_condition}")
            source_id = f"source:{talk_id}:{acoustic_condition}"
            pcm_path = output_root / "pcm" / acoustic_condition / f"{talk_id}.f32le"
            pcm_path.parent.mkdir(parents=True, exist_ok=True)
            with pcm_path.open("xb") as output:
                output.write(samples.tobytes(order="C"))
            hashes = prefix_hashes(pcm_path, sample_counts)
            provenance_path, provenance_sha256 = provenance_by_condition[acoustic_condition]
            source_rows.append(
                {
                    "source_id": source_id,
                    "talk_id": talk_id,
                    "acoustic_condition": acoustic_condition,
                    "source_pcm_path": str(pcm_path),
                    "source_pcm_sha256": sha256_file(pcm_path),
                    "pcm_format": "float32le_mono",
                    "sample_rate": sample_rate,
                    "total_sample_count": len(samples),
                    "materialization_kind": materialization_kind,
                    "upstream_audio_sha256": input_sha256,
                    "materializer_git_commit": git_commit,
                    "materializer_entrypoint_sha256": entrypoint_sha256,
                    "source_provenance_path": str(provenance_path),
                    "source_provenance_sha256": provenance_sha256,
                }
            )
            for event in sorted(talk_events, key=lambda value: value.event_id):
                for sequence_index, time_sec in enumerate(event_times[event.event_id]):
                    sample_count = int(round(time_sec * sample_rate))
                    prefix_rows.append(
                        {
                            "source_id": source_id,
                            "event_id": event.event_id,
                            "acoustic_condition": acoustic_condition,
                            "sequence_index": sequence_index,
                            "audio_time_sec": sample_count / sample_rate,
                            "prefix_id": (
                                f"prefix:{event.event_id}:{acoustic_condition}:{sequence_index}"
                            ),
                            "prefix_pcm_sha256": hashes[sample_count],
                            "sample_rate": sample_rate,
                            "sample_count": sample_count,
                        }
                    )
    schedule = CausalAudioSchedule.model_validate(
        {
            "schema_version": "acl6060_causal_audio_schedule_v3",
            "run_id": args.run_id,
            "expected_conditions": config.expected_conditions,
            "source_audio_roots": [str(output_root)],
            "sources": source_rows,
            "prefixes": prefix_rows,
        }
    )
    args.schedule_output.parent.mkdir(parents=True, exist_ok=True)
    with args.schedule_output.open("x", encoding="utf-8") as output:
        output.write(schedule.model_dump_json(indent=2) + "\n")
    print(
        json.dumps(
            {
                "talk_count": len(events_by_talk),
                "source_count": len(source_rows),
                "prefix_count": len(prefix_rows),
                "schedule_sha256": sha256_file(args.schedule_output),
            }
        )
    )


if __name__ == "__main__":
    main()
