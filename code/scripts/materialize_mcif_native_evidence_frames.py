#!/usr/bin/env python3
"""Materialize native-resolution MCIF evidence frames at frozen causal times."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Callable

import numpy as np
from PIL import Image


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_state_inventory(states: list[dict], source_manifest: dict) -> dict[str, dict]:
    if source_manifest.get("dataset") != "mcif":
        raise ValueError("Source manifest is not MCIF")
    if source_manifest.get("reference_files_extracted") is not False:
        raise ValueError("MCIF reference files must remain unextracted")
    if source_manifest.get("reference_content_inspected") is not False:
        raise ValueError("MCIF reference content must remain uninspected")

    talks = {row["talk_id"]: row for row in source_manifest.get("talks", [])}
    if len(talks) != source_manifest.get("talk_count"):
        raise ValueError("MCIF source manifest talk ids are not unique")
    state_ids = [(row.get("talk_id"), row.get("state_id")) for row in states]
    if len(state_ids) != len(set(state_ids)):
        raise ValueError("MCIF causal states contain duplicate identifiers")
    if {talk_id for talk_id, _ in state_ids} != set(talks):
        raise ValueError("MCIF causal-state and source-manifest talk ids differ")

    for talk_id in sorted(talks):
        rows = sorted(
            (row for row in states if row.get("talk_id") == talk_id),
            key=lambda row: row.get("state_id", -1),
        )
        if [row.get("state_id") for row in rows] != list(range(len(rows))):
            raise ValueError(f"Non-contiguous MCIF state ids for {talk_id}")
        if not rows or float(rows[0].get("availability_start_sec", -1)) != 0.0:
            raise ValueError(f"MCIF state timeline does not start at zero for {talk_id}")
        for previous, current in zip(rows, rows[1:]):
            if abs(
                float(previous["availability_end_sec"])
                - float(current["availability_start_sec"])
            ) > 1e-6:
                raise ValueError(f"Non-contiguous MCIF state intervals for {talk_id}")
        if abs(
            float(rows[-1]["availability_end_sec"])
            - float(talks[talk_id]["video_duration_sec"])
        ) > 1e-3:
            raise ValueError(f"MCIF state timeline duration mismatch for {talk_id}")
        for row in rows:
            if float(row["availability_end_sec"]) <= float(
                row["availability_start_sec"]
            ):
                raise ValueError(f"Non-positive MCIF state interval for {talk_id}")
            if abs(
                float(row["evidence_nominal_timestamp_sec"])
                - float(row["availability_start_sec"])
            ) > 1e-6:
                raise ValueError(f"Evidence timestamp is not causal for {talk_id}")
    return talks


def build_capture_schedule(
    states: list[dict],
    talks: dict[str, dict],
    *,
    capture_offset_sec: float,
) -> dict[tuple[str, int], dict[str, float]]:
    if capture_offset_sec < 0:
        raise ValueError("Capture offset must be non-negative")
    schedule = {}
    for talk_id in sorted(talks):
        duration = float(talks[talk_id]["video_duration_sec"])
        rows = sorted(
            (row for row in states if row["talk_id"] == talk_id),
            key=lambda row: row["state_id"],
        )
        capture_times = [
            min(
                float(row["evidence_nominal_timestamp_sec"]) + capture_offset_sec,
                duration - 1e-3,
            )
            for row in rows
        ]
        if any(timestamp < 0 for timestamp in capture_times):
            raise ValueError(f"Invalid evidence capture time for {talk_id}")
        if any(current <= previous for previous, current in zip(capture_times, capture_times[1:])):
            raise ValueError(f"Non-increasing evidence capture times for {talk_id}")
        for index, (row, capture_sec) in enumerate(zip(rows, capture_times, strict=True)):
            end_sec = duration if index + 1 == len(rows) else capture_times[index + 1]
            if end_sec <= capture_sec:
                raise ValueError(f"Non-positive corrected visual state for {talk_id}")
            schedule[(talk_id, int(row["state_id"]))] = {
                "capture_sec": round(capture_sec, 6),
                "availability_start_sec": round(capture_sec, 6),
                "availability_end_sec": round(end_sec, 6),
            }
    return schedule


def resolve_source_file(path_value: object, root: Path, label: str) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"Missing {label} path")
    path = Path(path_value)
    if not path.is_absolute():
        raise ValueError(f"{label} path must be absolute")
    resolved_root = root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or not resolved.is_relative_to(resolved_root):
        raise ValueError(f"{label} path escapes its source root")
    return resolved


def verify_source_videos(talks: dict[str, dict], video_root: Path) -> None:
    resolved_root = video_root.resolve(strict=True)
    for talk_id, talk in talks.items():
        video_path = (resolved_root / f"{talk_id}.mp4").resolve(strict=True)
        if not video_path.is_file() or not video_path.is_relative_to(resolved_root):
            raise ValueError(f"MCIF video path escapes the video root for {talk_id}")
        if sha256_file(video_path) != talk.get("video_sha256"):
            raise ValueError(f"MCIF source video hash mismatch for {talk_id}")


def extract_native_frame(video_path: Path, timestamp_sec: float, output_path: Path) -> None:
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{timestamp_sec:.6f}",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-frames:v",
        "1",
        "-vf",
        "format=rgb24",
        "-c:v",
        "png",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def alignment_mae(native_path: Path, detector_frame_path: Path) -> float:
    with Image.open(native_path) as native, Image.open(detector_frame_path) as detector:
        detector_rgb = detector.convert("RGB")
        resized = native.convert("RGB").resize(
            detector_rgb.size, resample=Image.Resampling.BILINEAR
        )
        difference = np.abs(
            np.asarray(resized, dtype=np.float32)
            - np.asarray(detector_rgb, dtype=np.float32)
        )
    return round(float(difference.mean()), 6)


def materialize_row(
    state: dict,
    talk: dict,
    *,
    state_root: Path,
    video_root: Path,
    output_root: Path,
    max_alignment_mae: float,
    capture_sec: float | None = None,
    corrected_availability_start_sec: float | None = None,
    corrected_availability_end_sec: float | None = None,
    capture_offset_sec: float = 0.0,
    source_video_hash_verified: bool = False,
    extractor: Callable[[Path, float, Path], None] = extract_native_frame,
) -> dict:
    talk_id = state["talk_id"]
    state_id = int(state["state_id"])
    detector_frame = resolve_source_file(
        state.get("evidence_frame_path"), state_root, "detector evidence frame"
    )
    if sha256_file(detector_frame) != state.get("evidence_frame_sha256"):
        raise ValueError(f"Detector evidence frame hash mismatch for {talk_id}:S{state_id:03d}")

    video_path = (video_root / f"{talk_id}.mp4").resolve(strict=True)
    if not video_path.is_file() or not video_path.is_relative_to(video_root.resolve(strict=True)):
        raise ValueError(f"MCIF video path escapes the video root for {talk_id}")
    if not source_video_hash_verified and sha256_file(video_path) != talk.get("video_sha256"):
        raise ValueError(f"MCIF source video hash mismatch for {talk_id}")

    relative_frame = Path("talks") / talk_id / "frames" / f"state_{state_id:03d}.png"
    output_path = output_root / relative_frame
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nominal_timestamp = float(state["evidence_nominal_timestamp_sec"])
    if capture_sec is None:
        capture_sec = nominal_timestamp
    if corrected_availability_start_sec is None:
        corrected_availability_start_sec = float(state["availability_start_sec"])
    if corrected_availability_end_sec is None:
        corrected_availability_end_sec = float(state["availability_end_sec"])
    if abs(capture_sec - corrected_availability_start_sec) > 1e-6:
        raise ValueError("Visual evidence cannot be available before its capture time")
    if corrected_availability_end_sec <= corrected_availability_start_sec:
        raise ValueError("Corrected visual availability interval must be positive")
    extractor(video_path, capture_sec, output_path)
    if not output_path.is_file():
        raise ValueError(f"Native evidence extraction produced no frame for {talk_id}:S{state_id:03d}")
    with Image.open(output_path) as image:
        image.verify()
    with Image.open(output_path) as image:
        width, height = image.size
    if width != int(talk["video_width"]) or height != int(talk["video_height"]):
        raise ValueError(f"Native evidence dimensions mismatch for {talk_id}:S{state_id:03d}")

    mae = alignment_mae(output_path, detector_frame)
    if mae > max_alignment_mae:
        raise ValueError(
            f"Native/detector evidence misalignment for {talk_id}:S{state_id:03d}: {mae}"
        )
    item_id = f"mcif:{talk_id}:S{state_id:03d}"
    return {
        "id": item_id,
        "lecture_id": talk_id,
        "state_id": state_id,
        "availability_start_sec": corrected_availability_start_sec,
        "availability_end_sec": corrected_availability_end_sec,
        "evidence_timestamp_sec": capture_sec,
        "detector_nominal_timestamp_sec": nominal_timestamp,
        "detector_nominal_availability_start_sec": float(
            state["availability_start_sec"]
        ),
        "detector_nominal_availability_end_sec": float(state["availability_end_sec"]),
        "capture_offset_sec": capture_offset_sec,
        "frame_path": relative_frame.as_posix(),
        "frame_sha256": sha256_file(output_path),
        "frame_width": width,
        "frame_height": height,
        "source_video_sha256": talk["video_sha256"],
        "detector_frame_sha256": state["evidence_frame_sha256"],
        "detector_frame_alignment_mae_8bit": mae,
        "source_transcript_consumed": False,
        "target_or_reference_consumed": False,
    }


def build_summary(
    rows: list[dict],
    *,
    source_manifest: dict,
    causal_states_sha256: str,
    max_alignment_mae: float,
    capture_offset_sec: float,
) -> dict:
    alignment_values = [row["detector_frame_alignment_mae_8bit"] for row in rows]
    return {
        "dataset": "mcif",
        "subset": source_manifest["subset"],
        "artifact": "native_resolution_causal_visual_evidence_v1",
        "status": "SOURCE_ONLY_NATIVE_EVIDENCE_NOT_ANNOTATION",
        "upstream_revision": source_manifest["upstream"]["revision"],
        "talk_count": len({row["lecture_id"] for row in rows}),
        "causal_state_count": len(rows),
        "causal_states_sha256": causal_states_sha256,
        "frame_binding_set_sha256": canonical_hash(
            [{"id": row["id"], "sha256": row["frame_sha256"]} for row in rows]
        ),
        "max_allowed_alignment_mae_8bit": max_alignment_mae,
        "timing_correction": {
            "detector_sampling_interval_sec": 1.0,
            "capture_offset_sec": capture_offset_sec,
            "availability_rule": "evidence_available_at_actual_capture_time",
            "pre_first_evidence_gap_sec_by_talk": capture_offset_sec,
        },
        "alignment_mae_8bit": {
            "min": min(alignment_values),
            "mean": round(sum(alignment_values) / len(alignment_values), 6),
            "max": max(alignment_values),
        },
        "source_transcript_consumed": any(
            row["source_transcript_consumed"] for row in rows
        ),
        "target_or_reference_consumed": any(
            row["target_or_reference_consumed"] for row in rows
        ),
        "interpretation": (
            "Native-resolution frames are source-only visual evidence extracted at "
            "the actual centers of the frozen 1-second detector sampling buckets. "
            "Availability is corrected to the capture time, so no visual evidence is "
            "available during the initial half-second. Detector thumbnails are used "
            "only for alignment auditing and not as the strong OCR input."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--causal-states", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-alignment-mae", type=float, default=12.0)
    parser.add_argument("--capture-offset-sec", type=float, default=0.5)
    args = parser.parse_args()
    if (
        args.workers <= 0
        or args.max_alignment_mae < 0
        or args.capture_offset_sec < 0
    ):
        raise ValueError("Workers must be positive and timing thresholds non-negative")
    if args.manifest_out.exists() or args.summary_out.exists():
        raise FileExistsError("Native evidence manifests must be created once")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError("Native evidence output root must be empty")

    states = load_jsonl(args.causal_states)
    source_manifest = load_json(args.source_manifest)
    talks = validate_state_inventory(states, source_manifest)
    schedule = build_capture_schedule(
        states, talks, capture_offset_sec=args.capture_offset_sec
    )
    verify_source_videos(talks, args.video_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                materialize_row,
                state,
                talks[state["talk_id"]],
                state_root=args.state_root,
                video_root=args.video_root,
                output_root=args.output_root,
                max_alignment_mae=args.max_alignment_mae,
                capture_sec=schedule[(state["talk_id"], int(state["state_id"]))][
                    "capture_sec"
                ],
                corrected_availability_start_sec=schedule[
                    (state["talk_id"], int(state["state_id"]))
                ]["availability_start_sec"],
                corrected_availability_end_sec=schedule[
                    (state["talk_id"], int(state["state_id"]))
                ]["availability_end_sec"],
                capture_offset_sec=args.capture_offset_sec,
                source_video_hash_verified=True,
            ): state
            for state in states
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if completed % 25 == 0 or completed == len(futures):
                print(json.dumps({"completed": completed, "total": len(futures)}), flush=True)
    order = {
        f"mcif:{state['talk_id']}:S{int(state['state_id']):03d}": index
        for index, state in enumerate(states)
    }
    rows.sort(key=lambda row: order[row["id"]])

    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    summary = build_summary(
        rows,
        source_manifest=source_manifest,
        causal_states_sha256=sha256_file(args.causal_states),
        max_alignment_mae=args.max_alignment_mae,
        capture_offset_sec=args.capture_offset_sec,
    )
    summary["manifest_sha256"] = sha256_file(args.manifest_out)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
