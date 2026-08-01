#!/usr/bin/env python3
"""Detect conservative MCIF visual-state candidates without reading references."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def extract_frames(video_path: Path, frame_dir: Path, interval_sec: float, frame_width: int) -> list[Path]:
    frame_dir.mkdir(parents=True, exist_ok=True)
    for path in frame_dir.glob("frame_*.jpg"):
        path.unlink()
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"fps=1/{interval_sec},scale={frame_width}:-2:flags=bilinear",
        "-q:v",
        "4",
        str(frame_dir / "frame_%05d.jpg"),
    ]
    subprocess.run(command, check=True)
    frames = sorted(frame_dir.glob("frame_*.jpg"))
    if not frames:
        raise ValueError(f"No frames extracted from {video_path}")
    return frames


def frame_signature(path: Path, size: tuple[int, int] = (96, 54)) -> np.ndarray:
    with Image.open(path) as image:
        gray = ImageOps.fit(image.convert("L"), size, method=Image.Resampling.BILINEAR)
        return np.asarray(gray, dtype=np.float32) / 255.0


def pair_metrics(previous: np.ndarray, current: np.ndarray, grid: tuple[int, int] = (6, 8)) -> dict:
    if previous.shape != current.shape:
        raise ValueError("Frame signatures must have matching shapes")
    rows, columns = grid
    height, width = previous.shape
    if height % rows or width % columns:
        raise ValueError(f"Signature shape {previous.shape} is not divisible by grid {grid}")
    difference = np.abs(current - previous)
    patch_values = difference.reshape(rows, height // rows, columns, width // columns).mean(axis=(1, 3))
    return {
        "global_mean_abs_diff": round(float(difference.mean()), 6),
        "patch_diff_p50": round(float(np.quantile(patch_values, 0.50)), 6),
        "patch_diff_p75": round(float(np.quantile(patch_values, 0.75)), 6),
        "patch_diff_p90": round(float(np.quantile(patch_values, 0.90)), 6),
        "changed_patch_fraction_ge_0_05": round(float(np.mean(patch_values >= 0.05)), 6),
    }


def score_pairs(
    frame_paths: list[Path],
    interval_sec: float,
    *,
    p75_threshold: float,
    changed_patch_fraction_threshold: float,
) -> list[dict]:
    signatures = [frame_signature(path) for path in frame_paths]
    rows = []
    for current_index, (previous, current) in enumerate(zip(signatures, signatures[1:]), start=1):
        metrics = pair_metrics(previous, current)
        metrics.update(
            {
                "current_sample_index": current_index,
                "nominal_timestamp_sec": round(current_index * interval_sec, 6),
                "is_candidate": (
                    metrics["patch_diff_p75"] >= p75_threshold
                    or metrics["changed_patch_fraction_ge_0_05"] >= changed_patch_fraction_threshold
                ),
            }
        )
        rows.append(metrics)
    return rows


def group_candidate_indices(indices: list[int], max_gap_samples: int) -> list[list[int]]:
    groups: list[list[int]] = []
    for index in indices:
        if not groups or index - groups[-1][-1] > max_gap_samples:
            groups.append([index])
        else:
            groups[-1].append(index)
    return groups


def build_candidate_states(
    pair_rows: list[dict],
    interval_sec: float,
    *,
    debounce_samples: int,
    stable_pairs: int,
    stability_p75_threshold: float,
    max_confirmation_samples: int,
) -> list[dict]:
    by_index = {row["current_sample_index"]: row for row in pair_rows}
    candidate_indices = [row["current_sample_index"] for row in pair_rows if row["is_candidate"]]
    groups = group_candidate_indices(candidate_indices, debounce_samples)
    states = []
    for state_id, group in enumerate(groups, start=1):
        first_index = group[0]
        last_index = group[-1]
        stable_run = 0
        unlock_index = None
        search_end = min(last_index + max_confirmation_samples, len(pair_rows))
        for current_index in range(last_index + 1, search_end + 1):
            row = by_index[current_index]
            if row["patch_diff_p75"] < stability_p75_threshold and not row["is_candidate"]:
                stable_run += 1
                if stable_run >= stable_pairs:
                    unlock_index = current_index
                    break
            else:
                stable_run = 0
        peak = max((by_index[index] for index in group), key=lambda row: row["patch_diff_p75"])
        states.append(
            {
                "state_id": state_id,
                "previous_confirmed_sample_sec": round(max(first_index - 1, 0) * interval_sec, 6),
                "transition_window_start_sec": round(first_index * interval_sec, 6),
                "transition_window_end_sec": round(last_index * interval_sec, 6),
                "unlock_sec": None if unlock_index is None else round(unlock_index * interval_sec, 6),
                "stable_confirmation": unlock_index is not None,
                "candidate_sample_count": len(group),
                "peak_sample_index": peak["current_sample_index"],
                "peak_patch_diff_p75": peak["patch_diff_p75"],
                "peak_changed_patch_fraction_ge_0_05": peak["changed_patch_fraction_ge_0_05"],
            }
        )
    return states


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_causal_states(
    talk_id: str,
    frame_paths: list[Path],
    candidate_states: list[dict],
    interval_sec: float,
    duration_sec: float,
) -> list[dict]:
    confirmed = [state for state in candidate_states if state["stable_confirmation"]]
    starts = [0.0] + [float(state["unlock_sec"]) for state in confirmed]
    source_states = [None] + confirmed
    causal_states = []
    for index, (start_sec, source_state) in enumerate(zip(starts, source_states)):
        end_sec = duration_sec if index + 1 == len(starts) else starts[index + 1]
        sample_index = min(int(round(start_sec / interval_sec)), len(frame_paths) - 1)
        frame_path = frame_paths[sample_index]
        row = {
            "talk_id": talk_id,
            "state_id": index,
            "availability_start_sec": round(start_sec, 6),
            "availability_end_sec": round(end_sec, 6),
            "evidence_sample_index": sample_index,
            "evidence_nominal_timestamp_sec": round(sample_index * interval_sec, 6),
            "evidence_frame_path": str(frame_path),
            "evidence_frame_sha256": sha256_file(frame_path),
            "transition_window_start_sec": None,
            "transition_window_end_sec": None,
        }
        if source_state is not None:
            row["transition_window_start_sec"] = source_state["transition_window_start_sec"]
            row["transition_window_end_sec"] = source_state["transition_window_end_sec"]
        causal_states.append(row)
    return causal_states


def make_transition_sheet(
    frame_paths: list[Path],
    states: list[dict],
    output_path: Path,
    interval_sec: float,
    *,
    cell_size: tuple[int, int] = (240, 135),
) -> None:
    columns = 4
    label_height = 24
    rows = max(len(states), 1)
    sheet = Image.new("RGB", (columns * cell_size[0], rows * (cell_size[1] + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for row_index, state in enumerate(states):
        start_index = int(round(state["transition_window_start_sec"] / interval_sec))
        end_index = int(round(state["transition_window_end_sec"] / interval_sec))
        unlock_index = (
            end_index
            if state["unlock_sec"] is None
            else int(round(state["unlock_sec"] / interval_sec))
        )
        indices = [max(start_index - 1, 0), start_index, end_index, min(unlock_index, len(frame_paths) - 1)]
        labels = ["before", "start", "end", "unlock" if state["unlock_sec"] is not None else "unconfirmed"]
        for column, (frame_index, label) in enumerate(zip(indices, labels)):
            with Image.open(frame_paths[frame_index]) as source:
                frame = ImageOps.pad(
                    source.convert("RGB"),
                    cell_size,
                    color="black",
                    method=Image.Resampling.LANCZOS,
                )
            x = column * cell_size[0]
            y = row_index * (cell_size[1] + label_height)
            sheet.paste(frame, (x, y))
            draw.text(
                (x + 4, y + cell_size[1] + 4),
                f"S{state['state_id']} {label} t={frame_index * interval_sec:.1f}s",
                fill="black",
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=90)


def audit_talk(row: dict, args: argparse.Namespace) -> dict:
    talk_id = row["talk_id"]
    talk_root = args.output_root / "talks" / talk_id
    frame_paths = extract_frames(
        Path(row["video"]["path"]),
        talk_root / "frames",
        args.interval_sec,
        args.frame_width,
    )
    pair_rows = score_pairs(
        frame_paths,
        args.interval_sec,
        p75_threshold=args.p75_threshold,
        changed_patch_fraction_threshold=args.changed_patch_fraction_threshold,
    )
    states = build_candidate_states(
        pair_rows,
        args.interval_sec,
        debounce_samples=args.debounce_samples,
        stable_pairs=args.stable_pairs,
        stability_p75_threshold=args.stability_p75_threshold,
        max_confirmation_samples=args.max_confirmation_samples,
    )
    causal_states = build_causal_states(
        talk_id,
        frame_paths,
        states,
        args.interval_sec,
        row["video"]["duration_sec"],
    )
    (talk_root / "pair_metrics.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in pair_rows), encoding="utf-8"
    )
    (talk_root / "candidate_states.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in states), encoding="utf-8"
    )
    (talk_root / "causal_states.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in causal_states), encoding="utf-8"
    )
    make_transition_sheet(
        frame_paths,
        states,
        args.output_root / "transition_sheets" / f"{talk_id}.jpg",
        args.interval_sec,
    )
    return {
        "talk_id": talk_id,
        "duration_sec": row["video"]["duration_sec"],
        "frame_count": len(frame_paths),
        "candidate_state_count": len(states),
        "confirmed_state_count": sum(state["stable_confirmation"] for state in states),
        "unconfirmed_state_count": sum(not state["stable_confirmation"] for state in states),
        "causal_state_count_including_initial": len(causal_states),
        "candidate_states_per_min": round(len(states) * 60.0 / row["video"]["duration_sec"], 6),
    }


def portable_summary(results: list[dict], args: argparse.Namespace) -> dict:
    return {
        "dataset": "mcif",
        "subset": "iwslt2026_translation_21",
        "audit_type": "reference_free_visual_state_candidates",
        "qa_staging": args.portable_staging_label,
        "parameters": {
            key: getattr(args, key)
            for key in (
                "interval_sec",
                "frame_width",
                "p75_threshold",
                "changed_patch_fraction_threshold",
                "debounce_samples",
                "stable_pairs",
                "stability_p75_threshold",
                "max_confirmation_samples",
            )
        },
        "talk_count": len(results),
        "video_duration_sec": round(sum(result["duration_sec"] for result in results), 6),
        "frame_count": sum(result["frame_count"] for result in results),
        "candidate_state_count": sum(result["candidate_state_count"] for result in results),
        "confirmed_state_count": sum(result["confirmed_state_count"] for result in results),
        "unconfirmed_state_count": sum(result["unconfirmed_state_count"] for result in results),
        "causal_state_count_including_initial": sum(
            result["causal_state_count_including_initial"] for result in results
        ),
        "candidate_states_per_min": round(
            sum(result["candidate_state_count"] for result in results)
            * 60.0
            / sum(result["duration_sec"] for result in results),
            6,
        ),
        "interpretation": (
            "Candidates and unlock times require transition-sheet QA; they are not "
            "ground-truth slide boundaries."
        ),
        "talks": sorted(results, key=lambda result: result["talk_id"]),
    }


def combine_causal_states(output_root: Path, talk_ids: list[str]) -> list[dict]:
    combined = []
    for talk_id in talk_ids:
        path = output_root / "talks" / talk_id / "causal_states.jsonl"
        rows = load_jsonl(path)
        if not rows:
            raise ValueError(f"Missing causal states for {talk_id}")
        if [row["state_id"] for row in rows] != list(range(len(rows))):
            raise ValueError(f"Non-contiguous causal state ids for {talk_id}")
        if any(row["talk_id"] != talk_id for row in rows):
            raise ValueError(f"Causal state talk id mismatch for {talk_id}")
        if rows[0]["availability_start_sec"] != 0.0:
            raise ValueError(f"First causal state does not start at zero for {talk_id}")
        for previous, current in zip(rows, rows[1:]):
            if previous["availability_end_sec"] != current["availability_start_sec"]:
                raise ValueError(f"Causal state intervals are not contiguous for {talk_id}")
        if any(row["availability_end_sec"] <= row["availability_start_sec"] for row in rows):
            raise ValueError(f"Non-positive causal state interval for {talk_id}")
        combined.extend(rows)
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--portable-summary-out", type=Path, required=True)
    parser.add_argument("--portable-staging-label", required=True)
    parser.add_argument("--talk-id", action="append", default=[])
    parser.add_argument("--interval-sec", type=float, default=1.0)
    parser.add_argument("--frame-width", type=int, default=320)
    parser.add_argument("--p75-threshold", type=float, default=0.03)
    parser.add_argument("--changed-patch-fraction-threshold", type=float, default=0.12)
    parser.add_argument("--debounce-samples", type=int, default=2)
    parser.add_argument("--stable-pairs", type=int, default=2)
    parser.add_argument("--stability-p75-threshold", type=float, default=0.02)
    parser.add_argument("--max-confirmation-samples", type=int, default=6)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    numeric_positive = (
        args.interval_sec,
        args.frame_width,
        args.stable_pairs,
        args.max_confirmation_samples,
        args.workers,
    )
    if any(value <= 0 for value in numeric_positive):
        raise ValueError("Sampling and worker parameters must be positive")
    rows = load_jsonl(args.inference_manifest)
    if args.talk_id:
        requested = set(args.talk_id)
        rows = [row for row in rows if row["talk_id"] in requested]
        found = {row["talk_id"] for row in rows}
        if found != requested:
            raise ValueError(f"Unknown talk ids: {sorted(requested - found)}")
    if not rows:
        raise ValueError("No inference rows selected")
    args.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(audit_talk, row, args): row["talk_id"] for row in rows}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps({"status": "detected", **result}), flush=True)
    summary = portable_summary(results, args)
    combined_causal_states = combine_causal_states(args.output_root, [row["talk_id"] for row in rows])
    if len(combined_causal_states) != summary["causal_state_count_including_initial"]:
        raise ValueError("Combined causal-state count differs from the portable summary")
    (args.output_root / "causal_states.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in combined_causal_states),
        encoding="utf-8",
    )
    (args.output_root / "visual_state_candidates.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    args.portable_summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.portable_summary_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "talk_count",
                    "candidate_state_count",
                    "confirmed_state_count",
                    "causal_state_count_including_initial",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
