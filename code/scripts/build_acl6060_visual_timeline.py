#!/usr/bin/env python3
"""Build a transcript-free ACL60/60 frame timeline and transition review set."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

from PIL import Image, ImageDraw, ImageOps

from scripts.detect_mcif_visual_state_candidates import frame_signature, pair_metrics


FRAME_PATTERN = re.compile(r"frame_(\d+(?:\.\d+)?)\.jpg")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_frame_timestamp(path: Path) -> float:
    match = FRAME_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected frame filename: {path.name}")
    timestamp = float(match.group(1))
    if not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError(f"Invalid frame timestamp: {path.name}")
    return timestamp


def numeric_talk_id(talk_id: str) -> str:
    value = talk_id.rsplit(".", 1)[-1]
    if not value.isdigit():
        raise ValueError(f"Cannot map ACL talk id to frame directory: {talk_id}")
    return value


def collect_observations(
    talk: dict,
    frame_split_root: Path,
    portable_staging_label: str,
) -> tuple[list[dict], list[dict]]:
    talk_id = talk["talk_id"]
    frame_dir = frame_split_root / numeric_talk_id(talk_id)
    frame_paths = sorted(frame_dir.glob("frame_*.jpg"), key=parse_frame_timestamp)
    expected = int(talk["segment_count"])
    if len(frame_paths) != expected:
        raise ValueError(f"Frame count mismatch for {talk_id}: {len(frame_paths)} != {expected}")
    timestamps = [parse_frame_timestamp(path) for path in frame_paths]
    if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
        raise ValueError(f"Frame timestamps are not strictly increasing for {talk_id}")
    duration_sec = float(talk["duration_sec"])
    if not timestamps or timestamps[-1] > duration_sec:
        raise ValueError(f"Frame timeline exceeds talk duration for {talk_id}")

    local_rows = []
    portable_rows = []
    for index, (path, timestamp) in enumerate(zip(frame_paths, timestamps)):
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
        frame_sha256 = sha256_file(path)
        row = {
            "dataset": "acl6060",
            "split": talk["split"],
            "talk_id": talk_id,
            "observation_id": f"{talk_id}:F{index:03d}",
            "observation_index": index,
            "observed_at_sec": timestamp,
            "causal_availability_sec": timestamp,
            "availability_start_sec": timestamp,
            "availability_end_sec": (
                duration_sec if index + 1 == len(timestamps) else timestamps[index + 1]
            ),
            "state_policy": "every_observation_no_backdating",
            "frame_path": str(path),
            "frame_sha256": frame_sha256,
            "frame_bytes": path.stat().st_size,
            "width": width,
            "height": height,
            "source": "do_slides_help_figshare_v2_real_acl_talk_frame",
        }
        local_rows.append(row)
        portable = dict(row)
        portable["frame_path"] = (
            f"{portable_staging_label}/{frame_split_root.name}/{frame_dir.name}/{path.name}"
        )
        portable_rows.append(portable)
    return local_rows, portable_rows


def score_observation_pairs(
    observations: list[dict],
    *,
    p75_threshold: float,
    changed_patch_fraction_threshold: float,
) -> tuple[list[dict], list[dict]]:
    signatures = [frame_signature(Path(row["frame_path"])) for row in observations]
    pair_rows = []
    candidates = []
    for current_index, (previous, current) in enumerate(zip(signatures, signatures[1:]), start=1):
        previous_row = observations[current_index - 1]
        current_row = observations[current_index]
        metrics = pair_metrics(previous, current)
        is_candidate = (
            metrics["patch_diff_p75"] >= p75_threshold
            or metrics["changed_patch_fraction_ge_0_05"]
            >= changed_patch_fraction_threshold
        )
        pair = {
            "talk_id": current_row["talk_id"],
            "previous_observation_id": previous_row["observation_id"],
            "current_observation_id": current_row["observation_id"],
            "previous_observed_at_sec": previous_row["observed_at_sec"],
            "current_observed_at_sec": current_row["observed_at_sec"],
            "is_candidate": is_candidate,
            **metrics,
        }
        pair_rows.append(pair)
        if is_candidate:
            candidates.append(
                {
                    **pair,
                    "candidate_id": f"{current_row['talk_id']}:T{len(candidates) + 1:03d}",
                    "transition_window_start_sec": previous_row["observed_at_sec"],
                    "transition_window_end_sec": current_row["observed_at_sec"],
                    "conservative_unlock_sec": current_row["observed_at_sec"],
                    "previous_frame_path": previous_row["frame_path"],
                    "previous_frame_sha256": previous_row["frame_sha256"],
                    "current_frame_path": current_row["frame_path"],
                    "current_frame_sha256": current_row["frame_sha256"],
                    "review_decision": "pending",
                    "review_note": "",
                }
            )
    return pair_rows, candidates


def portable_candidate(candidate: dict, path_map: dict[str, str]) -> dict:
    row = dict(candidate)
    row["previous_frame_path"] = path_map[candidate["previous_frame_path"]]
    row["current_frame_path"] = path_map[candidate["current_frame_path"]]
    return row


def select_negative_audit_rows(
    observations: list[dict],
    pair_rows: list[dict],
    *,
    p75_threshold: float,
    changed_patch_fraction_threshold: float,
    hard_count: int,
    random_count: int,
) -> list[dict]:
    by_id = {row["observation_id"]: row for row in observations}
    negatives = [
        {
            **row,
            "threshold_proximity": max(
                row["patch_diff_p75"] / p75_threshold if p75_threshold else 0.0,
                row["changed_patch_fraction_ge_0_05"] / changed_patch_fraction_threshold
                if changed_patch_fraction_threshold
                else 0.0,
            ),
        }
        for row in pair_rows
        if not row["is_candidate"]
    ]
    hard = sorted(
        negatives,
        key=lambda row: (-row["threshold_proximity"], row["current_observation_id"]),
    )[:hard_count]
    hard_ids = {row["current_observation_id"] for row in hard}
    remaining = [row for row in negatives if row["current_observation_id"] not in hard_ids]
    random_sample = sorted(
        remaining,
        key=lambda row: hashlib.sha256(
            f"acl6060-negative-audit-v1\0{row['current_observation_id']}".encode()
        ).digest(),
    )[:random_count]
    selected = [(row, "hard_negative") for row in hard]
    selected.extend((row, "hash_random_negative") for row in random_sample)
    output = []
    for index, (pair, selection) in enumerate(selected, start=1):
        previous = by_id[pair["previous_observation_id"]]
        current = by_id[pair["current_observation_id"]]
        output.append(
            {
                **pair,
                "audit_id": f"{current['talk_id']}:N{index:03d}",
                "selection": selection,
                "previous_frame_path": previous["frame_path"],
                "previous_frame_sha256": previous["frame_sha256"],
                "current_frame_path": current["frame_path"],
                "current_frame_sha256": current["frame_sha256"],
                "review_decision": "pending",
                "review_note": "",
            }
        )
    return output


def make_review_sheet(
    observations: list[dict],
    candidates: list[dict],
    output_path: Path,
    *,
    cell_size: tuple[int, int] = (320, 180),
) -> None:
    by_id = {row["observation_id"]: row for row in observations}
    label_height = 36
    columns = 3
    rows = max(len(candidates), 1)
    sheet = Image.new(
        "RGB",
        (columns * cell_size[0], rows * (cell_size[1] + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for row_index, candidate in enumerate(candidates):
        review_id = candidate.get("candidate_id", candidate.get("audit_id"))
        if review_id is None:
            raise ValueError("Review row lacks candidate_id or audit_id")
        current = by_id[candidate["current_observation_id"]]
        current_index = int(current["observation_index"])
        next_row = observations[min(current_index + 1, len(observations) - 1)]
        review_frames = [
            by_id[candidate["previous_observation_id"]],
            current,
            next_row,
        ]
        labels = ("before", "candidate", "after")
        for column, (frame_row, label) in enumerate(zip(review_frames, labels)):
            with Image.open(frame_row["frame_path"]) as source:
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
                (x + 4, y + cell_size[1] + 3),
                (
                    f"{review_id} {label} "
                    f"t={frame_row['observed_at_sec']:.2f}s"
                ),
                fill="black",
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=90)


def build_summary(
    talks: list[dict],
    observations: list[dict],
    pair_rows: list[dict],
    candidates: list[dict],
    negative_audit_rows: list[dict],
    args: argparse.Namespace,
) -> dict:
    talk_rows = []
    for talk in talks:
        talk_id = talk["talk_id"]
        talk_observations = [row for row in observations if row["talk_id"] == talk_id]
        talk_candidates = [row for row in candidates if row["talk_id"] == talk_id]
        talk_rows.append(
            {
                "talk_id": talk_id,
                "duration_sec": talk["duration_sec"],
                "frame_count": len(talk_observations),
                "candidate_count": len(talk_candidates),
                "first_observation_sec": talk_observations[0]["observed_at_sec"],
                "last_observation_sec": talk_observations[-1]["observed_at_sec"],
            }
        )
    return {
        "dataset": "acl6060",
        "split": args.split,
        "source": "do_slides_help_figshare_v2_real_acl_talk_frames",
        "audit_type": "reference_free_irregular_frame_transition_candidates",
        "status": "CAUSAL_OBSERVATIONS_READY_TRANSITION_CANDIDATES_REQUIRE_SEPARATE_QA",
        "transcript_metadata_consumed": False,
        "availability_rule": (
            "Every observed frame is a causal state from its filename timestamp until the "
            "next observation. Transition clustering is diagnostic and requires manual review."
        ),
        "parameters": {
            "signature_size": [96, 54],
            "patch_grid": [6, 8],
            "p75_threshold": args.p75_threshold,
            "changed_patch_fraction_threshold": args.changed_patch_fraction_threshold,
        },
        "talk_count": len(talks),
        "frame_count": len(observations),
        "pair_count": len(pair_rows),
        "candidate_count": len(candidates),
        "negative_audit_count": len(negative_audit_rows),
        "portable_frame_staging": args.portable_staging_label,
        "talks": talk_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--talk-manifest", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "eval"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--portable-frame-manifest-out", type=Path, required=True)
    parser.add_argument("--portable-candidates-out", type=Path, required=True)
    parser.add_argument("--portable-negative-audit-out", type=Path, required=True)
    parser.add_argument("--portable-summary-out", type=Path, required=True)
    parser.add_argument("--portable-staging-label", required=True)
    parser.add_argument("--p75-threshold", type=float, default=0.03)
    parser.add_argument("--changed-patch-fraction-threshold", type=float, default=0.12)
    parser.add_argument("--negative-hard-per-talk", type=int, default=6)
    parser.add_argument("--negative-random-per-talk", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.p75_threshold < 0 or not 0 <= args.changed_patch_fraction_threshold <= 1:
        raise ValueError("Transition thresholds are out of range")
    if args.negative_hard_per_talk < 0 or args.negative_random_per_talk < 0:
        raise ValueError("Negative audit sample counts must be non-negative")
    talks = [row for row in load_jsonl(args.talk_manifest) if row["split"] == args.split]
    if not talks:
        raise ValueError(f"No talks found for split {args.split}")
    talks.sort(key=lambda row: row["talk_id"])
    frame_split_root = args.frame_root / f"image_frames_{args.split}"

    local_observations = []
    portable_observations = []
    all_pair_rows = []
    all_candidates = []
    all_negative_audit_rows = []
    for talk in talks:
        local, portable = collect_observations(
            talk,
            frame_split_root,
            args.portable_staging_label,
        )
        pairs, candidates = score_observation_pairs(
            local,
            p75_threshold=args.p75_threshold,
            changed_patch_fraction_threshold=args.changed_patch_fraction_threshold,
        )
        local_observations.extend(local)
        portable_observations.extend(portable)
        all_pair_rows.extend(pairs)
        all_candidates.extend(candidates)
        negative_audit_rows = select_negative_audit_rows(
            local,
            pairs,
            p75_threshold=args.p75_threshold,
            changed_patch_fraction_threshold=args.changed_patch_fraction_threshold,
            hard_count=args.negative_hard_per_talk,
            random_count=args.negative_random_per_talk,
        )
        all_negative_audit_rows.extend(negative_audit_rows)
        make_review_sheet(
            local,
            candidates,
            args.output_root / "transition_sheets" / f"{talk['talk_id']}.jpg",
        )
        make_review_sheet(
            local,
            negative_audit_rows,
            args.output_root / "negative_audit_sheets" / f"{talk['talk_id']}.jpg",
        )

    path_map = {
        local["frame_path"]: portable["frame_path"]
        for local, portable in zip(local_observations, portable_observations)
    }
    portable_candidates = [portable_candidate(row, path_map) for row in all_candidates]
    portable_negative_audit = [
        portable_candidate(row, path_map) for row in all_negative_audit_rows
    ]
    summary = build_summary(
        talks,
        local_observations,
        all_pair_rows,
        all_candidates,
        all_negative_audit_rows,
        args,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_root / "frame_observations.jsonl", local_observations)
    write_jsonl(args.output_root / "pair_metrics.jsonl", all_pair_rows)
    write_jsonl(args.output_root / "transition_candidates.jsonl", all_candidates)
    write_jsonl(args.output_root / "negative_audit_sample.jsonl", all_negative_audit_rows)
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    write_jsonl(args.portable_frame_manifest_out, portable_observations)
    write_jsonl(args.portable_candidates_out, portable_candidates)
    write_jsonl(args.portable_negative_audit_out, portable_negative_audit)
    args.portable_summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.portable_summary_out.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "talk_count": len(talks),
                "frame_count": len(local_observations),
                "candidate_count": len(all_candidates),
                "negative_audit_count": len(all_negative_audit_rows),
            }
        )
    )


if __name__ == "__main__":
    main()
