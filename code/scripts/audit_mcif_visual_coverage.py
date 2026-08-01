#!/usr/bin/env python3
"""Sample MCIF videos and build a reference-free visual coverage audit."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def extract_samples(video_path: Path, frame_dir: Path, interval_sec: float, frame_width: int) -> list[Path]:
    frame_dir.mkdir(parents=True, exist_ok=True)
    for path in frame_dir.glob("sample_*.jpg"):
        path.unlink()
    output_pattern = frame_dir / "sample_%04d.jpg"
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"fps=1/{interval_sec},scale={frame_width}:-2:flags=lanczos",
        "-q:v",
        "3",
        str(output_pattern),
    ]
    subprocess.run(command, check=True)
    frames = sorted(frame_dir.glob("sample_*.jpg"))
    if not frames:
        raise ValueError(f"No frames sampled from {video_path}")
    return frames


def frame_signature(path: Path, size: tuple[int, int] = (64, 36)) -> np.ndarray:
    with Image.open(path) as image:
        gray = ImageOps.fit(image.convert("L"), size, method=Image.Resampling.BILINEAR)
        return np.asarray(gray, dtype=np.float32) / 255.0


def adjacent_differences(frame_paths: list[Path]) -> list[float]:
    signatures = [frame_signature(path) for path in frame_paths]
    return [
        float(np.mean(np.abs(current - previous)))
        for previous, current in zip(signatures, signatures[1:])
    ]


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values), quantile))


def summarize_differences(values: list[float], interval_sec: float) -> dict:
    return {
        "adjacent_pairs": len(values),
        "mean_abs_diff_mean": round(float(np.mean(values)) if values else 0.0, 6),
        "mean_abs_diff_p50": round(percentile(values, 0.5), 6),
        "mean_abs_diff_p90": round(percentile(values, 0.9), 6),
        "mean_abs_diff_max": round(max(values, default=0.0), 6),
        "near_duplicate_pairs_lt_0_005": sum(value < 0.005 for value in values),
        "large_change_candidates_ge_0_08": sum(value >= 0.08 for value in values),
        "candidate_changes_per_min": round(
            sum(value >= 0.08 for value in values) * 60.0 / (max(len(values), 1) * interval_sec), 6
        ),
    }


def make_contact_sheet(
    frame_paths: list[Path],
    output_path: Path,
    interval_sec: float,
    *,
    columns: int = 5,
    cell_size: tuple[int, int] = (320, 180),
) -> None:
    label_height = 22
    rows = math.ceil(len(frame_paths) / columns)
    sheet = Image.new("RGB", (columns * cell_size[0], rows * (cell_size[1] + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, frame_path in enumerate(frame_paths):
        with Image.open(frame_path) as source:
            frame = ImageOps.pad(
                source.convert("RGB"),
                cell_size,
                color="black",
                method=Image.Resampling.LANCZOS,
            )
        x = (index % columns) * cell_size[0]
        y = (index // columns) * (cell_size[1] + label_height)
        sheet.paste(frame, (x, y))
        draw.text((x + 5, y + cell_size[1] + 4), f"t={index * interval_sec:.1f}s", fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=90)


def audit_talk(row: dict, output_root: Path, interval_sec: float, frame_width: int) -> dict:
    talk_id = row["talk_id"]
    talk_root = output_root / "talks" / talk_id
    frame_paths = extract_samples(Path(row["video"]["path"]), talk_root / "frames", interval_sec, frame_width)
    differences = adjacent_differences(frame_paths)
    contact_sheet = output_root / "contact_sheets" / f"{talk_id}.jpg"
    make_contact_sheet(frame_paths, contact_sheet, interval_sec)
    sample_rows = [
        {
            "talk_id": talk_id,
            "sample_index": index,
            "nominal_timestamp_sec": round(index * interval_sec, 6),
            "frame_path": str(frame_path),
            "difference_from_previous": None if index == 0 else round(differences[index - 1], 6),
        }
        for index, frame_path in enumerate(frame_paths)
    ]
    sample_manifest = talk_root / "samples.jsonl"
    sample_manifest.write_text(
        "".join(json.dumps(sample, sort_keys=True) + "\n" for sample in sample_rows),
        encoding="utf-8",
    )
    return {
        "talk_id": talk_id,
        "duration_sec": row["video"]["duration_sec"],
        "sample_count": len(frame_paths),
        "sample_interval_sec": interval_sec,
        "contact_sheet": str(contact_sheet),
        "sample_manifest": str(sample_manifest),
        "difference_proxy": summarize_differences(differences, interval_sec),
    }


def portable_summary(results: list[dict], source_manifest: Path, staging_label: str) -> dict:
    all_proxy = [result["difference_proxy"] for result in results]
    total_pairs = sum(item["adjacent_pairs"] for item in all_proxy)
    total_large = sum(item["large_change_candidates_ge_0_08"] for item in all_proxy)
    return {
        "dataset": "mcif",
        "subset": "iwslt2026_translation_21",
        "audit_type": "reference_free_fixed_interval_visual_sampling",
        "source_inference_manifest": source_manifest.name,
        "qa_staging": staging_label,
        "talk_count": len(results),
        "video_duration_sec": round(sum(result["duration_sec"] for result in results), 6),
        "sample_count": sum(result["sample_count"] for result in results),
        "adjacent_pairs": total_pairs,
        "large_change_candidates_ge_0_08": total_large,
        "candidate_changes_per_min": round(
            total_large * 60.0 / sum(result["duration_sec"] for result in results), 6
        ),
        "interpretation": (
            "Pixel-difference counts are diagnostics only. They do not establish "
            "slide boundaries or semantic coverage."
        ),
        "talks": [
            {
                "talk_id": result["talk_id"],
                "duration_sec": result["duration_sec"],
                "sample_count": result["sample_count"],
                "sample_interval_sec": result["sample_interval_sec"],
                "difference_proxy": result["difference_proxy"],
            }
            for result in sorted(results, key=lambda item: item["talk_id"])
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--portable-summary-out", type=Path, required=True)
    parser.add_argument("--portable-staging-label", required=True)
    parser.add_argument("--interval-sec", type=float, default=10.0)
    parser.add_argument("--frame-width", type=int, default=480)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.interval_sec <= 0 or args.frame_width <= 0 or args.workers <= 0:
        raise ValueError("interval-sec, frame-width, and workers must be positive")
    rows = load_jsonl(args.inference_manifest)
    if not rows:
        raise ValueError("Inference manifest is empty")
    args.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                audit_talk,
                row,
                args.output_root,
                args.interval_sec,
                args.frame_width,
            ): row["talk_id"]
            for row in rows
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                json.dumps(
                    {
                        "talk_id": result["talk_id"],
                        "sample_count": result["sample_count"],
                        "status": "audited",
                    }
                ),
                flush=True,
            )
    summary = portable_summary(
        results,
        args.inference_manifest,
        args.portable_staging_label,
    )
    local_summary = args.output_root / "visual_coverage.json"
    local_summary.write_text(
        json.dumps({"results": results, "summary": summary}, indent=2) + "\n",
        encoding="utf-8",
    )
    args.portable_summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.portable_summary_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "summary": str(args.portable_summary_out),
                **{key: summary[key] for key in ("talk_count", "sample_count")},
            }
        )
    )


if __name__ == "__main__":
    main()
