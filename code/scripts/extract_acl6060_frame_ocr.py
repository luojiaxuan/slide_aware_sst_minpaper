#!/usr/bin/env python3
"""Extract deterministic Tesseract OCR from ACL60/60 frame observations."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import subprocess

from PIL import Image


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def parse_tsv(tsv: str, *, min_confidence: float, image_size: tuple[int, int]) -> list[dict]:
    width, height = image_size
    tokens = []
    for row in csv.DictReader(io.StringIO(tsv), delimiter="\t"):
        text = (row.get("text") or "").strip()
        try:
            confidence = float(row.get("conf", -1))
        except (TypeError, ValueError):
            confidence = -1.0
        if (
            confidence < min_confidence
            or not text
            or not any(character.isalnum() for character in text)
        ):
            continue
        left = int(row["left"])
        top = int(row["top"])
        token_width = int(row["width"])
        token_height = int(row["height"])
        tokens.append(
            {
                "text": text,
                "confidence": confidence,
                "line_key": ":".join(
                    row.get(key, "") for key in ("block_num", "par_num", "line_num")
                ),
                "bbox_px": [left, top, token_width, token_height],
                "bbox_norm": [
                    round(left / width, 6),
                    round(top / height, 6),
                    round(token_width / width, 6),
                    round(token_height / height, 6),
                ],
            }
        )
    return tokens


def group_lines(tokens: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for token in tokens:
        grouped.setdefault(token["line_key"], []).append(token)
    lines = []
    for line_key, line_tokens in grouped.items():
        line_tokens.sort(key=lambda token: token["bbox_px"][0])
        lines.append(
            {
                "line_key": line_key,
                "text": " ".join(token["text"] for token in line_tokens),
                "mean_confidence": round(
                    sum(token["confidence"] for token in line_tokens) / len(line_tokens),
                    6,
                ),
                "token_count": len(line_tokens),
                "top_norm": min(token["bbox_norm"][1] for token in line_tokens),
                "left_norm": min(token["bbox_norm"][0] for token in line_tokens),
            }
        )
    lines.sort(key=lambda line: (line["top_norm"], line["left_norm"]))
    return lines


def extract_observation(
    observation: dict,
    portable_root: Path,
    *,
    language: str,
    psm: int,
    min_confidence: float,
) -> dict:
    frame_path = portable_root / observation["frame_path"]
    if not frame_path.is_file():
        raise FileNotFoundError(frame_path)
    with Image.open(frame_path) as image:
        image_size = image.size
        image.verify()
    command = [
        "tesseract",
        str(frame_path),
        "stdout",
        "-l",
        language,
        "--psm",
        str(psm),
        "tsv",
    ]
    result = subprocess.run(command, capture_output=True, text=True, errors="replace")
    if result.returncode:
        raise RuntimeError(f"Tesseract failed for {frame_path}: {result.stderr[-500:]}")
    tokens = parse_tsv(result.stdout, min_confidence=min_confidence, image_size=image_size)
    lines = group_lines(tokens)
    return {
        "dataset": "acl6060",
        "split": observation["split"],
        "talk_id": observation["talk_id"],
        "observation_id": observation["observation_id"],
        "observed_at_sec": observation["observed_at_sec"],
        "availability_end_sec": observation["availability_end_sec"],
        "frame_path": observation["frame_path"],
        "frame_sha256": observation["frame_sha256"],
        "ocr_engine": "tesseract",
        "ocr_language": language,
        "ocr_psm": psm,
        "min_confidence": min_confidence,
        "token_count": len(tokens),
        "line_count": len(lines),
        "tokens": tokens,
        "lines": lines,
        "ocr_text": "\n".join(line["text"] for line in lines),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-observations", type=Path, required=True)
    parser.add_argument("--portable-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--language", default="eng")
    parser.add_argument("--psm", type=int, default=11)
    parser.add_argument("--min-confidence", type=float, default=50.0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")

    observations = load_jsonl(args.frame_observations)
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                extract_observation,
                observation,
                args.portable_root,
                language=args.language,
                psm=args.psm,
                min_confidence=args.min_confidence,
            ): observation["observation_id"]
            for observation in observations
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if completed % 25 == 0 or completed == len(futures):
                print(json.dumps({"completed": completed, "total": len(futures)}), flush=True)
    order = {row["observation_id"]: index for index, row in enumerate(observations)}
    rows.sort(key=lambda row: order[row["observation_id"]])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    tesseract_version = subprocess.run(
        ["tesseract", "--version"], capture_output=True, text=True, check=True
    ).stdout.splitlines()[0]
    talk_ids = list(dict.fromkeys(row["talk_id"] for row in rows))
    summary = {
        "dataset": "acl6060",
        "split": "dev",
        "artifact": "automatic_frame_ocr",
        "status": "AUTOMATIC_DIAGNOSTIC_NOT_HUMAN_ANNOTATION",
        "tesseract_version": tesseract_version,
        "language": args.language,
        "psm": args.psm,
        "min_confidence": args.min_confidence,
        "frame_count": len(rows),
        "zero_token_frame_count": sum(row["token_count"] == 0 for row in rows),
        "token_count": sum(row["token_count"] for row in rows),
        "line_count": sum(row["line_count"] for row in rows),
        "local_output": str(args.output),
        "local_output_sha256": sha256_file(args.output),
        "talks": [
            {
                "talk_id": talk_id,
                "frame_count": sum(row["talk_id"] == talk_id for row in rows),
                "zero_token_frame_count": sum(
                    row["talk_id"] == talk_id and row["token_count"] == 0 for row in rows
                ),
                "token_count": sum(
                    row["token_count"] for row in rows if row["talk_id"] == talk_id
                ),
            }
            for talk_id in talk_ids
        ],
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("frame_count", "token_count")}))


if __name__ == "__main__":
    main()
