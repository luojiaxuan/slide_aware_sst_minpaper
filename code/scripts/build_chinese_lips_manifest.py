#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--meta-json")
    group.add_argument("--meta-csv")
    parser.add_argument("--extracted-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow-missing-frame", action="store_true")
    args = parser.parse_args()

    extracted = Path(args.extracted_root)
    rows = _read_rows(args.meta_json, args.meta_csv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped = 0
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            wav_id = _clip_id(row)
            audio = extracted / f"processed_{args.split}" / f"{wav_id}.wav"
            lip_video = extracted / f"processed_{args.split}" / f"{wav_id}.mp4"
            frame = extracted / "first_frames" / f"{wav_id}_PPT.jpg"
            if not audio.exists() or not lip_video.exists() or (not frame.exists() and not args.allow_missing_frame):
                skipped += 1
                continue
            ocr_text = row.get("ocr_text") or []
            vl2_text = row.get("vl2_text") or []
            if isinstance(ocr_text, str):
                ocr_text = [ocr_text] if ocr_text else []
            if isinstance(vl2_text, str):
                vl2_text = [vl2_text] if vl2_text else []
            frame_paths = [str(frame)] if frame.exists() else []
            item = {
                "id": wav_id,
                "video_id": wav_id.rsplit("_", 1)[0],
                "video_path": str(lip_video),
                "audio_path": str(audio),
                "frame_paths": frame_paths,
                "start_sec": 0.0,
                "end_sec": 0.0,
                "zh_transcript": _transcript(row),
                "visual_context": {
                    "video_id": wav_id.rsplit("_", 1)[0],
                    "clip_id": wav_id,
                    "scene_summary": _scene_summary(row, ocr_text),
                    "ocr_text": ocr_text,
                    "objects": vl2_text,
                    "actions": [],
                    "spatial_relations": [],
                    "frame_ids": [Path(path).stem for path in frame_paths],
                    "metadata": {
                        "dataset": "BAAI/Chinese-LiPS",
                        "split": args.split,
                        "topic": row.get("TOPIC") or row.get("topic"),
                    },
                },
                "metadata": {
                    "dataset": "BAAI/Chinese-LiPS",
                    "split": args.split,
                    "source_wav": row.get("WAV") or row.get("wav_path"),
                    "source_ppt": row.get("PPT") or row.get("ppt_path"),
                    "source_face": row.get("FACE") or row.get("face_path"),
                },
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1
            if args.limit and kept >= args.limit:
                break
    print(f"Wrote {kept} rows to {out}")
    if skipped:
        print(f"Skipped {skipped} rows with missing assets")


def _read_rows(meta_json: str | None, meta_csv: str | None) -> list[dict[str, Any]]:
    if meta_json:
        return json.loads(Path(meta_json).read_text(encoding="utf-8"))
    assert meta_csv is not None
    with Path(meta_csv).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _clip_id(row: dict[str, Any]) -> str:
    if row.get("wav_path"):
        return Path(row["wav_path"]).stem
    return str(row.get("ID") or row.get("id") or row.get("clip_id"))


def _transcript(row: dict[str, Any]) -> str:
    return str(row.get("gt_text") or row.get("TEXT") or row.get("text") or "")


def _scene_summary(row: dict[str, Any], ocr_text: list[str]) -> str:
    if ocr_text:
        return "Slide first frame with OCR terms: " + "; ".join(ocr_text[:8])
    topic = row.get("TOPIC") or row.get("topic") or "unknown"
    return f"Slide first frame for Chinese-LiPS topic {topic}; OCR not provided in source metadata."


if __name__ == "__main__":
    main()
