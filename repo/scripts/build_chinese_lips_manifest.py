#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-json", required=True)
    parser.add_argument("--extracted-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    meta_path = Path(args.meta_json)
    extracted = Path(args.extracted_root)
    rows = json.loads(meta_path.read_text(encoding="utf-8"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            wav_id = Path(row["wav_path"]).stem
            audio = extracted / f"processed_{args.split}" / f"{wav_id}.wav"
            lip_video = extracted / f"processed_{args.split}" / f"{wav_id}.mp4"
            frame = extracted / "first_frames" / f"{wav_id}_PPT.jpg"
            if not audio.exists() or not lip_video.exists() or not frame.exists():
                continue
            ocr_text = row.get("ocr_text") or []
            vl2_text = row.get("vl2_text") or []
            item = {
                "id": wav_id,
                "video_id": wav_id.rsplit("_", 1)[0],
                "video_path": str(lip_video),
                "audio_path": str(audio),
                "frame_paths": [str(frame)],
                "start_sec": 0.0,
                "end_sec": 0.0,
                "zh_transcript": row.get("gt_text", ""),
                "visual_context": {
                    "video_id": wav_id.rsplit("_", 1)[0],
                    "clip_id": wav_id,
                    "scene_summary": "Slide first frame with OCR terms: " + "; ".join(ocr_text[:8]),
                    "ocr_text": ocr_text,
                    "objects": vl2_text,
                    "actions": [],
                    "spatial_relations": [],
                    "frame_ids": [frame.stem],
                    "metadata": {
                        "dataset": "BAAI/Chinese-LiPS",
                        "split": args.split,
                    },
                },
                "metadata": {
                    "dataset": "BAAI/Chinese-LiPS",
                    "split": args.split,
                },
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1
            if args.limit and kept >= args.limit:
                break
    print(f"Wrote {kept} rows to {out}")


if __name__ == "__main__":
    main()
