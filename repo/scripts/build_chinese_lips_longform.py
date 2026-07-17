#!/usr/bin/env python
"""Build long-form debug audio from Chinese-LiPS by concatenating same-video segments.

Chinese-LiPS (BAAI, CC BY-NC-SA 4.0) ships pre-segmented clips named
<video_id>_<seg>.wav. Concatenating all clips of one video in segment order yields a
continuous long-speech stream with inter-segment silence removed - useful as a
zh->en long-form SST debugging input where the developer can verify every word.

Outputs per video: <video_id>.longform.wav plus a JSONL manifest with each
segment's offset in the concatenated audio, its transcript, and its visual
annotations (PPT OCR / VL2 labels) carried over from meta_test.json.
"""
from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-json", required=True, help="Chinese-LiPS meta_test.json")
    parser.add_argument("--processed-dir", required=True, help="dir with <wav_id>.wav clips")
    parser.add_argument("--video-ids", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    rows = json.loads(Path(args.meta_json).read_text(encoding="utf-8"))
    processed = Path(args.processed_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_video: dict[str, list[dict]] = {vid: [] for vid in args.video_ids}
    for row in rows:
        wav_id = Path(row["wav_path"]).stem
        video_id = wav_id.rsplit("_", 1)[0]
        if video_id in by_video:
            by_video[video_id].append(row)

    for video_id, video_rows in by_video.items():
        if not video_rows:
            print(f"{video_id}: no segments found in meta, skipped")
            continue
        video_rows.sort(key=lambda r: int(Path(r["wav_path"]).stem.rsplit("_", 1)[1]))
        _concat_video(video_id, video_rows, processed, out_dir)


def _concat_video(video_id: str, rows: list[dict], processed: Path, out_dir: Path) -> None:
    out_wav = out_dir / f"{video_id}.longform.wav"
    out_manifest = out_dir / f"{video_id}.longform.jsonl"
    params = None
    offset = 0.0
    kept = missing = 0

    with out_manifest.open("w", encoding="utf-8") as mf:
        writer = None
        try:
            for row in rows:
                wav_id = Path(row["wav_path"]).stem
                clip = processed / f"{wav_id}.wav"
                if not clip.exists():
                    missing += 1
                    continue
                with wave.open(str(clip), "rb") as r:
                    if params is None:
                        params = r.getparams()
                        writer = wave.open(str(out_wav), "wb")
                        writer.setparams(params)
                    elif (
                        r.getframerate() != params.framerate
                        or r.getnchannels() != params.nchannels
                        or r.getsampwidth() != params.sampwidth
                    ):
                        raise ValueError(f"{clip}: wav params mismatch")
                    frames = r.readframes(r.getnframes())
                    duration = r.getnframes() / r.getframerate()
                writer.writeframes(frames)
                mf.write(
                    json.dumps(
                        {
                            "video_id": video_id,
                            "clip_id": wav_id,
                            "start": round(offset, 3),
                            "end": round(offset + duration, 3),
                            "zh_transcript": row.get("gt_text", ""),
                            "ocr_text": row.get("ocr_text") or [],
                            "vl2_text": row.get("vl2_text") or [],
                            "ppt_frame": row.get("ppt_path", ""),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                offset += duration
                kept += 1
        finally:
            if writer is not None:
                writer.close()

    print(
        f"{video_id}: {kept} segments concatenated ({offset/60:.1f} min), "
        f"{missing} missing clips -> {out_wav.name}"
    )


if __name__ == "__main__":
    main()
