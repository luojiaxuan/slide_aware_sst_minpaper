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
    parser.add_argument(
        "--timeline-dir",
        default=None,
        help="dir with <video_id>.orig_timeline.json (clip_id/orig_start/orig_end from the "
        "raw release's per-segment JSONs). When set, clips are placed on the original "
        "session timeline with real silence gaps restored, instead of back-to-back.",
    )
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
        timeline = _load_timeline(args.timeline_dir, video_id)
        _concat_video(video_id, video_rows, processed, out_dir, timeline)


def _load_timeline(timeline_dir: str | None, video_id: str) -> dict[str, dict] | None:
    if timeline_dir is None:
        return None
    path = Path(timeline_dir) / f"{video_id}.orig_timeline.json"
    if not path.exists():
        raise FileNotFoundError(f"--timeline-dir set but {path} not found")
    records = json.loads(path.read_text(encoding="utf-8"))
    return {r["clip_id"]: r for r in records}


def _concat_video(
    video_id: str,
    rows: list[dict],
    processed: Path,
    out_dir: Path,
    timeline: dict[str, dict] | None = None,
) -> None:
    out_wav = out_dir / f"{video_id}.longform.wav"
    out_manifest = out_dir / f"{video_id}.longform.jsonl"
    params = None
    offset = 0.0
    kept = missing = 0

    with out_manifest.open("w", encoding="utf-8") as mf:
        writer = None
        try:
            timeline_origin = None
            for row in rows:
                wav_id = Path(row["wav_path"]).stem
                clip = processed / f"{wav_id}.wav"
                if not clip.exists():
                    missing += 1
                    continue
                if timeline is not None and wav_id not in timeline:
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
                if timeline is not None:
                    seg = timeline[wav_id]
                    if timeline_origin is None:
                        timeline_origin = seg["orig_start"]
                    target = seg["orig_start"] - timeline_origin
                    gap = target - offset
                    if gap > 0.001:  # restore real inter-segment silence
                        writer.writeframes(
                            b"\x00" * (int(gap * params.framerate) * params.nchannels * params.sampwidth)
                        )
                        offset = target
                writer.writeframes(frames)
                record = {
                    "video_id": video_id,
                    "clip_id": wav_id,
                    "start": round(offset, 3),
                    "end": round(offset + duration, 3),
                    "zh_transcript": row.get("gt_text", ""),
                    "ocr_text": row.get("ocr_text") or [],
                    "vl2_text": row.get("vl2_text") or [],
                    "ppt_frame": row.get("ppt_path", ""),
                }
                if timeline is not None:
                    seg = timeline[wav_id]
                    record["orig_start"] = round(seg["orig_start"], 3)
                    record["orig_end"] = round(seg["orig_end"], 3)
                mf.write(json.dumps(record, ensure_ascii=False) + "\n")
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
