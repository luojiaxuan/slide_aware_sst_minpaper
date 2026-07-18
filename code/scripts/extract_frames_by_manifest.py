#!/usr/bin/env python
"""Extract per-segment frames from locally obtained talk videos.

Takes an mTEDx-V talk manifest (see build_mtedx_v_manifest.py) and a directory of
locally obtained videos named <talk_id>.<ext>. For each segment, samples frames at
a fixed rate inside [start, end] on the original video timeline with ffmpeg, and
writes a frames manifest JSONL compatible with the visual-context pipeline.

Videos are NOT downloaded by this script. Users must obtain media from the
original TEDx/YouTube sources themselves and comply with the applicable terms.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="talk-level manifest JSONL")
    parser.add_argument("--video-dir", required=True, help="dir with <talk_id>.<ext> videos")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fps", type=float, default=0.5, help="frames per second per segment")
    parser.add_argument("--max-frames-per-segment", type=int, default=8)
    parser.add_argument("--out-manifest", default=None)
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = Path(args.out_manifest or out_dir / "frames_manifest.jsonl")

    n_talks = n_missing = n_frames = 0
    with out_manifest.open("w", encoding="utf-8") as w:
        for line in Path(args.manifest).open(encoding="utf-8"):
            talk = json.loads(line)
            video = _find_video(video_dir, talk["talk_id"])
            if video is None:
                n_missing += 1
                continue
            n_talks += 1
            for seg in talk["segments"]:
                frame_paths = _extract_segment_frames(
                    video, out_dir, talk["talk_id"], seg, args.fps, args.max_frames_per_segment
                )
                n_frames += len(frame_paths)
                w.write(
                    json.dumps(
                        {
                            "talk_id": talk["talk_id"],
                            "segment_id": seg["segment_id"],
                            "start": seg["start"],
                            "end": seg["end"],
                            "frame_paths": frame_paths,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    print(f"talks: {n_talks} processed, {n_missing} missing video; frames: {n_frames}")
    print(f"frames manifest -> {out_manifest}")


def _find_video(video_dir: Path, talk_id: str) -> Path | None:
    for ext in VIDEO_EXTS:
        candidate = video_dir / f"{talk_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def _extract_segment_frames(
    video: Path, out_dir: Path, talk_id: str, seg: dict, fps: float, max_frames: int
) -> list[str]:
    seg_dir = out_dir / talk_id
    seg_dir.mkdir(parents=True, exist_ok=True)
    duration = max(seg["end"] - seg["start"], 1.0 / fps)
    n = min(max_frames, max(1, int(duration * fps)))
    paths: list[str] = []
    for i in range(n):
        t = seg["start"] + (i + 0.5) * duration / n
        frame_path = seg_dir / f"{talk_id}_{seg['segment_id']:04d}_{i:02d}.jpg"
        if not frame_path.exists():
            cmd = [
                "ffmpeg", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", str(video),
                "-frames:v", "1", "-q:v", "3", "-y", str(frame_path),
            ]
            subprocess.run(cmd, check=True)
        paths.append(str(frame_path))
    return paths


if __name__ == "__main__":
    main()
