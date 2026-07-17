#!/usr/bin/env python
"""Score per-talk visual signal (slide / on-screen-text presence) for mTEDx-V.

For each talk video (obtained locally by the user), samples frames at a fixed
interval and scores them with one of two backends:

- ``ocr``  : tesseract token counting (fast, no network; undercounts at <=480p
             and needs the right language packs for non-Latin scripts).
- ``vlm``  : a vision-language model classifies each frame into
             {slide, screen_text, props_or_media, speaker_only}. Requires
             ANTHROPIC_API_KEY (or compatible endpoint via ANTHROPIC_BASE_URL).

Outputs one JSON per talk plus an aggregate ``visual_signal.json`` with
input-side stratification tags (model-independent, decided before any
translation experiment) for full-set + per-stratum reporting.
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import subprocess
import tempfile
from pathlib import Path

VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov")
VLM_PROMPT = (
    "Classify this single video frame from a public talk. Answer with ONE word:\n"
    "slide  - a projected/presented slide or full-screen graphic with readable text dominates\n"
    "screen_text - some on-screen text is visible (captions, titles, partial slide)\n"
    "props_or_media - a physical object, demo, photo or video clip is the focus (no readable text)\n"
    "speaker_only - just speaker/stage/audience, no visual evidence"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", required=True, help="dir with *.talks.jsonl")
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--interval", type=float, default=20.0, help="seconds between frames")
    parser.add_argument("--backend", choices=["ocr", "vlm"], default="ocr")
    parser.add_argument("--vlm-model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--ocr-langs", default="eng", help="tesseract langs, e.g. eng+rus+ell")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "per_talk").mkdir(parents=True, exist_ok=True)
    video_dir = Path(args.video_dir)

    talks = []
    for mf in sorted(Path(args.manifest_dir).glob("*.talks.jsonl")):
        for line in mf.open(encoding="utf-8"):
            talks.append(json.loads(line))

    aggregate = {}
    for talk in talks:
        tid = talk["talk_id"]
        video = _find_video(video_dir, tid)
        if video is None:
            aggregate[tid] = {"status": "missing_video"}
            continue
        per_frame = _score_video(video, args)
        agg = _aggregate(per_frame)
        agg.update(status="ok", src_lang=talk["src_lang"], split=talk["split"])
        aggregate[tid] = agg
        (out_dir / "per_talk" / f"{tid}.json").write_text(
            json.dumps({"talk_id": tid, "frames": per_frame, **agg}, ensure_ascii=False, indent=1)
        )
        print(f"{tid} [{talk['src_lang']}/{talk['split']}]: {agg['visual_class']} "
              f"(text {agg['text_frame_rate']:.2f}, slide {agg['slide_frame_rate']:.2f})")

    (out_dir / "visual_signal.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=1)
    )


def _find_video(video_dir: Path, talk_id: str) -> Path | None:
    for ext in VIDEO_EXTS:
        p = video_dir / f"{talk_id}{ext}"
        if p.exists():
            return p
    return None


def _score_video(video: Path, args: argparse.Namespace) -> list[dict]:
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-i", str(video),
             "-vf", f"fps=1/{args.interval}", "-q:v", "3", f"{td}/f_%04d.jpg"],
            check=True,
        )
        frames = sorted(Path(td).glob("f_*.jpg"))
        out = []
        for i, frame in enumerate(frames):
            t = (i + 0.5) * args.interval
            if args.backend == "ocr":
                tokens = _ocr_tokens(frame, args.ocr_langs)
                label = ("slide" if len(tokens) >= 6
                         else "screen_text" if len(tokens) >= 2 else "speaker_only")
                out.append({"t": round(t, 1), "label": label, "ocr_tokens": len(tokens)})
            else:
                label = _vlm_label(frame, args.vlm_model)
                out.append({"t": round(t, 1), "label": label})
        return out


def _ocr_tokens(frame: Path, langs: str) -> list[str]:
    r = subprocess.run(
        ["tesseract", str(frame), "stdout", "-l", langs, "--psm", "11", "tsv"],
        capture_output=True, text=True, errors="ignore",
    )
    tokens = []
    for row in csv.DictReader(io.StringIO(r.stdout), delimiter="\t"):
        try:
            conf = float(row.get("conf", -1))
        except ValueError:
            conf = -1
        text = (row.get("text") or "").strip()
        if conf >= 60 and len(text) >= 3 and any(c.isalnum() for c in text):
            tokens.append(text)
    return tokens


def _vlm_label(frame: Path, model: str) -> str:
    import anthropic  # lazy: only needed for the vlm backend

    client = anthropic.Anthropic()
    image_b64 = base64.b64encode(frame.read_bytes()).decode()
    msg = client.messages.create(
        model=model,
        max_tokens=10,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                          "data": image_b64}},
            {"type": "text", "text": VLM_PROMPT},
        ]}],
    )
    label = msg.content[0].text.strip().split()[0].lower()
    valid = {"slide", "screen_text", "props_or_media", "speaker_only"}
    return label if label in valid else "speaker_only"


def _aggregate(per_frame: list[dict]) -> dict:
    n = max(len(per_frame), 1)
    rates = {
        label: sum(1 for f in per_frame if f["label"] == label) / n
        for label in ("slide", "screen_text", "props_or_media", "speaker_only")
    }
    slide_rate = rates["slide"]
    text_rate = rates["slide"] + rates["screen_text"]
    if slide_rate >= 0.25:
        visual_class = "slide_heavy"
    elif text_rate >= 0.10 or slide_rate >= 0.05:
        visual_class = "some_text"
    elif rates["props_or_media"] >= 0.10:
        visual_class = "props_only"
    else:
        visual_class = "near_zero"
    return {
        "n_frames": len(per_frame),
        "slide_frame_rate": round(slide_rate, 3),
        "text_frame_rate": round(text_rate, 3),
        "props_frame_rate": round(rates["props_or_media"], 3),
        "visual_class": visual_class,
    }


if __name__ == "__main__":
    main()
