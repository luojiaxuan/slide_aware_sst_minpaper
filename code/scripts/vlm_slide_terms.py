#!/usr/bin/env python
"""Extract slide terminology from talk frames with a VLM (OpenAI-compatible endpoint).

For each frame image, asks the VLM to list the readable terms/phrases on the
projected slide (ignoring stage/venue text), producing a JSON timeline compatible
with the kill-test items builder: [{"t": seconds, "terms": [...]}].

Replaces the tesseract lower bound that the kill test showed to be the
bottleneck (C >> A but B ~ A with OCR terms).
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import time
import urllib.request
from pathlib import Path

PROMPT = (
    "This is a frame from a conference talk video. Look at the projected slide "
    "or screen, if one is visible.\n"
    "List the distinct terms, named entities, and key phrases that are readable "
    "on the slide (not stage decorations, venue logos, or watermarks). Prefer "
    "content-bearing terminology; include numbers with their units/labels.\n"
    "Reply with ONLY a JSON array of strings, e.g. [\"term one\", \"term two\"]. "
    "If no slide text is readable, reply []."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", required=True, help="dir with f_NNN.jpg")
    parser.add_argument("--interval", type=float, default=20.0,
                        help="seconds between frames (fps=1/interval extraction)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--url", default="http://localhost:8902")
    parser.add_argument("--max-terms", type=int, default=20)
    args = parser.parse_args()

    frames = sorted(Path(args.frames_dir).glob("f_*.jpg"))
    timeline = []
    for i, frame in enumerate(frames):
        t = (i + 0.5) * args.interval
        terms = extract(frame, args)
        timeline.append({"t": round(t, 1), "terms": terms[:args.max_terms]})
        print(f"{frame.name} @{t:.0f}s: {len(terms)} terms "
              f"{terms[:5]}", flush=True)
    Path(args.out).write_text(json.dumps(timeline, ensure_ascii=False, indent=1))
    n = sum(1 for x in timeline if len(x["terms"]) >= 3)
    print(f"frames with >=3 terms: {n}/{len(timeline)} -> {args.out}")


def extract(frame: Path, args) -> list[str]:
    image_b64 = base64.b64encode(frame.read_bytes()).decode()
    body = json.dumps({
        "model": args.model, "temperature": 0.0, "max_tokens": 400,
        "messages": [{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            {"type": "text", "text": PROMPT},
        ]}],
    }).encode()
    req = urllib.request.Request(f"{args.url}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                text = json.load(r)["choices"][0]["message"]["content"]
            m = re.search(r"\[.*\]", text, re.S)
            if not m:
                return []
            terms = json.loads(m.group(0))
            return [str(x).strip() for x in terms if str(x).strip()]
        except Exception:
            if attempt == 2:
                return []
            time.sleep(5)
    return []


if __name__ == "__main__":
    main()
