#!/usr/bin/env python
"""Session-level streaming translation with drift-tolerant commit.

Moves from segment-level probing to full-session streaming: segments of one
talk are processed as one continuous stream with (i) cross-segment translation
context (the tail of previously committed output conditions each new segment),
(ii) slide hints pulled from the VLM timeline by wall-clock segment start, and
(iii) a drift-tolerant commit rule for serving-grade inference where identical
prefixes can yield different outputs (vLLM batching nondeterminism):

  - each READ step samples the full-prefix translation twice; the step
    hypothesis is the LCP of the two samples (averaging out serving jitter);
  - commits extend monotonically while the step hypothesis agrees with the
    committed prefix (case-insensitive); on disagreement the step is skipped
    (drift tolerated, nothing frozen); after --max-drift consecutive
    disagreements the tail is force-resynced to the current hypothesis.

Conditions: none | slide (prompt hints from the slide visible at segment start).
Outputs one JSONL row per segment with committed text, events, and the applied
hints, plus a session summary row.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--talk", required=True, help="talk-level manifest JSON line file (one talk)")
    parser.add_argument("--slide-timeline", default=None, help="VLM slide timeline JSON")
    parser.add_argument("--out", required=True)
    parser.add_argument("--condition", choices=["none", "slide"], default="none")
    parser.add_argument("--src-lang", default="Greek")
    parser.add_argument("--url", default="http://localhost:8901")
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--words-per-read", type=int, default=2)
    parser.add_argument("--chars-per-read", type=int, default=4)
    parser.add_argument("--context-tail-words", type=int, default=40)
    parser.add_argument("--max-drift", type=int, default=3)
    parser.add_argument("--lookback", type=float, default=90.0)
    args = parser.parse_args()

    talk = json.loads(Path(args.talk).read_text())
    timeline = json.loads(Path(args.slide_timeline).read_text()) if args.slide_timeline else []

    def slide_terms_at(t0: float) -> list[str]:
        best: list[str] = []
        for fr in timeline:
            if fr["t"] <= t0 and len(fr["terms"]) >= 2 and t0 - fr["t"] <= args.lookback:
                best = fr["terms"]
        return best[:10]

    out = Path(args.out).open("w", encoding="utf-8")
    session_committed: list[str] = []
    t_start = time.time()
    for seg in talk["segments"]:
        hints = slide_terms_at(seg["start"]) if args.condition == "slide" else []
        committed, events, drift_events = stream_segment(
            seg["transcript"], hints, session_committed, args)
        session_committed.extend(committed)
        out.write(json.dumps({
            "segment_id": seg["segment_id"], "start": seg["start"],
            "condition": args.condition, "hypothesis": " ".join(committed),
            "reference": seg["translation"], "events": events,
            "drift_resyncs": drift_events, "hints_used": hints,
        }, ensure_ascii=False) + "\n")
        out.flush()
        print(f"seg {seg['segment_id']}: {len(committed)}w "
              f"(session {len(session_committed)}w)", flush=True)
    out.write(json.dumps({"session_summary": True, "talk_id": talk["talk_id"],
                          "condition": args.condition,
                          "n_segments": len(talk["segments"]),
                          "total_words": len(session_committed),
                          "wall_s": round(time.time() - t_start, 1)},
                         ensure_ascii=False) + "\n")
    out.close()


def stream_segment(source: str, hints: list[str], session_tail: list[str],
                   args) -> tuple[list[str], list, int]:
    units = source_units(source, args)
    committed: list[str] = []
    events = []
    drift = 0
    resyncs = 0
    prev_hyp: list[str] = []
    context = " ".join(session_tail[-args.context_tail_words:])
    for n_read in range(1, len(units) + 1):
        prefix = join_units(units[:n_read], args)
        final = n_read == len(units)
        a = translate(prefix, hints, context, args)
        b = translate(prefix, hints, context, args)
        hyp = lcp(a, b)                       # jitter-averaged step hypothesis
        if final:
            best = a if len(a) >= len(b) else b
            k = len(lcp(committed, best))
            if len(best) > k:
                committed = committed + best[k:]
                events.append((n_read, len(committed)))
            break
        agree = lcp(prev_hyp, hyp)
        if len(agree) > len(committed) and _prefix_eq(agree, committed):
            committed = agree
            events.append((n_read, len(committed)))
            drift = 0
        elif len(agree) > 0 and not _prefix_eq(agree[:len(committed)], committed):
            drift += 1
            if drift >= args.max_drift:       # force resync: adopt current view
                committed = agree
                events.append((n_read, len(committed)))
                resyncs += 1
                drift = 0
        prev_hyp = hyp
    return committed, events, resyncs


def _prefix_eq(a: list[str], b: list[str]) -> bool:
    return [w.lower() for w in a[:len(b)]] == [w.lower() for w in b]


def source_units(text: str, args) -> list[str]:
    if args.src_lang == "Chinese":
        chars = list(text)
        return ["".join(chars[i:i + args.chars_per_read])
                for i in range(0, len(chars), args.chars_per_read)]
    words = text.split()
    return [" ".join(words[i:i + args.words_per_read])
            for i in range(0, len(words), args.words_per_read)]


def join_units(units: list[str], args) -> str:
    return "".join(units) if args.src_lang == "Chinese" else " ".join(units)


def lcp(a: list[str], b: list[str]) -> list[str]:
    out = []
    for x, y in zip(a, b):
        if x.lower() == y.lower():
            out.append(y)
        else:
            break
    return out


def translate(prefix: str, hints: list[str], context: str, args) -> list[str]:
    hint_block = (f"Terminology from the speaker's current slide: "
                  f"{', '.join(hints)}\n" if hints else "")
    ctx_block = (f"Preceding translation (context, do not repeat): ...{context}\n"
                 if context else "")
    prompt = (f"You are simultaneously translating a live {args.src_lang} talk "
              f"into English.\n{ctx_block}{hint_block}"
              f"Current sentence so far (may stop mid-sentence): {prefix}\n"
              f"Output the complete English translation of ONLY the current "
              f"sentence so far. No explanations - just the translation.")
    body = json.dumps({"model": args.model, "temperature": 0.0, "max_tokens": 200,
                       "messages": [{"role": "user", "content": prompt}],
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(f"{args.url}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                text = json.load(r)["choices"][0]["message"]["content"]
            return text.strip().split("\n")[0].strip().strip('"').split()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(5)
    return []


if __name__ == "__main__":
    main()
