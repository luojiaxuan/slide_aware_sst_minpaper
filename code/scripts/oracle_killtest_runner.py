#!/usr/bin/env python
"""Oracle anticipation kill test: incremental translation with term-hint conditions.

Simulates prompt-based simultaneous translation with a greedy commit policy over
an incrementally revealed source transcript, under four conditions:

  A none         no term hints
  B slide        terms OCR'd from the slide visible at segment start (realistic)
  C oracle       rare content words from the segment's reference (perfect
                 anticipation upper bound)
  D wrong        slide terms from a different moment/talk (faithfulness control)

Pre-registered decision rule: if C shows no term-recall/latency-quality gain
over A, the vision-as-anticipation premise dies; C >> A with B ~ A means
extraction (not the premise) is the bottleneck; gains in B confirm the realistic
channel. Metrics computed by oracle_killtest_analyze.py: AL, chrF, term recall.

Requires a local ollama server (default model qwen2.5:7b).
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

CONDITIONS = {"none": None, "slide": "slide_terms", "oracle": "oracle_terms", "wrong": "wrong_terms"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--words-per-read", type=int, default=2)
    parser.add_argument("--chars-per-read", type=int, default=4, help="for Chinese source")
    parser.add_argument("--max-new-words", type=int, default=12)
    args = parser.parse_args()

    items = json.load(open(args.items))
    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        for line in out_path.open():
            r = json.loads(line)
            done.add((r["id"], r["condition"]))

    with out_path.open("a", encoding="utf-8") as out:
        for it, cond in [(i, c) for i in items for c in CONDITIONS]:
            if (it["id"], cond) in done:
                continue
            t0 = time.time()
            result = run_incremental(it, cond, args)
            result["wall_s"] = round(time.time() - t0, 1)
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            print(f"{it['id']}/{cond}: {len(result['hypothesis'].split())}w "
                  f"in {result['wall_s']}s", flush=True)


def run_incremental(it: dict, cond: str, args) -> dict:
    src_units = source_units(it["source"], it["src_lang"], args)
    hints_key = CONDITIONS[cond]
    hints = it.get(hints_key) or [] if hints_key else []
    committed: list[str] = []
    events = []  # (n_units_read, n_target_words_total_after_step)

    for n_read in range(1, len(src_units) + 1):
        prefix = join_units(src_units[:n_read], it["src_lang"])
        final = n_read == len(src_units)
        new_words = step(prefix, committed, hints, it["src_lang"], final, args)
        if new_words:
            committed.extend(new_words)
            events.append((n_read, len(committed)))

    return {"id": it["id"], "condition": cond, "n_src_units": len(src_units),
            "events": events, "hypothesis": " ".join(committed),
            "reference": it["reference"], "oracle_terms": it["oracle_terms"],
            "hints_used": hints}


def source_units(text: str, lang: str, args) -> list[str]:
    if lang == "Chinese":
        chars = list(text)
        return ["".join(chars[i:i + args.chars_per_read])
                for i in range(0, len(chars), args.chars_per_read)]
    words = text.split()
    return [" ".join(words[i:i + args.words_per_read])
            for i in range(0, len(words), args.words_per_read)]


def join_units(units: list[str], lang: str) -> str:
    return "".join(units) if lang == "Chinese" else " ".join(units)


def step(prefix: str, committed: list[str], hints: list[str], lang: str,
         final: bool, args) -> list[str]:
    hint_block = (f"Terminology that may appear in this talk (from context): "
                  f"{', '.join(hints)}\n" if hints else "")
    task = ("The source is now COMPLETE. Output the remaining English words to "
            "finish the translation." if final else
            "Output ONLY the next English words that are already certain given "
            "the partial source. Never guess content not yet spoken. If nothing "
            "can be safely added yet, output exactly: NOTHING")
    prompt = (f"You are a professional simultaneous interpreter translating "
              f"{lang} speech into English, word by word as it arrives.\n"
              f"{hint_block}"
              f"Partial {lang} source so far: {prefix}\n"
              f"English committed so far: {' '.join(committed) if committed else '(nothing yet)'}\n"
              f"{task}\n"
              f"Reply with the new English words only - no quotes, no explanations.")
    text = ollama_generate(prompt, args)
    text = text.strip().split("\n")[0].strip().strip('"')
    if not text or text.upper().startswith("NOTHING"):
        return []
    words = text.split()
    if not final:
        words = words[:args.max_new_words]
    return words


def ollama_generate(prompt: str, args) -> str:
    body = json.dumps({"model": args.model, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0.0, "num_predict": 80}}).encode()
    req = urllib.request.Request(f"{args.ollama_url}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.load(r).get("response", "")
        except Exception:
            if attempt == 2:
                raise
            time.sleep(5)
    return ""


if __name__ == "__main__":
    main()
