#!/usr/bin/env python
"""Oracle anticipation kill test: incremental translation with term-hint conditions.

Simulates simultaneous translation with the Local Agreement commit policy
(Liu et al., 2020): at each READ step the model produces a full translation of
the source prefix, and the longest common prefix of two consecutive outputs
beyond the already-committed words is committed (monotonic, no retraction).
Four conditions:

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
    parser.add_argument("--api", choices=["ollama", "openai"], default="ollama")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--words-per-read", type=int, default=2)
    parser.add_argument("--chars-per-read", type=int, default=4, help="for Chinese source")
    parser.add_argument("--max-new-words", type=int, default=12)
    parser.add_argument("--conditions", default=None,
                        help="comma-separated subset of conditions to run")
    args = parser.parse_args()

    items = json.load(open(args.items))
    conds = list(CONDITIONS)
    if args.conditions:
        conds = [c for c in args.conditions.split(",") if c in CONDITIONS]
    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        for line in out_path.open():
            r = json.loads(line)
            done.add((r["id"], r["condition"]))

    with out_path.open("a", encoding="utf-8") as out:
        for it, cond in [(i, c) for i in items for c in conds]:
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

    prev_full: list[str] = []
    for n_read in range(1, len(src_units) + 1):
        prefix = join_units(src_units[:n_read], it["src_lang"])
        final = n_read == len(src_units)
        full = full_translation(prefix, hints, it["src_lang"], args)
        if final:
            agree = full if len(full) >= len(committed) else committed
        else:
            agree = lcp(prev_full, full)
        if len(agree) > len(committed) and [w.lower() for w in agree[:len(committed)]] == [w.lower() for w in committed]:
            committed = agree
            events.append((n_read, len(committed)))
        prev_full = full

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


def lcp(a: list[str], b: list[str]) -> list[str]:
    out = []
    for x, y in zip(a, b):
        if x.lower() == y.lower():
            out.append(y)
        else:
            break
    return out


def full_translation(prefix: str, hints: list[str], lang: str, args) -> list[str]:
    hint_block = (f"Terminology from the talk's slides/context that may appear "
                  f"soon: {', '.join(hints)}\n" if hints else "")
    prompt = (f"Translate this partial {lang} speech transcript into English.\n"
              f"{hint_block}"
              f"Partial source (speech so far, may stop mid-sentence): {prefix}\n"
              f"Output the complete English translation of ONLY what has been "
              f"spoken so far. No explanations, no notes - just the translation.")
    text = ollama_generate(prompt, args).strip().split("\n")[0].strip().strip('"')
    return text.split()


def ollama_generate(prompt: str, args) -> str:
    if args.api == "openai":
        body = json.dumps({"model": args.model, "temperature": 0.0, "max_tokens": 200,
                           "messages": [{"role": "user", "content": prompt}],
                           "chat_template_kwargs": {"enable_thinking": False}}).encode()
        req = urllib.request.Request(f"{args.ollama_url}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    return json.load(r)["choices"][0]["message"]["content"]
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(5)
        return ""
    body = json.dumps({"model": args.model, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0.0, "num_predict": 160}}).encode()
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
