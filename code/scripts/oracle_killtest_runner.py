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

CONDITIONS = {"none": None, "slide": "slide_terms", "oracle": "oracle_terms",
              "wrong": "wrong_terms",
              # gating prototypes: same hint source as "slide", but injected selectively
              "gated_oracle": "slide_terms",  # inject only if item["gate_on"] (hard-stratum oracle gate)
              "gated_llm": "slide_terms",     # inject once a runtime LLM relevance gate fires (sticky)
              "gated_unc": "slide_terms",     # inject once uncertainty/match signals fire (eq. gate, sticky)
              "bias": "slide_terms"}          # no prompt injection: decode-time logit bias on hint tokens


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
    parser.add_argument("--logit-bias", type=float, default=4.0,
                        help="bias value for the 'bias' condition")
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
    if cond == "gated_oracle" and not it.get("gate_on"):
        hints = []
    bias_map = build_bias(hints, args) if cond == "bias" else None
    llm_gate_open = cond not in ("gated_llm", "gated_unc")  # gated variants start closed
    gate_step = None
    stall_count = 0
    committed: list[str] = []
    events = []  # (n_units_read, n_target_words_total_after_step)

    prev_full: list[str] = []
    for n_read in range(1, len(src_units) + 1):
        prefix = join_units(src_units[:n_read], it["src_lang"])
        final = n_read == len(src_units)
        if cond == "gated_llm" and not llm_gate_open and hints:
            if ask_gate(prefix, hints, it["src_lang"], args):
                llm_gate_open = True
                gate_step = n_read
        if cond == "gated_unc" and not llm_gate_open and hints:
            # forward-looking need signals: (a) any instability (stall >= 1);
            # (b) long/rare word entering the source prefix (terminology incoming)
            recent = src_units[max(0, n_read - 2):n_read]
            long_word = any(len(w) >= 8 for u in recent for w in u.split())
            if stall_count >= 1 or long_word:
                llm_gate_open = True
                gate_step = n_read
        step_hints = [] if cond == "bias" else (hints if llm_gate_open else [])
        full = full_translation(prefix, step_hints, it["src_lang"], args, bias_map)
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
            "hints_used": hints, "gate_step": gate_step}


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


def hint_match(hints: list[str], hypothesis: list[str]) -> bool:
    """Stem-overlap between any hint word and the model's own partial hypothesis."""
    hyp_stems = {w.lower().rstrip("s")[:6] for w in hypothesis if len(w) >= 5}
    for h in hints:
        for w in h.split():
            if len(w) >= 5 and w.lower().rstrip("s")[:6] in hyp_stems:
                return True
    return False


def ask_gate(prefix: str, hints: list[str], lang: str, args) -> bool:
    prompt = (f"A simultaneous interpreter is translating {lang} speech into English.\n"
              f"Speech so far: {prefix}\n"
              f"Candidate terms read from the speaker's current slide: {', '.join(hints)}\n"
              f"Question: are these slide terms likely to appear in, or help translate, "
              f"the speech that is being given? Answer with exactly one word: YES or NO.")
    text = ollama_generate(prompt, args).strip().upper()
    return text.startswith("YES")


def build_bias(hints: list[str], args) -> dict:
    """Tokenize hint terms (with leading-space variants) into a logit_bias map."""
    ids = set()
    for h in hints:
        for variant in (h, " " + h):
            body = json.dumps({"model": args.model, "prompt": variant}).encode()
            req = urllib.request.Request(f"{args.ollama_url}/tokenize", data=body,
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    ids.update(json.load(r)["tokens"])
            except Exception:
                pass
    return {str(i): args.logit_bias for i in list(ids)[:250]}


def full_translation(prefix: str, hints: list[str], lang: str, args,
                     bias_map: dict | None = None) -> list[str]:
    hint_block = (f"Terminology from the talk's slides/context that may appear "
                  f"soon: {', '.join(hints)}\n" if hints else "")
    prompt = (f"Translate this partial {lang} speech transcript into English.\n"
              f"{hint_block}"
              f"Partial source (speech so far, may stop mid-sentence): {prefix}\n"
              f"Output the complete English translation of ONLY what has been "
              f"spoken so far. No explanations, no notes - just the translation.")
    text = ollama_generate(prompt, args, bias_map).strip().split("\n")[0].strip().strip('"')
    return text.split()


def ollama_generate(prompt: str, args, bias_map: dict | None = None) -> str:
    if args.api == "openai":
        payload = {"model": args.model, "temperature": 0.0, "max_tokens": 200,
                   "messages": [{"role": "user", "content": prompt}],
                   "chat_template_kwargs": {"enable_thinking": False}}
        if bias_map:
            payload["logit_bias"] = bias_map
        body = json.dumps(payload).encode()
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
