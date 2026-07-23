#!/usr/bin/env python
"""Deterministic streaming probe: hint-condition comparison, any target language.

Supersedes the vLLM-served runner for evaluation runs. Serving nondeterminism
(batched kernels returning different outputs for identical prefixes) destabilizes
Local Agreement commits and truncates outputs; local greedy transformers decoding
removes that confound (baseline chrF 38.8 -> 62.7 on the same el segments).

Conditions map to hint sources on each item:
  none    no hints (audio-only baseline)
  slide   item["slide_terms"]   -- real VLM-read slide terminology
  oracle  item["oracle_terms"]  -- reference rare terms (anticipation upper bound)
  wrong   item["wrong_terms"]   -- terms from a different slide (faithfulness control)

Target-language handling: --tgt-lang sets the prompt target and the Local
Agreement commit granularity (word-level for space-delimited targets,
character-level for zh/ja), matching how the target language is actually
consumed by a reader.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HINT_FIELD = {"none": None, "slide": "slide_terms",
              "oracle": "oracle_terms", "wrong": "wrong_terms"}
LANG_NAME = {"en": "English", "zh": "Chinese", "de": "German", "ja": "Japanese"}
CHAR_LEVEL = {"zh", "ja"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--conditions", default="none,slide,oracle,wrong")
    parser.add_argument("--tgt-lang", default="en")
    parser.add_argument("--words-per-read", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()

    items = json.load(open(args.items))
    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        for line in out_path.open():
            r = json.loads(line)
            done.add((r["id"], r["condition"]))

    with out_path.open("a", encoding="utf-8") as out:
        for cond in args.conditions.split(","):
            todo = [it for it in items if (it["id"], cond) not in done]
            if not todo:
                continue
            t0 = time.time()
            for i in range(0, len(todo), args.batch_size):
                for r in run_condition(todo[i:i + args.batch_size], cond,
                                       tok, model, args):
                    out.write(json.dumps(r, ensure_ascii=False) + "\n")
                out.flush()
                print(f"[{cond}] {min(i + args.batch_size, len(todo))}/"
                      f"{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    print("PROBE_DONE", flush=True)


def units_of(text: str, n_words: int) -> list[str]:
    words = text.split()
    return [" ".join(words[i:i + n_words]) for i in range(0, len(words), n_words)]


def tokenize_target(text: str, tgt: str) -> list[str]:
    """Commit granularity: characters for zh/ja, words otherwise."""
    return list(text.replace(" ", "")) if tgt in CHAR_LEVEL else text.split()


def join_target(pieces: list[str], tgt: str) -> str:
    return "".join(pieces) if tgt in CHAR_LEVEL else " ".join(pieces)


def run_condition(items, cond, tok, model, args):
    tgt = args.tgt_lang
    field = HINT_FIELD[cond]
    states = [{"it": it, "units": units_of(it["source"], args.words_per_read),
               "committed": [], "prev_full": [], "events": []} for it in items]
    max_steps = max(len(s["units"]) for s in states)

    for step in range(1, max_steps + 1):
        active = [s for s in states if step <= len(s["units"])]
        if not active:
            break
        prompts = []
        for s in active:
            hints = (s["it"].get(field) or []) if field else []
            hint_block = (f"Terminology from the talk's slides that may appear "
                          f"soon: {', '.join(hints)}\n" if hints else "")
            prompts.append(
                f"Translate this partial {s['it']['src_lang']} speech transcript "
                f"into {LANG_NAME.get(tgt, tgt)}.\n{hint_block}"
                f"Partial source (may stop mid-sentence): "
                f"{' '.join(s['units'][:step])}\n"
                f"Output the complete {LANG_NAME.get(tgt, tgt)} translation of "
                f"ONLY what has been spoken so far. No explanations - just the "
                f"translation.")
        for s, text in zip(active, generate_batch(prompts, tok, model, args)):
            full = tokenize_target(
                text.strip().split("\n")[0].strip().strip('"'), tgt)
            if step == len(s["units"]):
                k = lcp_len(s["committed"], full, tgt)
                if len(full) > k:
                    s["committed"] = s["committed"] + full[k:]
                    s["events"].append((step, len(s["committed"])))
            else:
                agree = full[:lcp_len(s["prev_full"], full, tgt)]
                if len(agree) > len(s["committed"]) and \
                        norm(agree[:len(s["committed"])], tgt) == norm(s["committed"], tgt):
                    s["committed"] = agree
                    s["events"].append((step, len(s["committed"])))
            s["prev_full"] = full

    return [{"id": s["it"]["id"], "condition": cond, "tgt_lang": tgt,
             "n_src_units": len(s["units"]), "events": s["events"],
             "hypothesis": join_target(s["committed"], tgt),
             "reference": s["it"]["reference"],
             "oracle_terms": s["it"].get("oracle_terms", []),
             "gloss_hits": s["it"].get("gloss_hits", []),
             "hints_used": (s["it"].get(field) or []) if field else []}
            for s in states]


def norm(pieces: list[str], tgt: str) -> list[str]:
    return pieces if tgt in CHAR_LEVEL else [w.lower() for w in pieces]


def lcp_len(a: list[str], b: list[str], tgt: str) -> int:
    n = 0
    for x, y in zip(norm(a, tgt), norm(b, tgt)):
        if x == y:
            n += 1
        else:
            break
    return n


def generate_batch(prompts, tok, model, args) -> list[str]:
    chats = [tok.apply_chat_template([{"role": "user", "content": p}],
                                     tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False) for p in prompts]
    enc = tok(chats, return_tensors="pt", padding=True,
              padding_side="left").to(model.device)
    with torch.no_grad():
        g = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                           do_sample=False, pad_token_id=tok.eos_token_id)
    return [tok.decode(g[i, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for i in range(len(prompts))]


if __name__ == "__main__":
    main()
