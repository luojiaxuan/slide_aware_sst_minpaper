#!/usr/bin/env python
"""Trie-constrained contextual biasing probe (injection-form study, final leg).

Compares three injection forms under the Local Agreement streaming protocol:
  none    audio-only baseline (no hints anywhere)
  prompt  hint terms in the prompt (known: powerful on hard, disruptive on easy)
  trie    hint terms ONLY as a trie-constrained logits bias (shallow-fusion
          style): entering a term's first token gets a small root bias; once
          the generated tail matches a term prefix, its continuation tokens get
          a large in-term bias. Prompt untouched -> no LCP perturbation.

Runs locally with transformers (greedy, batched per streaming step) so a custom
LogitsProcessor can be used. Designed for one 48GB GPU (Qwen3-14B bf16).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor


class TrieBias(LogitsProcessor):
    """Shallow-fusion trie bias over hint-term token sequences."""

    def __init__(self, tokenizer, terms: list[str], root_bias: float,
                 in_bias: float, max_tail: int = 10):
        self.root_bias = root_bias
        self.in_bias = in_bias
        self.max_tail = max_tail
        self.seqs: list[list[int]] = []
        for term in terms:
            for variant in (term, " " + term):
                ids = tokenizer.encode(variant, add_special_tokens=False)
                if 1 <= len(ids) <= 24:
                    self.seqs.append(ids)
        self.roots = {s[0] for s in self.seqs}

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        for b in range(input_ids.shape[0]):
            tail = input_ids[b, -self.max_tail:].tolist()
            boost: set[int] = set()
            # in-term continuations: any term prefix matching a suffix of tail
            for seq in self.seqs:
                for k in range(1, min(len(seq), len(tail)) + 1):
                    if len(seq) > k and tail[-k:] == seq[:k]:
                        boost.add(seq[k])
            for tok in boost:
                scores[b, tok] += self.in_bias
            for tok in self.roots:
                if tok not in boost:
                    scores[b, tok] += self.root_bias
        return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--conditions", default="none,prompt,trie")
    parser.add_argument("--root-bias", type=float, default=1.5)
    parser.add_argument("--in-bias", type=float, default=6.0)
    parser.add_argument("--words-per-read", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=160)
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
            results = run_condition(todo, cond, tok, model, args)
            for r in results:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[{cond}] {len(results)} items in {time.time()-t0:.0f}s",
                  flush=True)


def run_condition(items, cond, tok, model, args):
    states = []
    for it in items:
        words = it["source"].split()
        units = [" ".join(words[i:i + args.words_per_read])
                 for i in range(0, len(words), args.words_per_read)]
        states.append({"it": it, "units": units, "committed": [],
                       "prev_full": [], "events": [], "done": False})
    max_steps = max(len(s["units"]) for s in states)

    for step in range(1, max_steps + 1):
        active = [s for s in states if step <= len(s["units"])]
        if not active:
            break
        prompts, procs = [], []
        for s in active:
            prefix = " ".join(s["units"][:step])
            hints = s["it"].get("slide_terms") or []
            hint_block = (f"Terminology from the talk's slides that may appear "
                          f"soon: {', '.join(hints)}\n"
                          if cond == "prompt" and hints else "")
            prompts.append(
                f"Translate this partial {s['it']['src_lang']} speech "
                f"transcript into English.\n{hint_block}"
                f"Partial source (may stop mid-sentence): {prefix}\n"
                f"Output the complete English translation of ONLY what has "
                f"been spoken so far. No explanations - just the translation.")
        full_texts = generate_batch(prompts, active, cond, tok, model, args)
        for s, text in zip(active, full_texts):
            full = text.strip().split("\n")[0].strip().strip('"').split()
            final = step == len(s["units"])
            if final:
                k = lcp_len(s["committed"], full)
                if len(full) > k:
                    s["committed"] = s["committed"] + full[k:]
                    s["events"].append((step, len(s["committed"])))
            else:
                agree = full[:lcp_len(s["prev_full"], full)]
                if (len(agree) > len(s["committed"]) and
                        [w.lower() for w in agree[:len(s["committed"])]] ==
                        [w.lower() for w in s["committed"]]):
                    s["committed"] = agree
                    s["events"].append((step, len(s["committed"])))
            s["prev_full"] = full

    return [{"id": s["it"]["id"], "condition": cond,
             "n_src_units": len(s["units"]), "events": s["events"],
             "hypothesis": " ".join(s["committed"]),
             "reference": s["it"]["reference"],
             "oracle_terms": s["it"]["oracle_terms"],
             "hints_used": s["it"].get("slide_terms") or []} for s in states]


def generate_batch(prompts, active, cond, tok, model, args):
    chats = [tok.apply_chat_template(
        [{"role": "user", "content": p}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False) for p in prompts]
    enc = tok(chats, return_tensors="pt", padding=True,
              padding_side="left").to(model.device)
    processors = None
    if cond == "trie":
        # one shared processor per batch is wrong if hints differ; batch by item
        outs = []
        for i, s in enumerate(active):
            hints = s["it"].get("slide_terms") or []
            proc = ([TrieBias(tok, hints, args.root_bias, args.in_bias)]
                    if hints else None)
            single = tok(chats[i], return_tensors="pt").to(model.device)
            with torch.no_grad():
                g = model.generate(**single, max_new_tokens=args.max_new_tokens,
                                   do_sample=False,
                                   logits_processor=proc,
                                   pad_token_id=tok.eos_token_id)
            outs.append(tok.decode(g[0, single["input_ids"].shape[1]:],
                                   skip_special_tokens=True))
        return outs
    with torch.no_grad():
        g = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                           do_sample=False, pad_token_id=tok.eos_token_id)
    return [tok.decode(g[i, enc["input_ids"].shape[1]:],
                       skip_special_tokens=True) for i in range(len(prompts))]


def lcp_len(a: list[str], b: list[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x.lower() == y.lower():
            n += 1
        else:
            break
    return n


if __name__ == "__main__":
    main()
