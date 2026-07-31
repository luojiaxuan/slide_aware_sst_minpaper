#!/usr/bin/env python
"""Feasibility v2: does naive whole-slide context help streaming translation on
QUALITY and LATENCY (not terminology)?

Motivated by the reframe that the slide's value for streaming ST is (a) latency
-- 30-60s of anticipatory context lets the model commit earlier and more
confidently -- and (b) word-sense disambiguation (homophones/polysemy, esp.
zh->en), not rare-term recall. So we inject the WHOLE current slide as prose
context (no term extraction, no relevance selection -- deliberately naive; the
relevance selection is what RASST's retriever would add later) and measure chrF
and Average Lagging.

Conditions:
  full    non-streaming, all source read at once, NO slide  -> reference proxy
          (same model, so this isolates streaming+slide effects from capability)
  none    streaming, no slide                               -> baseline
  slide   streaming + whole current slide as context        -> naive vision
  wrong   streaming + a different slide's content           -> content control

Metrics per streaming condition: chrF vs the 'full' hypothesis, and Average
Lagging (Ma et al. 2019) in source READ units from the commit events.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CTX_FIELD = {"none": None, "slide": "slide_context", "wrong": "wrong_context"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--conditions", default="full,none,slide,wrong")
    ap.add_argument("--words-per-read", type=int, default=3)
    ap.add_argument("--chars-per-read", type=int, default=4, help="for Chinese source")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda:0").eval()

    items = json.load(open(args.items))
    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        for l in out_path.open():
            r = json.loads(l)
            done.add((r["id"], r["condition"]))

    with out_path.open("a", encoding="utf-8") as out:
        for cond in args.conditions.split(","):
            todo = [it for it in items if (it["id"], cond) not in done]
            for i in range(0, len(todo), args.batch_size):
                for r in run(todo[i:i + args.batch_size], cond, tok, model, args):
                    out.write(json.dumps(r, ensure_ascii=False) + "\n")
                out.flush()
                print(f"[{cond}] {min(i + args.batch_size, len(todo))}/{len(todo)}",
                      flush=True)
    print("PROBE_DONE", flush=True)


def units(text, n, src_lang="", char_n=4):
    if src_lang == "Chinese" or (not text.count(" ") and len(text) > 12):
        ch = list(text)
        return ["".join(ch[i:i + char_n]) for i in range(0, len(ch), char_n)] or [text]
    w = text.split()
    return [" ".join(w[i:i + n]) for i in range(0, len(w), n)] or [text]


def build_prompt(src_lang, prefix, ctx):
    ctx_block = ""
    if ctx:
        joined = "; ".join(ctx) if isinstance(ctx, list) else str(ctx)
        ctx_block = (f"[CURRENT SLIDE - context to disambiguate word senses; "
                     f"do NOT translate or output this, translate only the "
                     f"speech]\n{joined}\n[END SLIDE]\n")
    return (f"You are a simultaneous interpreter translating {src_lang} speech "
            f"into English.\n{ctx_block}"
            f"[SPEECH SO FAR - may stop mid-sentence]\n{prefix}\n[END SPEECH]\n"
            f"Output ONLY the English translation of the speech above. No "
            f"explanations, no source text, no slide text.")


def run(items, cond, tok, model, args):
    field = CTX_FIELD.get(cond)
    if cond == "full":
        # non-streaming reference: read everything, no slide
        prompts = [build_prompt(it["src_lang"], it["source"], None) for it in items]
        outs = generate(prompts, tok, model, args)
        return [{"id": it["id"], "condition": cond, "events": [],
                 "n_src_units": len(units(it["source"], args.words_per_read, it["src_lang"], args.chars_per_read)),
                 "hypothesis": clean(o), "reference": it.get("reference", ""),
                 "source": it["source"]} for it, o in zip(items, outs)]

    def _join(u, lang):
        return "".join(u) if lang == "Chinese" else " ".join(u)
    states = [{"it": it, "u": units(it["source"], args.words_per_read, it["src_lang"], args.chars_per_read),
               "c": [], "pf": [], "ev": [], "lang": it["src_lang"]} for it in items]
    maxs = max(len(s["u"]) for s in states)
    for step in range(1, maxs + 1):
        act = [s for s in states if step <= len(s["u"])]
        if not act:
            break
        prompts = []
        for s in act:
            ctx = (s["it"].get(field) or []) if field else None
            prompts.append(build_prompt(s["it"]["src_lang"],
                                        _join(s["u"][:step], s["lang"]), ctx))
        for s, o in zip(act, generate(prompts, tok, model, args)):
            full = clean(o).split()
            if step == len(s["u"]):
                k = len(s["c"])
                if len(full) > k:
                    s["c"] += full[k:]; s["ev"].append((step, len(s["c"])))
            else:
                agree = full[:lcp(s["pf"], full)]
                if len(agree) > len(s["c"]) and \
                        [w.lower() for w in agree[:len(s["c"])]] == [w.lower() for w in s["c"]]:
                    s["c"] = agree; s["ev"].append((step, len(s["c"])))
            s["pf"] = full
    return [{"id": s["it"]["id"], "condition": cond, "events": s["ev"],
             "n_src_units": len(s["u"]), "hypothesis": " ".join(s["c"]),
             "reference": s["it"].get("reference", ""),
             "ctx_used": (s["it"].get(field) or []) if field else []}
            for s in states]


def clean(t):
    return t.strip().split("\n")[0].strip().strip('"')


def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x.lower() == y.lower():
            n += 1
        else:
            break
    return n


def generate(prompts, tok, model, args):
    chats = [tok.apply_chat_template([{"role": "user", "content": p}],
             tokenize=False, add_generation_prompt=True, enable_thinking=False)
             for p in prompts]
    enc = tok(chats, return_tensors="pt", padding=True,
              padding_side="left").to(model.device)
    with torch.no_grad():
        g = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                           do_sample=False, pad_token_id=tok.eos_token_id)
    return [tok.decode(g[i, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for i in range(len(prompts))]


if __name__ == "__main__":
    main()
