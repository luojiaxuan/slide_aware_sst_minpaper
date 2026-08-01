#!/usr/bin/env python
"""Speech + vision streaming probe on Qwen3-Omni (the decisive feasibility test).

Why this design: all earlier probes fed a *transcript* to a text LLM, which
deletes the acoustic ambiguity that visual context is supposed to resolve
(if the transcript already says the right homophone, the slide has nothing to
add). Here the model consumes AUDIO incrementally, so the ambiguity is real,
and the slide is supplied as an IMAGE to the omni model's vision encoder.

Streaming protocol: audio is revealed in fixed-duration chunks; at each step the
model translates the audio heard so far, and Local Agreement (Liu et al. 2020)
commits the longest common prefix of two consecutive hypotheses. Latency is the
Average Lagging (Ma et al. 2019) over commit events, in chunks (= seconds).

Conditions:
  none    audio only                          -> baseline
  slide   audio + current slide image         -> vision
  wrong   audio + a different slide's image   -> content control (critical:
          separates real visual information from generic prompt perturbation)
  cross_talk  audio + a slide from another corpus/talk -> domain control
  blank       audio + a blank image                  -> vision-slot control

Run inside the sglang-omni container with HF_HOME pointing at the persistent
cache, e.g.
  HF_HOME=/data/hf_cache python3 omni_speech_vision_probe.py \
      --items items.json --audio-root /data/... --out runs.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True,
                    help="JSONL/JSON: id, audio, slide_image, wrong_image, "
                         "cross_talk_image, blank_image, reference, languages")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument("--model-revision")
    ap.add_argument("--conditions", default="none,slide,wrong")
    ap.add_argument("--chunk-s", type=float, default=1.0,
                    help="audio revealed per READ step (seconds)")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import (AutoConfig, Qwen3OmniMoeProcessor,
                              Qwen3OmniMoeThinkerForConditionalGeneration)

    torch.manual_seed(args.seed)
    processor = Qwen3OmniMoeProcessor.from_pretrained(
        args.model, revision=args.model_revision, trust_remote_code=True
    )
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cfg = AutoConfig.from_pretrained(
        args.model, revision=args.model_revision, trust_remote_code=True
    )
    model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        args.model, revision=args.model_revision, config=cfg.thinker_config,
        trust_remote_code=True,
        dtype="auto", device_map=args.device_map,
        attn_implementation=args.attn).eval()

    items = load_items(args.items)
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid shard index/count")
    items = [
        item for index, item in enumerate(items)
        if index % args.shard_count == args.shard_index
    ]
    if args.limit:
        items = items[:args.limit]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.open():
            r = json.loads(line)
            done.add((r["id"], r["condition"]))

    with out_path.open("a", encoding="utf-8") as out:
        for cond in [value.strip() for value in args.conditions.split(",") if value.strip()]:
            for i, it in enumerate(items):
                if (it["id"], cond) in done:
                    continue
                t0 = time.time()
                rec = stream_one(it, cond, processor, tokenizer, model, args)
                rec["wall_s"] = round(time.time() - t0, 1)
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                print(f"[{cond}] {i+1}/{len(items)} {it['id']} "
                      f"{rec['n_chunks']}ch {rec['wall_s']}s", flush=True)
    print("PROBE_DONE", flush=True)


def load_items(path: str) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def read_audio(path: str) -> tuple[np.ndarray, int]:
    import soundfile as sf
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 2:
        a = a.mean(axis=1)
    return np.nan_to_num(a.reshape(-1)), int(sr)


def stream_one(item, cond, processor, tokenizer, model, args) -> dict:
    audio, sr = read_audio(item["audio"])
    chunk = max(1, int(round(args.chunk_s * sr)))
    n_chunks = max(1, int(np.ceil(len(audio) / chunk)))
    image = None
    image_fields = {
        "slide": "slide_image",
        "wrong": "wrong_image",
        "cross_talk": "cross_talk_image",
        "blank": "blank_image",
    }
    if cond not in {"none", *image_fields}:
        raise ValueError(f"Unknown condition: {cond}")
    if cond in image_fields:
        image = item.get(image_fields[cond])
        if not image:
            raise ValueError(f"Missing image for {cond}: {item['id']}")

    committed: list[str] = []
    prev: list[str] = []
    events: list[tuple[int, int]] = []
    tgt = item.get("tgt_lang", "English")

    for step in range(1, n_chunks + 1):
        prefix_audio = audio[: step * chunk]
        text = translate_prefix(prefix_audio, sr, image, item, tgt,
                                processor, tokenizer, model, args)
        full = text.split()
        if step == n_chunks:
            k = len(committed)
            if len(full) > k:
                committed = committed + full[k:]
                events.append((step, len(committed)))
        else:
            agree = full[:lcp(prev, full)]
            if len(agree) > len(committed) and \
                    [w.lower() for w in agree[:len(committed)]] == \
                    [w.lower() for w in committed]:
                committed = agree
                events.append((step, len(committed)))
        prev = full

    return {"id": item["id"], "condition": cond, "n_chunks": n_chunks,
            "chunk_s": args.chunk_s, "events": events,
            "hypothesis": " ".join(committed),
            "reference": item.get("reference", ""),
            "image_used": image or "",
            "model": args.model,
            "model_revision": args.model_revision,
            "attention": args.attn,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "shard_index": args.shard_index,
            "shard_count": args.shard_count}


def translate_prefix(audio_prefix, sr, image_path, item, tgt,
                     processor, tokenizer, model, args) -> str:
    content: list[dict] = []
    if image_path:
        content.append({"type": "image", "image": image_path})
        content.append({"type": "text", "text":
                        "The image above is the slide currently on screen. "
                        "Use it only to resolve ambiguous words in the speech; "
                        "never translate or output slide content itself."})
    content.append({"type": "audio", "audio": audio_prefix})
    content.append({"type": "text", "text":
                    f"Translate the {item.get('src_lang','Chinese')} speech heard "
                    f"so far into {tgt}. The audio may stop mid-sentence; "
                    f"translate only what was actually said. Output only the "
                    f"{tgt} translation."})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages, add_generation_prompt=True,
                                         tokenize=False)
    images = [image_path] if image_path else None
    inputs = processor(text=text, audio=[audio_prefix], images=images,
                       videos=None, return_tensors="pt", padding=True,
                       sampling_rate=sr, use_audio_in_video=False)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    for k, v in list(inputs.items()):
        if hasattr(v, "to"):
            inputs[k] = v.to(device=device, dtype=dtype) if k == "input_features" \
                else v.to(device=device)
    with torch.inference_mode():
        gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                             do_sample=False,
                             pad_token_id=tokenizer.pad_token_id,
                             eos_token_id=tokenizer.eos_token_id)
    new = gen[0, inputs["input_ids"].shape[1]:]
    return clean(tokenizer.decode(new, skip_special_tokens=True))


def clean(t: str) -> str:
    import re
    t = re.sub(r"<think>.*?</think>", "", t or "", flags=re.DOTALL)
    for tok in ["<|im_end|>", "<|endoftext|>", "<think>", "</think>"]:
        t = t.replace(tok, "")
    t = t.strip().split("\n")[0].strip().strip('"')
    for p in ["Translation:", "Answer:", "译文：", "译文:"]:
        if t.startswith(p):
            t = t[len(p):].strip()
    return " ".join(t.split())


def lcp(a: list[str], b: list[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x.lower() == y.lower():
            n += 1
        else:
            break
    return n


if __name__ == "__main__":
    main()
