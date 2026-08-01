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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np


def main() -> None:
    import torch

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
    ap.add_argument("--batch-items", type=int, default=1,
                    help="number of active streaming items generated together")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.batch_items < 1:
        raise ValueError("batch-items must be positive")

    from transformers import (AutoConfig, Qwen3OmniMoeProcessor,
                              Qwen3OmniMoeThinkerForConditionalGeneration)

    torch.manual_seed(args.seed)
    processor = Qwen3OmniMoeProcessor.from_pretrained(
        args.model, revision=args.model_revision, trust_remote_code=True
    )
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
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
            todo = [
                (index, item) for index, item in enumerate(items)
                if (item["id"], cond) not in done
            ]
            for i, rec, wall_s in stream_many(
                todo,
                cond,
                processor,
                tokenizer,
                model,
                args,
                batch_size=args.batch_items,
            ):
                rec["wall_s"] = round(wall_s, 1)
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                done.add((rec["id"], cond))
                print(f"[{cond}] {i+1}/{len(items)} {rec['id']} "
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


@dataclass
class StreamState:
    source_index: int
    item: dict
    condition: str
    audio: np.ndarray
    sample_rate: int
    chunk_samples: int
    n_chunks: int
    image: str | None
    step: int = 0
    committed: list[str] = field(default_factory=list)
    previous: list[str] = field(default_factory=list)
    events: list[tuple[int, int]] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)


def prepare_stream_state(source_index: int, item: dict, cond: str, args) -> StreamState:
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
    return StreamState(
        source_index=source_index,
        item=item,
        condition=cond,
        audio=audio,
        sample_rate=sr,
        chunk_samples=chunk,
        n_chunks=n_chunks,
        image=image,
    )


def stream_many(
    indexed_items: list[tuple[int, dict]],
    cond: str,
    processor,
    tokenizer,
    model,
    args,
    *,
    batch_size: int,
) -> Iterator[tuple[int, dict, float]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    pending = iter(indexed_items)
    active: list[StreamState] = []

    def refill() -> None:
        while len(active) < batch_size:
            try:
                source_index, item = next(pending)
            except StopIteration:
                return
            active.append(prepare_stream_state(source_index, item, cond, args))

    refill()
    while active:
        prefixes = [
            state.audio[: (state.step + 1) * state.chunk_samples]
            for state in active
        ]
        texts = translate_prefix_batch(
            prefixes,
            [state.sample_rate for state in active],
            [state.image for state in active],
            [state.item for state in active],
            [state.item.get("tgt_lang", "English") for state in active],
            processor,
            tokenizer,
            model,
            args,
        )
        completed: list[tuple[int, dict, float]] = []
        remaining: list[StreamState] = []
        for state, text in zip(active, texts, strict=True):
            record = advance_stream_state(state, text, args)
            if record is None:
                remaining.append(state)
            else:
                completed.append(
                    (state.source_index, record, time.time() - state.started_at)
                )
        active = remaining
        refill()
        yield from completed


def advance_stream_state(state: StreamState, text: str, args) -> dict | None:
    state.step += 1
    full = text.split()
    if state.step == state.n_chunks:
        k = len(state.committed)
        if len(full) > k:
            state.committed.extend(full[k:])
            state.events.append((state.step, len(state.committed)))
    else:
        agree = full[:lcp(state.previous, full)]
        if len(agree) > len(state.committed) and \
                [word.lower() for word in agree[:len(state.committed)]] == \
                [word.lower() for word in state.committed]:
            state.committed = agree
            state.events.append((state.step, len(state.committed)))
    state.previous = full
    if state.step < state.n_chunks:
        return None
    return {
        "id": state.item["id"],
        "condition": state.condition,
        "n_chunks": state.n_chunks,
        "chunk_s": args.chunk_s,
        "events": state.events,
        "hypothesis": " ".join(state.committed),
        "reference": state.item.get("reference", ""),
        "image_used": state.image or "",
        "model": args.model,
        "model_revision": args.model_revision,
        "attention": args.attn,
        "max_new_tokens": args.max_new_tokens,
        "batch_items": args.batch_items,
        "seed": args.seed,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
    }


def stream_one(item, cond, processor, tokenizer, model, args) -> dict:
    result = list(
        stream_many(
            [(0, item)],
            cond,
            processor,
            tokenizer,
            model,
            args,
            batch_size=1,
        )
    )
    if len(result) != 1:
        raise RuntimeError("single-item stream produced an invalid result count")
    return result[0][1]


def translate_prefix(audio_prefix, sr, image_path, item, tgt,
                     processor, tokenizer, model, args) -> str:
    return translate_prefix_batch(
        [audio_prefix],
        [sr],
        [image_path],
        [item],
        [tgt],
        processor,
        tokenizer,
        model,
        args,
    )[0]


def translate_prefix_batch(audio_prefixes, sample_rates, image_paths, items, targets,
                           processor, tokenizer, model, args) -> list[str]:
    if not (
        len(audio_prefixes) == len(sample_rates) == len(image_paths)
        == len(items) == len(targets)
    ):
        raise ValueError("batched prefix inputs have different lengths")
    if not audio_prefixes:
        return []
    if len(set(sample_rates)) != 1:
        raise ValueError("all audio in a generation batch must share one sample rate")
    texts = [
        build_prompt(audio_prefix, image_path, item, target, processor)
        for audio_prefix, image_path, item, target in zip(
            audio_prefixes, image_paths, items, targets, strict=True
        )
    ]
    images = [path for path in image_paths if path]
    if images and len(images) != len(image_paths):
        raise ValueError("a generation batch cannot mix image and audio-only conditions")
    inputs = processor(
        text=texts,
        audio=list(audio_prefixes),
        images=images or None,
        videos=None,
        return_tensors="pt",
        padding=True,
        sampling_rate=sample_rates[0],
        use_audio_in_video=False,
    )
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    for key, value in list(inputs.items()):
        if hasattr(value, "to"):
            inputs[key] = value.to(device=device, dtype=dtype) \
                if key == "input_features" else value.to(device=device)
    import torch

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_width = inputs["input_ids"].shape[1]
    if generated.shape[0] != len(audio_prefixes):
        raise RuntimeError("model returned an unexpected generation batch size")
    return [
        clean(tokenizer.decode(row[prompt_width:], skip_special_tokens=True))
        for row in generated
    ]


def build_prompt(audio_prefix, image_path, item, tgt, processor) -> str:
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
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )


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
