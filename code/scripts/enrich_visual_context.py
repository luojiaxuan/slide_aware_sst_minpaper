#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from PIL import Image
from tqdm import tqdm

from slidesst.data.io import read_jsonl
from slidesst.data.schema import ChallengeItem, VisualContext


DEFAULT_PROMPT = """You are extracting slide context for simultaneous speech translation.
Inspect the slide image and return one compact JSON object with this schema:
{
  "ocr_text": ["visible slide text, title, formula, named entity, or term"],
  "scene_summary": "one short sentence describing the slide topic or visual content",
  "objects": ["salient visual objects or chart elements"],
  "actions": ["visible process/action if any"],
  "spatial_relations": ["important layout relation if any"]
}
Rules:
- Preserve visible Chinese text exactly when legible.
- Prefer short terms over full paragraphs.
- Each array must contain plain strings only.
- Do not output bounding boxes, coordinates, positions, or nested objects.
- Do not invent facts that are not visible.
- Keep at most 12 OCR text items and at most 8 items in each other array.
- Use [] for uncertain fields.
- Return JSON only. Do not wrap the JSON in Markdown fences.
"""


class SlideContextExtractor(Protocol):
    def extract(self, frame_path: str) -> dict[str, Any]: ...
    def extract_batch(self, frame_paths: list[str]) -> list[dict[str, Any]]: ...
    def prepare_batch(self, frame_paths: list[str]) -> "PreparedSlideBatch": ...
    def extract_prepared(self, prepared: "PreparedSlideBatch") -> list[dict[str, Any]]: ...


@dataclass
class PreparedSlideBatch:
    frame_paths: list[str]
    inputs: Any


@dataclass
class MockSlideContextExtractor:
    text_prefix: str = "mock visible term"

    def extract(self, frame_path: str) -> dict[str, Any]:
        frame_id = Path(frame_path).stem
        return {
            "ocr_text": [f"{self.text_prefix}: {frame_id}"],
            "scene_summary": f"Mock slide context for {frame_id}.",
            "objects": [],
            "actions": [],
            "spatial_relations": [],
        }

    def extract_batch(self, frame_paths: list[str]) -> list[dict[str, Any]]:
        return self.extract_prepared(self.prepare_batch(frame_paths))

    def prepare_batch(self, frame_paths: list[str]) -> PreparedSlideBatch:
        return PreparedSlideBatch(frame_paths=list(frame_paths), inputs=None)

    def extract_prepared(self, prepared: PreparedSlideBatch) -> list[dict[str, Any]]:
        return [self.extract(frame_path) for frame_path in prepared.frame_paths]


class QwenVLSlideContextExtractor:
    def __init__(
        self,
        *,
        model_id: str,
        cache_dir: str | None,
        device: str,
        dtype: str,
        max_new_tokens: int,
        prompt: str,
    ) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        dtype_value = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            dtype=dtype_value,
            trust_remote_code=True,
        )
        self.model.eval()
        if device != "auto":
            self.model.to(device)
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.prompt = prompt
        self._processor_lock = threading.Lock()

    def extract(self, frame_path: str) -> dict[str, Any]:
        return self.extract_batch([frame_path])[0]

    def extract_batch(self, frame_paths: list[str]) -> list[dict[str, Any]]:
        return self.extract_prepared(self.prepare_batch(frame_paths))

    def prepare_batch(self, frame_paths: list[str]) -> PreparedSlideBatch:
        images = [Image.open(frame_path).convert("RGB") for frame_path in frame_paths]
        try:
            texts = []
            with self._processor_lock:
                for image in images:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": self.prompt},
                            ],
                        }
                    ]
                    texts.append(
                        self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    )
                inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        finally:
            for image in images:
                image.close()
        return PreparedSlideBatch(frame_paths=list(frame_paths), inputs=inputs)

    def extract_prepared(self, prepared: PreparedSlideBatch) -> list[dict[str, Any]]:
        inputs = prepared.inputs
        if self.device != "auto":
            inputs = inputs.to(self.device)
        output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
        with self._processor_lock:
            decoded = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return [parse_context_json(text) for text in decoded]


@dataclass
class EnrichmentTask:
    entries: list[tuple[ChallengeItem, str]] | None = None
    passthrough: ChallengeItem | None = None

    @property
    def is_enrichment(self) -> bool:
        return self.entries is not None


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich visual contexts from slide frame images.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--provider", choices=["mock", "qwen_vl"], default="mock")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only-missing", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prefetch-batches", type=int, choices=[0, 1], default=0)
    parser.add_argument("--max-ocr-terms", type=int, default=24)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    prompt = Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else DEFAULT_PROMPT
    extractor = build_extractor(args, prompt)
    items = read_jsonl(args.input, ChallengeItem)
    if args.offset:
        items = items[args.offset :]
    if args.limit:
        items = items[: args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = read_done_ids(output_path) if args.resume else set()
    mode = "a" if args.resume and output_path.exists() else "w"

    tasks, skipped = build_tasks(items, done_ids, args)

    with output_path.open(mode, encoding="utf-8") as out:
        processed = process_tasks(tasks, extractor, out, args)
    print(f"Wrote {processed} enriched items to {output_path}; skipped {skipped}")


def build_tasks(
    items: list[ChallengeItem],
    done_ids: set[str],
    args: argparse.Namespace,
) -> tuple[list[EnrichmentTask], int]:
    tasks: list[EnrichmentTask] = []
    pending: list[tuple[ChallengeItem, str]] = []
    skipped = 0

    def flush_pending() -> None:
        if pending:
            tasks.append(EnrichmentTask(entries=list(pending)))
            pending.clear()

    for item in items:
        if item.id in done_ids:
            skipped += 1
            continue
        if args.only_missing and has_context(item):
            flush_pending()
            tasks.append(EnrichmentTask(passthrough=item))
            skipped += 1
            continue
        frame_path = first_frame(item)
        if frame_path is None:
            flush_pending()
            tasks.append(EnrichmentTask(passthrough=item))
            skipped += 1
            continue
        pending.append((item, frame_path))
        if len(pending) >= args.batch_size:
            flush_pending()
    flush_pending()
    return tasks, skipped


def process_tasks(
    tasks: list[EnrichmentTask],
    extractor: SlideContextExtractor,
    out: Any,
    args: argparse.Namespace,
) -> int:
    if args.prefetch_batches == 1:
        return process_tasks_with_prefetch(tasks, extractor, out, args)

    processed = 0
    with tqdm(total=sum(task_size(task) for task in tasks), desc="Enriching slide context") as progress:
        for task in tasks:
            processed += process_one_task(task, extractor, out, args)
            progress.update(task_size(task))
    return processed


def process_tasks_with_prefetch(
    tasks: list[EnrichmentTask],
    extractor: SlideContextExtractor,
    out: Any,
    args: argparse.Namespace,
) -> int:
    processed = 0
    future: Future[PreparedSlideBatch] | None = None
    future_index: int | None = None

    def enrichment_index_after(start: int) -> int | None:
        for idx in range(start, len(tasks)):
            if tasks[idx].is_enrichment:
                return idx
        return None

    def submit_prepare(executor: ThreadPoolExecutor, idx: int) -> Future[PreparedSlideBatch]:
        entries = require_enrichment_entries(tasks[idx])
        return executor.submit(extractor.prepare_batch, [frame_path for _, frame_path in entries])

    with ThreadPoolExecutor(max_workers=1) as executor:
        first_idx = enrichment_index_after(0)
        if first_idx is not None:
            future = submit_prepare(executor, first_idx)
            future_index = first_idx

        with tqdm(total=sum(task_size(task) for task in tasks), desc="Enriching slide context") as progress:
            for idx, task in enumerate(tasks):
                if not task.is_enrichment:
                    write_item(out, require_passthrough_item(task))
                    progress.update(1)
                    continue

                entries = require_enrichment_entries(task)
                if future is not None and future_index == idx:
                    prepared = future.result()
                    future = None
                    future_index = None
                else:
                    prepared = extractor.prepare_batch([frame_path for _, frame_path in entries])

                next_idx = enrichment_index_after(idx + 1)
                if next_idx is not None:
                    future = submit_prepare(executor, next_idx)
                    future_index = next_idx

                contexts = extractor.extract_prepared(prepared)
                write_enriched_batch(out, entries, contexts, args)
                processed += len(entries)
                progress.update(len(entries))
    return processed


def process_one_task(
    task: EnrichmentTask,
    extractor: SlideContextExtractor,
    out: Any,
    args: argparse.Namespace,
) -> int:
    if not task.is_enrichment:
        write_item(out, require_passthrough_item(task))
        return 0
    entries = require_enrichment_entries(task)
    frame_paths = [frame_path for _, frame_path in entries]
    contexts = extractor.extract_batch(frame_paths)
    write_enriched_batch(out, entries, contexts, args)
    return len(entries)


def write_enriched_batch(
    out: Any,
    entries: list[tuple[ChallengeItem, str]],
    contexts: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    actual_batch_size = len(entries)
    for (item, frame_path), context in zip(entries, contexts):
        apply_context(item, context, frame_path, args.provider, args.model_id, args.max_ocr_terms, actual_batch_size)
        write_item(out, item)


def task_size(task: EnrichmentTask) -> int:
    if task.entries is not None:
        return len(task.entries)
    return 1


def require_enrichment_entries(task: EnrichmentTask) -> list[tuple[ChallengeItem, str]]:
    if task.entries is None:
        raise ValueError("Expected an enrichment task")
    return task.entries


def require_passthrough_item(task: EnrichmentTask) -> ChallengeItem:
    if task.passthrough is None:
        raise ValueError("Expected a passthrough task")
    return task.passthrough


def build_extractor(args: argparse.Namespace, prompt: str) -> SlideContextExtractor:
    if args.provider == "mock":
        return MockSlideContextExtractor()
    return QwenVLSlideContextExtractor(
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        prompt=prompt,
    )


def parse_context_json(text: str) -> dict[str, Any]:
    payload = _extract_json_object(text)
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        parsed = {
            "ocr_text": [],
            "scene_summary": "",
            "objects": [],
            "actions": [],
            "spatial_relations": [],
        }
    return {
        "ocr_text": _as_str_list(parsed.get("ocr_text")),
        "scene_summary": str(parsed.get("scene_summary") or "").strip(),
        "objects": _as_str_list(parsed.get("objects")),
        "actions": _as_str_list(parsed.get("actions")),
        "spatial_relations": _as_str_list(parsed.get("spatial_relations")),
        "raw_output": text,
    }


def apply_context(
    item: ChallengeItem,
    context: dict[str, Any],
    frame_path: str,
    provider: str,
    model_id: str,
    max_ocr_terms: int,
    batch_size: int | None = None,
) -> None:
    visual = item.visual_context or VisualContext(video_id=item.lecture_id, clip_id=item.id)
    visual.ocr_text = _merge_terms(visual.ocr_text, context.get("ocr_text", []), max_ocr_terms)
    if context.get("scene_summary"):
        visual.scene_summary = str(context["scene_summary"])
    visual.objects = _merge_terms(visual.objects, context.get("objects", []), max_ocr_terms)
    visual.actions = _merge_terms(visual.actions, context.get("actions", []), max_ocr_terms)
    visual.spatial_relations = _merge_terms(
        visual.spatial_relations, context.get("spatial_relations", []), max_ocr_terms
    )
    frame_id = Path(frame_path).stem
    if frame_id not in visual.frame_ids:
        visual.frame_ids.append(frame_id)
    visual.metadata = {
        **visual.metadata,
        "context_enrichment": {
            "provider": provider,
            "model_id": model_id,
            "frame_path": frame_path,
            "batch_size": batch_size,
            "raw_output": context.get("raw_output", ""),
        },
    }
    item.visual_context = visual


def has_context(item: ChallengeItem) -> bool:
    visual = item.visual_context
    return bool(visual and (visual.ocr_text or visual.scene_summary or visual.objects or visual.actions))


def first_frame(item: ChallengeItem) -> str | None:
    if not item.video or not item.video.frame_paths:
        return None
    return item.video.frame_paths[0]


def read_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def write_item(handle: Any, item: ChallengeItem) -> None:
    handle.write(json.dumps(item.model_dump(mode="json"), ensure_ascii=False) + "\n")
    handle.flush()


def _extract_json_object(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = re.split(r"[;\n|]", value)
    elif isinstance(value, list):
        values = value
    else:
        values = [value]
    out: list[str] = []
    for item in values:
        text = _stringify_term(item)
        if text:
            out.append(text)
    return out


def _merge_terms(existing: list[str], new: Any, limit: int) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for term in [*existing, *_as_str_list(new)]:
        norm = re.sub(r"\s+", " ", term).strip()
        key = norm.lower()
        if not norm or key in seen:
            continue
        seen.add(key)
        merged.append(norm)
        if len(merged) >= limit:
            break
    return merged


def _stringify_term(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("text", "label", "description", "name", "value", "type"):
            if value.get(key):
                return str(value[key]).strip()
        return ""
    return str(value).strip()


if __name__ == "__main__":
    main()
