#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from slidesst.data.io import read_jsonl, write_jsonl
from slidesst.data.schema import ChallengeItem


SLICE_ORDER = [
    "ocr_support",
    "visual_non_ocr",
    "term_homophone",
    "distractor_risk",
    "latency_critical",
    "no_context",
]

TECHNICAL_CUES = (
    "线程",
    "现成",
    "基音",
    "基因",
    "卷积",
    "卷积核",
    "消息",
    "源域",
    "法向",
    "标识",
    "模型",
    "算法",
    "网络",
    "数据",
    "训练",
    "特征",
    "函数",
    "矩阵",
    "向量",
    "概率",
    "分布",
    "语音",
    "识别",
    "翻译",
    "系统",
)

DEICTIC_CUES = (
    "这个",
    "这里",
    "那里",
    "这张",
    "这页",
    "上面",
    "下面",
    "左边",
    "右边",
    "如图",
    "我们看",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample a human-translation diagnostic subset by slide-context evidence slices."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--stats-json", required=True)
    parser.add_argument("--target-size", type=int, default=500)
    parser.add_argument("--per-slice", type=int, default=100)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--low-ocr-chars", type=int, default=24)
    parser.add_argument("--distractor-ocr-chars", type=int, default=80)
    parser.add_argument("--ocr-overlap-threshold", type=float, default=0.15)
    parser.add_argument("--distractor-overlap-threshold", type=float, default=0.05)
    args = parser.parse_args()

    items = read_jsonl(args.input, ChallengeItem)
    specs = {
        item.id: classify_item(
            item,
            low_ocr_chars=args.low_ocr_chars,
            distractor_ocr_chars=args.distractor_ocr_chars,
            ocr_overlap_threshold=args.ocr_overlap_threshold,
            distractor_overlap_threshold=args.distractor_overlap_threshold,
        )
        for item in items
    }
    selected = sample_items(items, specs, target_size=args.target_size, per_slice=args.per_slice, seed=args.seed)
    for item in selected:
        slices = specs[item.id]["slices"]
        if item.visual_context:
            item.visual_context.metadata = {**item.visual_context.metadata, "diagnostic_slices": slices}

    write_jsonl(args.output_jsonl, selected)
    write_review_csv(args.output_csv, selected, specs)
    write_stats(args.stats_json, items, selected, specs)
    print(f"Wrote {len(selected)} diagnostic items to {args.output_jsonl}")
    print(f"Wrote review CSV to {args.output_csv}")
    print(f"Wrote stats to {args.stats_json}")


def classify_item(
    item: ChallengeItem,
    *,
    low_ocr_chars: int,
    distractor_ocr_chars: int,
    ocr_overlap_threshold: float,
    distractor_overlap_threshold: float,
) -> dict[str, object]:
    transcript = item.source_transcript or ""
    ocr_text = _ocr_text(item)
    ocr_chars = _signal_chars("".join(ocr_text))
    transcript_chars = _signal_chars(transcript)
    overlap = len(ocr_chars & transcript_chars) / max(1, len(ocr_chars))
    ocr_char_count = sum(len(text) for text in ocr_text)
    has_frame = bool(item.video and item.video.frame_paths)
    slices: list[str] = []

    if ocr_text and overlap >= ocr_overlap_threshold:
        slices.append("ocr_support")
    if has_frame and ocr_char_count <= low_ocr_chars:
        slices.append("visual_non_ocr")
    if _has_term_or_homophone_signal(item, transcript):
        slices.append("term_homophone")
    if ocr_char_count >= distractor_ocr_chars and overlap <= distractor_overlap_threshold:
        slices.append("distractor_risk")
    if _has_latency_signal(item, transcript):
        slices.append("latency_critical")
    if not slices:
        slices.append("no_context")

    return {
        "slices": slices,
        "ocr_char_count": ocr_char_count,
        "ocr_transcript_overlap": round(overlap, 4),
        "frame_count": len(item.video.frame_paths) if item.video else 0,
    }


def sample_items(
    items: list[ChallengeItem],
    specs: dict[str, dict[str, object]],
    *,
    target_size: int,
    per_slice: int,
    seed: int,
) -> list[ChallengeItem]:
    rng = random.Random(seed)
    by_slice: dict[str, list[ChallengeItem]] = defaultdict(list)
    for item in items:
        for slice_name in specs[item.id]["slices"]:
            by_slice[str(slice_name)].append(item)
    for bucket in by_slice.values():
        rng.shuffle(bucket)

    selected: list[ChallengeItem] = []
    selected_ids: set[str] = set()

    if per_slice > 0:
        for slice_name in SLICE_ORDER:
            for item in by_slice.get(slice_name, [])[:per_slice]:
                if len(selected) >= target_size:
                    break
                if item.id in selected_ids:
                    continue
                selected.append(item)
                selected_ids.add(item.id)
            if len(selected) >= target_size:
                break

    while len(selected) < target_size:
        added = False
        for slice_name in SLICE_ORDER:
            for item in by_slice.get(slice_name, []):
                if item.id in selected_ids:
                    continue
                selected.append(item)
                selected_ids.add(item.id)
                added = True
                break
            if len(selected) >= target_size:
                break
        if not added:
            break
    return selected


def write_review_csv(path: str | Path, items: list[ChallengeItem], specs: dict[str, dict[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "id",
        "lecture_id",
        "diagnostic_slices",
        "zh_transcript",
        "candidate_en",
        "human_en",
        "reference_status",
        "ocr_char_count",
        "ocr_transcript_overlap",
        "frame_count",
        "visual_summary",
        "ocr_text",
        "reviewer_slice_override",
        "reviewer_notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for item in items:
            spec = specs[item.id]
            visual = item.visual_context
            writer.writerow(
                {
                    "id": item.id,
                    "lecture_id": item.lecture_id,
                    "diagnostic_slices": "|".join(str(value) for value in spec["slices"]),
                    "zh_transcript": item.source_transcript,
                    "candidate_en": item.reference.translation or item.reference_translation or "",
                    "human_en": "",
                    "reference_status": item.reference.status,
                    "ocr_char_count": spec["ocr_char_count"],
                    "ocr_transcript_overlap": spec["ocr_transcript_overlap"],
                    "frame_count": spec["frame_count"],
                    "visual_summary": visual.scene_summary if visual else "",
                    "ocr_text": " | ".join(_ocr_text(item)),
                    "reviewer_slice_override": "",
                    "reviewer_notes": "",
                }
            )


def write_stats(
    path: str | Path,
    items: list[ChallengeItem],
    selected: list[ChallengeItem],
    specs: dict[str, dict[str, object]],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    all_counts: Counter[str] = Counter()
    selected_counts: Counter[str] = Counter()
    for item in items:
        all_counts.update(str(value) for value in specs[item.id]["slices"])
    for item in selected:
        selected_counts.update(str(value) for value in specs[item.id]["slices"])
    path.write_text(
        json.dumps(
            {
                "input_items": len(items),
                "selected_items": len(selected),
                "slice_counts_all": dict(sorted(all_counts.items())),
                "slice_counts_selected": dict(sorted(selected_counts.items())),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _ocr_text(item: ChallengeItem) -> list[str]:
    if not item.visual_context:
        return []
    return [text for text in item.visual_context.ocr_text if text]


def _signal_chars(text: str) -> set[str]:
    return {char.lower() for char in text if char.isalnum() or "\u4e00" <= char <= "\u9fff"}


def _has_term_or_homophone_signal(item: ChallengeItem, transcript: str) -> bool:
    if item.ambiguous_items:
        return True
    if any("term" in label.label_type or "homophone" in label.label_type for label in item.hard_labels):
        return True
    return any(cue in transcript for cue in TECHNICAL_CUES)


def _has_latency_signal(item: ChallengeItem, transcript: str) -> bool:
    if any(cue in transcript for cue in DEICTIC_CUES):
        return True
    return any(label.requires_visual for label in item.hard_labels)


if __name__ == "__main__":
    main()
