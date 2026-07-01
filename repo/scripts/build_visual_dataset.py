#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from slidesst.data.adapters.vasr import load_vasr_manifest
from slidesst.data.annotation_io import export_review_sheet
from slidesst.data.io import read_jsonl, write_jsonl
from slidesst.data.schema import ChallengeItem, HardLabel, StreamingUnit


DEICTIC_CUES = ("这个", "这里", "那里", "左边", "右边", "上面", "下面", "这条线", "这个模块")
ACTION_CUES = {
    "拧紧": ("action", ["tighten"], ["loosen"]),
    "打开": ("action", ["open"], ["close"]),
    "关闭": ("action", ["close"], ["open"]),
    "移动": ("action", ["move"], ["stop"]),
    "连接": ("action", ["connect"], ["disconnect"]),
    "对齐": ("action", ["align"], ["misalign"]),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--adapter", default="vasr")
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--mine-hard-labels", action="store_true")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text()) if args.config else {}
    paths = cfg.get("paths", {})
    manifest = args.manifest or paths.get("visual_manifest")
    out = Path(args.out or paths.get("candidate_jsonl", "outputs/visual/data/candidates.jsonl"))
    if manifest and Path(manifest).expanduser().exists():
        items = load_vasr_manifest(manifest)
    else:
        toy_path = Path(cfg.get("dataset", {}).get("toy_challenge_path", "examples/toy_visual_challenge.jsonl"))
        if not toy_path.is_absolute():
            toy_path = Path.cwd() / toy_path
        items = read_jsonl(toy_path, ChallengeItem)

    for item in items:
        item.streaming_units = item.streaming_units or make_streaming_units(item.source_transcript)
        if args.mine_hard_labels:
            item.hard_labels = item.hard_labels or mine_visual_hard_labels(item)
        if item.video and item.video.fps is None:
            item.video.fps = args.sample_fps

    write_jsonl(out, items)
    annotation_path = Path(paths.get("annotation_csv", out.parent.parent / "annotation" / "review_sheet.csv"))
    export_review_sheet(annotation_path, items)
    print(f"Wrote {len(items)} visual candidate items to {out}")
    print(f"Wrote review sheet to {annotation_path}")


def make_streaming_units(transcript: str, chars_per_step: int = 12) -> list[StreamingUnit]:
    return [
        StreamingUnit(t=float(idx + 1), partial_transcript=transcript[:end])
        for idx, end in enumerate(range(chars_per_step, len(transcript) + chars_per_step, chars_per_step))
    ]


def mine_visual_hard_labels(item: ChallengeItem) -> list[HardLabel]:
    labels: list[HardLabel] = []
    transcript = item.source_transcript
    if any(cue in transcript for cue in DEICTIC_CUES):
        labels.append(
            HardLabel(
                label_id=f"{item.id}_deixis_0",
                label_type="visual_deixis",
                source_span=next(cue for cue in DEICTIC_CUES if cue in transcript),
                gold_en=[],
                requires_visual=True,
                notes="Needs human annotation.",
            )
        )
    for cue, (label_type, gold, distractors) in ACTION_CUES.items():
        if cue in transcript:
            labels.append(
                HardLabel(
                    label_id=f"{item.id}_{label_type}_{len(labels)}",
                    label_type=label_type,
                    source_span=cue,
                    gold_en=gold,
                    distractor_en=distractors,
                    requires_visual=True,
                )
            )
    if item.visual_context and item.visual_context.ocr_text:
        labels.append(
            HardLabel(
                label_id=f"{item.id}_ocr_{len(labels)}",
                label_type="onscreen_text",
                source_span="; ".join(item.visual_context.ocr_text[:2]),
                gold_en=item.visual_context.ocr_text[:2],
                requires_visual=True,
                requires_ocr=True,
            )
        )
    return labels


if __name__ == "__main__":
    main()
