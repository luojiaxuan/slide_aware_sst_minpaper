#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import yaml

from slidesst.data.io import read_jsonl
from slidesst.data.schema import ChallengeItem, ModelOutput


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vision_zh_en.yaml")
    parser.add_argument("--mismatch", default="matched")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    items = read_jsonl(cfg["paths"]["challenge_jsonl"], ChallengeItem)
    run_dir = Path(cfg["paths"]["run_dir"]) / args.mismatch
    outputs = {
        condition: _load_outputs(run_dir / condition / "outputs.jsonl")
        for condition in cfg.get("conditions", [])
    }
    cases = []
    for item in items:
        cases.append(
            {
                "id": item.id,
                "source_transcript": item.source_transcript,
                "reference": item.reference.translation or item.reference_translation,
                "visual_context": item.visual_context.model_dump(mode="json") if item.visual_context else None,
                "hard_labels": [label.model_dump(mode="json") for label in item.hard_labels],
                "outputs": {condition: by_id[item.id].model_dump(mode="json") for condition, by_id in outputs.items() if item.id in by_id},
            }
        )
    out = Path(args.output or Path(cfg["paths"]["table_dir"]) / "case_studies.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"Wrote {len(cases)} case studies to {out}")


def _load_outputs(path: Path) -> dict[str, ModelOutput]:
    if not path.exists():
        return {}
    return {out.id: out for out in read_jsonl(path, ModelOutput)}


if __name__ == "__main__":
    main()
