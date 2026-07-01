#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from slidesst.data.adapters import load_challenge_items_from_config
from slidesst.data.mining import build_verified_items, copy_items, mine_hard_examples, write_annotation_csv

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["toy", "mine", "build"], default="toy")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--annotations", default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    paths = cfg["paths"]
    candidate_path = Path(paths.get("candidate_jsonl", "outputs/data/challenge_candidates.jsonl"))
    annotation_path = Path(args.annotations or paths.get("annotation_csv", "outputs/annotations/pilot_candidates.csv"))
    challenge_path = Path(paths["challenge_jsonl"])

    if args.stage == "toy":
        items = load_challenge_items_from_config(cfg)
        copy_items(challenge_path, items)
        write_annotation_csv(annotation_path, items)
        print(f"Wrote {len(items)} toy/fallback items to {challenge_path}")
        print(f"Wrote annotation CSV to {annotation_path}")
        return

    if args.stage == "mine":
        items = load_challenge_items_from_config(cfg)
        mined = mine_hard_examples(items, limit=args.limit)
        copy_items(candidate_path, mined)
        write_annotation_csv(annotation_path, mined)
        print(f"Wrote {len(mined)} candidate items to {candidate_path}")
        print(f"Wrote annotation CSV to {annotation_path}")
        return

    verified = build_verified_items(candidate_path, annotation_path)
    copy_items(challenge_path, verified)
    print(f"Wrote {len(verified)} verified items to {challenge_path}")


if __name__ == "__main__":
    main()
