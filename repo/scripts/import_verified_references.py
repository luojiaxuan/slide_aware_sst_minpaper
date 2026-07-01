#!/usr/bin/env python
from __future__ import annotations

import argparse

from slidesst.data.annotation_io import import_review_sheet
from slidesst.data.io import read_jsonl, write_jsonl
from slidesst.data.schema import ChallengeItem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    items = read_jsonl(args.input, ChallengeItem)
    reviewed = import_review_sheet(items, args.review)
    write_jsonl(args.output, reviewed)
    print(f"Wrote {len(reviewed)} verified reference items to {args.output}")


if __name__ == "__main__":
    main()
