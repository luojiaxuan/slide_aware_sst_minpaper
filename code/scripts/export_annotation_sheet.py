#!/usr/bin/env python
from __future__ import annotations

import argparse

from slidesst.data.annotation_io import export_review_sheet
from slidesst.data.io import read_jsonl
from slidesst.data.schema import ChallengeItem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    items = read_jsonl(args.input, ChallengeItem)
    export_review_sheet(args.output, items)
    print(f"Wrote review sheet to {args.output}")


if __name__ == "__main__":
    main()
