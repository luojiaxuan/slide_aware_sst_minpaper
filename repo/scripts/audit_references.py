#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from slidesst.data.io import read_jsonl
from slidesst.data.reference_audit import (
    audit_reference_items,
    summarize_reference_audits,
    write_reference_audit_csv,
    write_reference_audit_summary,
)
from slidesst.data.schema import ChallengeItem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", default=None)
    args = parser.parse_args()

    items = read_jsonl(args.input, ChallengeItem)
    audits = audit_reference_items(items)
    write_reference_audit_csv(args.output_csv, items, audits)
    summary = (
        write_reference_audit_summary(args.summary_json, audits)
        if args.summary_json
        else summarize_reference_audits(audits)
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
