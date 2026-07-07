#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge repaired visual-context rows into a challenge JSONL.")
    parser.add_argument("--base", required=True, help="Base challenge JSONL to preserve row order from.")
    parser.add_argument("--repair", action="append", required=True, help="Repair JSONL. Later files override earlier ones.")
    parser.add_argument("--output", required=True, help="Merged challenge JSONL output path.")
    parser.add_argument("--expected-rows", type=int, default=None)
    parser.add_argument("--expected-replacements", type=int, default=None)
    parser.add_argument("--log-json", default=None)
    args = parser.parse_args()

    result = merge_repairs(
        base=Path(args.base),
        repair_paths=[Path(path) for path in args.repair],
        output=Path(args.output),
        expected_rows=args.expected_rows,
        expected_replacements=args.expected_replacements,
    )
    if args.log_json:
        output_path = Path(args.log_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


def merge_repairs(
    *,
    base: Path,
    repair_paths: list[Path],
    output: Path,
    expected_rows: int | None = None,
    expected_replacements: int | None = None,
) -> dict[str, Any]:
    repaired: dict[str, dict[str, Any]] = {}
    source_counts: dict[str, int] = {}
    duplicate_repair_ids: set[str] = set()
    overridden_ids: set[str] = set()

    for path in repair_paths:
        count = 0
        seen_in_file: set[str] = set()
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                item_id = row.get("id")
                if not item_id:
                    raise ValueError(f"{path}:{line_no} has no id")
                if item_id in seen_in_file:
                    duplicate_repair_ids.add(str(item_id))
                if item_id in repaired:
                    overridden_ids.add(str(item_id))
                seen_in_file.add(str(item_id))
                repaired[str(item_id)] = row
                count += 1
        source_counts[str(path)] = count

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_suffix(output.suffix + ".tmp")
    total_rows = 0
    replaced_rows = 0
    duplicate_base_ids: set[str] = set()
    base_ids: set[str] = set()
    with base.open(encoding="utf-8") as src, tmp_output.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            item_id = row.get("id")
            if not item_id:
                raise ValueError(f"{base}:{line_no} has no id")
            item_id = str(item_id)
            if item_id in base_ids:
                duplicate_base_ids.add(item_id)
            base_ids.add(item_id)
            if item_id in repaired:
                row = repaired[item_id]
                replaced_rows += 1
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            total_rows += 1

    unused_repair_ids = sorted(set(repaired) - base_ids)
    if expected_rows is not None and total_rows != expected_rows:
        tmp_output.unlink(missing_ok=True)
        raise ValueError(f"expected {expected_rows} rows, got {total_rows}")
    if expected_replacements is not None and replaced_rows != expected_replacements:
        tmp_output.unlink(missing_ok=True)
        raise ValueError(f"expected {expected_replacements} replacements, got {replaced_rows}")
    if duplicate_base_ids or duplicate_repair_ids or unused_repair_ids:
        tmp_output.unlink(missing_ok=True)
        raise ValueError(
            "merge validation failed: "
            f"duplicate_base_ids={sorted(duplicate_base_ids)[:10]}, "
            f"duplicate_repair_ids={sorted(duplicate_repair_ids)[:10]}, "
            f"unused_repair_ids={unused_repair_ids[:10]}"
        )

    tmp_output.replace(output)
    return {
        "base": str(base),
        "output": str(output),
        "repair_sources": source_counts,
        "repair_unique_ids": len(repaired),
        "repair_overridden_ids": len(overridden_ids),
        "rows": total_rows,
        "replaced_rows": replaced_rows,
    }


if __name__ == "__main__":
    main()
