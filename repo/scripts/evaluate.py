#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import yaml

from slidesst.data.io import read_jsonl
from slidesst.data.schema import ChallengeItem, ModelOutput
from slidesst.eval.metrics import evaluate_hard_cases


def load_aliases(path: str | None) -> dict[str, list[str]]:
    if not path:
        return {}
    alias_path = Path(path)
    if not alias_path.exists():
        return {}
    data = yaml.safe_load(alias_path.read_text(encoding="utf-8")) or {}
    return {str(k): [str(x) for x in v] for k, v in data.items()}


def output_path(run_dir: Path, mismatch: str, condition: str) -> Path:
    nested = run_dir / mismatch / condition / "outputs.jsonl"
    if nested.exists():
        return nested
    if mismatch == "matched":
        return run_dir / condition / "outputs.jsonl"
    return nested


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    items = read_jsonl(cfg["paths"]["challenge_jsonl"], ChallengeItem)
    aliases = load_aliases(cfg.get("evaluation", {}).get("output_aliases_path"))
    rows = []
    robustness_rows = []
    run_dir = Path(cfg["paths"]["run_dir"])
    mismatch_modes = cfg.get("evaluation", {}).get("mismatch_modes", ["matched"])
    for mismatch in mismatch_modes:
        for condition in cfg["conditions"]:
            out_path = output_path(run_dir, mismatch, condition)
            if not out_path.exists():
                continue
            outputs = read_jsonl(out_path, ModelOutput)
            m = evaluate_hard_cases(items, outputs, aliases=aliases)
            row = {
                "mismatch": mismatch,
                "condition": condition,
                "bleu": m.bleu,
                "hda": m.hda,
                "term_precision": m.term_precision,
                "term_recall": m.term_recall,
                "term_f1": m.term_f1,
                "context_overuse_rate": m.context_overuse_rate,
                "wrong_slide_adoption_rate": m.wrong_slide_adoption_rate,
                "avg_wall_time_sec": m.avg_wall_time_sec,
                "n": m.n,
            }
            rows.append(row)
            if mismatch != "matched":
                robustness_rows.append(row)
    table_dir = Path(cfg["paths"]["table_dir"])
    table_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row for row in rows if row["mismatch"] == "matched"]).to_csv(table_dir / "main_results.csv", index=False)
    pd.DataFrame(robustness_rows).to_csv(table_dir / "robustness.csv", index=False)
    print(f"Wrote {table_dir / 'main_results.csv'}")
    print(f"Wrote {table_dir / 'robustness.csv'}")


if __name__ == "__main__":
    main()
