#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import yaml


def dataframe_to_latex(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = [
        "\\begin{tabular}{" + "l" * len(columns) + "}",
        "\\toprule",
        " & ".join(_latex_escape(str(col)) for col in columns) + " \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        cells = [_format_cell(row[col]) for col in columns]
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def _format_cell(value) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return _latex_escape(str(value))


def _latex_escape(text: str) -> str:
    return text.replace("_", "\\_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    table_dir = Path(cfg["paths"]["table_dir"])
    for name in ("main_results", "robustness"):
        csv_path = table_dir / f"{name}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        tex = dataframe_to_latex(df)
        (table_dir / f"{name}.tex").write_text(tex, encoding="utf-8")
        print(f"Wrote {table_dir / f'{name}.tex'}")


if __name__ == "__main__":
    main()
