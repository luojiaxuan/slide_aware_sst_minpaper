#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vision_zh_en.yaml")
    parser.add_argument("--mismatch", default="matched")
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    conditions = args.conditions or cfg["conditions"]

    subprocess.run([sys.executable, "scripts/build_visual_context_index.py", "--config", args.config, "--mismatch", args.mismatch], check=True)
    for condition in conditions:
        cmd = [sys.executable, "scripts/run_stream_translate.py", "--config", args.config, "--condition", condition, "--mismatch", args.mismatch]
        if args.limit:
            cmd.extend(["--limit", str(args.limit)])
        subprocess.run(cmd, check=True)
    subprocess.run([sys.executable, "scripts/evaluate.py", "--config", args.config], check=True)
    subprocess.run([sys.executable, "scripts/make_paper_tables.py", "--config", args.config], check=True)


if __name__ == "__main__":
    main()
