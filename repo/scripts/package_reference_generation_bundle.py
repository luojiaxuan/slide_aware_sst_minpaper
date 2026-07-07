#!/usr/bin/env python
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--hf-repo", required=True)
    parser.add_argument("--source-hf-revision", required=True)
    parser.add_argument("--source-hf-tag", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-version", required=True)
    parser.add_argument("--generation-input", required=True)
    parser.add_argument("--generation-limit", type=int, required=True)
    parser.add_argument("--selected-batch-size", type=int, required=True)
    parser.add_argument("--refs-jsonl", default="pilot_100_refs_repaired.jsonl")
    parser.add_argument("--audit-csv", default="pilot_100_reference_audit_repaired.csv")
    parser.add_argument("--audit-summary-json", default="pilot_100_reference_audit_summary_repaired.json")
    parser.add_argument("--hf-subdir", default="reference_generation/qwen3_32b_hf_revision_a837704")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    hf_subdir = Path(args.hf_subdir)
    data_dir = output_dir / hf_subdir
    data_dir.mkdir(parents=True, exist_ok=True)

    bundled_files = []
    refs_src = source_dir / args.refs_jsonl
    refs_dst = data_dir / f"{args.refs_jsonl}.gz"
    with refs_src.open("rb") as src, gzip.open(refs_dst, "wb") as dst:
        shutil.copyfileobj(src, dst)
    bundled_files.append(refs_dst)

    for name in (args.audit_csv, args.audit_summary_json):
        dst = data_dir / name
        shutil.copy2(source_dir / name, dst)
        bundled_files.append(dst)

    manifest = {
        "artifact": args.artifact,
        "hf_repo": args.hf_repo,
        "source_hf_revision": args.source_hf_revision,
        "source_hf_tag": args.source_hf_tag,
        "model": args.model,
        "prompt_version": args.prompt_version,
        "generation_input": args.generation_input,
        "generation_limit": args.generation_limit,
        "selected_batch_size": args.selected_batch_size,
        "audit_summary": json.loads((source_dir / args.audit_summary_json).read_text(encoding="utf-8")),
        "files": [_file_record(path, output_dir) for path in bundled_files],
    }
    manifest_path = output_dir / f"manifest_{args.artifact}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme_path = output_dir / f"README_{args.artifact}.md"
    readme_path.write_text(_readme(args, manifest), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "manifest": str(manifest_path)}, ensure_ascii=False, indent=2))


def _file_record(path: Path, root: Path) -> dict:
    data = path.read_bytes()
    rows = None
    if path.name.endswith(".jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            rows = sum(1 for line in f if line.strip())
    elif path.suffix == ".csv":
        rows = max(0, sum(1 for _ in path.open("r", encoding="utf-8")) - 1)
    return {
        "path": str(path.relative_to(root)),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "rows": rows,
    }


def _readme(args: argparse.Namespace, manifest: dict) -> str:
    summary = manifest["audit_summary"]
    return f"""# {args.artifact}

Private pseudo-reference artifact derived from the private Chinese-LiPS Qwen3-VL context bundle.

- Source HF revision: {args.source_hf_revision}
- Source HF tag: {args.source_hf_tag}
- Model: {args.model}
- Rows: {args.generation_limit}
- Selected batch size: {args.selected_batch_size}
- Audit summary: {json.dumps(summary, ensure_ascii=False)}
- Access: keep private because upstream Chinese-LiPS is gated.
"""


if __name__ == "__main__":
    main()
