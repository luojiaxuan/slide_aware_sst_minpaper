#!/usr/bin/env python
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_FILES = {
    "data/challenge_verified_qwen3_vl_context.jsonl.gz": "outputs/chinese_lips_train/data/challenge_verified_qwen3_vl_context.jsonl",
    "index/evidence_qwen3_vl_context.jsonl.gz": "outputs/chinese_lips_train/index/evidence_qwen3_vl_context.jsonl",
    "annotation/diagnostic_sample_500_qwen3_vl_context.jsonl.gz": "outputs/chinese_lips_train/annotation/diagnostic_sample_500_qwen3_vl_context.jsonl",
    "annotation/diagnostic_sample_500_qwen3_vl_context.csv": "outputs/chinese_lips_train/annotation/diagnostic_sample_500_qwen3_vl_context.csv",
    "annotation/diagnostic_sample_500_qwen3_vl_context.stats.json": "outputs/chinese_lips_train/annotation/diagnostic_sample_500_qwen3_vl_context.stats.json",
    "qa/qwen3_vl_context_qa.json": "outputs/chinese_lips_train/qa/qwen3_vl_context_qa.json",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Package the Qwen3-VL Chinese-LiPS derived dataset for HF upload.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hf-repo-id", required=True)
    parser.add_argument("--source-git-commit", required=True)
    parser.add_argument("--upstream-revision", required=True)
    parser.add_argument("--variant", default="qwen3_vl_context_v1")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = package_bundle(
        repo_root=repo_root,
        output_dir=output_dir,
        hf_repo_id=args.hf_repo_id,
        source_git_commit=args.source_git_commit,
        upstream_revision=args.upstream_revision,
        variant=args.variant,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


def package_bundle(
    *,
    repo_root: Path,
    output_dir: Path,
    hf_repo_id: str,
    source_git_commit: str,
    upstream_revision: str,
    variant: str,
) -> dict[str, Any]:
    file_records = []
    for relative_output, relative_input in DEFAULT_FILES.items():
        source = repo_root / relative_input
        destination = output_dir / relative_output
        destination.parent.mkdir(parents=True, exist_ok=True)
        if relative_output.endswith(".gz"):
            gzip_copy(source, destination)
        else:
            shutil.copyfile(source, destination)
        file_records.append(record_file(destination, output_root=output_dir, source=relative_input))

    qa_path = repo_root / DEFAULT_FILES["qa/qwen3_vl_context_qa.json"]
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hf_repo_id": hf_repo_id,
        "variant": variant,
        "source_git_repo": "https://github.com/luojiaxuan/slide_aware_sst_minpaper",
        "source_git_commit": source_git_commit,
        "upstream_dataset": "BAAI/Chinese-LiPS",
        "upstream_revision": upstream_revision,
        "upstream_license": "cc-by-nc-sa-4.0",
        "access_note": (
            "Derived from gated Chinese-LiPS data. Upload only to a private or otherwise access-controlled "
            "Hugging Face dataset repo unless the upstream maintainers explicitly permit wider redistribution."
        ),
        "qa_summary": {
            "rows": qa.get("rows"),
            "unique_ids": qa.get("unique_ids"),
            "missing_raw_output": qa.get("missing_raw_output"),
            "raw_parse_failures": qa.get("raw_parse_failures"),
            "evidence_rows": qa.get("evidence", {}).get("rows"),
        },
        "files": file_records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(render_readme(manifest), encoding="utf-8")
    return manifest


def gzip_copy(source: Path, destination: Path) -> None:
    with source.open("rb") as src, destination.open("wb") as raw_dst:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_dst, mtime=0) as dst:
            shutil.copyfileobj(src, dst)


def record_file(path: Path, *, output_root: Path, source: str) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(output_root)),
        "source": source,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "rows": count_rows(path),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_rows(path: Path) -> int | None:
    if path.suffix == ".gz":
        opener = gzip.open
    else:
        opener = open
    if not (path.name.endswith(".jsonl") or path.name.endswith(".jsonl.gz") or path.suffix == ".csv"):
        return None
    with opener(path, "rt", encoding="utf-8", newline="") as f:
        rows = sum(1 for _ in f)
    if path.suffix == ".csv":
        return max(0, rows - 1)
    return rows


def render_readme(manifest: dict[str, Any]) -> str:
    qa = manifest["qa_summary"]
    files = "\n".join(
        f"| `{record['path']}` | `{record['source']}` | {record['rows']} | `{record['sha256']}` |"
        for record in manifest["files"]
    )
    return f"""---
license: cc-by-nc-sa-4.0
language:
- zh
task_categories:
- automatic-speech-recognition
- translation
pretty_name: Slide Context SST Chinese-LiPS Qwen3-VL Context
tags:
- simultaneous-translation
- speech-translation
- visual-context
- qwen3-vl
---

# Slide Context SST Chinese-LiPS Qwen3-VL Context

This dataset bundle contains derived slide/context artifacts for the
`slide_aware_sst_minpaper` project. It is derived from `BAAI/Chinese-LiPS` and
Qwen3-VL slide-frame enrichment.

## Access and Redistribution

The upstream dataset is gated and licensed as `cc-by-nc-sa-4.0`. Its access
terms restrict redistribution of derived works outside the research group unless
the upstream maintainers grant permission. Keep this Hugging Face repo private
or otherwise access-controlled unless that permission is obtained.

## Provenance

- Source Git repo: {manifest['source_git_repo']}
- Source Git commit: `{manifest['source_git_commit']}`
- Upstream dataset: `BAAI/Chinese-LiPS`
- Upstream revision: `{manifest['upstream_revision']}`
- Variant: `{manifest['variant']}`

## QA Summary

| Metric | Value |
| --- | ---: |
| Challenge rows | {qa['rows']} |
| Unique ids | {qa['unique_ids']} |
| Missing raw model outputs | {qa['missing_raw_output']} |
| Raw parse failures | {qa['raw_parse_failures']} |
| Evidence rows | {qa['evidence_rows']} |

## Files

| File | Source path | Rows | SHA-256 |
| --- | --- | ---: | --- |
{files}

See `manifest.json` for machine-readable provenance and checksums.
"""


if __name__ == "__main__":
    main()
