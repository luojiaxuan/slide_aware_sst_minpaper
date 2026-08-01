#!/usr/bin/env python3
"""Validate and package the private Chinese-LiPS visual-control diagnostic."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


CONDITIONS = ("none", "slide", "wrong", "cross_talk", "blank")
OPTIONAL_RUNTIME_FILES = (
    "launch_gpu_snapshot.csv",
    "supervisor.log",
    "worker_0.log",
    "worker_1.log",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gzip_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_file, destination.open("xb") as raw_output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_output, mtime=0) as output:
            shutil.copyfileobj(input_file, output)


def file_record(path: Path, output_root: Path, *, rows: int | None = None) -> dict:
    return {
        "path": path.relative_to(output_root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "rows": rows,
    }


def validate_run(run_root: Path) -> tuple[dict, dict, list[tuple[Path, int]]]:
    completion_path = run_root / "completion.json"
    analysis_path = run_root / "analysis_summary_v1.json"
    completion_bytes = completion_path.read_bytes()
    analysis_bytes = analysis_path.read_bytes()
    completion = json.loads(completion_bytes)
    analysis = json.loads(analysis_bytes)
    if completion.get("status") != "COMPLETE":
        raise ValueError("visual-control run is not complete")
    if tuple(completion.get("conditions", [])) != CONDITIONS:
        raise ValueError("completion condition matrix differs from frozen v1")
    if analysis.get("completion_sha256") != hashlib.sha256(completion_bytes).hexdigest():
        raise ValueError("analysis does not bind the completion bytes")

    shard_records: list[tuple[Path, int]] = []
    rows = []
    for output in completion.get("outputs", []):
        shard = Path(output["output"])
        if shard.resolve(strict=True).parent != run_root.resolve(strict=True):
            raise ValueError("completion references a shard outside the run root")
        payload = shard.read_bytes()
        if hashlib.sha256(payload).hexdigest() != output["output_sha256"]:
            raise ValueError(f"shard hash mismatch: {shard.name}")
        shard_rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
        if len(shard_rows) != output["record_count"]:
            raise ValueError(f"shard record count mismatch: {shard.name}")
        if output["record_count"] != output["expected_record_count"]:
            raise ValueError(f"shard did not reach its expected count: {shard.name}")
        rows.extend(shard_rows)
        shard_records.append((shard, len(shard_rows)))
    expected_record_count = analysis.get("record_count")
    if not shard_records or len(rows) != expected_record_count:
        raise ValueError("completion shards do not match the analysis record count")

    keys = [(row["id"], row["condition"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("visual-control shards contain duplicate item/condition keys")
    matrix: dict[str, dict[str, dict]] = {}
    for row in rows:
        matrix.setdefault(row["id"], {})[row["condition"]] = row
    if len(matrix) != analysis.get("item_count"):
        raise ValueError("visual-control item count differs from analysis")
    for item_id, by_condition in matrix.items():
        if set(by_condition) != set(CONDITIONS):
            raise ValueError(f"incomplete visual-control matrix: {item_id}")
        references = {row["reference"] for row in by_condition.values()}
        models = {row["model"] for row in by_condition.values()}
        revisions = {row["model_revision"] for row in by_condition.values()}
        if references != {next(iter(references))} or len(references) != 1:
            raise ValueError(f"reference mismatch across conditions: {item_id}")
        if models != {completion["model"]} or revisions != {completion["model_revision"]}:
            raise ValueError(f"model identity mismatch across conditions: {item_id}")
    return completion, analysis, shard_records


def render_readme(manifest: dict) -> str:
    return f"""---
license: cc-by-nc-sa-4.0
language:
- zh
- en
task_categories:
- automatic-speech-recognition
- translation
tags:
- simultaneous-translation
- audio-visual
- visual-context
- qwen3-omni
---

# Chinese-LiPS Qwen3-Omni Visual-Control Diagnostic

Private single-talk mechanism diagnostic for `slide_aware_sst_minpaper`. This is
not a paper-grade benchmark result and its references are machine drafts.

## Access

The artifact is derived from gated Chinese-LiPS data. Keep this dataset repo
private unless the upstream maintainers explicitly permit redistribution.

## Provenance

- Run Git commit: `{manifest['run_git_commit']}`
- Packaging Git commit: `{manifest['packaging_git_commit']}`
- Upstream probe: `{manifest['upstream_probe_repo']}@{manifest['upstream_probe_revision']}`
- Model: `{manifest['model']}@{manifest['model_revision']}`
- Input SHA256: `{manifest['input_sha256']}`
- Records: `{manifest['record_count']}` across `{manifest['item_count']}` items
- Bootstrap samples: `{manifest['bootstrap_samples']}`

`manifest.json` contains exact file hashes. Runtime logs and utilization snapshots
are included for execution provenance; raw Chinese-LiPS media and control images
are not included.
"""


def package_bundle(
    *,
    run_root: Path,
    output_dir: Path,
    hf_repo_id: str,
    run_git_commit: str,
    packaging_git_commit: str,
    upstream_probe_repo: str,
    upstream_probe_revision: str,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"bundle output already exists: {output_dir}")
    completion, analysis, shard_records = validate_run(run_root)
    output_dir.mkdir(parents=True)
    files = []
    for name in ("completion.json", "analysis_summary_v1.json"):
        destination = output_dir / name
        shutil.copyfile(run_root / name, destination)
        files.append(file_record(destination, output_dir))
    for shard, row_count in shard_records:
        destination = output_dir / "runs" / f"{shard.name}.gz"
        gzip_copy(shard, destination)
        files.append(file_record(destination, output_dir, rows=row_count))
    for name in OPTIONAL_RUNTIME_FILES:
        source = run_root / name
        if source.is_file():
            destination = output_dir / "runtime" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            files.append(file_record(destination, output_dir))

    manifest = {
        "schema_version": "chinese_lips_visual_control_diagnostic_bundle_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "hf_repo_id": hf_repo_id,
        "run_git_commit": run_git_commit,
        "packaging_git_commit": packaging_git_commit,
        "upstream_probe_repo": upstream_probe_repo,
        "upstream_probe_revision": upstream_probe_revision,
        "upstream_license": "cc-by-nc-sa-4.0",
        "private_required": True,
        "scope": analysis["scope"],
        "model": completion["model"],
        "model_revision": completion["model_revision"],
        "input_sha256": completion["items_sha256"],
        "item_count": analysis["item_count"],
        "record_count": analysis["record_count"],
        "bootstrap_samples": analysis["contrasts"][0]["bootstrap_samples"],
        "files": files,
    }
    readme_path = output_dir / "README.md"
    readme_path.write_text(render_readme(manifest), encoding="utf-8")
    files.append(file_record(readme_path, output_dir))
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hf-repo-id", required=True)
    parser.add_argument("--run-git-commit", required=True)
    parser.add_argument("--packaging-git-commit", required=True)
    parser.add_argument("--upstream-probe-repo", required=True)
    parser.add_argument("--upstream-probe-revision", required=True)
    args = parser.parse_args()
    manifest = package_bundle(
        run_root=args.run_root,
        output_dir=args.output_dir,
        hf_repo_id=args.hf_repo_id,
        run_git_commit=args.run_git_commit,
        packaging_git_commit=args.packaging_git_commit,
        upstream_probe_repo=args.upstream_probe_repo,
        upstream_probe_revision=args.upstream_probe_revision,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
