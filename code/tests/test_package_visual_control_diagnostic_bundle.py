import gzip
import hashlib
import json

import pytest

from scripts.package_visual_control_diagnostic_bundle import CONDITIONS, package_bundle


def write_run(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    rows = [
        {
            "id": "item-1",
            "condition": condition,
            "model": "fixture/model",
            "model_revision": "a" * 40,
            "reference": "reference",
            "hypothesis": condition,
        }
        for condition in CONDITIONS
    ]
    shard_path = run_root / "runs_shard_0.jsonl"
    shard_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    completion = {
        "status": "COMPLETE",
        "conditions": list(CONDITIONS),
        "model": "fixture/model",
        "model_revision": "a" * 40,
        "items_sha256": "b" * 64,
        "outputs": [
            {
                "output": str(shard_path),
                "output_sha256": hashlib.sha256(shard_path.read_bytes()).hexdigest(),
                "record_count": len(rows),
                "expected_record_count": len(rows),
            }
        ],
    }
    completion_path = run_root / "completion.json"
    completion_path.write_text(json.dumps(completion) + "\n", encoding="utf-8")
    analysis = {
        "scope": "private_story_diagnostic_not_paper_gold",
        "item_count": 1,
        "record_count": len(rows),
        "completion_sha256": hashlib.sha256(completion_path.read_bytes()).hexdigest(),
        "contrasts": [{"bootstrap_samples": 10_000}],
    }
    (run_root / "analysis_summary_v1.json").write_text(
        json.dumps(analysis) + "\n",
        encoding="utf-8",
    )
    return run_root, shard_path


def test_package_visual_control_bundle_validates_and_gzips_outputs(tmp_path):
    run_root, _ = write_run(tmp_path)
    output_dir = tmp_path / "bundle"
    manifest = package_bundle(
        run_root=run_root,
        output_dir=output_dir,
        hf_repo_id="owner/private-repo",
        run_git_commit="1" * 40,
        packaging_git_commit="2" * 40,
        upstream_probe_repo="owner/probe",
        upstream_probe_revision="3" * 40,
    )
    assert manifest["record_count"] == 5
    assert manifest["private_required"] is True
    with gzip.open(output_dir / "runs" / "runs_shard_0.jsonl.gz", "rt") as source:
        assert len(source.readlines()) == 5
    assert (output_dir / "README.md").is_file()
    assert (output_dir / "manifest.json").is_file()


def test_package_visual_control_bundle_rejects_model_revision_drift(tmp_path):
    run_root, shard_path = write_run(tmp_path)
    rows = [json.loads(line) for line in shard_path.read_text().splitlines()]
    rows[0]["model_revision"] = "c" * 40
    shard_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    completion_path = run_root / "completion.json"
    completion = json.loads(completion_path.read_text())
    completion["outputs"][0]["output_sha256"] = hashlib.sha256(
        shard_path.read_bytes()
    ).hexdigest()
    completion_path.write_text(json.dumps(completion) + "\n", encoding="utf-8")
    analysis_path = run_root / "analysis_summary_v1.json"
    analysis = json.loads(analysis_path.read_text())
    analysis["completion_sha256"] = hashlib.sha256(completion_path.read_bytes()).hexdigest()
    analysis_path.write_text(json.dumps(analysis) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="model identity mismatch"):
        package_bundle(
            run_root=run_root,
            output_dir=tmp_path / "bundle",
            hf_repo_id="owner/private-repo",
            run_git_commit="1" * 40,
            packaging_git_commit="2" * 40,
            upstream_probe_repo="owner/probe",
            upstream_probe_revision="3" * 40,
        )
