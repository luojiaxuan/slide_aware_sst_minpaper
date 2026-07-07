import gzip
import json
import subprocess
import sys


def test_package_hf_dataset_bundle_writes_manifest_and_card(tmp_path):
    repo_root = tmp_path / "repo"
    output_dir = tmp_path / "hf_bundle"
    files = {
        "outputs/chinese_lips_train/data/challenge_verified_qwen3_vl_context.jsonl": '{"id":"a"}\n{"id":"b"}\n',
        "outputs/chinese_lips_train/index/evidence_qwen3_vl_context.jsonl": '{"evidence_id":"e1"}\n',
        "outputs/chinese_lips_train/annotation/diagnostic_sample_500_qwen3_vl_context.jsonl": '{"id":"a"}\n',
        "outputs/chinese_lips_train/annotation/diagnostic_sample_500_qwen3_vl_context.csv": "id\n a\n",
        "outputs/chinese_lips_train/annotation/diagnostic_sample_500_qwen3_vl_context.stats.json": '{"selected_items":1}\n',
        "outputs/chinese_lips_train/qa/qwen3_vl_context_qa.json": json.dumps(
            {
                "rows": 2,
                "unique_ids": 2,
                "missing_raw_output": 0,
                "raw_parse_failures": 0,
                "evidence": {"rows": 1},
            }
        )
        + "\n",
    }
    for relative_path, content in files.items():
        path = repo_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/package_hf_dataset_bundle.py",
            "--repo-root",
            str(repo_root),
            "--output-dir",
            str(output_dir),
            "--hf-repo-id",
            "owner/slide-context-sst-chinese-lips",
            "--source-git-commit",
            "abc123",
            "--upstream-revision",
            "upstream123",
        ],
        check=True,
    )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["hf_repo_id"] == "owner/slide-context-sst-chinese-lips"
    assert manifest["source_git_commit"] == "abc123"
    assert manifest["upstream_revision"] == "upstream123"
    rows_by_path = {record["path"]: record["rows"] for record in manifest["files"]}
    assert rows_by_path["data/challenge_verified_qwen3_vl_context.jsonl.gz"] == 2
    assert rows_by_path["index/evidence_qwen3_vl_context.jsonl.gz"] == 1
    with gzip.open(output_dir / "data/challenge_verified_qwen3_vl_context.jsonl.gz", "rt", encoding="utf-8") as f:
        assert f.read() == files["outputs/chinese_lips_train/data/challenge_verified_qwen3_vl_context.jsonl"]
    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    assert "Keep this Hugging Face repo private" in readme
    assert "abc123" in readme
