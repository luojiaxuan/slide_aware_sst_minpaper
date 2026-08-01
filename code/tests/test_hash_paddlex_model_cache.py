import importlib.util
from pathlib import Path

import pytest


def load_module():
    script = Path(__file__).parents[1] / "scripts" / "hash_paddlex_model_cache.py"
    spec = importlib.util.spec_from_file_location("hash_paddlex_model_cache", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_manifest_hashes_every_unique_model(tmp_path):
    module = load_module()
    for name, value in (("model-a", b"a"), ("model-b", b"b")):
        root = tmp_path / f"{name}_safetensors"
        root.mkdir()
        (root / "weights.bin").write_bytes(value)
    config = {
        "model_source": "huggingface",
        "inference_engine": "paddle_dynamic",
        "models": {"first": "model-a", "second": "model-b", "third": "model-a"},
    }
    manifest = module.build_manifest(config, tmp_path)
    assert [record["requested_name"] for record in manifest["unique_models"]] == [
        "model-a",
        "model-b",
    ]
    assert [record["resolved_name"] for record in manifest["unique_models"]] == [
        "model-a_safetensors",
        "model-b_safetensors",
    ]
    assert all(record["file_count"] == 1 for record in manifest["unique_models"])
    assert len(manifest["model_set_sha256"]) == 64
    assert str(tmp_path) not in str(manifest)


def test_model_manifest_rejects_symlink(tmp_path):
    module = load_module()
    root = tmp_path / "model-a_safetensors"
    root.mkdir()
    target = tmp_path / "weights.bin"
    target.write_bytes(b"weights")
    (root / "weights.bin").symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        module.hash_model_directory(tmp_path, "model-a", "paddle_dynamic")
