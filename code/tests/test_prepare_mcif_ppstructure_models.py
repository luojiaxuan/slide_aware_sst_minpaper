import importlib.util
import json
from pathlib import Path

import pytest


def load_module():
    scripts = Path(__file__).parents[1] / "scripts"
    script = scripts / "prepare_mcif_ppstructure_models.py"
    spec = importlib.util.spec_from_file_location(
        "prepare_mcif_ppstructure_models", script
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    import sys

    sys.path.insert(0, str(scripts))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


class FakePipeline:
    def __init__(self):
        self.closed = False

    def export_paddlex_config_to_yaml(self, path: str) -> None:
        Path(path).write_text("pipeline: frozen\n", encoding="utf-8")

    def close(self) -> None:
        self.closed = True


def test_prepare_writes_atomic_model_and_resolved_manifests(tmp_path, monkeypatch):
    module = load_module()
    config_path = (
        Path(__file__).parents[1]
        / "configs"
        / "mcif_ppstructurev3_source_screen_v1.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    models_root = tmp_path / "official_models"
    for name in set(config["models"].values()):
        model_root = models_root / name
        model_root.mkdir(parents=True)
        (model_root / "weights.bin").write_bytes(name.encode())
    pipeline = FakePipeline()

    def factory(frozen, device):
        assert frozen == config
        assert device == "gpu:0"
        return pipeline, {
            "paddleocr": "3.7.0",
            "paddlex": "3.7.0",
            "paddlepaddle": "3.3.0",
        }

    monkeypatch.setenv("PADDLE_PDX_MODEL_SOURCE", "huggingface")
    resolved = tmp_path / "resolved.yaml"
    manifest_path = tmp_path / "models.json"
    manifest = module.prepare(
        config,
        device="gpu:0",
        official_models_root=models_root,
        resolved_config_out=resolved,
        model_manifest_out=manifest_path,
        create_pipeline_fn=factory,
    )
    assert pipeline.closed
    assert resolved.read_text(encoding="utf-8") == "pipeline: frozen\n"
    assert manifest_path.is_file()
    assert manifest["models"] == config["models"]
    assert len(manifest["unique_models"]) == len(set(config["models"].values()))
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(FileExistsError):
        module.prepare(
            config,
            device="gpu:0",
            official_models_root=models_root,
            resolved_config_out=resolved,
            model_manifest_out=manifest_path,
            create_pipeline_fn=factory,
        )
