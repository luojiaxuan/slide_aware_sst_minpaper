#!/usr/bin/env python3
"""Create a portable byte-level manifest for frozen PaddleX model directories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def resolved_model_name(requested_name: str, engine: str) -> str:
    if engine in {"paddle_dynamic", "transformers"}:
        return f"{requested_name}_safetensors"
    if engine == "paddle_static":
        return requested_name
    raise ValueError(f"Unsupported PaddleX engine: {engine}")


def hash_model_directory(
    root: Path, requested_name: str, engine: str
) -> dict[str, Any]:
    resolved_name = resolved_model_name(requested_name, engine)
    model_root = root / resolved_name
    if not model_root.is_dir() or model_root.is_symlink():
        raise FileNotFoundError(f"Missing plain PaddleX model directory: {model_root}")
    files = []
    for path in sorted(model_root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"PaddleX model contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(model_root).as_posix()
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not files:
        raise ValueError(f"PaddleX model directory is empty: {model_root}")
    return {
        "requested_name": requested_name,
        "resolved_name": resolved_name,
        "file_count": len(files),
        "size_bytes": sum(item["size_bytes"] for item in files),
        "tree_sha256": canonical_hash(files),
        "files": files,
    }


def build_manifest(config: dict[str, Any], official_models_root: Path) -> dict[str, Any]:
    models = config.get("models") or {}
    if not models:
        raise ValueError("Config contains no PaddleX models")
    engine = config.get("inference_engine")
    unique_models = [
        hash_model_directory(official_models_root, name, engine)
        for name in sorted(set(models.values()))
    ]
    return {
        "schema_version": 1,
        "artifact": "paddlex_model_file_manifest",
        "model_source": config.get("model_source"),
        "config_sha256": canonical_hash(config),
        "models": models,
        "unique_models": unique_models,
        "model_set_sha256": canonical_hash(
            [
                {
                    "requested_name": model["requested_name"],
                    "resolved_name": model["resolved_name"],
                    "tree_sha256": model["tree_sha256"],
                }
                for model in unique_models
            ]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--official-models-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = build_manifest(
        load_json(args.config), args.official_models_root.resolve(strict=True)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "unique_models"}, sort_keys=True))


if __name__ == "__main__":
    main()
