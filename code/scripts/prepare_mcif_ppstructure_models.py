#!/usr/bin/env python3
"""Prepare and freeze the PaddleX models used by the MCIF structure screen."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable

from hash_paddlex_model_cache import build_manifest
from run_mcif_ppstructure_screen import (
    canonical_hash,
    create_pipeline,
    load_json,
    validate_config,
)


def prepare(
    config: dict[str, Any],
    *,
    device: str,
    official_models_root: Path,
    resolved_config_out: Path,
    model_manifest_out: Path,
    create_pipeline_fn: Callable = create_pipeline,
) -> dict[str, Any]:
    validate_config(config)
    if os.environ.get("PADDLE_PDX_MODEL_SOURCE", "huggingface").lower() != config[
        "model_source"
    ]:
        raise ValueError("PaddleX model source differs from the frozen config")
    if resolved_config_out.exists() or model_manifest_out.exists():
        raise FileExistsError("Frozen PaddleX preparation outputs must be created once")

    resolved_config_out.parent.mkdir(parents=True, exist_ok=True)
    model_manifest_out.parent.mkdir(parents=True, exist_ok=True)
    resolved_tmp = resolved_config_out.with_name(resolved_config_out.name + ".tmp")
    manifest_tmp = model_manifest_out.with_name(model_manifest_out.name + ".tmp")
    if resolved_tmp.exists() or manifest_tmp.exists():
        raise FileExistsError("Stale PaddleX preparation temporary output")

    pipeline, package_versions = create_pipeline_fn(config, device)
    try:
        pipeline.export_paddlex_config_to_yaml(str(resolved_tmp))
    finally:
        pipeline.close()
    manifest = build_manifest(config, official_models_root.resolve(strict=True))
    manifest["package_versions"] = package_versions
    manifest["config_sha256"] = canonical_hash(config)
    manifest_tmp.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(resolved_tmp, resolved_config_out)
    os.replace(manifest_tmp, model_manifest_out)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--official-models-root", type=Path, required=True)
    parser.add_argument("--resolved-config-out", type=Path, required=True)
    parser.add_argument("--model-manifest-out", type=Path, required=True)
    parser.add_argument("--device", default="gpu:0")
    args = parser.parse_args()
    manifest = prepare(
        load_json(args.config),
        device=args.device,
        official_models_root=args.official_models_root,
        resolved_config_out=args.resolved_config_out,
        model_manifest_out=args.model_manifest_out,
    )
    print(
        json.dumps(
            {key: value for key, value in manifest.items() if key != "unique_models"},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
