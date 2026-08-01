#!/usr/bin/env python3
"""Prepare and optionally execute one contract-driven Phase-A SimulStream run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import yaml

from slidesst.phase_a_contract import condition_contexts, load_context_bundle


FORBIDDEN_INFERENCE_MARKERS = ("reference", "source_transcript", "tagged_terminology")
REQUIRED_RUNTIME_LOCK_KEYS = {
    "container_image",
    "container_image_digest",
    "python_version",
    "cuda_version",
    "torch_version",
    "qwen_asr_version",
    "vllm_version",
    "transformers_version",
    "simulstream_revision",
    "omnisteval_revision",
    "comet_model",
    "comet_revision",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def hf_snapshot_path(hf_home: Path, repo_id: str, revision: str) -> Path:
    repo_dir = "models--" + repo_id.replace("/", "--")
    return hf_home / "hub" / repo_dir / "snapshots" / revision


def load_inference_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("Inference view is empty")
    for row in rows:
        leaked = {
            key
            for key in row
            if any(marker in key.lower() for marker in FORBIDDEN_INFERENCE_MARKERS)
        }
        if leaked:
            raise ValueError(f"Inference view leaks scoring-only keys: {sorted(leaked)}")
        audio = Path(row["audio_path"])
        paper = Path(row["paper_pdf_path"])
        if sha256_file(audio) != row["audio_sha256"]:
            raise ValueError(f"Audio hash mismatch: {audio}")
        if sha256_file(paper) != row["paper_pdf_sha256"]:
            raise ValueError(f"Paper hash mismatch: {paper}")
        if audio.stem != row["talk_id"]:
            raise ValueError(f"Inference talk/audio mismatch: {audio}")
    return rows


def load_runtime_lock(path: Path, contract: dict) -> dict:
    lock = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(lock, dict) or set(lock) != REQUIRED_RUNTIME_LOCK_KEYS:
        raise ValueError("Runtime lock has an invalid schema")
    if any(not isinstance(lock[key], str) or not lock[key] for key in REQUIRED_RUNTIME_LOCK_KEYS):
        raise ValueError("Runtime lock values must be non-empty strings")
    if not lock["container_image_digest"].startswith("sha256:"):
        raise ValueError("Runtime lock container_image_digest must be a sha256 digest")
    if lock["simulstream_revision"] != contract["runner"]["toolkit"]["revision"]:
        raise ValueError("Runtime lock SimulStream revision differs from contract")
    if lock["omnisteval_revision"] != contract["runner"]["evaluation"]["revision"]:
        raise ValueError("Runtime lock OmniSTEval revision differs from contract")
    return lock


def verify_model_snapshots(contract: dict, hf_home: Path) -> dict[str, str]:
    paths = {}
    for role in ("asr", "forced_aligner", "mt"):
        model = contract["runner"]["models"][role]
        path = hf_snapshot_path(hf_home, model["repo"], model["revision"])
        if not path.is_dir():
            raise FileNotFoundError(f"Missing pinned {role} snapshot: {path}")
        repo_dir = "models--" + model["repo"].replace("/", "--")
        main_ref = hf_home / "hub" / repo_dir / "refs" / "main"
        if not main_ref.is_file() or main_ref.read_text(encoding="utf-8").strip() != model["revision"]:
            raise ValueError(f"HF cache main ref is not pinned for {role}: {main_ref}")
        paths[role] = str(path.resolve())
    return paths


def prepare_run(args: argparse.Namespace) -> Path:
    contract = yaml.safe_load(args.contract.read_text(encoding="utf-8"))
    condition = args.condition
    condition_config = contract["conditions"][condition]
    if condition_config["launcher_status"].startswith("blocked_"):
        raise ValueError(f"{condition} launch blocked: {condition_config['launcher_status']}")
    runtime_lock = load_runtime_lock(args.runtime_lock, contract)
    expected_upstream = contract["runner"]["upstream"]["revision"]
    actual_upstream = git_head(args.upstream_root)
    if actual_upstream != expected_upstream:
        raise ValueError(f"Upstream revision mismatch: expected {expected_upstream}, got {actual_upstream}")
    expected_simulstream = contract["runner"]["toolkit"]["revision"]
    actual_simulstream = git_head(args.simulstream_root)
    if actual_simulstream != expected_simulstream:
        raise ValueError(
            f"SimulStream revision mismatch: expected {expected_simulstream}, got {actual_simulstream}"
        )
    expected_omnisteval = contract["runner"]["evaluation"]["revision"]
    actual_omnisteval = git_head(args.omnisteval_root)
    if actual_omnisteval != expected_omnisteval:
        raise ValueError(
            f"OmniSTEval revision mismatch: expected {expected_omnisteval}, got {actual_omnisteval}"
        )

    rows = load_inference_rows(args.inference_view)
    wav_lines = [row["audio_path"] for row in rows]
    args.output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=args.output_root,
        prefix=".phase_a_wav_list.",
        suffix=".txt",
        delete=False,
    ) as handle:
        handle.write("\n".join(wav_lines) + "\n")
        staging_wav_list = Path(handle.name)
    try:
        bundle = load_context_bundle(args.context_bundle, staging_wav_list)
    finally:
        staging_wav_list.unlink(missing_ok=True)

    max_tokens = contract["context_contract"]["max_injected_tokens_per_channel"]
    condition_contexts(bundle["packets"], condition, max_tokens)
    model_paths = verify_model_snapshots(contract, args.hf_home)
    hashes = {
        "contract": sha256_file(args.contract),
        "inference_view": sha256_file(args.inference_view),
        "context_bundle": sha256_file(args.context_bundle),
        "runtime_lock": sha256_file(args.runtime_lock),
    }
    chunk_ms = contract["runner"]["policy"]["primary_chunk_ms"]
    run_id = (
        f"acl6060_dev_{condition.lower()}_seg{chunk_ms}_"
        f"{hashes['contract'][:10]}_{hashes['inference_view'][:10]}_"
        f"{hashes['context_bundle'][:10]}_{hashes['runtime_lock'][:10]}"
    )
    run_dir = args.output_root / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir()

    wav_list = run_dir / "wav_list.txt"
    wav_list.write_text("\n".join(wav_lines) + "\n", encoding="utf-8")
    processor_config = {
        "type": contract["runner"]["adapter_class"],
        "speech_chunk_size": chunk_ms / 1000,
        "latency_unit": contract["runner"]["policy"]["latency_unit"],
        "detokenizer_type": "simuleval",
        "asr_model_name": model_paths["asr"],
        "llm_model_name": model_paths["mt"],
        "source_lang": "English",
        "target_lang": "Chinese",
        "min_start_seconds": contract["runner"]["policy"]["min_start_seconds"],
        "max_history_utterances": 0,
        "max_new_tokens": 100,
        "temperature": 0.0,
        "repetition_penalty": 1.05,
        "ner_results_path": None,
        "abstract_results_path": None,
        "context_condition": condition,
        "context_packet_path": str(args.context_bundle.resolve()),
        "wav_list_file": str(wav_list.resolve()),
        "max_context_tokens": max_tokens,
    }
    config_path = run_dir / "speech_processor.yaml"
    config_path.write_text(yaml.safe_dump(processor_config, sort_keys=False), encoding="utf-8")
    manifest = {
        "run_id": run_id,
        "condition": condition,
        "status": "prepared",
        "talk_ids": [row["talk_id"] for row in rows],
        "talk_audio_sha256": {row["talk_id"]: row["audio_sha256"] for row in rows},
        "hashes": hashes,
        "upstream_revision": actual_upstream,
        "simulstream_revision": actual_simulstream,
        "omnisteval_revision": actual_omnisteval,
        "model_paths": model_paths,
        "compiler": bundle["compiler"],
        "runtime_lock": runtime_lock,
        "reference_paths_passed_to_process": False,
        "command": [
            "simulstream_inference",
            "--speech-processor-config",
            str(config_path.resolve()),
            "--wav-list-file",
            str(wav_list.resolve()),
            "--src-lang",
            "English",
            "--tgt-lang",
            "Chinese",
            "--metrics-log-file",
            str((run_dir / "metrics.jsonl").resolve()),
        ],
    }
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.execute:
        env = os.environ.copy()
        pythonpath = [
            str(args.repo_src.resolve()),
            str(args.upstream_root.resolve()),
            str(args.simulstream_root.resolve()),
        ]
        if env.get("PYTHONPATH"):
            pythonpath.append(env["PYTHONPATH"])
        env.update(
            PYTHONPATH=os.pathsep.join(pythonpath),
            HF_HOME=str(args.hf_home.resolve()),
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
        )
        subprocess.run(manifest["command"], cwd=args.upstream_root, env=env, check=True)
        metrics = run_dir / "metrics.jsonl"
        if not metrics.is_file() or metrics.stat().st_size == 0:
            raise RuntimeError("Inference completed without a non-empty metrics.jsonl")
        manifest["status"] = "inference_complete_unscored"
        manifest["metrics_sha256"] = sha256_file(metrics)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--condition", choices=("C0", "C1", "C2", "C3"), required=True)
    parser.add_argument("--inference-view", type=Path, required=True)
    parser.add_argument("--context-bundle", type=Path, required=True)
    parser.add_argument("--runtime-lock", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--simulstream-root", type=Path, required=True)
    parser.add_argument("--omnisteval-root", type=Path, required=True)
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--repo-src", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(prepare_run(args))


if __name__ == "__main__":
    main()
