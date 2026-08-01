#!/usr/bin/env python3
"""Run resumable speech-vision control shards on an explicit GPU allocation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_count(path: Path) -> int:
    return sum(1 for line in path.open(encoding="utf-8") if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--conditions", default="none,slide,wrong,cross_talk,blank")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--chunk-s", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    gpu_ids = [value.strip() for value in args.gpu_ids.split(",") if value.strip()]
    if not gpu_ids:
        raise ValueError("At least one GPU id is required")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("GPU ids must be unique")
    if args.workers_per_gpu < 1:
        raise ValueError("workers-per-gpu must be positive")
    worker_gpu_ids = [
        gpu_id
        for gpu_id in gpu_ids
        for _ in range(args.workers_per_gpu)
    ]
    conditions = [value.strip() for value in args.conditions.split(",") if value.strip()]
    rows = [line for line in args.items.read_text(encoding="utf-8").splitlines() if line]
    args.run_root.mkdir(parents=True, exist_ok=True)
    workers = []
    for shard_index, gpu_id in enumerate(worker_gpu_ids):
        output = args.run_root / f"runs_shard_{shard_index}.jsonl"
        log_path = args.run_root / f"worker_{shard_index}.log"
        command = [
            sys.executable,
            str(Path(__file__).with_name("omni_speech_vision_probe.py")),
            "--items",
            str(args.items),
            "--out",
            str(output),
            "--model",
            args.model,
            "--model-revision",
            args.model_revision,
            "--conditions",
            ",".join(conditions),
            "--chunk-s",
            str(args.chunk_s),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--device-map",
            "cuda:0",
            "--shard-count",
            str(len(worker_gpu_ids)),
            "--shard-index",
            str(shard_index),
            "--seed",
            str(args.seed),
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu_id
        log = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        workers.append((shard_index, process, log, output, log_path))

    exit_codes: dict[int, int] = {}
    try:
        remaining = {shard_index for shard_index, *_ in workers}
        while remaining:
            for shard_index, process, log, output, log_path in workers:
                if shard_index not in remaining:
                    continue
                code = process.poll()
                if code is None:
                    continue
                log.close()
                exit_codes[shard_index] = code
                remaining.remove(shard_index)
                if code != 0:
                    for peer_index, peer, _, _, _ in workers:
                        if peer_index in remaining and peer.poll() is None:
                            peer.terminate()
                    break
            if any(code != 0 for code in exit_codes.values()):
                for peer_index, peer, peer_log, _, _ in workers:
                    if peer_index not in remaining:
                        continue
                    try:
                        code = peer.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        peer.kill()
                        code = peer.wait()
                    if not peer_log.closed:
                        peer_log.close()
                    exit_codes[peer_index] = code
                remaining.clear()
                break
            if remaining:
                time.sleep(5)
    finally:
        for _, process, log, _, _ in workers:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if not log.closed:
                log.close()

    if len(exit_codes) != len(workers):
        raise RuntimeError(
            f"Missing worker exit codes: {len(exit_codes)}/{len(workers)}"
        )
    failures = sorted(
        (index, code) for index, code in exit_codes.items() if code != 0
    )
    if failures:
        raise RuntimeError(f"Worker failures: {failures}")
    outputs = []
    for shard_index, _, _, output, log_path in workers:
        shard_items = sum(
            index % len(worker_gpu_ids) == shard_index for index in range(len(rows))
        )
        expected = shard_items * len(conditions)
        actual = line_count(output)
        if actual != expected:
            raise RuntimeError(
                f"Shard {shard_index} produced {actual}/{expected} records"
            )
        outputs.append(
            {
                "shard_index": shard_index,
                "gpu_id": worker_gpu_ids[shard_index],
                "record_count": actual,
                "expected_record_count": expected,
                "output": str(output),
                "output_sha256": sha256_file(output),
                "log": str(log_path),
            }
        )
    completion = {
        "status": "COMPLETE",
        "completed_at_unix": time.time(),
        "items": str(args.items),
        "items_sha256": sha256_file(args.items),
        "conditions": conditions,
        "gpu_ids": gpu_ids,
        "workers_per_gpu": args.workers_per_gpu,
        "worker_gpu_ids": worker_gpu_ids,
        "model": args.model,
        "model_revision": args.model_revision,
        "outputs": outputs,
    }
    (args.run_root / "completion.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(completion))


if __name__ == "__main__":
    main()
