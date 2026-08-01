#!/usr/bin/env python3
"""Validate and analyze the five-condition speech-vision control matrix."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import sacrebleu


CONDITIONS = ("none", "slide", "wrong", "cross_talk", "blank")
CONTRASTS = (
    ("slide", "none"),
    ("wrong", "none"),
    ("slide", "wrong"),
    ("wrong", "cross_talk"),
    ("cross_talk", "blank"),
    ("blank", "none"),
    ("slide", "cross_talk"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def average_lagging(row: dict) -> float:
    source_steps = int(row["n_chunks"])
    events = row.get("events") or []
    if source_steps <= 0 or not events:
        return float(source_steps) * float(row["chunk_s"])
    delays: list[int] = []
    previous_total = 0
    for source_step, target_total in events:
        if target_total < previous_total:
            raise ValueError(f"Non-monotonic commit events for {row['id']}")
        delays.extend([int(source_step)] * (int(target_total) - previous_total))
        previous_total = int(target_total)
    if not delays:
        return float(source_steps) * float(row["chunk_s"])
    target_steps = len(delays)
    rate = target_steps / source_steps
    tau = next(
        (index + 1 for index, delay in enumerate(delays) if delay >= source_steps),
        target_steps,
    )
    lag = sum(delays[index] - index / rate for index in range(tau)) / tau
    return float(lag) * float(row["chunk_s"])


def load_matrix(run_root: Path) -> tuple[list[str], dict[str, dict[str, dict]]]:
    rows = []
    for path in sorted(run_root.glob("runs_shard_*.jsonl")):
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        )
    if not rows:
        raise ValueError("No shard output records found")
    matrix: dict[str, dict[str, dict]] = {}
    for row in rows:
        item_id = row["id"]
        condition = row["condition"]
        if condition not in CONDITIONS:
            raise ValueError(f"Unexpected condition: {condition}")
        if condition in matrix.setdefault(item_id, {}):
            raise ValueError(f"Duplicate record: {item_id}/{condition}")
        matrix[item_id][condition] = row
    item_ids = sorted(matrix)
    for item_id in item_ids:
        missing = set(CONDITIONS) - set(matrix[item_id])
        if missing:
            raise ValueError(f"Incomplete item {item_id}: missing {sorted(missing)}")
        revisions = {matrix[item_id][condition].get("model_revision") for condition in CONDITIONS}
        references = {matrix[item_id][condition].get("reference") for condition in CONDITIONS}
        if len(revisions) != 1 or None in revisions:
            raise ValueError(f"Unfrozen model revision for {item_id}: {revisions}")
        if len(references) != 1:
            raise ValueError(f"Reference mismatch for {item_id}")
    return item_ids, matrix


def corpus_chrf(rows: list[dict]) -> float:
    return float(
        sacrebleu.corpus_chrf(
            [row["hypothesis"] for row in rows],
            [[row["reference"] for row in rows]],
        ).score
    )


def chrf_statistics(rows: list[dict]) -> np.ndarray:
    metric = sacrebleu.metrics.CHRF()
    return np.asarray(
        metric._extract_corpus_statistics(
            [row["hypothesis"] for row in rows],
            [[row["reference"] for row in rows]],
        ),
        dtype=np.int64,
    )


def corpus_chrf_from_statistics(statistics: np.ndarray) -> float:
    metric = sacrebleu.metrics.CHRF()
    return float(metric._compute_score_from_stats(statistics.sum(axis=0)).score)


def summarize_condition(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "corpus_chrf": corpus_chrf(rows),
        "mean_sentence_chrf": float(
            np.mean(
                [
                    sacrebleu.sentence_chrf(row["hypothesis"], [row["reference"]]).score
                    for row in rows
                ]
            )
        ),
        "mean_al_sec": float(np.mean([average_lagging(row) for row in rows])),
        "mean_wall_sec": float(np.mean([float(row["wall_s"]) for row in rows])),
    }


def paired_bootstrap(
    item_ids: list[str],
    matrix: dict[str, dict[str, dict]],
    first: str,
    second: str,
    samples: int,
    seed: int,
) -> dict:
    first_rows = [matrix[item_id][first] for item_id in item_ids]
    second_rows = [matrix[item_id][second] for item_id in item_ids]
    first_chrf_statistics = chrf_statistics(first_rows)
    second_chrf_statistics = chrf_statistics(second_rows)
    observed_chrf = corpus_chrf_from_statistics(
        first_chrf_statistics
    ) - corpus_chrf_from_statistics(second_chrf_statistics)
    first_al = np.asarray([average_lagging(row) for row in first_rows])
    second_al = np.asarray([average_lagging(row) for row in second_rows])
    observed_al = float(np.mean(first_al - second_al))
    rng = np.random.default_rng(seed)
    chrf_deltas = np.empty(samples)
    al_deltas = np.empty(samples)
    for sample_index in range(samples):
        indices = rng.integers(0, len(item_ids), len(item_ids))
        chrf_deltas[sample_index] = corpus_chrf_from_statistics(
            first_chrf_statistics[indices]
        ) - corpus_chrf_from_statistics(second_chrf_statistics[indices])
        al_deltas[sample_index] = float(np.mean((first_al - second_al)[indices]))
    return {
        "first": first,
        "second": second,
        "delta_corpus_chrf": observed_chrf,
        "delta_corpus_chrf_ci95": np.quantile(chrf_deltas, [0.025, 0.975]).tolist(),
        "delta_al_sec": observed_al,
        "delta_al_sec_ci95": np.quantile(al_deltas, [0.025, 0.975]).tolist(),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
    }


def _paired_bootstrap_job(args: tuple) -> dict:
    return paired_bootstrap(*args)


def analyze_contrasts(
    item_ids: list[str],
    matrix: dict[str, dict[str, dict]],
    samples: int,
    seed: int,
    workers: int,
) -> list[dict]:
    jobs = [
        (item_ids, matrix, first, second, samples, seed + index)
        for index, (first, second) in enumerate(CONTRASTS)
    ]
    if workers == 1:
        return [_paired_bootstrap_job(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_paired_bootstrap_job, jobs))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(len(CONTRASTS), os.cpu_count() or 1),
        help="Parallel contrast workers; use 1 for serial execution.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    completion_path = args.run_root / "completion.json"
    if not completion_path.is_file():
        raise ValueError("Run is not complete: completion.json is absent")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("status") != "COMPLETE":
        raise ValueError("Run completion status is not COMPLETE")
    item_ids, matrix = load_matrix(args.run_root)
    conditions = {
        condition: summarize_condition(
            [matrix[item_id][condition] for item_id in item_ids]
        )
        for condition in CONDITIONS
    }
    contrasts = analyze_contrasts(
        item_ids,
        matrix,
        args.bootstrap_samples,
        args.seed,
        min(args.workers, len(CONTRASTS)),
    )
    summary = {
        "schema_version": "speech_vision_probe_visual_control_analysis_v1",
        "scope": "private_story_diagnostic_not_paper_gold",
        "item_count": len(item_ids),
        "record_count": len(item_ids) * len(CONDITIONS),
        "conditions": conditions,
        "contrasts": contrasts,
        "inference_warning": (
            "Segment bootstrap intervals are descriptive because all items come "
            "from one talk and are not independent talk-level replicates."
        ),
        "completion_sha256": sha256_file(completion_path),
    }
    output_path = args.run_root / "analysis_summary_v1.json"
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output_path), "item_count": len(item_ids)}))


if __name__ == "__main__":
    main()
