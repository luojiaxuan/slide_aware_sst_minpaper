#!/usr/bin/env python3
"""Analyze the MCIF exploratory raw-image versus OCR screen."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import sacrebleu


CONDITIONS = ("audio_only", "ocr", "raw_image", "wrong_image")
PRIMARY_CONTRASTS = (("raw_image", "ocr"), ("raw_image", "wrong_image"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def sentence_chrf(hypothesis: str, reference: str) -> float:
    return float(sacrebleu.sentence_chrf(hypothesis, [reference]).score)


def prefix_auc(row: dict, reference: str) -> float:
    prefixes = row.get("prefix_hypotheses") or []
    if len(prefixes) != int(row["n_chunks"]):
        raise ValueError(f"Incomplete prefix history: {row['id']}/{row['condition']}")
    times = [0.0]
    scores = [0.0]
    for expected_step, prefix in enumerate(prefixes, 1):
        if prefix["step"] != expected_step or prefix["audio_time_sec"] <= times[-1]:
            raise ValueError(f"Invalid prefix order: {row['id']}/{row['condition']}")
        times.append(float(prefix["audio_time_sec"]))
        scores.append(sentence_chrf(prefix["hypothesis"], reference))
    duration = times[-1]
    area = sum(
        (scores[index - 1] + scores[index])
        * (times[index] - times[index - 1])
        / 2.0
        for index in range(1, len(times))
    )
    return area / duration


def load_matrix(run_root: Path, expected_revision: str) -> dict[tuple[str, str, str], dict]:
    rows = []
    for path in sorted(run_root.glob("runs_shard_*.jsonl")):
        rows.extend(load_jsonl(path))
    if not rows:
        raise ValueError("No result shards found")
    matrix = {}
    for row in rows:
        key = (row["screen_id"], row["acoustic_condition"], row["condition"])
        if key in matrix:
            raise ValueError(f"Duplicate result: {key}")
        if row["condition"] not in CONDITIONS:
            raise ValueError(f"Unexpected condition: {row['condition']}")
        if row.get("model_revision") != expected_revision:
            raise ValueError(f"Model revision drift: {row.get('model_revision')}")
        if row.get("reference") not in (None, ""):
            raise ValueError(f"Reference was exposed to inference: {row['id']}")
        matrix[key] = row
    return matrix


def cluster_bootstrap(
    metric_rows: list[dict],
    first: str,
    second: str,
    samples: int,
    seed: int,
) -> dict:
    by_talk: dict[str, list[float]] = defaultdict(list)
    for row in metric_rows:
        by_talk[row["talk_id"]].append(
            row["conditions"][first]["prefix_auc_sentence_chrf"]
            - row["conditions"][second]["prefix_auc_sentence_chrf"]
        )
    talks = sorted(by_talk)
    talk_means = np.asarray([np.mean(by_talk[talk]) for talk in talks])
    observed = float(np.mean(talk_means))
    rng = np.random.default_rng(seed)
    draws = np.empty(samples)
    for index in range(samples):
        sampled = rng.integers(0, len(talks), len(talks))
        draws[index] = float(np.mean(talk_means[sampled]))
    return {
        "first": first,
        "second": second,
        "talk_macro_delta_prefix_auc_sentence_chrf": observed,
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "talk_count": len(talks),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
    }


def analyze(args: argparse.Namespace) -> dict:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    expected_revision = config["inference"]["model_revision"]
    selection = load_jsonl(args.selection)
    by_screen = {row["screen_id"]: row for row in selection}
    if len(by_screen) != len(selection):
        raise ValueError("Duplicate scorer selection rows")
    matrix = load_matrix(args.run_root, expected_revision)
    acoustic_conditions = [
        row["condition_id"] for row in config["acoustic_conditions"]
    ]
    expected = {
        (screen_id, acoustic, condition)
        for screen_id in by_screen
        for acoustic in acoustic_conditions
        for condition in CONDITIONS
    }
    if set(matrix) != expected:
        missing = sorted(expected - set(matrix))
        extra = sorted(set(matrix) - expected)
        raise ValueError(f"Incomplete matrix: missing={missing[:5]} extra={extra[:5]}")

    metric_rows = []
    positive_rows = []
    for screen_id, scorer in sorted(by_screen.items()):
        for acoustic in acoustic_conditions:
            condition_metrics = {}
            for condition in CONDITIONS:
                result = matrix[(screen_id, acoustic, condition)]
                reference = scorer["target_reference_zh"]
                condition_metrics[condition] = {
                    "final_sentence_chrf": sentence_chrf(result["hypothesis"], reference),
                    "prefix_auc_sentence_chrf": prefix_auc(result, reference),
                    "final_hypothesis": result["hypothesis"],
                }
            primary_positive = (
                condition_metrics["raw_image"]["prefix_auc_sentence_chrf"]
                > condition_metrics["ocr"]["prefix_auc_sentence_chrf"]
                and condition_metrics["raw_image"]["prefix_auc_sentence_chrf"]
                > condition_metrics["wrong_image"]["prefix_auc_sentence_chrf"]
            )
            metric_row = {
                "screen_id": screen_id,
                "candidate_id": scorer["candidate_id"],
                "candidate_text": scorer["candidate_text"],
                "talk_id": scorer["talk_id"],
                "segment_id": scorer["segment_id"],
                "acoustic_condition": acoustic,
                "conditions": condition_metrics,
                "primary_positive": primary_positive,
            }
            metric_rows.append(metric_row)
            if primary_positive:
                positive_rows.append(
                    {
                        **metric_row,
                        "source_reference_en": scorer["source_reference_en"],
                        "target_reference_zh": scorer["target_reference_zh"],
                        "formal_human_validation_status": "PENDING",
                    }
                )

    condition_summary = {}
    for acoustic in acoustic_conditions:
        rows = [row for row in metric_rows if row["acoustic_condition"] == acoustic]
        condition_summary[acoustic] = {
            condition: {
                "mean_final_sentence_chrf": float(
                    np.mean(
                        [row["conditions"][condition]["final_sentence_chrf"] for row in rows]
                    )
                ),
                "mean_prefix_auc_sentence_chrf": float(
                    np.mean(
                        [
                            row["conditions"][condition]["prefix_auc_sentence_chrf"]
                            for row in rows
                        ]
                    )
                ),
            }
            for condition in CONDITIONS
        }
        condition_summary[acoustic]["primary_positive_count"] = sum(
            row["primary_positive"] for row in rows
        )

    bootstraps = {}
    for acoustic_index, acoustic in enumerate(acoustic_conditions):
        rows = [row for row in metric_rows if row["acoustic_condition"] == acoustic]
        bootstraps[acoustic] = [
            cluster_bootstrap(
                rows,
                first,
                second,
                args.bootstrap_samples,
                args.seed + acoustic_index * 10 + contrast_index,
            )
            for contrast_index, (first, second) in enumerate(PRIMARY_CONTRASTS)
        ]

    positive_by_screen = Counter(
        row["screen_id"] for row in positive_rows
    )
    positive_in_both = sum(
        count == len(acoustic_conditions) for count in positive_by_screen.values()
    )
    summary = {
        "schema_version": "mcif_beyond_ocr_exploratory_analysis_v1",
        "scope": config["scope"],
        "candidate_count": len(selection),
        "result_count": len(matrix),
        "model_revision": expected_revision,
        "conditions": condition_summary,
        "talk_cluster_bootstrap": bootstraps,
        "positive_candidate_acoustic_pairs": len(positive_rows),
        "unique_positive_candidates": len(positive_by_screen),
        "positive_in_both_acoustic_conditions": positive_in_both,
        "positive_in_exactly_one_acoustic_condition": (
            len(positive_by_screen) - positive_in_both
        ),
        "decision_rule": config["decision_rule"],
        "selection_sha256": sha256_file(args.selection),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.output_root / "candidate_metrics.jsonl").open("w", encoding="utf-8") as out:
        for row in metric_rows:
            out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (args.output_root / "positive_candidates.jsonl").open("w", encoding="utf-8") as out:
        for row in positive_rows:
            out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()
    print(json.dumps(analyze(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
