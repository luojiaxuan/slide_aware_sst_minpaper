import numpy as np
import pytest

from scripts.analyze_probe_visual_control_matrix import (
    analyze_contrasts,
    average_lagging,
    chrf_statistics,
    corpus_chrf,
    corpus_chrf_from_statistics,
    paired_bootstrap,
)


def test_average_lagging_uses_commit_events_and_chunk_seconds():
    row = {
        "id": "item-1",
        "n_chunks": 4,
        "chunk_s": 0.5,
        "events": [[2, 2], [4, 4]],
    }
    assert average_lagging(row) == pytest.approx(5 / 6)


def test_average_lagging_falls_back_to_full_source_for_empty_output():
    row = {"id": "item-1", "n_chunks": 4, "chunk_s": 0.5, "events": []}
    assert average_lagging(row) == pytest.approx(2.0)


def test_parallel_contrast_analysis_matches_serial_output():
    item_ids = ["item-1", "item-2"]
    matrix = {}
    for item_index, item_id in enumerate(item_ids):
        matrix[item_id] = {}
        for condition_index, condition in enumerate(
            ("none", "slide", "wrong", "cross_talk", "blank")
        ):
            matrix[item_id][condition] = {
                "id": item_id,
                "hypothesis": f"translation {item_index + condition_index}",
                "reference": f"reference {item_index}",
                "n_chunks": 4,
                "chunk_s": 0.5,
                "events": [[2, 2], [4, 4]],
                "wall_s": 1.0,
            }

    serial = analyze_contrasts(item_ids, matrix, samples=5, seed=17, workers=1)
    parallel = analyze_contrasts(item_ids, matrix, samples=5, seed=17, workers=2)

    assert parallel == serial


def test_precomputed_chrf_statistics_match_sacrebleu_corpus_score():
    rows = [
        {"hypothesis": "the quick fox", "reference": "the quick brown fox"},
        {"hypothesis": "jumps high", "reference": "jumps very high"},
    ]

    assert corpus_chrf_from_statistics(chrf_statistics(rows)) == pytest.approx(
        corpus_chrf(rows)
    )


def test_precomputed_bootstrap_matches_naive_resampled_corpus_chrf():
    item_ids = ["item-1", "item-2", "item-3"]
    matrix = {
        item_id: {
            "slide": {
                "id": item_id,
                "hypothesis": f"slide translation {index}",
                "reference": f"reference translation {index}",
                "n_chunks": 4,
                "chunk_s": 0.5,
                "events": [[2, 2], [4, 4]],
            },
            "none": {
                "id": item_id,
                "hypothesis": f"audio output {index}",
                "reference": f"reference translation {index}",
                "n_chunks": 4,
                "chunk_s": 0.5,
                "events": [[2, 2], [4, 4]],
            },
        }
        for index, item_id in enumerate(item_ids)
    }
    samples = 11
    seed = 23
    result = paired_bootstrap(item_ids, matrix, "slide", "none", samples, seed)
    first_rows = [matrix[item_id]["slide"] for item_id in item_ids]
    second_rows = [matrix[item_id]["none"] for item_id in item_ids]
    rng = np.random.default_rng(seed)
    naive_deltas = []
    for _ in range(samples):
        indices = rng.integers(0, len(item_ids), len(item_ids))
        naive_deltas.append(
            corpus_chrf([first_rows[index] for index in indices])
            - corpus_chrf([second_rows[index] for index in indices])
        )

    assert result["delta_corpus_chrf_ci95"] == pytest.approx(
        np.quantile(naive_deltas, [0.025, 0.975])
    )
