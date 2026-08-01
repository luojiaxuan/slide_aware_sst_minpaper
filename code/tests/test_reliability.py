import json
import random

import pytest

from slidesst.data.reliability import (
    _draw_cluster_sample,
    build_confusion_matrix,
    cluster_bootstrap_percentile_ci,
    reliability_report,
    reliability_report_from_confusion,
)


CATEGORIES = ["no", "uncertain", "yes"]


def test_hand_calculated_three_by_three_fixture():
    matrix = [
        [4, 1, 0],
        [0, 3, 1],
        [1, 0, 2],
    ]

    report = reliability_report_from_confusion(matrix, CATEGORIES)

    assert report["categories"] == CATEGORIES
    assert report["confusion_matrix"] == matrix
    assert report["row_totals"] == [5, 4, 3]
    assert report["column_totals"] == [5, 4, 3]
    assert report["exact_agreement"] == {"value": 0.75, "status": "ok"}
    assert [entry["value"] for entry in report["category_specific_agreement"]] == pytest.approx(
        [0.8, 0.75, 2.0 / 3.0]
    )
    assert report["gwet_ac1"]["value"] == pytest.approx(61.0 / 97.0)
    assert report["gwet_ac1"]["status"] == "ok"
    assert report["cohen_kappa"]["value"] == pytest.approx(29.0 / 47.0)
    assert report["cohen_kappa"]["status"] == "ok"
    json.dumps(report)


def test_confusion_matrix_uses_fixed_category_order():
    pairs = [("yes", "no"), ("no", "yes"), ("yes", "yes")]

    assert build_confusion_matrix(pairs, ["yes", "no"]) == [[1, 1], [1, 0]]
    assert build_confusion_matrix(pairs, ["no", "yes"]) == [[0, 1], [1, 1]]


def test_category_specific_agreement_is_none_for_zero_denominator():
    report = reliability_report([("yes", "yes")], ["yes", "no"])

    assert report["category_specific_agreement"][1] == {
        "category": "no",
        "value": None,
        "status": "undefined_zero_denominator",
    }


def test_constant_marginal_kappa_is_explicitly_undefined():
    report = reliability_report([("yes", "yes")] * 4, ["yes", "no"])

    assert report["exact_agreement"] == {"value": 1.0, "status": "ok"}
    assert report["cohen_kappa"] == {
        "value": None,
        "status": "undefined_zero_expected_disagreement",
    }
    assert report["gwet_ac1"] == {"value": 1.0, "status": "ok"}


def test_cluster_draw_never_splits_a_talk():
    talk_ids = ["talk-a", "talk-b"]
    grouped = {
        "talk-a": [("a-1", "a-1"), ("a-2", "a-2")],
        "talk-b": [("b-1", "b-1"), ("b-2", "b-2"), ("b-3", "b-3")],
    }

    drawn_talks, sampled_pairs = _draw_cluster_sample(talk_ids, grouped, random.Random(7))

    expected = [pair for talk_id in drawn_talks for pair in grouped[talk_id]]
    assert sampled_pairs == expected
    for talk_id in talk_ids:
        draw_count = drawn_talks.count(talk_id)
        for pair in grouped[talk_id]:
            assert sampled_pairs.count(pair) == draw_count


def test_cluster_bootstrap_is_reproducible_with_fixed_seed():
    items = [
        ("talk-a", "yes", "yes"),
        ("talk-a", "yes", "no"),
        ("talk-b", "no", "no"),
        ("talk-b", "uncertain", "no"),
        ("talk-c", "yes", "uncertain"),
        ("talk-c", "uncertain", "uncertain"),
    ]

    first = cluster_bootstrap_percentile_ci(items, CATEGORIES, n_resamples=200, seed=42)
    second = cluster_bootstrap_percentile_ci(items, CATEGORIES, n_resamples=200, seed=42)

    assert first == second
    assert first["method"] == "talk_cluster_percentile"
    assert first["confidence_level"] == 0.95
    assert first["n_talks"] == 3
    assert first["n_items"] == 6
    assert first["metrics"]["exact_agreement"]["valid_resamples"] == 200
    json.dumps(first)
