from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping, Sequence
from typing import TypeAlias


LabelPair: TypeAlias = tuple[str, str]
ClusteredLabelPair: TypeAlias = tuple[str, str, str]
JsonObject: TypeAlias = dict[str, object]


def _validate_categories(categories: Sequence[str]) -> list[str]:
    ordered = list(categories)
    if not ordered:
        raise ValueError("categories must not be empty")
    if any(not isinstance(category, str) or not category for category in ordered):
        raise ValueError("categories must contain non-empty strings")
    if len(set(ordered)) != len(ordered):
        raise ValueError("categories must be unique")
    return ordered


def build_confusion_matrix(
    label_pairs: Iterable[LabelPair],
    categories: Sequence[str],
) -> list[list[int]]:
    """Build an annotator-A-by-annotator-B matrix in the supplied category order."""
    ordered = _validate_categories(categories)
    category_index = {category: index for index, category in enumerate(ordered)}
    matrix = [[0 for _ in ordered] for _ in ordered]

    for item_index, pair in enumerate(label_pairs):
        try:
            label_a, label_b = pair
        except (TypeError, ValueError) as exc:
            raise ValueError(f"label pair {item_index} must contain exactly two labels") from exc
        if label_a not in category_index:
            raise ValueError(f"unknown annotator-A label at item {item_index}: {label_a!r}")
        if label_b not in category_index:
            raise ValueError(f"unknown annotator-B label at item {item_index}: {label_b!r}")
        matrix[category_index[label_a]][category_index[label_b]] += 1

    return matrix


def _validate_confusion_matrix(
    confusion_matrix: Sequence[Sequence[int]],
    categories: Sequence[str],
) -> tuple[list[list[int]], list[str]]:
    ordered = _validate_categories(categories)
    if len(confusion_matrix) != len(ordered):
        raise ValueError("confusion matrix size must match categories")

    matrix: list[list[int]] = []
    for row in confusion_matrix:
        if len(row) != len(ordered):
            raise ValueError("confusion matrix must be square")
        normalized_row = list(row)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in normalized_row):
            raise ValueError("confusion matrix counts must be non-negative integers")
        matrix.append(normalized_row)
    return matrix, ordered


def _metric(value: float | None, status: str) -> JsonObject:
    return {"value": value, "status": status}


def reliability_report_from_confusion(
    confusion_matrix: Sequence[Sequence[int]],
    categories: Sequence[str],
) -> JsonObject:
    """Compute agreement metrics from a fixed-order confusion matrix."""
    matrix, ordered = _validate_confusion_matrix(confusion_matrix, categories)
    n_items = sum(sum(row) for row in matrix)
    row_totals = [sum(row) for row in matrix]
    column_totals = [sum(matrix[row][column] for row in range(len(ordered))) for column in range(len(ordered))]
    diagonal = sum(matrix[index][index] for index in range(len(ordered)))

    if n_items == 0:
        exact = _metric(None, "undefined_no_items")
        ac1 = _metric(None, "undefined_no_items")
        kappa = _metric(None, "undefined_no_items")
    else:
        observed_agreement = diagonal / n_items
        exact = _metric(observed_agreement, "ok")

        if len(ordered) == 1:
            ac1 = _metric(None, "undefined_single_category")
        else:
            averaged_marginals = [
                (row_totals[index] + column_totals[index]) / (2 * n_items)
                for index in range(len(ordered))
            ]
            chance_agreement_ac1 = sum(
                marginal * (1.0 - marginal) for marginal in averaged_marginals
            ) / (len(ordered) - 1)
            denominator_ac1 = 1.0 - chance_agreement_ac1
            if math.isclose(denominator_ac1, 0.0, abs_tol=1e-15):
                ac1 = _metric(None, "undefined_zero_expected_disagreement")
            else:
                ac1 = _metric(
                    (observed_agreement - chance_agreement_ac1) / denominator_ac1,
                    "ok",
                )

        chance_agreement_kappa = sum(
            row_totals[index] * column_totals[index] for index in range(len(ordered))
        ) / (n_items * n_items)
        denominator_kappa = 1.0 - chance_agreement_kappa
        if math.isclose(denominator_kappa, 0.0, abs_tol=1e-15):
            kappa = _metric(None, "undefined_zero_expected_disagreement")
        else:
            kappa = _metric(
                (observed_agreement - chance_agreement_kappa) / denominator_kappa,
                "ok",
            )

    category_agreement: list[JsonObject] = []
    for index, category in enumerate(ordered):
        denominator = row_totals[index] + column_totals[index]
        if denominator == 0:
            result = _metric(None, "undefined_zero_denominator")
        else:
            result = _metric(2.0 * matrix[index][index] / denominator, "ok")
        category_agreement.append({"category": category, **result})

    return {
        "categories": ordered,
        "n_items": n_items,
        "confusion_matrix": matrix,
        "row_totals": row_totals,
        "column_totals": column_totals,
        "exact_agreement": exact,
        "category_specific_agreement": category_agreement,
        "gwet_ac1": ac1,
        "cohen_kappa": kappa,
    }


def reliability_report(
    label_pairs: Iterable[LabelPair],
    categories: Sequence[str],
) -> JsonObject:
    """Build the confusion matrix and return JSON-serializable agreement metrics."""
    ordered = _validate_categories(categories)
    matrix = build_confusion_matrix(label_pairs, ordered)
    return reliability_report_from_confusion(matrix, ordered)


def _group_clustered_pairs(
    item_pairs: Iterable[ClusteredLabelPair],
) -> tuple[list[str], dict[str, list[LabelPair]]]:
    talk_ids: list[str] = []
    grouped: dict[str, list[LabelPair]] = {}
    for item_index, item in enumerate(item_pairs):
        try:
            talk_id, label_a, label_b = item
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"clustered item {item_index} must contain talk_id and two labels"
            ) from exc
        if not isinstance(talk_id, str) or not talk_id:
            raise ValueError(f"talk_id at item {item_index} must be a non-empty string")
        if talk_id not in grouped:
            talk_ids.append(talk_id)
            grouped[talk_id] = []
        grouped[talk_id].append((label_a, label_b))
    return talk_ids, grouped


def _draw_cluster_sample(
    talk_ids: Sequence[str],
    grouped_pairs: Mapping[str, Sequence[LabelPair]],
    rng: random.Random,
) -> tuple[list[str], list[LabelPair]]:
    drawn_talk_ids = [rng.choice(talk_ids) for _ in talk_ids]
    sampled_pairs = [pair for talk_id in drawn_talk_ids for pair in grouped_pairs[talk_id]]
    return drawn_talk_ids, sampled_pairs


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between zero and one")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    return ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight


def _bootstrap_interval(
    estimate: float | None,
    values: Sequence[float],
    *,
    confidence_level: float,
    n_resamples: int,
) -> JsonObject:
    if not values:
        return {
            "estimate": estimate,
            "lower": None,
            "upper": None,
            "valid_resamples": 0,
            "total_resamples": n_resamples,
            "status": "undefined_no_valid_resamples",
        }

    tail_probability = (1.0 - confidence_level) / 2.0
    status = "ok" if len(values) == n_resamples else "partial_undefined_resamples"
    return {
        "estimate": estimate,
        "lower": _percentile(values, tail_probability),
        "upper": _percentile(values, 1.0 - tail_probability),
        "valid_resamples": len(values),
        "total_resamples": n_resamples,
        "status": status,
    }


def _metric_value(report: Mapping[str, object], metric_name: str) -> float | None:
    metric = report[metric_name]
    if not isinstance(metric, Mapping):
        raise TypeError(f"metric {metric_name!r} is not a mapping")
    value = metric["value"]
    if value is not None and not isinstance(value, float):
        raise TypeError(f"metric {metric_name!r} is not numeric")
    return value


def cluster_bootstrap_percentile_ci(
    item_pairs: Iterable[ClusteredLabelPair],
    categories: Sequence[str],
    *,
    n_resamples: int = 2_000,
    seed: int = 0,
    confidence_level: float = 0.95,
) -> JsonObject:
    """Return percentile intervals after resampling complete talk clusters."""
    ordered = _validate_categories(categories)
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples <= 0:
        raise ValueError("n_resamples must be a positive integer")
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")

    talk_ids, grouped_pairs = _group_clustered_pairs(item_pairs)
    if not talk_ids:
        raise ValueError("item_pairs must contain at least one item")

    observed_pairs = [pair for talk_id in talk_ids for pair in grouped_pairs[talk_id]]
    estimate_report = reliability_report(observed_pairs, ordered)
    samples: dict[str, list[float]] = {
        "exact_agreement": [],
        "gwet_ac1": [],
        "cohen_kappa": [],
    }
    category_samples: list[list[float]] = [[] for _ in ordered]
    rng = random.Random(seed)

    # note (luojiaxuan): Each draw copies every item from each selected talk, including
    # repeated copies when a talk is sampled more than once, then rebuilds all metrics.
    for _ in range(n_resamples):
        _, sampled_pairs = _draw_cluster_sample(talk_ids, grouped_pairs, rng)
        sampled_report = reliability_report(sampled_pairs, ordered)
        for metric_name in samples:
            value = _metric_value(sampled_report, metric_name)
            if value is not None:
                samples[metric_name].append(value)

        sampled_categories = sampled_report["category_specific_agreement"]
        if not isinstance(sampled_categories, list):
            raise TypeError("category-specific agreement is not a list")
        for index, category_result in enumerate(sampled_categories):
            if not isinstance(category_result, Mapping):
                raise TypeError("category-specific agreement entry is not a mapping")
            value = category_result["value"]
            if value is not None:
                if not isinstance(value, float):
                    raise TypeError("category-specific agreement is not numeric")
                category_samples[index].append(value)

    metrics = {
        metric_name: _bootstrap_interval(
            _metric_value(estimate_report, metric_name),
            metric_samples,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
        )
        for metric_name, metric_samples in samples.items()
    }
    estimate_categories = estimate_report["category_specific_agreement"]
    if not isinstance(estimate_categories, list):
        raise TypeError("category-specific agreement is not a list")
    category_intervals = []
    for index, category in enumerate(ordered):
        category_estimate = estimate_categories[index]
        if not isinstance(category_estimate, Mapping):
            raise TypeError("category-specific agreement entry is not a mapping")
        estimate = category_estimate["value"]
        if estimate is not None and not isinstance(estimate, float):
            raise TypeError("category-specific agreement is not numeric")
        category_intervals.append(
            {
                "category": category,
                **_bootstrap_interval(
                    estimate,
                    category_samples[index],
                    confidence_level=confidence_level,
                    n_resamples=n_resamples,
                ),
            }
        )

    return {
        "method": "talk_cluster_percentile",
        "confidence_level": confidence_level,
        "seed": seed,
        "n_resamples": n_resamples,
        "n_talks": len(talk_ids),
        "n_items": len(observed_pairs),
        "metrics": metrics,
        "category_specific_agreement": category_intervals,
    }
