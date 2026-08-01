import pytest

from scripts.analyze_probe_visual_control_matrix import analyze_contrasts, average_lagging


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
