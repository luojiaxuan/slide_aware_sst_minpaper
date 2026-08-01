import pytest

from scripts.analyze_probe_visual_control_matrix import average_lagging


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
