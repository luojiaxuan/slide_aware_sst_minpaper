import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from scripts.build_acl6060_visual_timeline import (
    collect_observations,
    parse_frame_timestamp,
    portable_candidate,
    score_observation_pairs,
    select_negative_audit_rows,
)


def write_frame(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((54, 96, 3), value, dtype=np.uint8)).save(path)


def test_parse_frame_timestamp_is_strict():
    assert parse_frame_timestamp(Path("frame_12.34.jpg")) == 12.34
    with pytest.raises(ValueError, match="Unexpected frame filename"):
        parse_frame_timestamp(Path("slide_12.34.jpg"))


def test_collect_observations_is_transcript_free_and_conservative(tmp_path):
    root = tmp_path / "image_frames_dev"
    write_frame(root / "110" / "frame_4.50.jpg", 0)
    write_frame(root / "110" / "frame_9.75.jpg", 255)
    talk = {
        "talk_id": "2022.acl-long.110",
        "split": "dev",
        "duration_sec": 12.0,
        "segment_count": 2,
    }

    local, portable = collect_observations(talk, root, "frame-cache")

    assert [row["observed_at_sec"] for row in local] == [4.5, 9.75]
    assert local[0]["causal_availability_sec"] == 4.5
    assert local[0]["availability_start_sec"] == 4.5
    assert local[0]["availability_end_sec"] == 9.75
    assert local[1]["availability_end_sec"] == 12.0
    assert local[0]["state_policy"] == "every_observation_no_backdating"
    assert portable[0]["frame_path"] == "frame-cache/image_frames_dev/110/frame_4.50.jpg"
    assert "sentence" not in json.dumps(portable)


def test_collect_observations_rejects_count_mismatch(tmp_path):
    root = tmp_path / "image_frames_dev"
    write_frame(root / "110" / "frame_4.50.jpg", 0)
    talk = {
        "talk_id": "2022.acl-long.110",
        "split": "dev",
        "duration_sec": 12.0,
        "segment_count": 2,
    }

    with pytest.raises(ValueError, match="Frame count mismatch"):
        collect_observations(talk, root, "frame-cache")


def test_pair_candidate_uses_observation_interval_and_current_unlock(tmp_path):
    paths = []
    for name, value in (("frame_1.00.jpg", 0), ("frame_5.00.jpg", 255), ("frame_8.00.jpg", 255)):
        path = tmp_path / name
        write_frame(path, value)
        paths.append(path)
    observations = [
        {
            "talk_id": "talk-a",
            "observation_id": f"talk-a:F{index:03d}",
            "observation_index": index,
            "observed_at_sec": timestamp,
            "frame_path": str(path),
            "frame_sha256": str(index) * 64,
        }
        for index, (timestamp, path) in enumerate(zip((1.0, 5.0, 8.0), paths))
    ]

    pairs, candidates = score_observation_pairs(
        observations,
        p75_threshold=0.03,
        changed_patch_fraction_threshold=0.12,
    )

    assert len(pairs) == 2
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["transition_window_start_sec"] == 1.0
    assert candidate["transition_window_end_sec"] == 5.0
    assert candidate["conservative_unlock_sec"] == 5.0
    assert candidate["review_decision"] == "pending"

    portable = portable_candidate(
        candidate,
        {str(paths[0]): "old.jpg", str(paths[1]): "new.jpg"},
    )
    assert portable["previous_frame_path"] == "old.jpg"
    assert portable["current_frame_path"] == "new.jpg"


def test_negative_audit_combines_hard_and_stable_random_rows():
    observations = [
        {
            "talk_id": "talk-a",
            "observation_id": f"talk-a:F{index:03d}",
            "observation_index": index,
            "observed_at_sec": float(index),
            "frame_path": f"frame-{index}.jpg",
            "frame_sha256": str(index) * 64,
        }
        for index in range(7)
    ]
    pairs = [
        {
            "talk_id": "talk-a",
            "previous_observation_id": f"talk-a:F{index - 1:03d}",
            "current_observation_id": f"talk-a:F{index:03d}",
            "patch_diff_p75": index / 100.0,
            "changed_patch_fraction_ge_0_05": 0.0,
            "is_candidate": index == 6,
        }
        for index in range(1, 7)
    ]

    first = select_negative_audit_rows(
        observations,
        pairs,
        p75_threshold=0.10,
        changed_patch_fraction_threshold=0.20,
        hard_count=2,
        random_count=2,
    )
    second = select_negative_audit_rows(
        observations,
        pairs,
        p75_threshold=0.10,
        changed_patch_fraction_threshold=0.20,
        hard_count=2,
        random_count=2,
    )

    assert [row["current_observation_id"] for row in first[:2]] == [
        "talk-a:F005",
        "talk-a:F004",
    ]
    assert [row["current_observation_id"] for row in first] == [
        row["current_observation_id"] for row in second
    ]
    assert {row["selection"] for row in first} == {
        "hard_negative",
        "hash_random_negative",
    }
    assert all(not row["is_candidate"] for row in first)
