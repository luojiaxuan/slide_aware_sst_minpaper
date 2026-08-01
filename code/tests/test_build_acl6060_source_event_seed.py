from scripts.build_acl6060_source_event_seed import select_rows, validate_seed


def make_contract():
    return {
        "schema_version": "test-v1",
        "selection": {
            "transition_candidate_states_per_talk": 2,
            "hash_random_states_per_talk": 2,
            "audio_window_before_evidence_sec": 5.0,
            "audio_window_after_evidence_sec": 60.0,
            "salt": "test-salt",
        },
        "annotation": {"audio_prefix_step_sec": 0.96},
    }


def make_observations(talk_id):
    return [
        {
            "talk_id": talk_id,
            "observation_id": f"{talk_id}:F{index:03d}",
            "frame_path": f"frames/{index}.jpg",
            "frame_sha256": str(index) * 64,
            "causal_availability_sec": float(index * 10),
        }
        for index in range(8)
    ]


def test_balanced_seed_is_deterministic_and_source_only():
    talk = {
        "talk_id": "talk-a",
        "split": "dev",
        "duration_sec": 65.0,
        "audio_sha256": "a" * 64,
    }
    observations = make_observations("talk-a")
    candidates = [
        {
            "talk_id": "talk-a",
            "current_observation_id": f"talk-a:F{index:03d}",
        }
        for index in range(4)
    ]

    first = select_rows(observations, candidates, talk, make_contract())
    second = select_rows(observations, candidates, talk, make_contract())

    assert [row["observation_id"] for row in first] == [
        row["observation_id"] for row in second
    ]
    assert len(first) == 4
    assert [row["selection_stratum"] for row in first].count("transition_candidate") == 2
    assert [row["selection_stratum"] for row in first].count("hash_random_observation") == 2
    assert all(row["event_status"] == "pending" for row in first)
    assert all("target" not in key for row in first for key in row)
    assert max(row["suggested_audio_window_end_sec"] for row in first) <= 65.0
    validate_seed(first, [talk], make_contract())


def test_validation_rejects_duplicate_observations():
    talk = {
        "talk_id": "talk-a",
        "split": "dev",
        "duration_sec": 65.0,
        "audio_sha256": "a" * 64,
    }
    observations = make_observations("talk-a")
    candidates = [
        {
            "talk_id": "talk-a",
            "current_observation_id": f"talk-a:F{index:03d}",
        }
        for index in range(4)
    ]
    rows = select_rows(observations, candidates, talk, make_contract())
    rows[-1]["observation_id"] = rows[0]["observation_id"]

    try:
        validate_seed(rows, [talk], make_contract())
    except ValueError as error:
        assert "observations must be unique" in str(error)
    else:
        raise AssertionError("Expected duplicate observation validation failure")
