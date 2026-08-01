import numpy as np
from PIL import Image

from scripts.detect_mcif_visual_state_candidates import (
    build_causal_states,
    build_candidate_states,
    combine_causal_states,
    group_candidate_indices,
    pair_metrics,
)


def test_patch_metric_ignores_one_dynamic_patch_but_detects_full_change():
    original = np.zeros((54, 96), dtype=np.float32)
    speaker_motion = original.copy()
    speaker_motion[:9, :12] = 1.0
    new_slide = np.ones((54, 96), dtype=np.float32)

    local = pair_metrics(original, speaker_motion)
    global_change = pair_metrics(original, new_slide)

    assert local["patch_diff_p75"] == 0.0
    assert local["changed_patch_fraction_ge_0_05"] < 0.03
    assert global_change["patch_diff_p75"] == 1.0
    assert global_change["changed_patch_fraction_ge_0_05"] == 1.0


def test_candidate_groups_and_conservative_unlock():
    pair_rows = []
    for index in range(1, 9):
        is_candidate = index in {2, 3, 7}
        pair_rows.append(
            {
                "current_sample_index": index,
                "patch_diff_p75": 0.8 if is_candidate else 0.0,
                "changed_patch_fraction_ge_0_05": 1.0 if is_candidate else 0.0,
                "is_candidate": is_candidate,
            }
        )

    assert group_candidate_indices([2, 3, 7], max_gap_samples=2) == [[2, 3], [7]]
    states = build_candidate_states(
        pair_rows,
        interval_sec=1.0,
        debounce_samples=2,
        stable_pairs=2,
        stability_p75_threshold=0.02,
        max_confirmation_samples=4,
    )

    assert len(states) == 2
    assert states[0]["transition_window_start_sec"] == 2.0
    assert states[0]["transition_window_end_sec"] == 3.0
    assert states[0]["unlock_sec"] == 5.0
    assert states[0]["stable_confirmation"] is True
    assert states[1]["unlock_sec"] is None


def test_causal_states_unlock_only_after_confirmation(tmp_path):
    frames = []
    for index in range(6):
        path = tmp_path / f"frame_{index}.jpg"
        Image.fromarray(np.full((12, 12, 3), index, dtype=np.uint8)).save(path)
        frames.append(path)
    candidates = [
        {
            "state_id": 1,
            "transition_window_start_sec": 2.0,
            "transition_window_end_sec": 2.0,
            "unlock_sec": 4.0,
            "stable_confirmation": True,
        },
        {
            "state_id": 2,
            "transition_window_start_sec": 5.0,
            "transition_window_end_sec": 5.0,
            "unlock_sec": None,
            "stable_confirmation": False,
        },
    ]

    states = build_causal_states("talk-a", frames, candidates, 1.0, 6.0)

    assert len(states) == 2
    assert states[0]["availability_start_sec"] == 0.0
    assert states[0]["availability_end_sec"] == 4.0
    assert states[1]["availability_start_sec"] == 4.0
    assert states[1]["evidence_sample_index"] == 4
    assert states[1]["transition_window_start_sec"] == 2.0
    assert len(states[1]["evidence_frame_sha256"]) == 64


def test_combine_causal_states_preserves_talk_order_and_checks_intervals(tmp_path):
    for talk_id in ("talk-b", "talk-a"):
        path = tmp_path / "talks" / talk_id / "causal_states.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(
            "\n".join(
                [
                    '{"talk_id":"%s","state_id":0,"availability_start_sec":0.0,"availability_end_sec":2.0}'
                    % talk_id,
                    '{"talk_id":"%s","state_id":1,"availability_start_sec":2.0,"availability_end_sec":3.0}'
                    % talk_id,
                ]
            )
            + "\n"
        )

    rows = combine_causal_states(tmp_path, ["talk-b", "talk-a"])

    assert [row["talk_id"] for row in rows] == ["talk-b", "talk-b", "talk-a", "talk-a"]
