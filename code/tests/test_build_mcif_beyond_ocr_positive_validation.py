import pytest

from scripts import build_mcif_beyond_ocr_positive_validation as validation


def metric(screen_id, acoustic, positive):
    return {
        "screen_id": screen_id,
        "acoustic_condition": acoustic,
        "primary_positive": positive,
    }


def test_robust_screens_requires_both_acoustic_conditions():
    rows = [
        metric("both", "clean", True),
        metric("both", "babble_p5_s0", True),
        metric("clean-only", "clean", True),
        metric("clean-only", "babble_p5_s0", False),
    ]
    assert validation.robust_screens(rows, 1) == ["both"]
    with pytest.raises(ValueError, match="count differs"):
        validation.robust_screens(rows, 2)


def test_role_randomization_is_deterministic_and_namespaced():
    values = ["raw_image", "wrong_image", "ocr", "audio_only"]
    first = validation.deterministic_order(values, "seed", "role-a")
    assert first == validation.deterministic_order(values, "seed", "role-a")
    assert first != validation.deterministic_order(values, "seed", "role-b")


def test_pending_outcome_has_every_acoustic_slot():
    row = validation.pending_outcome("item")
    pairs = {
        (entry["acoustic_condition"], entry["slot"])
        for entry in row["slot_judgments"]
    }
    assert pairs == {
        (acoustic, slot)
        for acoustic in validation.ACOUSTIC_CONDITIONS
        for slot in ("A", "B", "C", "D")
    }
