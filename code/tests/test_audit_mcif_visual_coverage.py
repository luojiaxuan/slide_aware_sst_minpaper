from pathlib import Path

import numpy as np
from PIL import Image

from scripts.audit_mcif_visual_coverage import (
    adjacent_differences,
    make_contact_sheet,
    portable_summary,
    summarize_differences,
)


def write_frame(path: Path, value: int) -> None:
    Image.fromarray(np.full((90, 160, 3), value, dtype=np.uint8)).save(path)


def test_difference_proxy_detects_duplicate_and_large_change(tmp_path):
    frames = [tmp_path / f"{index}.jpg" for index in range(3)]
    write_frame(frames[0], 0)
    write_frame(frames[1], 0)
    write_frame(frames[2], 255)

    differences = adjacent_differences(frames)
    summary = summarize_differences(differences, interval_sec=10.0)

    assert differences[0] == 0.0
    assert differences[1] > 0.99
    assert summary["near_duplicate_pairs_lt_0_005"] == 1
    assert summary["large_change_candidates_ge_0_08"] == 1


def test_contact_sheet_and_portable_summary(tmp_path):
    frames = [tmp_path / f"{index}.jpg" for index in range(3)]
    for index, frame in enumerate(frames):
        write_frame(frame, index * 80)
    sheet = tmp_path / "sheet.jpg"

    make_contact_sheet(frames, sheet, interval_sec=10.0, columns=2, cell_size=(160, 90))
    summary = portable_summary(
        [
            {
                "talk_id": "talk-a",
                "duration_sec": 30.0,
                "sample_count": 3,
                "sample_interval_sec": 10.0,
                "difference_proxy": summarize_differences([0.0, 1.0], interval_sec=10.0),
            }
        ],
        Path("inference.jsonl"),
        "ResearchStudio/data/mcif/qa",
    )

    assert sheet.is_file()
    assert Image.open(sheet).size == (320, 224)
    assert summary["talk_count"] == 1
    assert summary["large_change_candidates_ge_0_08"] == 1
    assert "diagnostics only" in summary["interpretation"]
