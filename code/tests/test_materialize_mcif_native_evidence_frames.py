import hashlib
import importlib.util
from pathlib import Path

from PIL import Image
import pytest


def load_module():
    script = (
        Path(__file__).parents[1]
        / "scripts"
        / "materialize_mcif_native_evidence_frames.py"
    )
    spec = importlib.util.spec_from_file_location(
        "materialize_mcif_native_evidence_frames", script
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest(video_hash: str) -> dict:
    return {
        "dataset": "mcif",
        "subset": "test",
        "talk_count": 1,
        "reference_files_extracted": False,
        "reference_content_inspected": False,
        "upstream": {"revision": "rev"},
        "talks": [
            {
                "talk_id": "talk-a",
                "video_duration_sec": 2.0,
                "video_width": 8,
                "video_height": 4,
                "video_sha256": video_hash,
            }
        ],
    }


def state(frame: Path, frame_hash: str) -> dict:
    return {
        "talk_id": "talk-a",
        "state_id": 0,
        "availability_start_sec": 0.0,
        "availability_end_sec": 2.0,
        "evidence_nominal_timestamp_sec": 0.0,
        "evidence_frame_path": str(frame),
        "evidence_frame_sha256": frame_hash,
    }


def test_materialize_row_binds_native_frame_and_alignment(tmp_path):
    module = load_module()
    state_root = tmp_path / "states"
    video_root = tmp_path / "videos"
    output_root = tmp_path / "native"
    state_root.mkdir()
    video_root.mkdir()
    frame = state_root / "detector.jpg"
    video = video_root / "talk-a.mp4"
    Image.new("RGB", (4, 2), (10, 20, 30)).save(frame, quality=100)
    video.write_bytes(b"source-video")
    talk = source_manifest(sha256(video))["talks"][0]

    def fake_extractor(_video, _timestamp, output):
        Image.new("RGB", (8, 4), (10, 20, 30)).save(output)

    row = module.materialize_row(
        state(frame, sha256(frame)),
        talk,
        state_root=state_root,
        video_root=video_root,
        output_root=output_root,
        max_alignment_mae=2.0,
        extractor=fake_extractor,
    )
    assert row["id"] == "mcif:talk-a:S000"
    assert row["frame_path"] == "talks/talk-a/frames/state_000.png"
    assert row["frame_width"] == 8
    assert row["frame_height"] == 4
    assert row["detector_frame_alignment_mae_8bit"] <= 2.0
    assert not row["source_transcript_consumed"]
    assert not row["target_or_reference_consumed"]


def test_capture_schedule_delays_availability_until_frame_center():
    module = load_module()
    rows = [
        {
            "talk_id": "talk-a",
            "state_id": 0,
            "evidence_nominal_timestamp_sec": 0.0,
        },
        {
            "talk_id": "talk-a",
            "state_id": 1,
            "evidence_nominal_timestamp_sec": 1.0,
        },
    ]
    talks = {"talk-a": {"video_duration_sec": 2.0}}
    schedule = module.build_capture_schedule(rows, talks, capture_offset_sec=0.5)
    assert schedule[("talk-a", 0)] == {
        "capture_sec": 0.5,
        "availability_start_sec": 0.5,
        "availability_end_sec": 1.5,
    }
    assert schedule[("talk-a", 1)] == {
        "capture_sec": 1.5,
        "availability_start_sec": 1.5,
        "availability_end_sec": 2.0,
    }


@pytest.mark.parametrize("mutation", ["reference", "duplicate", "timestamp"])
def test_validate_state_inventory_fails_closed(mutation, tmp_path):
    module = load_module()
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"frame")
    rows = [state(frame, sha256(frame))]
    manifest = source_manifest("video-hash")
    if mutation == "reference":
        manifest["reference_content_inspected"] = True
    elif mutation == "duplicate":
        rows.append(dict(rows[0]))
    elif mutation == "timestamp":
        rows[0]["evidence_nominal_timestamp_sec"] = 1.0
    with pytest.raises(ValueError):
        module.validate_state_inventory(rows, manifest)


def test_materialize_row_rejects_hash_or_alignment_mismatch(tmp_path):
    module = load_module()
    state_root = tmp_path / "states"
    video_root = tmp_path / "videos"
    state_root.mkdir()
    video_root.mkdir()
    frame = state_root / "detector.jpg"
    video = video_root / "talk-a.mp4"
    Image.new("RGB", (4, 2), "black").save(frame)
    video.write_bytes(b"source-video")
    talk = source_manifest(sha256(video))["talks"][0]

    def white_extractor(_video, _timestamp, output):
        Image.new("RGB", (8, 4), "white").save(output)

    with pytest.raises(ValueError, match="hash mismatch"):
        module.materialize_row(
            state(frame, "bad"),
            talk,
            state_root=state_root,
            video_root=video_root,
            output_root=tmp_path / "bad-hash",
            max_alignment_mae=12.0,
            extractor=white_extractor,
        )
    with pytest.raises(ValueError, match="misalignment"):
        module.materialize_row(
            state(frame, sha256(frame)),
            talk,
            state_root=state_root,
            video_root=video_root,
            output_root=tmp_path / "bad-alignment",
            max_alignment_mae=12.0,
            extractor=white_extractor,
        )


def test_verify_source_videos_hashes_each_talk(tmp_path):
    module = load_module()
    video = tmp_path / "talk-a.mp4"
    video.write_bytes(b"source-video")
    manifest = source_manifest(sha256(video))
    talks = {"talk-a": manifest["talks"][0]}
    module.verify_source_videos(talks, tmp_path)
    talks["talk-a"]["video_sha256"] = "bad"
    with pytest.raises(ValueError, match="hash mismatch"):
        module.verify_source_videos(talks, tmp_path)


def test_summary_is_source_only_and_hash_bound():
    module = load_module()
    rows = [
        {
            "id": "mcif:talk-a:S000",
            "lecture_id": "talk-a",
            "frame_sha256": "frame-hash",
            "detector_frame_alignment_mae_8bit": 1.25,
            "source_transcript_consumed": False,
            "target_or_reference_consumed": False,
        }
    ]
    summary = module.build_summary(
        rows,
        source_manifest=source_manifest("video-hash"),
        causal_states_sha256="states-hash",
        max_alignment_mae=12.0,
        capture_offset_sec=0.5,
    )
    assert summary["causal_state_count"] == 1
    assert summary["talk_count"] == 1
    assert summary["alignment_mae_8bit"]["max"] == 1.25
    assert summary["timing_correction"]["capture_offset_sec"] == 0.5
    assert not summary["source_transcript_consumed"]
    assert not summary["target_or_reference_consumed"]
