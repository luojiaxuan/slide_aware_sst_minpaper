import json
import wave
import zipfile
from pathlib import Path

import pytest

from scripts.materialize_mcif_subset import (
    assert_inference_safe,
    build_inference_rows,
    extract_inference_files,
    load_segments,
    portable_summary,
    sha256_file,
)


def write_wave(path: Path, duration_sec: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\x00\x00" * int(16000 * duration_sec))


def make_archive(path: Path, talk_ids: list[str]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for talk_id in talk_ids:
            wave_path = path.parent / f"{talk_id}.wav"
            write_wave(wave_path)
            archive.write(wave_path, f"mcif-long-trans/audio/{talk_id}.wav")
            archive.writestr(f"mcif-long-trans/pdf/{talk_id}.pdf", b"%PDF-test")
        rows = [
            {"wav": f"{talk_id}.wav", "offset": 0.1, "duration": 0.5, "speaker_id": index}
            for index, talk_id in enumerate(talk_ids)
        ]
        import yaml

        archive.writestr("mcif-long-trans/audio-segments.yaml", yaml.safe_dump(rows))
        archive.writestr("mcif-long-trans/ref/zh.txt", "target must stay hidden")


def test_extract_inference_files_keeps_references_out(tmp_path):
    archive = tmp_path / "mcif.zip"
    output = tmp_path / "materialized"
    make_archive(archive, ["talk-a", "talk-b"])

    extracted = extract_inference_files(archive, ["talk-a", "talk-b"], output)

    assert len(extracted) == 5
    assert (output / "audio" / "talk-a.wav").is_file()
    assert (output / "pdf" / "talk-b.pdf").is_file()
    assert not (output / "ref").exists()
    assert "target must stay hidden" not in "".join(path.read_text(errors="ignore") for path in output.rglob("*.*"))


def test_build_inference_rows_and_portable_summary(tmp_path):
    archive = tmp_path / "mcif.zip"
    output = tmp_path / "materialized"
    talk_ids = ["talk-a"]
    make_archive(archive, talk_ids)
    extract_inference_files(archive, talk_ids, output)
    video = output / "video" / "talk-a.mp4"
    video.parent.mkdir()
    video.write_bytes(b"video")
    segments = load_segments(output / "metadata" / "audio-segments.yaml", talk_ids)

    rows = build_inference_rows(
        talk_ids,
        output,
        segments,
        "owner/mcif",
        "revision",
        video_probe=lambda path: {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "duration_sec": 1.0,
            "width": 1280,
            "height": 720,
            "avg_frame_rate": "30/1",
        },
    )
    summary = portable_summary(
        rows,
        {
            "repo": "owner/mcif",
            "revision": "revision",
            "license": "cc-by-4.0",
            "iwslt2026_archive": {"sha256": "archive-hash"},
        },
        "ResearchStudio/data/mcif",
    )

    assert rows[0]["talk_id"] == "talk-a"
    assert rows[0]["alignment"]["segment_count"] == 1
    assert summary["talk_count"] == 1
    assert summary["reference_files_extracted"] is False
    assert "reference" not in json.dumps(rows)


def test_inference_guard_rejects_leakage():
    with pytest.raises(ValueError, match="Forbidden inference field"):
        assert_inference_safe({"talk_id": "x", "target_translation": "hidden"})
