import json
import wave
import zipfile
from pathlib import Path

import pytest

from scripts.build_acl6060_phase_a_views import assert_inference_safe, build_views
from scripts.freeze_phase_a_sources import (
    mcif_iwslt_archive_snapshot,
    next_link,
    sha256_file,
    wave_metadata,
    xml_talk_counts,
)


def write_wave(path: Path, frames: int = 160) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\x00\x00" * frames)


def test_wave_metadata_and_hash(tmp_path):
    path = tmp_path / "talk.wav"
    write_wave(path)

    metadata = wave_metadata(path)

    assert metadata["sample_rate_hz"] == 16000
    assert metadata["channels"] == 1
    assert metadata["duration_sec"] == 0.01
    assert len(sha256_file(path)) == 64


def test_xml_talk_counts(tmp_path):
    path = tmp_path / "source.xml"
    path.write_text(
        "<mteval><srcset><doc docid='talk-a'><abstract>A.</abstract>"
        "<seg id='1'>One.</seg><seg id='2'>Two.</seg></doc></srcset></mteval>",
        encoding="utf-8",
    )

    assert xml_talk_counts(path) == {"talk-a": 2}


def test_xml_talk_counts_recovers_malformed_xml(tmp_path):
    path = tmp_path / "source.xml"
    path.write_text(
        "<mteval><srcset><doc docid='talk-a'><seg id='1'>A & B</seg>"
        "<seg id='2'>Two.</seg></doc></srcset></mteval>",
        encoding="utf-8",
    )

    assert xml_talk_counts(path) == {"talk-a": 2}


def test_mcif_iwslt_archive_freezes_talk_ids_without_reading_references(tmp_path):
    path = tmp_path / "mcif.zip"
    talk_ids = {f"talk-{index:02d}" for index in range(21)}
    with zipfile.ZipFile(path, "w") as archive:
        for talk_id in talk_ids:
            archive.writestr(f"mcif-long-trans/audio/{talk_id}.wav", b"audio")
            archive.writestr(f"mcif-long-trans/pdf/{talk_id}.pdf", b"pdf")
        archive.writestr("mcif-long-trans/audio-segments.yaml", b"metadata")
        for language in ("en", "zh", "de", "it"):
            archive.writestr(f"mcif-long-trans/ref/{language}.txt", b"do-not-read")

    snapshot = mcif_iwslt_archive_snapshot(path, talk_ids | {"another-talk"})

    assert snapshot["talk_ids"] == sorted(talk_ids)
    assert snapshot["talk_count"] == 21
    assert snapshot["reference_content_inspected"] is False


def test_next_link_parser():
    header = '<https://example.test/page2>; rel="next", <https://example.test/page9>; rel="last"'
    assert next_link(header) == "https://example.test/page2"
    assert next_link(None) is None


def test_phase_a_views_keep_references_out_of_inference(tmp_path):
    root = tmp_path / "acl"
    paper_dir = tmp_path / "papers"
    (root / "dev" / "full_wavs").mkdir(parents=True)
    (root / "dev" / "text" / "xml").mkdir(parents=True)
    (root / "dev" / "text" / "tagged_terminology").mkdir(parents=True)
    paper_dir.mkdir()
    write_wave(root / "dev" / "full_wavs" / "talk-a.wav")
    (paper_dir / "talk-a.pdf").write_bytes(b"pdf")
    source_xml = root / "dev" / "text" / "xml" / "ACL.6060.dev.en-xx.en.xml"
    source_xml.write_text(
        "<mteval><srcset><doc docid='talk-a'><abstract>Public paper abstract.</abstract>"
        "<seg id='1'>Gold transcript.</seg></doc></srcset></mteval>",
        encoding="utf-8",
    )
    (root / "dev" / "text" / "xml" / "ACL.6060.dev.en-xx.zh.xml").write_text("<mteval/>")
    for language in ("en", "zh"):
        (root / "dev" / "text" / "tagged_terminology" / f"ACL.6060.dev.tagged.en-xx.{language}.txt").write_text(
            "[term]\n", encoding="utf-8"
        )
    talks = [
        {
            "split": "dev",
            "talk_id": "talk-a",
            "audio_relpath": "dev/full_wavs/talk-a.wav",
            "audio_sha256": "audio-hash",
            "duration_sec": 0.01,
            "paper_sha256": "paper-hash",
        }
    ]

    inference, scoring = build_views(root, paper_dir, talks, "dev")

    assert inference[0]["paper_abstract"] == "Public paper abstract."
    assert inference[0]["segment_count"] == 1
    assert "source_transcript" not in json.dumps(inference[0])
    assert "tagged_terminology" not in json.dumps(inference[0])
    assert scoring[0]["target_xml_path"].endswith("ACL.6060.dev.en-xx.zh.xml")


def test_inference_guard_rejects_reference_fields():
    with pytest.raises(ValueError, match="reference"):
        assert_inference_safe({"talk_id": "x", "reference_path": "hidden.txt"})
