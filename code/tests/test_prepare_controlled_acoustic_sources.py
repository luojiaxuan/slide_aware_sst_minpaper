import io
import json
import tarfile
import wave
import zipfile
from pathlib import Path

import pytest

from scripts.prepare_controlled_acoustic_sources import (
    build_source_rows,
    extract_tar_members,
    extract_zip_members,
    portable_summary,
    safe_member_path,
    select_musan_members,
)


def wav_bytes(frames: int = 1600, channels: int = 1) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\x00\x00" * frames * channels)
    return buffer.getvalue()


def test_selection_is_disjoint_and_deterministic():
    inventory = [
        {"member": f"musan/speech/source-{index}.wav", "bytes": 1000}
        for index in range(8)
    ]
    selection = {
        "salt": "test-v1",
        "categories": {
            "babble_speech": {
                "prefix": "musan/speech/",
                "min_bytes": 0,
                "development": 2,
                "confirmatory": 3,
            }
        },
    }

    first = select_musan_members(inventory, selection)
    second = select_musan_members(list(reversed(inventory)), selection)

    assert first == second
    assert len(first) == 5
    assert {row["member"] for row in first if row["split"] == "development"}.isdisjoint(
        {row["member"] for row in first if row["split"] == "confirmatory"}
    )


def test_selective_extract_and_portable_rows(tmp_path):
    tar_path = tmp_path / "musan.tar.gz"
    zip_path = tmp_path / "rirs.zip"
    speech = wav_bytes()
    rir = wav_bytes(channels=2)
    with tarfile.open(tar_path, "w:gz") as archive:
        info = tarfile.TarInfo("musan/speech/a.wav")
        info.size = len(speech)
        archive.addfile(info, io.BytesIO(speech))
        hidden = tarfile.TarInfo("musan/speech/not-selected.wav")
        hidden.size = len(speech)
        archive.addfile(hidden, io.BytesIO(speech))
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("RIRS/rir.wav", rir)

    output = tmp_path / "sources"
    musan_rows = [
        {
            "member": "musan/speech/a.wav",
            "bytes": len(speech),
            "category": "babble_speech",
            "split": "development",
        }
    ]
    rir_rows = [
        {
            "member": "RIRS/rir.wav",
            "bytes": len(rir),
            "category": "rir",
            "split": "confirmatory",
            "channel": 1,
        }
    ]
    extract_tar_members(tar_path, musan_rows, output)
    extract_zip_members(zip_path, rir_rows, output)
    rows = build_source_rows(musan_rows, rir_rows, output, "ResearchStudio/noise")
    summary = portable_summary(
        {"sources": {}, "selection": {}},
        rows,
        "ResearchStudio/noise",
    )

    assert len(rows) == 2
    assert not (output / "musan/speech/not-selected.wav").exists()
    assert next(row for row in rows if row["category"] == "rir")["selected_channel"] == 1
    assert summary["development_confirmatory_overlap"] is False
    assert str(tmp_path) not in json.dumps(summary)


def test_rejects_unsafe_member():
    with pytest.raises(ValueError, match="Unsafe archive member"):
        safe_member_path("../escape.wav")
