import wave
from pathlib import Path

import numpy as np
import pytest

from scripts.materialize_full_talk_corruptions import (
    activity_mask,
    apply_rir,
    assert_inference_safe,
    clean_audio_fields,
    circular_audio,
    energy_vad_v1,
    materialize_condition,
    mix_at_snr,
    portable_row,
    read_pcm16_wav,
    sha256_file,
    stable_rng,
    stable_seed,
    write_pcm16_wav,
)


def test_activity_mask_uses_source_intervals():
    mask = activity_mask(
        [{"offset_sec": 0.1, "end_sec": 0.3}, {"offset_sec": 0.5, "end_sec": 0.7}],
        sample_rate=10,
        sample_count=10,
    )

    assert mask.tolist() == [False, True, True, False, False, True, True, False, False, False]


def test_energy_vad_and_flat_acl_audio_schema():
    samples = np.zeros(32000, dtype=np.float32)
    samples[8000:24000] = 0.2

    mask, definition = energy_vad_v1(samples, sample_rate=16000)
    path, digest = clean_audio_fields({"audio_path": "/tmp/a.wav", "audio_sha256": "abc"})

    assert np.mean(mask[9000:23000]) == 1.0
    assert np.mean(mask[:6000]) == 0.0
    assert definition["name"] == "energy_vad_v1"
    assert path == Path("/tmp/a.wav")
    assert digest == "abc"


def test_mix_reaches_target_snr_and_guards_peak():
    clean = np.full(16000, 0.5, dtype=np.float32)
    noise = np.linspace(-1.0, 1.0, 16000, dtype=np.float32)
    active = np.ones(len(clean), dtype=bool)

    mixed, measurements = mix_at_snr(clean, noise, active, target_snr_db=5.0)

    assert measurements["achieved_snr_db"] == pytest.approx(5.0, abs=1e-6)
    assert np.max(np.abs(mixed)) <= 0.990001


def test_rir_is_trimmed_without_changing_output_length():
    clean = np.zeros(4000, dtype=np.float32)
    clean[100:3000] = 0.2
    active = clean != 0
    rir = np.zeros(800, dtype=np.float32)
    rir[200] = 1.0
    rir[400] = 0.3

    reverberant, measurements = apply_rir(clean, active, rir)

    assert len(reverberant) == len(clean)
    assert measurements["rir_leading_trim_samples"] == 184
    assert np.max(np.abs(reverberant)) <= 0.990001


def test_circular_audio_and_rng_are_deterministic():
    source = np.array([1, 2, 3], dtype=np.float32)

    assert circular_audio(source, 7, 2).tolist() == [3, 1, 2, 3, 1, 2, 3]
    faded = circular_audio(source, 7, 2, wrap_fade_samples=1)
    assert faded.tolist() == [1.5, 0.5, 2.0, 1.5, 0.5, 2.0, 3.0]
    assert stable_rng(7, "talk", "condition").integers(100000) == stable_rng(
        7, "talk", "condition"
    ).integers(100000)
    assert stable_seed(7, "talk", "condition") == 10744359677044736681


def test_wav_round_trip_and_inference_guard(tmp_path):
    path = tmp_path / "audio.wav"
    values = np.linspace(-0.5, 0.5, 1000, dtype=np.float32)
    write_pcm16_wav(path, values, 16000)

    loaded, sample_rate = read_pcm16_wav(path)

    assert sample_rate == 16000
    assert loaded == pytest.approx(values, abs=1 / 32768)
    with wave.open(str(path), "rb") as audio:
        assert audio.getnframes() == 1000
    with pytest.raises(ValueError, match="Forbidden field"):
        assert_inference_safe({"target_translation": "hidden"})


def test_materialize_babble_condition_from_flat_inference_row(tmp_path):
    clean_path = tmp_path / "clean.wav"
    clean = np.sin(np.linspace(0, 100, 32000, dtype=np.float32)) * 0.2
    write_pcm16_wav(clean_path, clean, 16000)
    pool = []
    for index in range(5):
        source_path = tmp_path / f"source-{index}.wav"
        source = np.sin(np.linspace(index, index + 300, 40000, dtype=np.float32)) * 0.1
        write_pcm16_wav(source_path, source, 16000)
        pool.append(
            {
                "source_id": f"source-{index}",
                "category": "babble_speech",
                "split": "development",
                "path": str(source_path),
                "sha256": sha256_file(source_path),
            }
        )
    talk = {
        "talk_id": "talk-a",
        "audio_path": str(clean_path),
        "audio_sha256": sha256_file(clean_path),
    }

    result = materialize_condition(
        talk,
        {"condition_id": "babble_p5_s0", "kind": "babble", "snr_db": 5},
        pool,
        "development",
        20260801,
        tmp_path / "output",
    )
    portable = portable_row(result, "ResearchStudio/corruptions")

    assert result["measurements"]["achieved_snr_db"] == pytest.approx(5.0, abs=1e-6)
    assert result["activity_definition"]["name"] == "energy_vad_v1"
    assert len(result["sources"]) == 5
    assert Path(result["output_audio_path"]).is_file()
    assert str(tmp_path) not in str(portable)
