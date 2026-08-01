import wave

import pytest

from scripts.build_acl6060_simulstream_inputs import locate_segment


def write_wave(path, samples):
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(10)
        audio.writeframes(b"".join(int(sample).to_bytes(2, "little", signed=True) for sample in samples))


def test_locate_segment_recovers_exact_full_wav_offset(tmp_path):
    full = tmp_path / "talk.wav"
    segment = tmp_path / "sent_1.wav"
    write_wave(full, [0, 1, 2, 3, 4, 5, 6])
    write_wave(segment, [2, 3, 4])

    timing, cursor = locate_segment(full, segment, 0)

    assert timing == {"offset": 0.2, "duration": 0.3, "speaker_id": "NA", "wav": "talk.wav"}
    assert cursor == 6


def test_locate_segment_rejects_non_slice(tmp_path):
    full = tmp_path / "talk.wav"
    segment = tmp_path / "sent_1.wav"
    write_wave(full, [0, 1, 2, 3])
    write_wave(segment, [9, 9])

    with pytest.raises(ValueError, match="not an exact slice"):
        locate_segment(full, segment, 0)
