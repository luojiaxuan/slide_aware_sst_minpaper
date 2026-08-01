import wave

import numpy as np

from scripts.build_acl6060_source_annotation_manifest import build_rows
from scripts.build_acl6060_simulstream_inputs import sha256_file


def write_wav(path, values, sample_rate=1000):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(np.asarray(values, dtype="<i2").tobytes())


def test_build_rows_reads_source_xml_and_recovers_offsets(tmp_path):
    root = tmp_path / "acl"
    full = np.arange(100, dtype=np.int16)
    full_path = root / "dev" / "full_wavs" / "talk.110.wav"
    write_wav(full_path, full)
    write_wav(root / "dev" / "segmented_wavs" / "gold" / "sent_1.wav", full[10:20])
    write_wav(root / "dev" / "segmented_wavs" / "gold" / "sent_2.wav", full[40:55])
    xml_path = root / "dev" / "text" / "xml" / "ACL.6060.dev.en-xx.en.xml"
    xml_path.parent.mkdir(parents=True)
    xml_path.write_text(
        '<mteval><doc docid="talk.110"><seg id="1">first source</seg>'
        '<seg id="2">second source</seg></doc></mteval>'
    )
    talk_manifest = [
        {
            "talk_id": "talk.110",
            "split": "dev",
            "segment_count": 2,
            "audio_sha256": sha256_file(full_path),
        }
    ]

    rows = build_rows(root, "dev", talk_manifest)

    assert [row["source_text"] for row in rows] == ["first source", "second source"]
    assert [row["offset_sec"] for row in rows] == [0.01, 0.04]
    assert [row["duration_sec"] for row in rows] == [0.01, 0.015]
    assert all("target" not in key for row in rows for key in row)
    assert all("reference" not in key for row in rows for key in row)
