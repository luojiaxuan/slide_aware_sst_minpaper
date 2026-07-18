import json
import subprocess
import sys
from pathlib import Path


def test_build_chinese_lips_manifest_from_csv(tmp_path):
    extracted = tmp_path / "extracted"
    (extracted / "processed_train").mkdir(parents=True)
    (extracted / "first_frames").mkdir()
    (extracted / "processed_train" / "001_24_F_JKYS_044.wav").write_bytes(b"")
    (extracted / "processed_train" / "001_24_F_JKYS_044.mp4").write_bytes(b"")
    (extracted / "first_frames" / "001_24_F_JKYS_044_PPT.jpg").write_bytes(b"")
    meta = tmp_path / "meta_train.csv"
    meta.write_text(
        "ID,TOPIC,WAV,PPT,FACE,TEXT\n"
        "001_24_F_JKYS_044,JKYS,src.wav,src_ppt.mp4,src_face.mp4,我们讨论健康。\n",
        encoding="utf-8",
    )
    out = tmp_path / "manifest.jsonl"

    subprocess.run(
        [
            sys.executable,
            "scripts/build_chinese_lips_manifest.py",
            "--meta-csv",
            str(meta),
            "--extracted-root",
            str(extracted),
            "--out",
            str(out),
            "--split",
            "train",
        ],
        check=True,
    )

    row = json.loads(out.read_text(encoding="utf-8"))
    assert row["id"] == "001_24_F_JKYS_044"
    assert row["zh_transcript"] == "我们讨论健康。"
    assert row["visual_context"]["ocr_text"] == []
    assert "OCR not provided" in row["visual_context"]["scene_summary"]
