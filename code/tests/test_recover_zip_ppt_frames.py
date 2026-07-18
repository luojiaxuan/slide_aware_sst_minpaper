import binascii
import importlib.util
import struct
import sys
import zlib
from pathlib import Path


def test_recover_ppt_frames_from_split_local_headers(tmp_path):
    module = _load_script()
    archive = (
        b"PK\x07\x08"
        + _local_file("train/001/FACE/001_FACE.mp4", b"face-video")
        + _local_file("train/001/PPT/001_PPT.mp4", b"ppt-video")
    )
    split = len(archive) - 4
    part1 = tmp_path / "train.z01"
    part2 = tmp_path / "train.z02"
    part1.write_bytes(archive[:split])
    part2.write_bytes(archive[split:])
    out_dir = tmp_path / "frames"
    manifest = tmp_path / "manifest.jsonl"

    def fake_extractor(video_path, frame_path, ffmpeg_bin, member_name):
        assert video_path.read_bytes() == b"ppt-video"
        assert ffmpeg_bin == "ffmpeg"
        assert member_name == "train/001/PPT/001_PPT.mp4"
        frame_path.write_bytes(b"jpg")

    counts = module.recover_ppt_frames(
        parts=[part1, part2],
        out_dir=out_dir,
        tmp_dir=tmp_path / "tmp",
        suffix="_PPT.mp4",
        contains="/PPT/",
        limit=None,
        overwrite=False,
        manifest_out=manifest,
        failure_log=None,
        ffmpeg_bin="ffmpeg",
        progress_every=0,
        frame_extractor=fake_extractor,
    )

    assert counts == {
        "members_seen": 2,
        "ppt_seen": 1,
        "frames_written": 1,
        "frames_skipped": 0,
        "failures": 0,
    }
    assert (out_dir / "001_PPT.jpg").read_bytes() == b"jpg"
    assert "train/001/PPT/001_PPT.mp4" in manifest.read_text(encoding="utf-8")


def _load_script():
    script = Path("scripts/recover_zip_ppt_frames.py")
    spec = importlib.util.spec_from_file_location("recover_zip_ppt_frames", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _local_file(name: str, data: bytes) -> bytes:
    compressor = zlib.compressobj(level=6, wbits=-15)
    compressed = compressor.compress(data) + compressor.flush()
    raw_name = name.encode("utf-8")
    header = struct.pack(
        "<IHHHHHIIIHH",
        0x04034B50,
        20,
        0x800,
        8,
        0,
        0,
        binascii.crc32(data) & 0xFFFFFFFF,
        len(compressed),
        len(data),
        len(raw_name),
        0,
    )
    return header + raw_name + compressed
