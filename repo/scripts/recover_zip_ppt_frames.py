#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, BinaryIO


LOCAL_FILE_HEADER = b"PK\x03\x04"
CENTRAL_DIRECTORY_HEADER = b"PK\x01\x02"
END_OF_CENTRAL_DIRECTORY = b"PK\x05\x06"
CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class LocalZipMember:
    name: str
    compression: int
    compressed_size: int
    uncompressed_size: int
    header_offset: int


class SplitReader:
    def __init__(self, parts: list[Path]) -> None:
        if not parts:
            raise ValueError("At least one split archive part is required")
        self.parts = parts
        self.part_index = -1
        self.current: BinaryIO | None = None
        self.current_size = 0
        self.offset = 0
        self._advance_part()

    def close(self) -> None:
        if self.current is not None:
            self.current.close()
            self.current = None

    def read(self, size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while remaining > 0 and self.current is not None:
            chunk = self.current.read(remaining)
            if chunk:
                chunks.append(chunk)
                remaining -= len(chunk)
                self.offset += len(chunk)
                continue
            self._advance_part()
        return b"".join(chunks)

    def read_exact(self, size: int) -> bytes:
        data = self.read(size)
        if len(data) != size:
            raise EOFError(f"Expected {size} bytes, got {len(data)}")
        return data

    def skip(self, size: int) -> None:
        remaining = size
        while remaining > 0 and self.current is not None:
            position = self.current.tell()
            available = self.current_size - position
            if available <= 0:
                self._advance_part()
                continue
            step = min(remaining, available)
            self.current.seek(step, 1)
            self.offset += step
            remaining -= step
        if remaining:
            raise EOFError(f"Archive ended with {remaining} bytes left to skip")

    def _advance_part(self) -> None:
        if self.current is not None:
            self.current.close()
        self.part_index += 1
        if self.part_index >= len(self.parts):
            self.current = None
            self.current_size = 0
            return
        part = self.parts[self.part_index]
        self.current = part.open("rb")
        self.current_size = part.stat().st_size


def iter_local_members(parts: list[Path]) -> list[LocalZipMember]:
    reader = SplitReader(parts)
    members: list[LocalZipMember] = []
    try:
        while True:
            member = read_next_member(reader)
            if member is None:
                break
            members.append(member)
            reader.skip(member.compressed_size)
    finally:
        reader.close()
    return members


def read_next_member(reader: SplitReader) -> LocalZipMember | None:
    signature, offset = _find_next_signature(reader)
    if signature is None or signature in {CENTRAL_DIRECTORY_HEADER, END_OF_CENTRAL_DIRECTORY}:
        return None
    if signature != LOCAL_FILE_HEADER:
        raise ValueError(f"Unexpected ZIP signature {signature!r} at byte offset {offset}")

    fields = struct.unpack("<HHHHHIIIHH", reader.read_exact(26))
    _, flag, compression, _, _, _, compressed_size, uncompressed_size, name_len, extra_len = fields
    raw_name = reader.read_exact(name_len)
    reader.skip(extra_len)
    if flag & 0x8:
        raise ValueError(f"{raw_name!r} uses data descriptors; streaming recovery cannot skip it safely")
    if compressed_size == 0xFFFFFFFF or uncompressed_size == 0xFFFFFFFF:
        raise ValueError(f"{raw_name!r} uses ZIP64 local sizes; this script expects 32-bit local sizes")
    encoding = "utf-8" if flag & 0x800 else "cp437"
    name = raw_name.decode(encoding, errors="replace")
    return LocalZipMember(
        name=name,
        compression=compression,
        compressed_size=compressed_size,
        uncompressed_size=uncompressed_size,
        header_offset=offset,
    )


def recover_ppt_frames(
    parts: list[Path],
    out_dir: Path,
    tmp_dir: Path,
    suffix: str,
    contains: str,
    limit: int | None,
    overwrite: bool,
    manifest_out: Path | None,
    failure_log: Path | None,
    ffmpeg_bin: str,
    progress_every: int,
    frame_extractor: Callable[[Path, Path, str, str], None] | None = None,
) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    if frame_extractor is None:
        frame_extractor = extract_first_frame

    counts = {
        "members_seen": 0,
        "ppt_seen": 0,
        "frames_written": 0,
        "frames_skipped": 0,
        "failures": 0,
    }
    manifest_handle = manifest_out.open("w", encoding="utf-8") if manifest_out else None
    failure_handle = failure_log.open("w", encoding="utf-8") if failure_log else None
    reader = SplitReader(parts)
    try:
        with tempfile.TemporaryDirectory(prefix="recover_zip_ppt_", dir=tmp_dir) as work:
            work_dir = Path(work)
            while True:
                member = read_next_member(reader)
                if member is None:
                    break
                counts["members_seen"] += 1
                normalized_name = member.name.replace("\\", "/")
                is_match = normalized_name.endswith(suffix) and contains in normalized_name
                if not is_match:
                    reader.skip(member.compressed_size)
                    _print_progress(counts, progress_every)
                    continue

                counts["ppt_seen"] += 1
                frame_path = out_dir / f"{Path(normalized_name).stem}.jpg"
                if frame_path.exists() and not overwrite:
                    reader.skip(member.compressed_size)
                    counts["frames_skipped"] += 1
                    _write_manifest(manifest_handle, member, frame_path, parts, "skipped")
                    if limit and counts["ppt_seen"] >= limit:
                        break
                    _print_progress(counts, progress_every)
                    continue

                tmp_video = work_dir / f"{Path(normalized_name).stem}.mp4"
                try:
                    write_member_payload(reader, member, tmp_video)
                    frame_extractor(tmp_video, frame_path, ffmpeg_bin, normalized_name)
                    counts["frames_written"] += 1
                    _write_manifest(manifest_handle, member, frame_path, parts, "written")
                except Exception as exc:  # pragma: no cover - exercised by real corrupt media.
                    counts["failures"] += 1
                    if frame_path.exists():
                        frame_path.unlink()
                    _write_failure(failure_handle, member, exc)
                    print(f"Failed to recover {normalized_name}: {exc}", file=sys.stderr)
                finally:
                    if tmp_video.exists():
                        tmp_video.unlink()

                if limit and counts["ppt_seen"] >= limit:
                    break
                _print_progress(counts, progress_every)
    finally:
        reader.close()
        if manifest_handle:
            manifest_handle.close()
        if failure_handle:
            failure_handle.close()
    return counts


def write_member_payload(reader: SplitReader, member: LocalZipMember, out_path: Path) -> None:
    remaining = member.compressed_size
    with out_path.open("wb") as out:
        if member.compression == 0:
            while remaining:
                chunk = reader.read(min(CHUNK_SIZE, remaining))
                if not chunk:
                    raise EOFError(f"Unexpected EOF while reading {member.name}")
                out.write(chunk)
                remaining -= len(chunk)
            return
        if member.compression != 8:
            reader.skip(remaining)
            raise ValueError(f"Unsupported ZIP compression method {member.compression} for {member.name}")

        decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
        written = 0
        while remaining:
            chunk = reader.read(min(CHUNK_SIZE, remaining))
            if not chunk:
                raise EOFError(f"Unexpected EOF while reading {member.name}")
            remaining -= len(chunk)
            data = decompressor.decompress(chunk)
            written += len(data)
            out.write(data)
        tail = decompressor.flush()
        written += len(tail)
        out.write(tail)
    if member.uncompressed_size and written != member.uncompressed_size:
        raise ValueError(
            f"Recovered {written} bytes for {member.name}, expected {member.uncompressed_size}"
        )


def extract_first_frame(video_path: Path, frame_path: Path, ffmpeg_bin: str, member_name: str) -> None:
    if shutil.which(ffmpeg_bin) is None:
        raise FileNotFoundError(f"ffmpeg binary not found: {ffmpeg_bin}")
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(frame_path),
        ],
        check=True,
    )


def _find_next_signature(reader: SplitReader) -> tuple[bytes | None, int]:
    signature = reader.read(4)
    if len(signature) < 4:
        return None, reader.offset
    if signature in {LOCAL_FILE_HEADER, CENTRAL_DIRECTORY_HEADER, END_OF_CENTRAL_DIRECTORY}:
        return signature, reader.offset - 4

    window = signature
    while True:
        byte = reader.read(1)
        if not byte:
            return None, reader.offset
        window = (window + byte)[-4:]
        if window in {LOCAL_FILE_HEADER, CENTRAL_DIRECTORY_HEADER, END_OF_CENTRAL_DIRECTORY}:
            return window, reader.offset - 4


def _write_manifest(
    handle: object,
    member: LocalZipMember,
    frame_path: Path,
    parts: list[Path],
    status: str,
) -> None:
    if handle is None:
        return
    record = {
        "member": member.name,
        "frame_path": str(frame_path),
        "status": status,
        "compression": member.compression,
        "compressed_size": member.compressed_size,
        "uncompressed_size": member.uncompressed_size,
        "source_parts": [str(part) for part in parts],
    }
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def _write_failure(handle: object, member: LocalZipMember, exc: Exception) -> None:
    if handle is None:
        return
    handle.write(
        json.dumps(
            {
                "member": member.name,
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    handle.flush()


def _print_progress(counts: dict[str, int], progress_every: int) -> None:
    if progress_every <= 0:
        return
    if counts["members_seen"] % progress_every == 0:
        print(
            "progress "
            f"members_seen={counts['members_seen']} "
            f"ppt_seen={counts['ppt_seen']} "
            f"frames_written={counts['frames_written']} "
            f"frames_skipped={counts['frames_skipped']} "
            f"failures={counts['failures']}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover PPT first-frame images from split ZIP files without a central directory."
    )
    parser.add_argument("--parts", nargs="+", required=True, help="Split ZIP parts in byte order.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tmp-dir", required=True)
    parser.add_argument("--manifest-out")
    parser.add_argument("--failure-log")
    parser.add_argument("--suffix", default="_PPT.mp4")
    parser.add_argument("--contains", default="/PPT/")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = recover_ppt_frames(
        parts=[Path(part) for part in args.parts],
        out_dir=Path(args.out_dir),
        tmp_dir=Path(args.tmp_dir),
        suffix=args.suffix,
        contains=args.contains,
        limit=args.limit,
        overwrite=args.overwrite,
        manifest_out=Path(args.manifest_out) if args.manifest_out else None,
        failure_log=Path(args.failure_log) if args.failure_log else None,
        ffmpeg_bin=args.ffmpeg_bin,
        progress_every=args.progress_every,
    )
    print(json.dumps(counts, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
