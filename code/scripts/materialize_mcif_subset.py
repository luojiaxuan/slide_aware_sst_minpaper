#!/usr/bin/env python3
"""Materialize the frozen MCIF translation subset without exposing references."""

from __future__ import annotations

import argparse
import binascii
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Callable

import yaml


FORBIDDEN_INFERENCE_KEYS = ("reference", "transcript", "translation", "target_text", "tagged_terminology")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def zip_member_matches(path: Path, info: zipfile.ZipInfo) -> bool:
    if not path.is_file() or path.stat().st_size != info.file_size:
        return False
    checksum = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            checksum = binascii.crc32(chunk, checksum)
    return checksum & 0xFFFFFFFF == info.CRC


def extract_inference_files(archive_path: Path, talk_ids: list[str], output_root: Path) -> dict[str, Path]:
    requested = {"mcif-long-trans/audio-segments.yaml": output_root / "metadata" / "audio-segments.yaml"}
    for talk_id in talk_ids:
        requested[f"mcif-long-trans/audio/{talk_id}.wav"] = output_root / "audio" / f"{talk_id}.wav"
        requested[f"mcif-long-trans/pdf/{talk_id}.pdf"] = output_root / "pdf" / f"{talk_id}.pdf"

    extracted: dict[str, Path] = {}
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        missing = sorted(set(requested) - names)
        if missing:
            raise ValueError(f"MCIF archive is missing inference-safe files: {missing}")
        for member, destination in requested.items():
            info = archive.getinfo(member)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not zip_member_matches(destination, info):
                temporary = destination.with_suffix(destination.suffix + ".part")
                temporary.unlink(missing_ok=True)
                with archive.open(info) as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                if not zip_member_matches(temporary, info):
                    temporary.unlink(missing_ok=True)
                    raise ValueError(f"Extracted MCIF member failed CRC verification: {member}")
                temporary.replace(destination)
            extracted[member] = destination

    if (output_root / "ref").exists():
        raise ValueError("Reference directory must not exist under the MCIF inference root")
    return extracted


def hf_resolve_url(repo_id: str, revision: str, repo_path: str) -> str:
    encoded_repo = urllib.parse.quote(repo_id, safe="/")
    encoded_revision = urllib.parse.quote(revision, safe="")
    encoded_path = urllib.parse.quote(repo_path, safe="/")
    return (
        f"https://huggingface.co/datasets/{encoded_repo}/resolve/"
        f"{encoded_revision}/{encoded_path}?download=true"
    )


def verify_download(path: Path, expected_bytes: int, expected_sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"Unexpected size for {path}: {path.stat().st_size} != {expected_bytes}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"Unexpected SHA256 for {path}: {actual} != {expected_sha256}")


def download_file(
    url: str,
    destination: Path,
    expected_bytes: int,
    expected_sha256: str,
    *,
    retries: int = 4,
) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        verify_download(destination, expected_bytes, expected_sha256)
        return "verified-existing"

    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, retries + 1):
        existing = temporary.stat().st_size if temporary.exists() else 0
        headers = {"User-Agent": "slide-aware-sst-mcif-materializer/1"}
        if existing:
            headers["Range"] = f"bytes={existing}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                status = getattr(response, "status", 200)
                mode = "ab" if existing and status == 206 else "wb"
                with temporary.open(mode) as output:
                    shutil.copyfileobj(response, output, length=4 * 1024 * 1024)
            if temporary.stat().st_size == expected_bytes:
                verify_download(temporary, expected_bytes, expected_sha256)
                temporary.replace(destination)
                return "downloaded"
            if temporary.stat().st_size > expected_bytes:
                temporary.unlink()
        except (OSError, urllib.error.URLError, ValueError) as error:
            if attempt == retries:
                raise RuntimeError(f"Failed to download {url}") from error
        time.sleep(attempt * 2)
    raise RuntimeError(f"Failed to download {url}")


def audio_metadata(path: Path) -> dict:
    with wave.open(str(path), "rb") as audio:
        frames = audio.getnframes()
        sample_rate = audio.getframerate()
        return {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "sample_rate_hz": sample_rate,
            "channels": audio.getnchannels(),
            "duration_sec": round(frames / sample_rate, 6),
        }


def ffprobe_video(path: Path) -> dict:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=codec_type,width,height,avg_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    video_streams = [stream for stream in payload.get("streams", []) if stream.get("codec_type") == "video"]
    if len(video_streams) != 1:
        raise ValueError(f"Expected one video stream in {path}, found {len(video_streams)}")
    stream = video_streams[0]
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "duration_sec": round(float(payload["format"]["duration"]), 6),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "avg_frame_rate": stream.get("avg_frame_rate"),
    }


def load_segments(path: Path, talk_ids: list[str]) -> dict[str, list[dict]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("MCIF audio-segments.yaml must contain a list")
    grouped: dict[str, list[dict]] = defaultdict(list)
    allowed = set(talk_ids)
    for row in raw:
        if not isinstance(row, dict) or not {"wav", "offset", "duration"}.issubset(row):
            raise ValueError(f"Invalid MCIF audio segment row: {row!r}")
        talk_id = Path(str(row["wav"])).stem
        if talk_id not in allowed:
            raise ValueError(f"Unexpected talk in MCIF segment metadata: {talk_id}")
        offset = float(row["offset"])
        duration = float(row["duration"])
        if offset < 0 or duration <= 0:
            raise ValueError(f"Invalid MCIF segment timing for {talk_id}: {row!r}")
        grouped[talk_id].append(
            {
                "segment_id": len(grouped[talk_id]),
                "offset_sec": round(offset, 6),
                "duration_sec": round(duration, 6),
                "end_sec": round(offset + duration, 6),
                "speaker_id": str(row.get("speaker_id", "")),
            }
        )
    if set(grouped) != allowed:
        raise ValueError(f"MCIF segment metadata talk ids differ: {sorted(allowed - set(grouped))}")
    for talk_id, segments in grouped.items():
        offsets = [segment["offset_sec"] for segment in segments]
        if offsets != sorted(offsets):
            raise ValueError(f"MCIF segment offsets are not monotonic for {talk_id}")
    return dict(grouped)


def assert_inference_safe(value: object, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            lowered = str(key).lower()
            if any(forbidden in lowered for forbidden in FORBIDDEN_INFERENCE_KEYS):
                raise ValueError(f"Forbidden inference field at {path}.{key}")
            assert_inference_safe(nested, f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            assert_inference_safe(nested, f"{path}[{index}]")


def build_inference_rows(
    talk_ids: list[str],
    output_root: Path,
    segments: dict[str, list[dict]],
    repo_id: str,
    revision: str,
    video_probe: Callable[[Path], dict] = ffprobe_video,
) -> list[dict]:
    rows = []
    for talk_id in talk_ids:
        audio_path = output_root / "audio" / f"{talk_id}.wav"
        video_path = output_root / "video" / f"{talk_id}.mp4"
        paper_path = output_root / "pdf" / f"{talk_id}.pdf"
        audio = audio_metadata(audio_path)
        video = video_probe(video_path)
        talk_segments = segments[talk_id]
        if talk_segments[-1]["end_sec"] > audio["duration_sec"] + 0.1:
            raise ValueError(f"MCIF segments exceed audio duration for {talk_id}")
        row = {
            "dataset": "mcif",
            "subset": "iwslt2026_translation_21",
            "talk_id": talk_id,
            "upstream": {"repo_id": repo_id, "revision": revision},
            "audio": {"path": str(audio_path), **audio},
            "video": {"path": str(video_path), **video},
            "paper": {
                "path": str(paper_path),
                "bytes": paper_path.stat().st_size,
                "sha256": sha256_file(paper_path),
            },
            "alignment": {
                "audio_video_duration_delta_sec": round(video["duration_sec"] - audio["duration_sec"], 6),
                "segment_count": len(talk_segments),
                "last_segment_end_sec": talk_segments[-1]["end_sec"],
            },
            "segments": talk_segments,
        }
        assert_inference_safe(row)
        rows.append(row)
    return rows


def portable_summary(rows: list[dict], sources: dict, output_root_label: str) -> dict:
    return {
        "dataset": "mcif",
        "subset": "iwslt2026_translation_21",
        "upstream": {
            "repo": sources["repo"],
            "revision": sources["revision"],
            "license": sources["license"],
            "iwslt_archive_sha256": sources["iwslt2026_archive"]["sha256"],
        },
        "talk_count": len(rows),
        "segment_count": sum(row["alignment"]["segment_count"] for row in rows),
        "audio_duration_sec": round(sum(row["audio"]["duration_sec"] for row in rows), 6),
        "video_duration_sec": round(sum(row["video"]["duration_sec"] for row in rows), 6),
        "local_staging": output_root_label,
        "reference_files_extracted": False,
        "reference_content_inspected": False,
        "talks": [
            {
                "talk_id": row["talk_id"],
                "segments": row["alignment"]["segment_count"],
                "audio_bytes": row["audio"]["bytes"],
                "audio_sha256": row["audio"]["sha256"],
                "audio_duration_sec": row["audio"]["duration_sec"],
                "video_bytes": row["video"]["bytes"],
                "video_sha256": row["video"]["sha256"],
                "video_duration_sec": row["video"]["duration_sec"],
                "video_width": row["video"]["width"],
                "video_height": row["video"]["height"],
                "paper_bytes": row["paper"]["bytes"],
                "paper_sha256": row["paper"]["sha256"],
            }
            for row in rows
        ],
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources-manifest", type=Path, required=True)
    parser.add_argument("--files-manifest", type=Path, required=True)
    parser.add_argument("--iwslt-archive", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--portable-manifest-out", type=Path)
    parser.add_argument(
        "--portable-staging-label",
        default="ResearchStudio/data/vision-aware-sst/mcif/materialized",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    source_document = json.loads(args.sources_manifest.read_text(encoding="utf-8"))
    sources = source_document["mcif"]
    archive_contract = sources["iwslt2026_archive"]
    if sha256_file(args.iwslt_archive) != archive_contract["sha256"]:
        raise ValueError("MCIF IWSLT archive does not match the frozen SHA256")
    talk_ids = list(sources["translation_subset_talk_ids"])
    if talk_ids != archive_contract["talk_ids"]:
        raise ValueError("MCIF source manifest has inconsistent translation talk ids")

    extract_inference_files(args.iwslt_archive, talk_ids, args.output_root)
    file_rows = {row["path"]: row for row in load_jsonl(args.files_manifest)}
    downloads = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for talk_id in talk_ids:
            repo_path = f"MCIF_DATA/LONG_VIDEOS/{talk_id}.mp4"
            remote = file_rows.get(repo_path)
            if not remote or not remote.get("sha256"):
                raise ValueError(f"Frozen MCIF file metadata missing for {repo_path}")
            destination = args.output_root / "video" / f"{talk_id}.mp4"
            future = pool.submit(
                download_file,
                hf_resolve_url(sources["repo"], sources["revision"], repo_path),
                destination,
                int(remote["bytes"]),
                str(remote["sha256"]),
            )
            downloads.append((talk_id, future))
        for talk_id, future in downloads:
            print(f"video {talk_id}: {future.result()}")

    segments = load_segments(args.output_root / "metadata" / "audio-segments.yaml", talk_ids)
    rows = build_inference_rows(talk_ids, args.output_root, segments, sources["repo"], sources["revision"])
    inference_path = args.output_root / "manifests" / "inference.jsonl"
    summary = portable_summary(rows, sources, args.portable_staging_label)
    write_jsonl(inference_path, rows)
    write_json(args.output_root / "manifests" / "materialization.json", summary)
    if args.portable_manifest_out:
        write_json(args.portable_manifest_out, summary)
    output_keys = ("talk_count", "segment_count", "audio_duration_sec", "video_duration_sec")
    print(
        json.dumps(
            {
                "inference_manifest": str(inference_path),
                **{key: summary[key] for key in output_keys},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
