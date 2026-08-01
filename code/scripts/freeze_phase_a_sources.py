#!/usr/bin/env python3
"""Freeze ACL60/60 local assets and MCIF remote metadata for Phase A."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.parse
import urllib.request
import wave
import zipfile
from pathlib import Path

from lxml import etree


ACL_ARCHIVE_URL = "https://aclanthology.org/attachments/2023.iwslt-1.2.dataset.zip"
ACL_PAPER_URL = "https://aclanthology.org/2023.iwslt-1.2/"
ACL_LICENSE_URL = "https://aclanthology.org/faq/copyright/"
MCIF_REPO = "FBK-MT/MCIF"
MCIF_IWSLT_ARCHIVE_URL = (
    "https://fbk.sharepoint.com/:u:/s/MTUnit/"
    "IQCPqOBxtZXTTKHJrxIK1Om2AYNRrW_Gtj3IfqhtDNab8_A?e=f6vURw"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wave_metadata(path: Path) -> dict:
    with wave.open(str(path), "rb") as audio:
        frames = audio.getnframes()
        sample_rate = audio.getframerate()
        return {
            "sample_rate_hz": sample_rate,
            "channels": audio.getnchannels(),
            "sample_width_bytes": audio.getsampwidth(),
            "frames": frames,
            "duration_sec": round(frames / sample_rate, 6),
        }


def parse_xml(path: Path) -> tuple[etree._Element, int]:
    parser = etree.XMLParser(recover=True, resolve_entities=False, no_network=True)
    root = etree.parse(str(path), parser).getroot()
    return root, len(parser.error_log)


def xml_talk_counts(path: Path) -> dict[str, int]:
    root, _ = parse_xml(path)
    counts = {}
    for doc in root.iter("doc"):
        talk_id = doc.attrib["docid"]
        counts[talk_id] = sum(1 for _ in doc.iter("seg"))
    return counts


def archive_metadata(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        files = [item for item in archive.infolist() if not item.is_dir()]
    return {
        "url": ACL_ARCHIVE_URL,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "file_count": len(files),
        "uncompressed_bytes": sum(item.file_size for item in files),
    }


def critical_acl_files(root: Path) -> list[dict]:
    paths = []
    for split in ("dev", "eval"):
        paths.extend((root / split / "full_wavs").glob("*.wav"))
        paths.append(root / split / "FILE_ORDER")
        for section in ("txt", "xml", "tagged_terminology"):
            paths.extend((root / split / "text" / section).glob("*"))
    rows = []
    for path in sorted(set(paths)):
        if not path.is_file() or path.name == ".DS_Store":
            continue
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def acl_snapshot(archive: Path, root: Path, paper_dir: Path) -> tuple[dict, list[dict], list[dict]]:
    talks = []
    split_summary = {}
    for split in ("dev", "eval"):
        order = [line.strip() for line in (root / split / "FILE_ORDER").read_text().splitlines() if line.strip()]
        source_xml = root / split / "text" / "xml" / f"ACL.6060.{split}.en-xx.en.xml"
        xml_root, recovery_errors = parse_xml(source_xml)
        segment_counts = {
            doc.attrib["docid"]: sum(1 for _ in doc.iter("seg")) for doc in xml_root.iter("doc")
        }
        if set(order) != set(segment_counts):
            raise ValueError(f"{split} FILE_ORDER and source XML talk ids differ")

        text_lines = {}
        for language in ("en", "zh"):
            path = root / split / "text" / "txt" / f"ACL.6060.{split}.en-xx.{language}.txt"
            text_lines[language] = sum(1 for _ in path.open(encoding="utf-8"))
        expected_segments = sum(segment_counts.values())
        if set(text_lines.values()) != {expected_segments}:
            raise ValueError(f"{split} text line count does not match XML segments")

        for talk_id in order:
            audio = root / split / "full_wavs" / f"{talk_id}.wav"
            if not audio.exists():
                raise FileNotFoundError(audio)
            paper = paper_dir / f"{talk_id}.pdf"
            row = {
                "dataset": "acl6060",
                "split": split,
                "talk_id": talk_id,
                "segment_count": segment_counts[talk_id],
                "audio_relpath": audio.relative_to(root).as_posix(),
                "audio_bytes": audio.stat().st_size,
                "audio_sha256": sha256_file(audio),
                "paper_url": f"https://aclanthology.org/{talk_id}.pdf",
                **wave_metadata(audio),
            }
            if paper.exists():
                row.update(
                    paper_local=True,
                    paper_bytes=paper.stat().st_size,
                    paper_sha256=sha256_file(paper),
                )
            else:
                row["paper_local"] = False
            talks.append(row)

        split_summary[split] = {
            "talk_ids": order,
            "talk_count": len(order),
            "segment_count": expected_segments,
            "gold_segment_wavs": len(list((root / split / "segmented_wavs" / "gold").glob("*.wav"))),
            "shas_segment_wavs": len(list((root / split / "segmented_wavs" / "shas").glob("*.wav"))),
            "text_lines": text_lines,
            "source_xml_recovery_errors": recovery_errors,
        }

    files = critical_acl_files(root)
    summary = {
        "dataset": "acl6060",
        "source_type": "official_acl_anthology_attachment",
        "paper_url": ACL_PAPER_URL,
        "license": "CC BY 4.0",
        "license_url": ACL_LICENSE_URL,
        "archive": archive_metadata(archive),
        "splits": split_summary,
        "critical_file_count": len(files),
        "local_staging": "ResearchStudio/data/vision-aware-sst/acl6060",
        "reference_policy": {
            "dev": "score_only; never mount in inference process",
            "eval": "frozen but unopened until route passes dev futility gate",
            "tagged_terminology": "evaluation labels only; forbidden as C1 runtime input",
        },
    }
    return summary, talks, files


def fetch_json(url: str) -> tuple[object, dict[str, str]]:
    request = urllib.request.Request(url, headers={"User-Agent": "slide-aware-sst-source-freezer/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response), dict(response.headers.items())


def next_link(header: str | None) -> str | None:
    if not header:
        return None
    for part in header.split(","):
        match = re.match(r'\s*<([^>]+)>;\s*rel="next"', part)
        if match:
            return match.group(1)
    return None


def hf_tree(repo_id: str, revision: str) -> list[dict]:
    encoded_repo = urllib.parse.quote(repo_id, safe="/")
    encoded_revision = urllib.parse.quote(revision, safe="")
    url = f"https://huggingface.co/api/datasets/{encoded_repo}/tree/{encoded_revision}?recursive=true&expand=true"
    items = []
    while url:
        page, headers = fetch_json(url)
        items.extend(page)
        url = next_link(headers.get("Link") or headers.get("link"))
    return items


def mcif_iwslt_archive_snapshot(path: Path, media_talk_ids: set[str]) -> dict:
    with zipfile.ZipFile(path) as archive:
        files = [item for item in archive.infolist() if not item.is_dir()]
    audio_pattern = re.compile(r"^mcif-long-trans/audio/([^/]+)\.wav$")
    paper_pattern = re.compile(r"^mcif-long-trans/pdf/([^/]+)\.pdf$")
    audio_ids = {match.group(1) for item in files if (match := audio_pattern.match(item.filename))}
    paper_ids = {match.group(1) for item in files if (match := paper_pattern.match(item.filename))}
    if audio_ids != paper_ids:
        raise ValueError("MCIF IWSLT archive audio/PDF talk ids differ")
    if len(audio_ids) != 21:
        raise ValueError(f"Expected 21 MCIF translation talks, found {len(audio_ids)}")
    if not audio_ids.issubset(media_talk_ids):
        raise ValueError("MCIF IWSLT talk ids are not a subset of the HF media pool")
    required_paths = {
        "mcif-long-trans/audio-segments.yaml",
        "mcif-long-trans/ref/en.txt",
        "mcif-long-trans/ref/zh.txt",
        "mcif-long-trans/ref/de.txt",
        "mcif-long-trans/ref/it.txt",
    }
    archive_paths = {item.filename for item in files}
    if not required_paths.issubset(archive_paths):
        raise ValueError("MCIF IWSLT archive is missing required metadata/reference files")
    return {
        "url": MCIF_IWSLT_ARCHIVE_URL,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "file_count": len(files),
        "uncompressed_bytes": sum(item.file_size for item in files),
        "talk_ids": sorted(audio_ids),
        "talk_count": len(audio_ids),
        "audio_definition_path": "mcif-long-trans/audio-segments.yaml",
        "reference_paths": sorted(required_paths - {"mcif-long-trans/audio-segments.yaml"}),
        "reference_content_inspected": False,
    }


def mcif_snapshot(revision: str, iwslt_archive: Path) -> tuple[dict, list[dict]]:
    repo_url = f"https://huggingface.co/api/datasets/{MCIF_REPO}/revision/{revision}"
    metadata, _ = fetch_json(repo_url)
    if metadata["sha"] != revision:
        raise ValueError(f"MCIF revision endpoint returned {metadata['sha']} instead of {revision}")

    files = []
    for item in hf_tree(MCIF_REPO, revision):
        if item.get("type") != "file":
            continue
        lfs = item.get("lfs") or {}
        files.append(
            {
                "path": item["path"],
                "bytes": item.get("size"),
                "git_oid": item.get("oid"),
                "sha256": lfs.get("oid"),
            }
        )
    files.sort(key=lambda row: row["path"])
    audio_pattern = re.compile(r"^MCIF_DATA/LONG_AUDIOS/([^/]+)\.wav$")
    video_pattern = re.compile(r"^MCIF_DATA/LONG_VIDEOS/([^/]+)\.mp4$")
    audio_ids = {match.group(1) for row in files if (match := audio_pattern.match(row["path"]))}
    video_ids = {match.group(1) for row in files if (match := video_pattern.match(row["path"]))}
    if audio_ids != video_ids:
        raise ValueError("MCIF long audio/video talk ids differ")
    if len(audio_ids) != 100:
        raise ValueError(f"Expected 100 MCIF long media talks, found {len(audio_ids)}")
    translation = mcif_iwslt_archive_snapshot(iwslt_archive, audio_ids)

    summary = {
        "dataset": "mcif",
        "repo": MCIF_REPO,
        "repo_url": f"https://huggingface.co/datasets/{MCIF_REPO}",
        "revision": revision,
        "last_modified": metadata.get("lastModified"),
        "license": metadata.get("cardData", {}).get("license"),
        "private": metadata.get("private"),
        "gated": metadata.get("gated"),
        "file_count": len(files),
        "total_bytes": sum(row["bytes"] or 0 for row in files),
        "long_media_talk_ids": sorted(audio_ids),
        "long_media_talk_count": len(audio_ids),
        "translation_subset_talk_count": translation["talk_count"],
        "translation_subset_talk_ids": translation["talk_ids"],
        "iwslt2026_archive": translation,
        "reference_policy": (
            "archive and filenames frozen only; reference content remains unopened and must not be mounted "
            "in inference before the frozen-run commit"
        ),
    }
    return summary, files


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acl-archive", type=Path, required=True)
    parser.add_argument("--acl-root", type=Path, required=True)
    parser.add_argument("--acl-paper-dir", type=Path, required=True)
    parser.add_argument("--mcif-revision", required=True)
    parser.add_argument("--mcif-iwslt-archive", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    acl, talks, acl_files = acl_snapshot(args.acl_archive, args.acl_root, args.acl_paper_dir)
    mcif, mcif_files = mcif_snapshot(args.mcif_revision, args.mcif_iwslt_archive)
    write_json(args.out_dir / "phase_a_sources_20260731.json", {"acl6060": acl, "mcif": mcif})
    write_jsonl(args.out_dir / "acl6060_talks_20260731.jsonl", talks)
    write_jsonl(args.out_dir / "acl6060_critical_files_20260731.jsonl", acl_files)
    write_jsonl(args.out_dir / f"mcif_files_{args.mcif_revision[:8]}.jsonl", mcif_files)
    print(f"ACL60/60: {len(talks)} talks, {len(acl_files)} critical files")
    print(
        f"MCIF: {mcif['long_media_talk_count']} media talks, "
        f"{mcif['translation_subset_talk_count']} frozen translation talks, "
        f"{len(mcif_files)} remote files"
    )


if __name__ == "__main__":
    main()
