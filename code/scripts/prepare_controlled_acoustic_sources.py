#!/usr/bin/env python3
"""Verify and selectively materialize frozen OpenSLR acoustic sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import wave
import zipfile
from pathlib import Path, PurePosixPath


def file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path: Path, contract: dict) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(contract["bytes"]):
        raise ValueError(f"Archive size differs for {path}")
    for algorithm in ("md5", "sha256"):
        if file_digest(path, algorithm) != contract[algorithm]:
            raise ValueError(f"Archive {algorithm} differs for {path}")


def safe_member_path(member: str) -> PurePosixPath:
    path = PurePosixPath(member)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe archive member: {member}")
    return path


def tar_wav_inventory(path: Path) -> list[dict]:
    rows = []
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            if member.isfile() and member.name.lower().endswith(".wav"):
                safe_member_path(member.name)
                rows.append({"member": member.name, "bytes": member.size})
    return rows


def zip_wav_inventory(path: Path) -> list[dict]:
    rows = []
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            if not member.is_dir() and member.filename.lower().endswith(".wav"):
                safe_member_path(member.filename)
                rows.append({"member": member.filename, "bytes": member.file_size})
    return rows


def rank_key(salt: str, category: str, member: str) -> str:
    value = f"{salt}\0{category}\0{member}".encode()
    return hashlib.sha256(value).hexdigest()


def select_musan_members(inventory: list[dict], selection: dict) -> list[dict]:
    selected = []
    used_members: set[str] = set()
    salt = selection["salt"]
    for category, rule in selection["categories"].items():
        candidates = [
            row
            for row in inventory
            if row["member"].startswith(rule["prefix"])
            and row["bytes"] >= int(rule.get("min_bytes", 0))
        ]
        candidates.sort(key=lambda row: rank_key(salt, category, row["member"]))
        cursor = 0
        for split in ("development", "confirmatory"):
            count = int(rule[split])
            split_rows = candidates[cursor : cursor + count]
            if len(split_rows) != count:
                raise ValueError(
                    f"Not enough {category} candidates for {split}: "
                    f"{len(split_rows)} != {count}"
                )
            cursor += count
            for row in split_rows:
                if row["member"] in used_members:
                    raise ValueError(f"MUSAN member selected twice: {row['member']}")
                used_members.add(row["member"])
                selected.append({**row, "category": category, "split": split})
    return selected


def extract_tar_members(archive_path: Path, rows: list[dict], output_root: Path) -> None:
    wanted = {row["member"] for row in rows}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if member.name not in wanted:
                continue
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Could not read archive member: {member.name}")
            destination = output_root / safe_member_path(member.name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".part")
            with source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
            temporary.replace(destination)
            wanted.remove(member.name)
    if wanted:
        raise ValueError(f"Selected MUSAN members were not extracted: {sorted(wanted)}")


def extract_zip_members(archive_path: Path, rows: list[dict], output_root: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        missing = {row["member"] for row in rows} - names
        if missing:
            raise ValueError(f"Selected SLR28 members are missing: {sorted(missing)}")
        for row in rows:
            destination = output_root / safe_member_path(row["member"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".part")
            with archive.open(row["member"]) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
            temporary.replace(destination)


def wav_metadata(path: Path) -> dict:
    with wave.open(str(path), "rb") as audio:
        sample_rate = audio.getframerate()
        frames = audio.getnframes()
        return {
            "bytes": path.stat().st_size,
            "sha256": file_digest(path, "sha256"),
            "sample_rate_hz": sample_rate,
            "channels": audio.getnchannels(),
            "sample_width_bytes": audio.getsampwidth(),
            "frames": frames,
            "duration_sec": round(frames / sample_rate, 6),
        }


def build_source_rows(
    selected_musan: list[dict],
    rir_rows: list[dict],
    output_root: Path,
    portable_staging_label: str,
) -> list[dict]:
    rows = []
    for upstream, selected in (("openslr17", selected_musan), ("openslr28", rir_rows)):
        for item in selected:
            path = output_root / safe_member_path(item["member"])
            metadata = wav_metadata(path)
            if metadata["sample_rate_hz"] != 16000 or metadata["sample_width_bytes"] != 2:
                raise ValueError(f"Expected 16 kHz PCM16 source: {path}")
            source_id = f"{upstream}:{item['member']}"
            row = {
                "source_id": source_id,
                "upstream": upstream,
                "upstream_member": item["member"],
                "category": item["category"],
                "split": item["split"],
                "path": str(path),
                "staging_path": f"{portable_staging_label}/{item['member']}",
                **metadata,
            }
            if "channel" in item:
                channel = int(item["channel"])
                if not 0 <= channel < metadata["channels"]:
                    raise ValueError(f"Invalid RIR channel for {source_id}: {channel}")
                row["selected_channel"] = channel
            rows.append(row)
    return sorted(rows, key=lambda row: (row["split"], row["category"], row["source_id"]))


def portable_summary(contract: dict, rows: list[dict], portable_staging_label: str) -> dict:
    categories = {}
    for split in ("development", "confirmatory"):
        categories[split] = {
            category: sum(row["split"] == split and row["category"] == category for row in rows)
            for category in sorted({row["category"] for row in rows})
        }
    portable_rows = [
        {key: value for key, value in row.items() if key != "path"}
        for row in rows
    ]
    return {
        "schema_version": 1,
        "sources": contract["sources"],
        "selection": contract["selection"],
        "staging": portable_staging_label,
        "source_count": len(rows),
        "counts": categories,
        "development_confirmatory_overlap": bool(
            {row["source_id"] for row in rows if row["split"] == "development"}
            & {row["source_id"] for row in rows if row["split"] == "confirmatory"}
        ),
        "source_files": portable_rows,
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--musan-archive", type=Path, required=True)
    parser.add_argument("--rir-archive", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--portable-summary-out", type=Path, required=True)
    parser.add_argument("--portable-staging-label", required=True)
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    verify_archive(args.musan_archive, contract["sources"]["openslr17"]["archive"])
    verify_archive(args.rir_archive, contract["sources"]["openslr28"]["archive"])
    musan_inventory = tar_wav_inventory(args.musan_archive)
    selected_musan = select_musan_members(musan_inventory, contract["selection"])
    rir_inventory = {row["member"]: row for row in zip_wav_inventory(args.rir_archive)}
    rir_rows = []
    for split, specification in contract["rirs"].items():
        member = specification["member"]
        if member not in rir_inventory:
            raise ValueError(f"Frozen RIR is missing from SLR28: {member}")
        rir_rows.append(
            {
                **rir_inventory[member],
                "category": "rir",
                "split": split,
                "channel": specification["channel"],
            }
        )

    extract_tar_members(args.musan_archive, selected_musan, args.output_root)
    extract_zip_members(args.rir_archive, rir_rows, args.output_root)
    rows = build_source_rows(
        selected_musan,
        rir_rows,
        args.output_root,
        args.portable_staging_label,
    )
    if portable_summary(contract, rows, args.portable_staging_label)[
        "development_confirmatory_overlap"
    ]:
        raise ValueError("Development and confirmatory acoustic sources overlap")
    write_jsonl(args.output_root / "source_pool.jsonl", rows)
    summary = portable_summary(contract, rows, args.portable_staging_label)
    write_json(args.output_root / "source_pool_summary.json", summary)
    write_json(args.portable_summary_out, summary)
    print(json.dumps({"source_count": len(rows), "counts": summary["counts"]}))


if __name__ == "__main__":
    main()
