#!/usr/bin/env python3
"""Build an ACL60/60 source-only segment manifest for annotation diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.build_acl6060_simulstream_inputs import locate_segment, xml_docs


FORBIDDEN_KEYS = ("reference", "target", "translation")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def assert_source_only(row: dict) -> None:
    for key in row:
        lowered = key.lower()
        if any(forbidden in lowered for forbidden in FORBIDDEN_KEYS):
            raise ValueError(f"Forbidden source annotation key: {key}")


def build_rows(acl_root: Path, split: str, talk_manifest: list[dict]) -> list[dict]:
    source_xml = acl_root / split / "text" / "xml" / f"ACL.6060.{split}.en-xx.en.xml"
    source_docs = xml_docs(source_xml)
    expected = {row["talk_id"]: row for row in talk_manifest if row["split"] == split}
    if {talk_id for talk_id, _ in source_docs} != set(expected):
        raise ValueError("Source XML talk ids differ from frozen talk manifest")

    rows = []
    global_index = 1
    for talk_id, source_segments in source_docs:
        talk = expected[talk_id]
        if len(source_segments) != int(talk["segment_count"]):
            raise ValueError(f"Source segment count mismatch for {talk_id}")
        full_audio = acl_root / split / "full_wavs" / f"{talk_id}.wav"
        if sha256_file(full_audio) != talk["audio_sha256"]:
            raise ValueError(f"Full audio hash mismatch for {talk_id}")
        cursor = 0
        for talk_segment_index, source_text in enumerate(source_segments):
            segment_audio = (
                acl_root / split / "segmented_wavs" / "gold" / f"sent_{global_index}.wav"
            )
            timing, cursor = locate_segment(full_audio, segment_audio, cursor)
            row = {
                "dataset": "acl6060",
                "split": split,
                "talk_id": talk_id,
                "segment_id": f"{talk_id}:S{talk_segment_index:03d}",
                "talk_segment_index": talk_segment_index,
                "global_segment_index": global_index,
                "offset_sec": round(float(timing["offset"]), 6),
                "duration_sec": round(float(timing["duration"]), 6),
                "end_sec": round(float(timing["offset"] + timing["duration"]), 6),
                "source_text": source_text,
                "source_audio_id": full_audio.name,
                "source_audio_sha256": talk["audio_sha256"],
                "segment_audio_sha256": sha256_file(segment_audio),
            }
            assert_source_only(row)
            rows.append(row)
            global_index += 1
    return rows


def build_summary(
    rows: list[dict],
    output: Path,
    acl_root: Path,
    split: str,
) -> dict:
    talk_ids = list(dict.fromkeys(row["talk_id"] for row in rows))
    return {
        "dataset": "acl6060",
        "split": split,
        "artifact": "source_only_annotation_segments",
        "status": "LOCAL_ANNOTATION_ONLY_NOT_INFERENCE_INPUT",
        "talk_count": len(talk_ids),
        "segment_count": len(rows),
        "talks": [
            {
                "talk_id": talk_id,
                "segment_count": sum(row["talk_id"] == talk_id for row in rows),
            }
            for talk_id in talk_ids
        ],
        "local_output": str(output),
        "local_output_sha256": sha256_file(output),
        "source_xml_sha256": sha256_file(
            acl_root / split / "text" / "xml" / f"ACL.6060.{split}.en-xx.en.xml"
        ),
        "source_text_consumed": True,
        "target_or_reference_consumed": False,
        "intended_use": "source-side annotation and automatic anticipation diagnostics only",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acl-root", type=Path, required=True)
    parser.add_argument("--talk-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("dev",), default="dev")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()

    rows = build_rows(args.acl_root, args.split, load_jsonl(args.talk_manifest))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = build_summary(rows, args.output, args.acl_root, args.split)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("talk_count", "segment_count")}))


if __name__ == "__main__":
    main()
