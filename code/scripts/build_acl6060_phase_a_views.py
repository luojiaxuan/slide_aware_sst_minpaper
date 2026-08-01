#!/usr/bin/env python3
"""Build physically separated ACL60/60 inference and scoring views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lxml import etree


FORBIDDEN_INFERENCE_KEYS = {
    "reference",
    "reference_path",
    "source_transcript",
    "source_transcript_path",
    "tagged_terminology",
    "tagged_terminology_path",
}


def source_docs(path: Path) -> dict[str, dict]:
    parser = etree.XMLParser(recover=True, resolve_entities=False, no_network=True)
    root = etree.parse(str(path), parser).getroot()
    docs = {}
    for doc in root.iter("doc"):
        talk_id = doc.attrib["docid"]
        abstract = doc.findtext("abstract") or ""
        docs[talk_id] = {"abstract": abstract.strip(), "segment_count": sum(1 for _ in doc.iter("seg"))}
    return docs


def load_snapshot(path: Path, split: str) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if row["split"] == split]


def assert_inference_safe(row: dict) -> None:
    leaked = FORBIDDEN_INFERENCE_KEYS.intersection(row)
    if leaked:
        raise ValueError(f"Inference row leaks scoring-only fields: {sorted(leaked)}")


def build_views(root: Path, paper_dir: Path, talks: list[dict], split: str) -> tuple[list[dict], list[dict]]:
    source_xml = root / split / "text" / "xml" / f"ACL.6060.{split}.en-xx.en.xml"
    docs = source_docs(source_xml)
    inference_rows = []
    scoring_rows = []
    for talk in talks:
        talk_id = talk["talk_id"]
        audio = root / talk["audio_relpath"]
        paper = paper_dir / f"{talk_id}.pdf"
        if not audio.exists() or not paper.exists():
            raise FileNotFoundError(audio if not audio.exists() else paper)
        inference = {
            "dataset": "acl6060",
            "split": split,
            "talk_id": talk_id,
            "audio_path": str(audio.resolve()),
            "audio_sha256": talk["audio_sha256"],
            "duration_sec": talk["duration_sec"],
            "segment_count": docs[talk_id]["segment_count"],
            "paper_pdf_path": str(paper.resolve()),
            "paper_pdf_sha256": talk["paper_sha256"],
            "paper_abstract": docs[talk_id]["abstract"],
            "context_sources": {
                "C0": [],
                "C1": ["paper_pdf_auto_terms"],
                "C2": ["paper_abstract", "paper_pdf_auto_entities"],
                "C3": ["paper_pdf_auto_phrase_boost", "paper_pdf_pretranslated_bm25"],
            },
        }
        assert_inference_safe(inference)
        inference_rows.append(inference)
        scoring_rows.append(
            {
                "dataset": "acl6060",
                "split": split,
                "talk_id": talk_id,
                "source_xml_path": str(source_xml.resolve()),
                "target_xml_path": str(
                    (root / split / "text" / "xml" / f"ACL.6060.{split}.en-xx.zh.xml").resolve()
                ),
                "tagged_source_path": str(
                    (
                        root
                        / split
                        / "text"
                        / "tagged_terminology"
                        / f"ACL.6060.{split}.tagged.en-xx.en.txt"
                    ).resolve()
                ),
                "tagged_target_path": str(
                    (
                        root
                        / split
                        / "text"
                        / "tagged_terminology"
                        / f"ACL.6060.{split}.tagged.en-xx.zh.txt"
                    ).resolve()
                ),
            }
        )
    return inference_rows, scoring_rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acl-root", type=Path, required=True)
    parser.add_argument("--acl-paper-dir", type=Path, required=True)
    parser.add_argument("--talk-snapshot", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "eval"], default="dev")
    parser.add_argument("--inference-out", type=Path, required=True)
    parser.add_argument("--scoring-out", type=Path, required=True)
    args = parser.parse_args()

    talks = load_snapshot(args.talk_snapshot, args.split)
    inference, scoring = build_views(args.acl_root, args.acl_paper_dir, talks, args.split)
    write_jsonl(args.inference_out, inference)
    write_jsonl(args.scoring_out, scoring)
    print(f"wrote {len(inference)} inference rows and {len(scoring)} scoring rows for {args.split}")


if __name__ == "__main__":
    main()
