#!/usr/bin/env python3
"""Build a self-contained researcher prescreen for strict beyond-OCR candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any


RELATION_WORDS = re.compile(
    r"\b(left|right|above|below|top|bottom|arrow|line|color|colour|connect|"
    r"flow|between|row|column|count|icon|bubble|box|circle|highlight|inner|"
    r"outer|first|second|third|legend|axis|chart|diagram)\b",
    re.IGNORECASE,
)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def token_overlap(candidate: str, ocr_text: str) -> dict[str, Any]:
    candidate_tokens = tokens(candidate)
    ocr_tokens = set(tokens(ocr_text))
    matched = [token for token in candidate_tokens if token in ocr_tokens]
    coverage = len(matched) / max(len(candidate_tokens), 1)
    return {
        "candidate_tokens": candidate_tokens,
        "matched_tokens": matched,
        "coverage": round(coverage, 6),
    }


def priority_score(visual: dict, mapping: dict) -> float:
    fields = {
        origin.get("descriptor_field", "")
        for origin in visual["proposed_evidence_origins"]
    }
    description = " ".join(
        origin.get("descriptor_text", "")
        for origin in visual["proposed_evidence_origins"]
    )
    overlap = token_overlap(
        visual["candidate_source_en"], visual["current_slide_r0_text"]
    )["coverage"]
    count = len(tokens(visual["candidate_source_en"]))
    return (
        (3.0 if "spatial_relations" in fields else 0.0)
        + (2.0 if "actions" in fields else 0.0)
        + (2.0 if overlap == 0 else 1.0 if overlap < 1 else 0.0)
        + (2.0 if RELATION_WORDS.search(description) else 0.0)
        + (1.0 if 2 <= count <= 4 else 0.0)
        + min(float(mapping["lead_lower_bound_sec"]), 30.0) / 30.0
    )


def choose_priority(rows: list[dict], max_per_talk: int) -> set[str]:
    selected: set[str] = set()
    seen_segments: set[str] = set()
    talk_counts: dict[str, int] = {}
    ordered = sorted(
        rows,
        key=lambda row: (
            -row["_priority_score"],
            row["mapping"]["talk_id"],
            row["visual"]["candidate_source_en"],
            row["mapping"]["candidate_id"],
        ),
    )
    for row in ordered:
        mapping = row["mapping"]
        talk_id = mapping["talk_id"]
        segment_id = mapping["segment_id"]
        if segment_id in seen_segments or talk_counts.get(talk_id, 0) >= max_per_talk:
            continue
        selected.add(mapping["candidate_id"])
        seen_segments.add(segment_id)
        talk_counts[talk_id] = talk_counts.get(talk_id, 0) + 1
    return selected


def hashed(value: dict) -> dict:
    output = dict(value)
    output["row_sha256"] = canonical_sha256(output)
    return output


def validate_source_row(row: dict) -> None:
    claimed = row.get("row_sha256")
    payload = {key: value for key, value in row.items() if key != "row_sha256"}
    if claimed != canonical_sha256(payload):
        raise ValueError(f"Source row hash differs: {row.get('item_id', 'mapping')}")


def build(args: argparse.Namespace) -> dict:
    if args.output_root.exists():
        raise FileExistsError(f"Output root already exists: {args.output_root}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if file_sha256(args.config) != args.expected_config_sha256:
        raise ValueError("Prescreen config hash differs")

    visual_root = args.workspace_root / "visual_validator_view"
    target_root = args.workspace_root / "target_author_view"
    mapping_root = args.workspace_root / "scorer_private"
    visual_rows = load_jsonl(visual_root / "validation_items.jsonl")
    target_rows = load_jsonl(target_root / "annotation_items.jsonl")
    mapping_rows = load_jsonl(mapping_root / "item_mapping.jsonl")
    expected_source = int(config["expected_source_items"])
    if not (
        len(visual_rows) == len(target_rows) == len(mapping_rows) == expected_source
    ):
        raise ValueError("Prescreen source count differs")
    for row in visual_rows + target_rows + mapping_rows:
        validate_source_row(row)

    visual_by_id = {row["item_id"]: row for row in visual_rows}
    target_by_id = {row["item_id"]: row for row in target_rows}
    if len(visual_by_id) != expected_source or len(target_by_id) != expected_source:
        raise ValueError("Prescreen source ids are not unique")

    joined = []
    for mapping in mapping_rows:
        visual = visual_by_id[mapping["visual_item_id"]]
        target = target_by_id[mapping["target_item_id"]]
        if visual["candidate_source_en"] != target["candidate_source_en"]:
            raise ValueError("Visual and target candidate differ")
        duration = float(mapping["source_segment_end_sec"]) - float(
            mapping["source_segment_offset_sec"]
        )
        if float(mapping["lead_lower_bound_sec"]) < float(config["minimum_lead_sec"]):
            continue
        if not (
            float(config["minimum_segment_duration_sec"])
            <= duration
            <= float(config["maximum_segment_duration_sec"])
        ):
            continue
        joined.append(
            {
                "mapping": mapping,
                "visual": visual,
                "target": target,
                "duration": duration,
                "_priority_score": priority_score(visual, mapping),
            }
        )

    priority_ids = choose_priority(
        joined, int(config["priority_max_candidates_per_talk"])
    )
    if len(joined) != int(config["expected_review_items"]):
        raise ValueError(f"Review item count differs: {len(joined)}")
    if len(priority_ids) != int(config["expected_priority_items"]):
        raise ValueError(f"Priority item count differs: {len(priority_ids)}")

    ordered = sorted(
        joined,
        key=lambda row: (
            0 if row["mapping"]["candidate_id"] in priority_ids else 1,
            -row["_priority_score"],
            row["mapping"]["candidate_id"],
        ),
    )
    args.output_root.mkdir(parents=True)
    media_root = args.output_root / "media"
    media_root.mkdir()
    copied_media: dict[str, dict] = {}
    review_items = []
    for index, row in enumerate(ordered, 1):
        mapping = row["mapping"]
        visual = row["visual"]
        target = row["target"]
        source_media = visual_root / visual["current_slide"]["path"]
        expected_media_sha = visual["current_slide"]["sha256"]
        if file_sha256(source_media) != expected_media_sha:
            raise ValueError("Prescreen source media hash differs")
        if expected_media_sha not in copied_media:
            destination = media_root / f"{expected_media_sha[:16]}.png"
            shutil.copy2(source_media, destination)
            if file_sha256(destination) != expected_media_sha:
                raise ValueError("Prescreen media copy differs")
            copied_media[expected_media_sha] = {
                "path": f"media/{destination.name}",
                "sha256": expected_media_sha,
                "bytes": destination.stat().st_size,
            }
        overlap = token_overlap(
            visual["candidate_source_en"], visual["current_slide_r0_text"]
        )
        review_items.append(
            hashed(
                {
                    "schema_version": "mcif_strict_beyond_ocr_research_item_v1",
                    "review_index": index,
                    "queue": "A" if mapping["candidate_id"] in priority_ids else "B",
                    "candidate_id": mapping["candidate_id"],
                    "candidate_source_en": visual["candidate_source_en"],
                    "candidate_kind": visual["candidate_kind"],
                    "talk_id": mapping["talk_id"],
                    "segment_id": mapping["segment_id"],
                    "state_id": mapping["current_state_id"],
                    "lead_lower_bound_sec": mapping["lead_lower_bound_sec"],
                    "segment_duration_sec": round(row["duration"], 6),
                    "source_reference_en": target["source_reference_en"],
                    "target_reference_zh": target["target_reference_zh"],
                    "flat_ocr_text": visual["current_slide_r0_text"],
                    "structured_text_blocks": visual["current_slide_r1_blocks"],
                    "machine_proposed_evidence": visual["proposed_evidence_origins"],
                    "candidate_ocr_token_overlap": overlap,
                    "slide": copied_media[expected_media_sha],
                    "source_visual_item_id": mapping["visual_item_id"],
                    "source_target_item_id": mapping["target_item_id"],
                    "source_mapping_row_sha256": mapping["row_sha256"],
                }
            )
        )

    packet_core = {
        "schema_version": "mcif_strict_beyond_ocr_research_packet_v1",
        "scope": config["scope"],
        "config_sha256": args.expected_config_sha256,
        "source_workspace_sha256": file_sha256(args.workspace_root / "SHA256SUMS"),
        "item_count": len(review_items),
        "priority_item_count": len(priority_ids),
        "media_count": len(copied_media),
        "items_sha256": canonical_sha256(review_items),
    }
    packet_id = f"mcif-strict-beyond-ocr-prescreen-v1-{canonical_sha256(packet_core)[:12]}"
    packet = {**packet_core, "packet_id": packet_id, "items": review_items}
    template = args.template.read_text(encoding="utf-8")
    packet_json = json.dumps(packet, ensure_ascii=False, separators=(",", ":")).replace(
        "</script>", "<\\/script>"
    )
    html = template.replace("__PACKET_JSON__", packet_json)
    if "__PACKET_JSON__" in html:
        raise ValueError("Prescreen template packet placeholder remains")
    (args.output_root / "index.html").write_text(html, encoding="utf-8")
    manifest = {
        **packet_core,
        "packet_id": packet_id,
        "builder_git_commit": args.builder_git_commit,
        "source_workspace": str(args.workspace_root),
        "config": str(args.config),
        "review_output_schema": "mcif_strict_beyond_ocr_researcher_review_v1",
        "status": "READY_FOR_RESEARCHER_PRESCREEN_NO_LABELS",
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rules_source = args.rules.read_text(encoding="utf-8")
    (args.output_root / "REVIEW_RULES.md").write_text(rules_source, encoding="utf-8")
    checksums = []
    for path in sorted(args.output_root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            checksums.append(f"{file_sha256(path)}  {path.relative_to(args.output_root)}")
    (args.output_root / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--builder-git-commit", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), ensure_ascii=False, indent=2, sort_keys=True))
