#!/usr/bin/env python3
"""Extract private MCIF references and build automatic target-event candidates."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any
import zipfile

import yaml

from scripts.analyze_acl6060_ocr_anticipation import (
    is_contiguous_subsequence,
    normalized_tokens,
    source_candidates,
)
from scripts.build_mcif_visual_token_controls import (
    canonical_sha256,
    file_sha256,
    git_head_clean,
    load_jsonl,
    unique_by_id,
)


LANGUAGES = ("en", "zh", "de", "it")
REFERENCE_MEMBERS = {
    language: f"mcif-long-trans/ref/{language}.txt" for language in LANGUAGES
}
SEGMENT_MEMBER = "mcif-long-trans/audio-segments.yaml"
SAFE_TALK_ID = re.compile(r"[A-Za-z0-9_-]+")


def sha256_bytes(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


def parse_segment_metadata(raw: bytes) -> list[dict[str, Any]]:
    parsed = yaml.safe_load(raw)
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("MCIF segment metadata is not a non-empty list")
    rows = []
    talk_indices: dict[str, int] = defaultdict(int)
    closed_talks = set()
    previous_talk = None
    for global_index, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise ValueError("MCIF segment metadata contains a non-object row")
        wav = item.get("wav")
        if not isinstance(wav, str) or not wav.endswith(".wav"):
            raise ValueError("MCIF segment row has an invalid WAV name")
        talk_id = wav[:-4]
        if SAFE_TALK_ID.fullmatch(talk_id) is None:
            raise ValueError("MCIF segment row has an unsafe talk id")
        if previous_talk is not None and talk_id != previous_talk:
            closed_talks.add(previous_talk)
        if talk_id in closed_talks:
            raise ValueError("MCIF segment rows for a talk are not contiguous")
        offset = float(item.get("offset"))
        duration = float(item.get("duration"))
        if offset < 0 or duration <= 0:
            raise ValueError("MCIF segment timing is invalid")
        segment_index = talk_indices[talk_id]
        talk_indices[talk_id] += 1
        rows.append(
            {
                "global_segment_index": global_index,
                "talk_id": talk_id,
                "talk_segment_index": segment_index,
                "speaker_id": str(item.get("speaker_id")),
                "offset_sec": offset,
                "duration_sec": duration,
                "end_sec": offset + duration,
            }
        )
        previous_talk = talk_id
    by_talk: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_talk[row["talk_id"]].append(row)
    for talk_id, talk_rows in by_talk.items():
        ordered = sorted(talk_rows, key=lambda row: row["talk_segment_index"])
        if any(
            right["offset_sec"] < left["offset_sec"]
            for left, right in zip(ordered, ordered[1:])
        ):
            raise ValueError(f"MCIF segment offsets decrease within {talk_id}")
    return rows


def talk_blocks(segments: list[dict[str, Any]]) -> list[tuple[str, int, int]]:
    blocks = []
    start = 0
    while start < len(segments):
        talk_id = segments[start]["talk_id"]
        end = start + 1
        while end < len(segments) and segments[end]["talk_id"] == talk_id:
            end += 1
        blocks.append((talk_id, start, end))
        start = end
    return blocks


def parse_reference_lines(
    raw: bytes,
    *,
    language: str,
    segments: list[dict[str, Any]],
) -> tuple[list[str], dict[str, int]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"MCIF {language} reference is not UTF-8") from exc
    lines = text.splitlines()
    if len(lines) != len(segments):
        raise ValueError(f"MCIF {language} reference count differs from segment metadata")
    output = list(lines)
    wrapped_talks = 0
    for talk_id, start, end in talk_blocks(segments):
        starts_wrapped = output[start].startswith('"')
        ends_wrapped = output[end - 1].endswith('"')
        if starts_wrapped != ends_wrapped:
            raise ValueError(f"MCIF {language} wrapper quotes are unbalanced for {talk_id}")
        if starts_wrapped:
            wrapped_talks += 1
            output[start] = output[start][1:]
            output[end - 1] = output[end - 1][:-1]
        for index in range(start, end):
            output[index] = output[index].replace('""', '"').strip()
            if not output[index]:
                raise ValueError(f"MCIF {language} contains an empty reference row")
    return output, {"physical_lines": len(lines), "wrapped_talks": wrapped_talks}


def load_archive(
    archive: Path,
) -> tuple[list[dict[str, Any]], dict[str, bytes], dict[str, Any]]:
    with zipfile.ZipFile(archive) as source:
        names = set(source.namelist())
        required = {SEGMENT_MEMBER, *REFERENCE_MEMBERS.values()}
        if not required.issubset(names):
            raise ValueError("MCIF archive lacks required segment/reference members")
        raw_members = {SEGMENT_MEMBER: source.read(SEGMENT_MEMBER)}
        raw_members.update(
            {member: source.read(member) for member in REFERENCE_MEMBERS.values()}
        )
    segments = parse_segment_metadata(raw_members[SEGMENT_MEMBER])
    parsing = {}
    references = {}
    for language, member in REFERENCE_MEMBERS.items():
        references[language], parsing[language] = parse_reference_lines(
            raw_members[member],
            language=language,
            segments=segments,
        )
    rows = []
    for index, segment in enumerate(segments):
        row = {
            "schema_version": "mcif_iwslt2026_reference_segment_v1",
            **segment,
            "segment_id": f"mcif:{segment['talk_id']}:SEG{segment['talk_segment_index']:03d}",
            "source_reference_en": references["en"][index],
            "target_reference_zh": references["zh"][index],
            "target_reference_de": references["de"][index],
            "target_reference_it": references["it"][index],
            "official_reference_consumed": True,
            "model_output_consumed": False,
        }
        row["row_sha256"] = canonical_sha256(row)
        rows.append(row)
    report = {
        "segments": len(rows),
        "talks": len({row["talk_id"] for row in rows}),
        "member_sha256": {
            member: sha256_bytes(raw) for member, raw in sorted(raw_members.items())
        },
        "reference_parsing": parsing,
    }
    return rows, raw_members, report


def validate_source_inputs(
    segments: list[dict[str, Any]],
    ladder: list[dict[str, Any]],
    inference_rows: list[dict[str, Any]],
    *,
    expected_segments: int,
    expected_talks: int,
) -> None:
    if len(segments) != expected_segments:
        raise ValueError("MCIF parsed reference segment count differs from contract")
    segment_talks = {row["talk_id"] for row in segments}
    if len(segment_talks) != expected_talks:
        raise ValueError("MCIF parsed reference talk count differs from contract")
    if len(unique_by_id(ladder, "source evidence ladder")) != len(ladder):
        raise ValueError("MCIF source evidence ladder has duplicate ids")
    ladder_talks = {row.get("lecture_id") for row in ladder}
    inference_by_talk = {row.get("talk_id"): row for row in inference_rows}
    if segment_talks != ladder_talks or segment_talks != set(inference_by_talk):
        raise ValueError("MCIF segment, ladder and inference talk sets differ")
    segment_counts = Counter(row["talk_id"] for row in segments)
    for talk_id in sorted(segment_talks):
        inference = inference_by_talk[talk_id]
        inference_segments = inference.get("segments") or []
        if len(inference_segments) != segment_counts[talk_id]:
            raise ValueError(f"MCIF inference segment count differs for {talk_id}")
        expected = [
            (
                row["talk_segment_index"],
                round(row["offset_sec"], 9),
                round(row["duration_sec"], 9),
            )
            for row in segments
            if row["talk_id"] == talk_id
        ]
        observed = [
            (
                int(row["segment_id"]),
                round(float(row["offset_sec"]), 9),
                round(float(row["duration_sec"]), 9),
            )
            for row in inference_segments
        ]
        if expected != observed:
            raise ValueError(f"MCIF inference segment timing differs for {talk_id}")
    for row in ladder:
        if row.get("schema_version") != "mcif_source_evidence_ladder_v1":
            raise ValueError("Unexpected MCIF evidence ladder schema")
        if row.get("row_sha256") != canonical_sha256(
            {key: value for key, value in row.items() if key != "row_sha256"}
        ):
            raise ValueError(f"MCIF evidence ladder row hash mismatch: {row.get('id')}")
        if row.get("source_transcript_consumed") is not False or row.get(
            "target_or_reference_consumed"
        ) is not False:
            raise ValueError("MCIF source evidence ladder consumed outcome data")
    for talk_id in sorted(segment_talks):
        states = sorted(
            (row for row in ladder if row["lecture_id"] == talk_id),
            key=lambda row: row["state_id"],
        )
        if [row["state_id"] for row in states] != list(range(len(states))):
            raise ValueError(f"MCIF evidence state ids are not contiguous for {talk_id}")
        previous_start = None
        for row in states:
            start = float(row["availability_start_sec"])
            end = float(row["availability_end_sec"])
            if start < 0 or end <= start:
                raise ValueError(f"MCIF evidence interval is invalid for {row['id']}")
            if previous_start is not None and start <= previous_start:
                raise ValueError(f"MCIF evidence state times do not increase for {talk_id}")
            previous_start = start


def maximal_candidates(candidates: set[tuple[str, str]]) -> list[tuple[str, str]]:
    phrases = [candidate for candidate in candidates if candidate[0] == "phrase"]
    tokens = [candidate for candidate in candidates if candidate[0] == "token"]
    maximal_phrases = []
    for candidate in phrases:
        candidate_tokens = tuple(candidate[1].split())
        if any(
            len(other[1].split()) > len(candidate_tokens)
            and is_contiguous_subsequence(candidate_tokens, tuple(other[1].split()))
            for other in phrases
        ):
            continue
        maximal_phrases.append(candidate)
    covered = {
        token
        for _, phrase in maximal_phrases
        for token in phrase.split()
    }
    uncovered_tokens = [candidate for candidate in tokens if candidate[1] not in covered]
    return sorted(
        maximal_phrases + uncovered_tokens,
        key=lambda candidate: (candidate[0], candidate[1]),
    )


def build_candidates(
    segments: list[dict[str, Any]],
    ladder: list[dict[str, Any]],
    *,
    max_ngram: int,
) -> list[dict[str, Any]]:
    states_by_talk: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ladder:
        states_by_talk[row["lecture_id"]].append(row)
    segments_by_talk: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in segments:
        segments_by_talk[row["talk_id"]].append(row)
    output = []
    for talk_id in sorted(segments_by_talk):
        talk_segments = sorted(
            segments_by_talk[talk_id], key=lambda row: row["talk_segment_index"]
        )
        states = sorted(states_by_talk[talk_id], key=lambda row: row["state_id"])
        state_times = [float(row["availability_start_sec"]) for row in states]
        visible = [
            source_candidates(normalized_tokens(row["r0_flat_ocr"]["model_input_text"]), max_ngram)
            for row in states
        ]
        first_source: dict[tuple[str, str], dict[str, Any]] = {}
        for segment in talk_segments:
            for candidate in source_candidates(
                normalized_tokens(segment["source_reference_en"]), max_ngram
            ):
                first_source.setdefault(candidate, segment)
        talk_index = 0
        for segment in talk_segments:
            state_index = bisect_right(state_times, float(segment["offset_sec"])) - 1
            if state_index < 0:
                continue
            if float(segment["offset_sec"]) >= float(
                states[state_index]["availability_end_sec"]
            ):
                continue
            intersections = {
                candidate
                for candidate in visible[state_index]
                if first_source.get(candidate, {}).get("segment_id") == segment["segment_id"]
            }
            for candidate in maximal_candidates(intersections):
                earliest_state_index = state_index
                while (
                    earliest_state_index > 0
                    and candidate in visible[earliest_state_index - 1]
                    and float(states[earliest_state_index - 1]["availability_end_sec"])
                    >= float(states[earliest_state_index]["availability_start_sec"])
                ):
                    earliest_state_index -= 1
                current_state = states[state_index]
                earliest_state = states[earliest_state_index]
                lead = float(segment["offset_sec"]) - float(
                    earliest_state["availability_start_sec"]
                )
                talk_index += 1
                row = {
                    "schema_version": "mcif_target_event_candidate_v1",
                    "status": "AUTOMATIC_REFERENCE_AWARE_CANDIDATE_NOT_GOLD_EVENT",
                    "candidate_id": f"mcif:{talk_id}:C{talk_index:03d}",
                    "talk_id": talk_id,
                    "segment_id": segment["segment_id"],
                    "talk_segment_index": segment["talk_segment_index"],
                    "source_segment_offset_sec": segment["offset_sec"],
                    "source_segment_end_sec": segment["end_sec"],
                    "source_reference_en": segment["source_reference_en"],
                    "target_reference_zh": segment["target_reference_zh"],
                    "target_reference_de": segment["target_reference_de"],
                    "target_reference_it": segment["target_reference_it"],
                    "candidate_kind": candidate[0],
                    "normalized_source_candidate": candidate[1],
                    "candidate_token_count": len(candidate[1].split()),
                    "current_state_id": current_state["id"],
                    "current_state_row_sha256": current_state["row_sha256"],
                    "current_evidence_available_sec": current_state[
                        "availability_start_sec"
                    ],
                    "earliest_contiguous_state_id": earliest_state["id"],
                    "earliest_contiguous_state_row_sha256": earliest_state["row_sha256"],
                    "earliest_contiguous_evidence_sec": earliest_state[
                        "availability_start_sec"
                    ],
                    "lead_lower_bound_sec": round(lead, 6),
                    "candidate_eligibility": None,
                    "acceptable_target_realizations_zh": [],
                    "forbidden_target_realizations_zh": [],
                    "acceptable_target_realizations_de": [],
                    "forbidden_target_realizations_de": [],
                    "acceptable_target_realizations_it": [],
                    "forbidden_target_realizations_it": [],
                    "audio_insufficient_until_sec": None,
                    "audio_first_sufficient_sec": None,
                    "annotator_id": None,
                    "annotation_note": "",
                    "official_reference_consumed": True,
                    "model_output_consumed": False,
                }
                row["row_sha256"] = canonical_sha256(row)
                output.append(row)
    return output


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_checksums(root: Path) -> tuple[int, str]:
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    checksum_path = root / "SHA256SUMS"
    checksum_path.write_text(
        "".join(
            f"{file_sha256(path)}  {path.relative_to(root).as_posix()}\n" for path in paths
        ),
        encoding="utf-8",
    )
    return len(paths), file_sha256(checksum_path)


def build_bundle(
    output_root: Path,
    *,
    segments: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    raw_members: dict[str, bytes],
    archive_report: dict[str, Any],
    archive_sha256: str,
    ladder_sha256: str,
    inference_manifest_sha256: str,
    builder_git_commit: str,
    max_ngram: int,
) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError("MCIF outcome candidate output root must not already exist")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    try:
        raw_root = temporary / "raw"
        raw_root.mkdir()
        for member, raw in sorted(raw_members.items()):
            filename = "audio-segments.yaml" if member == SEGMENT_MEMBER else Path(member).name
            (raw_root / filename).write_bytes(raw)
        segments_path = temporary / "reference_segments.jsonl"
        candidates_path = temporary / "candidate_events.jsonl"
        write_jsonl(segments_path, segments)
        write_jsonl(candidates_path, candidates)
        leads = [float(row["lead_lower_bound_sec"]) for row in candidates]
        report = {
            "schema_version": "mcif_outcome_candidate_inventory_report_v1",
            "status": "PRIVATE_REFERENCE_AWARE_CANDIDATES_PENDING_HUMAN_FREEZE",
            "builder_git_commit": builder_git_commit,
            "archive_sha256": archive_sha256,
            "ladder_sha256": ladder_sha256,
            "inference_manifest_sha256": inference_manifest_sha256,
            "segments": len(segments),
            "talks": len({row["talk_id"] for row in segments}),
            "candidates": len(candidates),
            "candidate_talks": len({row["talk_id"] for row in candidates}),
            "candidate_kind_distribution": dict(
                sorted(Counter(row["candidate_kind"] for row in candidates).items())
            ),
            "candidate_lead_ge_5_sec": sum(value >= 5 for value in leads),
            "candidate_lead_ge_10_sec": sum(value >= 10 for value in leads),
            "candidate_lead_max_sec": None if not leads else max(leads),
            "max_ngram": max_ngram,
            "archive_members": archive_report,
            "reference_segments_sha256": file_sha256(segments_path),
            "candidate_events_sha256": file_sha256(candidates_path),
            "official_reference_consumed": True,
            "model_output_consumed": False,
            "human_event_labels_complete": False,
            "audio_sufficiency_labels_complete": False,
            "interpretation": (
                "Exact source-reference/OCR overlaps are annotation candidates only. "
                "They are not gold events, acceptable target realizations, or ST results."
            ),
        }
        report_path = temporary / "report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "README.md").write_text(
            "# MCIF Outcome Candidate Inventory V1\n\n"
            "Private outcome-side extraction of the official IWSLT 2026 MCIF English, "
            "Chinese, German and Italian references, aligned to 919 segment timings. "
            "Automatic candidates are first-occurrence exact lexical overlaps between "
            "the English reference and causally available flat OCR. They require human "
            "eligibility, target-realization and audio-sufficiency annotation. This "
            "bundle must never be mounted into inference.\n\n"
            f"- segments / talks: {report['segments']} / {report['talks']}\n"
            f"- automatic candidates: {report['candidates']}\n"
            f"- candidate SHA256: `{report['candidate_events_sha256']}`\n",
            encoding="utf-8",
        )
        checksum_entries, checksum_sha256 = write_checksums(temporary)
        os.rename(temporary, output_root)
        return {
            **report,
            "checksum_entries": checksum_entries,
            "checksum_manifest_sha256": checksum_sha256,
            "bundle_files": sum(1 for path in output_root.rglob("*") if path.is_file()),
            "bundle_bytes": sum(
                path.stat().st_size for path in output_root.rglob("*") if path.is_file()
            ),
        }
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--ladder", type=Path, required=True)
    parser.add_argument("--expected-ladder-sha256", required=True)
    parser.add_argument("--inference-manifest", type=Path, required=True)
    parser.add_argument("--expected-inference-manifest-sha256", required=True)
    parser.add_argument("--code-repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-ngram", type=int, default=4)
    parser.add_argument("--expected-segments", type=int, default=919)
    parser.add_argument("--expected-talks", type=int, default=21)
    args = parser.parse_args()
    for path, expected, label in (
        (args.archive, args.expected_archive_sha256, "archive"),
        (args.ladder, args.expected_ladder_sha256, "ladder"),
        (
            args.inference_manifest,
            args.expected_inference_manifest_sha256,
            "inference manifest",
        ),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"MCIF {label} hash differs from the frozen input")
    if args.max_ngram < 1:
        raise ValueError("max-ngram must be positive")
    builder_git_commit = git_head_clean(args.code_repo)
    segments, raw_members, archive_report = load_archive(args.archive)
    ladder = load_jsonl(args.ladder)
    inference_rows = load_jsonl(args.inference_manifest)
    validate_source_inputs(
        segments,
        ladder,
        inference_rows,
        expected_segments=args.expected_segments,
        expected_talks=args.expected_talks,
    )
    candidates = build_candidates(segments, ladder, max_ngram=args.max_ngram)
    report = build_bundle(
        args.output_root,
        segments=segments,
        candidates=candidates,
        raw_members=raw_members,
        archive_report=archive_report,
        archive_sha256=args.expected_archive_sha256,
        ladder_sha256=args.expected_ladder_sha256,
        inference_manifest_sha256=args.expected_inference_manifest_sha256,
        builder_git_commit=builder_git_commit,
        max_ngram=args.max_ngram,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
