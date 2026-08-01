#!/usr/bin/env python3
"""Measure source-side lexical anticipation visible in ACL60/60 slide OCR."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from statistics import median


TOKEN_RE = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*")
STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "among",
    "another",
    "because",
    "before",
    "being",
    "between",
    "both",
    "could",
    "during",
    "first",
    "from",
    "have",
    "into",
    "more",
    "most",
    "other",
    "over",
    "should",
    "some",
    "such",
    "than",
    "that",
    "their",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "under",
    "using",
    "very",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
    "your",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def normalized_tokens(text: str) -> tuple[str, ...]:
    return tuple(match.group(0).lower().replace("'s", "") for match in TOKEN_RE.finditer(text))


def source_candidates(tokens: tuple[str, ...], max_ngram: int) -> set[tuple[str, str]]:
    candidates: set[tuple[str, str]] = set()
    for token in tokens:
        if len(token) >= 5 and token not in STOPWORDS:
            candidates.add(("token", token))
    for width in range(2, max_ngram + 1):
        for start in range(len(tokens) - width + 1):
            phrase_tokens = tokens[start : start + width]
            if sum(token not in STOPWORDS for token in phrase_tokens) < 1:
                continue
            if sum(len(token) for token in phrase_tokens) < 8:
                continue
            candidates.add(("phrase", " ".join(phrase_tokens)))
    return candidates


def observation_candidates(row: dict, max_ngram: int) -> dict[tuple[str, str], dict]:
    candidates: dict[tuple[str, str], dict] = {}
    for token in row["tokens"]:
        normalized = normalized_tokens(token["text"])
        if len(normalized) != 1:
            continue
        word = normalized[0]
        cleaned_surface = TOKEN_RE.fullmatch(token["text"].strip(".,:;()[]{}"))
        is_acronym = bool(cleaned_surface and cleaned_surface.group(0).isupper() and len(word) >= 2)
        if (len(word) >= 5 and word not in STOPWORDS) or is_acronym:
            candidates[("token", word)] = {
                "surface_text": token["text"],
                "ocr_confidence": float(token["confidence"]),
            }
    for line in row["lines"]:
        tokens = normalized_tokens(line["text"])
        for kind, text in source_candidates(tokens, max_ngram):
            if kind != "phrase":
                continue
            key = (kind, text)
            evidence = {
                "surface_text": line["text"],
                "ocr_confidence": float(line["mean_confidence"]),
            }
            if (
                key not in candidates
                or evidence["ocr_confidence"] > candidates[key]["ocr_confidence"]
            ):
                candidates[key] = evidence
    return candidates


def first_source_occurrences(segments: list[dict], max_ngram: int) -> dict[tuple[str, str], dict]:
    first: dict[tuple[str, str], dict] = {}
    for segment in sorted(segments, key=lambda row: row["talk_segment_index"]):
        for candidate in source_candidates(normalized_tokens(segment["source_text"]), max_ngram):
            first.setdefault(candidate, segment)
    return first


def quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return round(float(ordered[index]), 3)


def analyze_talk(
    talk_id: str,
    observations: list[dict],
    segments: list[dict],
    *,
    max_ngram: int,
) -> list[dict]:
    observations = sorted(observations, key=lambda row: row["observed_at_sec"])
    times = [float(row["observed_at_sec"]) for row in observations]
    visible = [observation_candidates(row, max_ngram) for row in observations]
    first_source = first_source_occurrences(segments, max_ngram)
    visual_union = set().union(*(set(candidates) for candidates in visible))
    opportunities = []
    for candidate in sorted(visual_union & set(first_source)):
        segment = first_source[candidate]
        source_time = float(segment["offset_sec"])
        current_index = bisect_right(times, source_time) - 1
        if current_index < 0 or candidate not in visible[current_index]:
            continue
        current = observations[current_index]
        if source_time >= float(current["availability_end_sec"]):
            raise ValueError(f"Observation interval lookup failed for {talk_id}")
        earliest_index = current_index
        while earliest_index > 0 and candidate in visible[earliest_index - 1]:
            earliest_index -= 1
        earliest = observations[earliest_index]
        lead = source_time - float(earliest["observed_at_sec"])
        evidence = visible[current_index][candidate]
        opportunities.append(
            {
                "dataset": "acl6060",
                "split": "dev",
                "status": "AUTOMATIC_SOURCE_ONLY_DIAGNOSTIC",
                "talk_id": talk_id,
                "candidate_kind": candidate[0],
                "normalized_text": candidate[1],
                "surface_text": evidence["surface_text"],
                "token_count": len(candidate[1].split()),
                "first_spoken_segment_id": segment["segment_id"],
                "first_spoken_segment_start_sec": source_time,
                "first_spoken_segment_text": segment["source_text"],
                "current_observation_id": current["observation_id"],
                "current_frame_path": current["frame_path"],
                "current_frame_sha256": current["frame_sha256"],
                "current_observed_at_sec": current["observed_at_sec"],
                "earliest_contiguous_observation_id": earliest["observation_id"],
                "earliest_contiguous_visible_sec": earliest["observed_at_sec"],
                "lead_lower_bound_sec": round(lead, 3),
                "ocr_confidence_at_first_spoken": round(evidence["ocr_confidence"], 3),
            }
        )
    return opportunities


def seed_coverage(seed_rows: list[dict], opportunities: list[dict]) -> list[dict]:
    by_observation: dict[str, list[dict]] = defaultdict(list)
    for row in opportunities:
        by_observation[row["current_observation_id"]].append(row)
    coverage = []
    for seed in seed_rows:
        future = [
            row
            for row in by_observation.get(seed["observation_id"], [])
            if float(row["first_spoken_segment_start_sec"]) > float(seed["t_evidence_sec"])
            and float(row["first_spoken_segment_start_sec"])
            <= float(seed["suggested_audio_window_end_sec"])
        ]
        coverage.append(
            {
                "packet_id": seed["packet_id"],
                "talk_id": seed["talk_id"],
                "observation_id": seed["observation_id"],
                "selection_stratum": seed["selection_stratum"],
                "automatic_future_candidate_count": len(future),
                "automatic_future_phrase_count": sum(
                    row["candidate_kind"] == "phrase" for row in future
                ),
                "max_lead_lower_bound_sec": max(
                    (row["lead_lower_bound_sec"] for row in future), default=None
                ),
                "candidate_ids": [
                    f'{row["candidate_kind"]}:{row["normalized_text"]}' for row in future
                ],
            }
        )
    return coverage


def is_contiguous_subsequence(shorter: tuple[str, ...], longer: tuple[str, ...]) -> bool:
    return any(
        longer[start : start + len(shorter)] == shorter
        for start in range(len(longer) - len(shorter) + 1)
    )


def collapse_nested_candidates(opportunities: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in opportunities:
        grouped[
            (row["talk_id"], row["first_spoken_segment_id"], row["current_observation_id"])
        ].append(row)
    collapsed = []
    sorted_groups = sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            item[1][0]["first_spoken_segment_start_sec"],
            item[0][2],
        ),
    )
    talk_event_indices: dict[str, int] = defaultdict(int)
    for (talk_id, _, _), rows in sorted_groups:
        talk_event_indices[talk_id] += 1
        event_id = f"{talk_id}:E{talk_event_indices[talk_id]:03d}"
        phrases = [row for row in rows if row["candidate_kind"] == "phrase"]
        phrase_tokens = [tuple(row["normalized_text"].split()) for row in phrases]
        maximal_phrases = []
        for row, tokens in zip(phrases, phrase_tokens):
            if any(
                len(other) > len(tokens) and is_contiguous_subsequence(tokens, other)
                for other in phrase_tokens
            ):
                continue
            maximal_phrases.append(row)
        covered_tokens = {
            token
            for row in maximal_phrases
            for token in row["normalized_text"].split()
        }
        uncovered_tokens = [
            row
            for row in rows
            if row["candidate_kind"] == "token"
            and row["normalized_text"] not in covered_tokens
        ]
        for row in maximal_phrases + uncovered_tokens:
            collapsed.append({**row, "automatic_event_id": event_id})
    collapsed.sort(
        key=lambda row: (
            row["talk_id"],
            row["first_spoken_segment_start_sec"],
            row["candidate_kind"],
            row["normalized_text"],
        )
    )
    return collapsed


def select_audit_sample(
    opportunities: list[dict], *, per_talk: int, top_lead_count: int
) -> list[dict]:
    by_talk: dict[str, list[dict]] = defaultdict(list)
    for row in opportunities:
        by_talk[row["talk_id"]].append(row)
    selected = []
    for talk_id in sorted(by_talk):
        representatives = {}
        for row in by_talk[talk_id]:
            event_id = row["automatic_event_id"]
            rank = (
                row["candidate_kind"] == "phrase",
                row["token_count"],
                row["ocr_confidence_at_first_spoken"],
            )
            if event_id not in representatives or rank > representatives[event_id][0]:
                representatives[event_id] = (rank, row)
        candidates = [item[1] for item in representatives.values()]
        highest = sorted(
            candidates,
            key=lambda row: (-row["lead_lower_bound_sec"], row["automatic_event_id"]),
        )[:top_lead_count]
        highest_ids = {row["automatic_event_id"] for row in highest}
        remainder = [row for row in candidates if row["automatic_event_id"] not in highest_ids]
        remainder.sort(
            key=lambda row: hashlib.sha256(
                f'acl6060-anticipation-audit-v1:{row["automatic_event_id"]}'.encode()
            ).hexdigest()
        )
        random_count = max(0, per_talk - len(highest))
        for stratum, rows in (("highest_lead", highest), ("hash_random", remainder[:random_count])):
            for row in rows:
                selected.append(
                    {
                        **row,
                        "audit_selection_stratum": stratum,
                        "audit_status": "pending",
                        "ocr_text_verified": None,
                        "frame_timing_plausible": None,
                        "semantic_anticipation_useful": None,
                        "audit_note": "",
                    }
                )
    return selected


def build_summary(
    raw_opportunities: list[dict],
    collapsed_opportunities: list[dict],
    coverage: list[dict],
    audit_sample: list[dict],
    *,
    ocr_path: Path,
    source_segments_path: Path,
    output_path: Path,
    collapsed_output_path: Path,
    coverage_path: Path,
    audit_sample_path: Path,
) -> dict:
    talk_ids = sorted({row["talk_id"] for row in collapsed_opportunities})
    leads = [float(row["lead_lower_bound_sec"]) for row in collapsed_opportunities]
    phrase_leads = [
        float(row["lead_lower_bound_sec"])
        for row in collapsed_opportunities
        if row["candidate_kind"] == "phrase"
    ]

    def counts(rows: list[dict]) -> dict:
        return {
            "opportunity_count": len(rows),
            "token_count": sum(row["candidate_kind"] == "token" for row in rows),
            "phrase_count": sum(row["candidate_kind"] == "phrase" for row in rows),
            "lead_ge_5s_count": sum(float(row["lead_lower_bound_sec"]) >= 5 for row in rows),
            "lead_ge_10s_count": sum(float(row["lead_lower_bound_sec"]) >= 10 for row in rows),
            "lead_ge_30s_count": sum(float(row["lead_lower_bound_sec"]) >= 30 for row in rows),
        }

    return {
        "dataset": "acl6060",
        "split": "dev",
        "artifact": "ocr_first_spoken_anticipation_diagnostic",
        "status": "AUTOMATIC_SOURCE_ONLY_DIAGNOSTIC_REQUIRES_HUMAN_AUDIT",
        "interpretation_limit": (
            "Measures OCR-sufficient lexical headroom only; it does not establish translation gains "
            "or raw-vision gains beyond OCR."
        ),
        "timing_limit": (
            "Source segment start is used as the earliest possible spoken time, so reported lead is "
            "a conservative lower bound for within-segment mentions."
        ),
        "raw_ngram_match_count": len(raw_opportunities),
        "nonredundant_candidate_count": len(collapsed_opportunities),
        "independent_segment_frame_event_count": len(
            {row["automatic_event_id"] for row in collapsed_opportunities}
        ),
        **counts(collapsed_opportunities),
        "lead_lower_bound_sec": {
            "min": round(min(leads), 3) if leads else None,
            "p25": quantile(leads, 0.25),
            "median": round(median(leads), 3) if leads else None,
            "p75": quantile(leads, 0.75),
            "p90": quantile(leads, 0.9),
            "max": round(max(leads), 3) if leads else None,
            "phrase_median": round(median(phrase_leads), 3) if phrase_leads else None,
        },
        "seed_packet_count": len(coverage),
        "seed_packet_with_future_candidate_count": sum(
            row["automatic_future_candidate_count"] > 0 for row in coverage
        ),
        "seed_packet_with_future_phrase_count": sum(
            row["automatic_future_phrase_count"] > 0 for row in coverage
        ),
        "pending_audit_sample_count": len(audit_sample),
        "talks": [
            {
                "talk_id": talk_id,
                "raw_ngram_match_count": sum(
                    row["talk_id"] == talk_id for row in raw_opportunities
                ),
                "independent_segment_frame_event_count": len(
                    {
                        row["automatic_event_id"]
                        for row in collapsed_opportunities
                        if row["talk_id"] == talk_id
                    }
                ),
                **counts(
                    [row for row in collapsed_opportunities if row["talk_id"] == talk_id]
                ),
            }
            for talk_id in talk_ids
        ],
        "provenance": {
            "ocr_sha256": sha256_file(ocr_path),
            "source_segments_sha256": sha256_file(source_segments_path),
            "raw_opportunities_local_path": str(output_path),
            "raw_opportunities_sha256": sha256_file(output_path),
            "nonredundant_opportunities_local_path": str(collapsed_output_path),
            "nonredundant_opportunities_sha256": sha256_file(collapsed_output_path),
            "seed_coverage_local_path": str(coverage_path),
            "seed_coverage_sha256": sha256_file(coverage_path),
            "pending_audit_sample_local_path": str(audit_sample_path),
            "pending_audit_sample_sha256": sha256_file(audit_sample_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocr", type=Path, required=True)
    parser.add_argument("--source-segments", type=Path, required=True)
    parser.add_argument("--event-seed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nonredundant-output", type=Path, required=True)
    parser.add_argument("--seed-coverage-out", type=Path, required=True)
    parser.add_argument("--audit-sample-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--max-ngram", type=int, default=4)
    parser.add_argument("--audit-per-talk", type=int, default=10)
    parser.add_argument("--audit-top-lead-per-talk", type=int, default=5)
    args = parser.parse_args()
    if args.max_ngram < 2:
        raise ValueError("max-ngram must be at least 2")

    ocr_rows = load_jsonl(args.ocr)
    source_rows = load_jsonl(args.source_segments)
    by_talk_ocr: dict[str, list[dict]] = defaultdict(list)
    by_talk_source: dict[str, list[dict]] = defaultdict(list)
    for row in ocr_rows:
        by_talk_ocr[row["talk_id"]].append(row)
    for row in source_rows:
        by_talk_source[row["talk_id"]].append(row)
    if set(by_talk_ocr) != set(by_talk_source):
        raise ValueError("OCR and source segment talk ids differ")

    opportunities = []
    for talk_id in sorted(by_talk_ocr):
        opportunities.extend(
            analyze_talk(
                talk_id,
                by_talk_ocr[talk_id],
                by_talk_source[talk_id],
                max_ngram=args.max_ngram,
            )
        )
    opportunities.sort(
        key=lambda row: (
            row["talk_id"],
            row["first_spoken_segment_start_sec"],
            row["candidate_kind"],
            row["normalized_text"],
        )
    )
    collapsed = collapse_nested_candidates(opportunities)
    coverage = seed_coverage(load_jsonl(args.event_seed), collapsed)
    audit_sample = select_audit_sample(
        collapsed,
        per_talk=args.audit_per_talk,
        top_lead_count=args.audit_top_lead_per_talk,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in opportunities),
        encoding="utf-8",
    )
    args.nonredundant_output.parent.mkdir(parents=True, exist_ok=True)
    args.nonredundant_output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in collapsed),
        encoding="utf-8",
    )
    args.seed_coverage_out.parent.mkdir(parents=True, exist_ok=True)
    args.seed_coverage_out.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in coverage),
        encoding="utf-8",
    )
    args.audit_sample_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_sample_out.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in audit_sample),
        encoding="utf-8",
    )
    summary = build_summary(
        opportunities,
        collapsed,
        coverage,
        audit_sample,
        ocr_path=args.ocr,
        source_segments_path=args.source_segments,
        output_path=args.output,
        collapsed_output_path=args.nonredundant_output,
        coverage_path=args.seed_coverage_out,
        audit_sample_path=args.audit_sample_out,
    )
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "raw_ngram_match_count": summary["raw_ngram_match_count"],
                "independent_segment_frame_event_count": summary[
                    "independent_segment_frame_event_count"
                ],
                "nonredundant_candidate_count": summary["nonredundant_candidate_count"],
                "phrase_count": summary["phrase_count"],
                "seed_packet_with_future_candidate_count": summary[
                    "seed_packet_with_future_candidate_count"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
