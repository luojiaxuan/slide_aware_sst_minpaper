#!/usr/bin/env python3
"""Build and validate physically separated ACL60/60 annotation v2 stages."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import shutil

from scripts.materialize_acl6060_source_event_workspace import extract_pcm_window


GLOBAL_FORBIDDEN_KEY_PARTS = ("transcript", "target", "reference", "translation", "model_output")
LOCK_EXCLUDED_FIELDS = {
    "question_lock_sha256",
    "authoring_lock_sha256",
    "author_review_lock_sha256",
    "audio_annotation_lock_sha256",
    "frame_annotation_lock_sha256",
    "adjudication_lock_sha256",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_stage_media(
    rows: list[dict], root: Path, path_key: str, hash_key: str
) -> None:
    for row in rows:
        path = root / row[path_key]
        if not path.is_file():
            raise FileNotFoundError(f'Missing stage media: {row["packet_id"]}: {path}')
        if sha256_file(path) != row[hash_key]:
            raise ValueError(f'Stage media hash mismatch: {row["packet_id"]}: {path_key}')


def verify_frame_stage_media(
    rows: list[dict], packet_rows: list[dict], root: Path
) -> None:
    packets = index_packets_for_v2(packet_rows)
    for row in rows:
        packet = packets[row["packet_id"]]
        path = root / row["frame_path"]
        if not path.is_file():
            raise FileNotFoundError(f'Missing stage media: {row["packet_id"]}: {path}')
        frame_sha256 = sha256_file(path)
        if frame_sha256 != packet["workspace_frame_sha256"]:
            raise ValueError(f'Frame media hash mismatch: {row["packet_id"]}')
        if row["frame_binding_hmac"] != frame_binding_hmac(
            row["packet_id"], frame_sha256
        ):
            raise ValueError(f'Frame media binding mismatch: {row["packet_id"]}')


def verify_sequential_interaction_log(rows: list[dict], path: Path) -> None:
    events = load_jsonl(path)
    if not events or events[0].get("event_type") != "session_started":
        raise ValueError("Interaction log lacks session start")
    validator_ids = {row["validator_id"] for row in rows}
    if validator_ids != {events[0].get("validator_id")}:
        raise ValueError("Interaction log validator id mismatch")
    pending_tasks = []
    for row in rows:
        pending = dict(row)
        pending.update(
            {
                "question_only_status": "pending",
                "question_only_answer_option_id": None,
                "question_only_submitted_at_utc": None,
                "question_only_lock_sha256": None,
                "audio_annotation_status": "pending",
                "prefix_judgments": [],
                "audio_submitted_at_utc": None,
                "interaction_log_tail_sha256": None,
                "interaction_log_sha256": None,
                "sequential_delivery_backend": None,
                "audio_annotation_lock_sha256": None,
            }
        )
        pending_tasks.append(pending)
    if events[0].get("task_sheet_sha256") != canonical_hash({"tasks": pending_tasks}):
        raise ValueError("Interaction log task sheet mismatch")
    if any(row.get("interaction_log_sha256") != sha256_file(path) for row in rows):
        raise ValueError("Interaction log file hash mismatch")
    previous = None
    previous_time = None
    completed = {}
    task_ids = [row["packet_id"] for row in rows]
    task_index = {row["packet_id"]: row for row in rows}
    task_position = 0
    state = "question"
    step_index = 0
    release_seen = False
    for index, event in enumerate(events):
        payload = {key: value for key, value in event.items() if key != "event_sha256"}
        if event.get("event_index") != index:
            raise ValueError(f"Non-contiguous interaction event index: {index}")
        if event.get("previous_event_sha256") != previous:
            raise ValueError(f"Broken interaction event chain: {index}")
        if event.get("event_sha256") != canonical_hash(payload):
            raise ValueError(f"Interaction event hash mismatch: {index}")
        previous = event["event_sha256"]
        event_time = parse_utc(event.get("server_time_utc"))
        if previous_time is not None and event_time < previous_time:
            raise ValueError(f"Interaction event timestamps are not monotonic: {index}")
        previous_time = event_time
        event_type = event.get("event_type")
        if index == 0:
            continue
        if event_type == "session_started":
            raise ValueError("Duplicate interaction session start")
        if task_position >= len(task_ids):
            raise ValueError(f"Interaction event follows final completion: {index}")
        packet_id = event.get("packet_id")
        expected_packet_id = task_ids[task_position]
        if packet_id != expected_packet_id:
            raise ValueError(
                f"Interaction packet order mismatch: {packet_id} != {expected_packet_id}"
            )
        grid = expected_prefix_grid(task_index[packet_id])
        if state == "question":
            if event_type != "question_only_submitted":
                raise ValueError(f"Question-only event missing before audio: {packet_id}")
            state = "prefix"
            step_index = 0
            release_seen = False
            continue
        if state == "prefix":
            if event_type == "item_completed":
                raise ValueError(f"Item completion is out of order: {packet_id}")
            if event_type == "prefix_released":
                if event.get("step_index") != step_index:
                    raise ValueError(f"Out-of-order prefix release: {packet_id}")
                if abs(float(event.get("prefix_end_sec")) - grid[step_index]) > 1e-5:
                    raise ValueError(f"Invalid prefix release boundary: {packet_id}")
                release_seen = True
                continue
            if event_type != "prefix_submitted" or not release_seen:
                raise ValueError(f"Prefix submission lacks its release: {packet_id}")
            if event.get("step_index") != step_index:
                raise ValueError(f"Out-of-order prefix submission: {packet_id}")
            if abs(float(event.get("prefix_end_sec")) - grid[step_index]) > 1e-5:
                raise ValueError(f"Invalid prefix submission boundary: {packet_id}")
            step_index += 1
            release_seen = False
            state = "complete" if step_index == len(grid) else "prefix"
            continue
        if state != "complete" or event_type != "item_completed":
            raise ValueError(f"Item completion is out of order: {packet_id}")
        if packet_id in completed:
            raise ValueError(f'Duplicate completed event: {packet_id}')
        completed[packet_id] = event
        task_position += 1
        state = "question"
    if task_position != len(task_ids) or state != "question":
        raise ValueError("Interaction log ended before all ordered tasks completed")
    if set(completed) != {row["packet_id"] for row in rows}:
        raise ValueError("Interaction log and completed audio packet ids differ")
    for row in rows:
        event = completed[row["packet_id"]]
        if row["interaction_log_tail_sha256"] != event["event_sha256"]:
            raise ValueError(f'Interaction log tail mismatch: {row["packet_id"]}')
        packet_events = [
            candidate
            for candidate in events
            if candidate.get("packet_id") == row["packet_id"]
        ]
        questions = [
            candidate
            for candidate in packet_events
            if candidate["event_type"] == "question_only_submitted"
        ]
        prefixes = [
            candidate
            for candidate in packet_events
            if candidate["event_type"] == "prefix_submitted"
        ]
        grid = expected_prefix_grid(row)
        if len(questions) != 1 or len(prefixes) != len(grid):
            raise ValueError(f'Incomplete interaction trajectory: {row["packet_id"]}')
        if (
            row["question_only_status"] != questions[0]["status"]
            or row["question_only_answer_option_id"] != questions[0]["option_id"]
            or row["question_only_submitted_at_utc"] != questions[0]["server_time_utc"]
        ):
            raise ValueError(f'Question-only row/log mismatch: {row["packet_id"]}')
        for index, (prefix, expected_time) in enumerate(zip(prefixes, grid)):
            releases = [
                candidate
                for candidate in packet_events
                if candidate["event_type"] == "prefix_released"
                and candidate["step_index"] == index
                and candidate["event_index"] > questions[0]["event_index"]
                and candidate["event_index"] < prefix["event_index"]
            ]
            if (
                prefix["step_index"] != index
                or abs(float(prefix["prefix_end_sec"]) - expected_time) > 1e-5
                or not releases
                or any(
                    abs(float(release["prefix_end_sec"]) - expected_time) > 1e-5
                    for release in releases
                )
                or row["prefix_judgments"][index]["status"] != prefix["status"]
                or row["prefix_judgments"][index]["option_id"] != prefix["option_id"]
            ):
                raise ValueError(f'Invalid interaction step: {row["packet_id"]}: {index}')
        if (
            row["audio_submitted_at_utc"] != event["server_time_utc"]
            or row["sequential_delivery_backend"] != "acl6060_audio_gate_v1"
            or event["event_index"] <= prefixes[-1]["event_index"]
        ):
            raise ValueError(f'Completed row/log mismatch: {row["packet_id"]}')


def canonical_hash(value: dict) -> str:
    filtered = {key: child for key, child in value.items() if key not in LOCK_EXCLUDED_FIELDS}
    payload = json.dumps(filtered, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assert_no_forbidden_keys(value: object, forbidden_parts: tuple[str, ...]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = key.lower()
            if any(part in lowered for part in forbidden_parts):
                raise ValueError(f"Forbidden key in stage view: {key}")
            assert_no_forbidden_keys(child, forbidden_parts)
    elif isinstance(value, list):
        for child in value:
            assert_no_forbidden_keys(child, forbidden_parts)


def index_unique(rows: list[dict], key: str = "packet_id") -> dict[str, dict]:
    indexed = {}
    for row in rows:
        value = row[key]
        if value in indexed:
            raise ValueError(f"Duplicate {key}: {value}")
        indexed[value] = row
    return indexed


def parse_utc(value: str) -> datetime:
    if not value:
        raise ValueError("Missing UTC timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"Timestamp lacks timezone: {value}")
    return parsed


def packet_dir_name(packet_id: str) -> str:
    return packet_id.replace(":", "__")


def blinding_secret() -> str:
    secret = os.environ.get("ACL6060_V2_BLINDING_SECRET")
    if secret is None or len(secret) < 32:
        raise ValueError("ACL6060_V2_BLINDING_SECRET must contain at least 32 characters")
    return secret


def blinded_packet_id(source_packet_id: str) -> str:
    secret = blinding_secret()
    digest = hmac.new(secret.encode(), source_packet_id.encode(), hashlib.sha256).hexdigest()
    return f"ACLDEV-{digest[:16]}"


def frame_binding_hmac(packet_id: str, frame_sha256: str) -> str:
    digest = hmac.new(
        blinding_secret().encode(),
        f"acl6060-v2-frame-binding:{packet_id}:{frame_sha256}".encode(),
        hashlib.sha256,
    ).hexdigest()
    return f"FRAME-{digest}"


def index_packets_for_v2(packet_rows: list[dict]) -> dict[str, dict]:
    indexed = {}
    for packet in packet_rows:
        identifier = blinded_packet_id(packet["packet_id"])
        if identifier in indexed:
            raise ValueError(f"Duplicate blinded packet id: {identifier}")
        indexed[identifier] = packet
    return indexed


def prepare_author_rows(packet_rows: list[dict], author_id: str) -> list[dict]:
    rows = []
    for packet in packet_rows:
        row = {
            "schema_version": "acl6060_source_event_annotation_v2",
            "stage": "frame_only_question_authoring",
            "packet_id": blinded_packet_id(packet["packet_id"]),
            "frame_path": f'packets/{blinded_packet_id(packet["packet_id"])}/frame.jpg',
            "frame_binding_hmac": frame_binding_hmac(
                blinded_packet_id(packet["packet_id"]),
                packet["workspace_frame_sha256"],
            ),
            "author_id": author_id,
            "authoring_status": "pending",
            "source_question": None,
            "source_options": [],
            "canonical_answer_index": None,
            "evidence_subtypes": [],
            "evidence_region": None,
            "term_or_entity": None,
            "negative_labels": [],
            "exclusion_labels": [],
            "authoring_note": "",
            "authoring_completed_at_utc": None,
            "question_locked_at_utc": None,
            "question_lock_sha256": None,
            "authoring_lock_sha256": None,
        }
        assert_no_forbidden_keys(row, GLOBAL_FORBIDDEN_KEY_PARTS + ("audio",))
        rows.append(row)
    rows.sort(
        key=lambda row: hashlib.sha256(
            f'acl6060-v2-author-order:{row["packet_id"]}'.encode()
        ).hexdigest()
    )
    return rows


def materialize_author_view(
    packet_rows: list[dict], workspace_root: Path, output_root: Path, author_id: str
) -> list[dict]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output root is not empty: {output_root}")
    rows = prepare_author_rows(packet_rows, author_id)
    packets = index_packets_for_v2(packet_rows)
    for row in rows:
        packet = packets[row["packet_id"]]
        source = workspace_root / packet["workspace_frame_path"]
        destination = output_root / row["frame_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        frame_sha256 = sha256_file(destination)
        if frame_sha256 != packet["workspace_frame_sha256"]:
            raise ValueError(f'Copied frame hash mismatch: {row["packet_id"]}')
        if row["frame_binding_hmac"] != frame_binding_hmac(
            row["packet_id"], frame_sha256
        ):
            raise ValueError(f'Copied frame binding mismatch: {row["packet_id"]}')
    write_jsonl(output_root / "authoring.jsonl", rows)
    return rows


def question_lock_payload(row: dict) -> dict:
    return {
        "schema_version": row["schema_version"],
        "packet_id": row["packet_id"],
        "frame_binding_hmac": row["frame_binding_hmac"],
        "source_question": row["source_question"],
        "source_options": row["source_options"],
        "canonical_answer_index": row["canonical_answer_index"],
        "evidence_subtypes": row["evidence_subtypes"],
        "evidence_region": row["evidence_region"],
        "term_or_entity": row["term_or_entity"],
    }


def authoring_lock_payload(row: dict) -> dict:
    return {
        "schema_version": row["schema_version"],
        "packet_id": row["packet_id"],
        "frame_path": row["frame_path"],
        "frame_binding_hmac": row["frame_binding_hmac"],
        "author_id": row["author_id"],
        "authoring_status": row["authoring_status"],
        "source_question": row["source_question"],
        "source_options": row["source_options"],
        "canonical_answer_index": row["canonical_answer_index"],
        "evidence_subtypes": row["evidence_subtypes"],
        "evidence_region": row["evidence_region"],
        "term_or_entity": row["term_or_entity"],
        "negative_labels": row["negative_labels"],
        "exclusion_labels": row["exclusion_labels"],
        "authoring_note": row["authoring_note"],
        "authoring_completed_at_utc": row["authoring_completed_at_utc"],
        "question_locked_at_utc": row["question_locked_at_utc"],
    }


def validate_author_row(row: dict, config: dict) -> None:
    allowed = set(config["question_authoring"]["authoring_status"])
    if row["authoring_status"] not in allowed or row["authoring_status"] == "pending":
        raise ValueError(f'Incomplete author status: {row["packet_id"]}')
    parse_utc(row["authoring_completed_at_utc"])
    if row["authoring_status"] != "candidate":
        if row["authoring_status"] == "negative_no_visual_question":
            if not row["negative_labels"] or not set(row["negative_labels"]) <= set(
                config["negative_labels"]
            ):
                raise ValueError(f'Invalid author negative labels: {row["packet_id"]}')
        elif not row["exclusion_labels"] or not set(row["exclusion_labels"]) <= set(
            config["exclusion_labels"]
        ):
            raise ValueError(f'Invalid author exclusion labels: {row["packet_id"]}')
        if not row["authoring_note"].strip():
            raise ValueError(f'Negative/excluded author row lacks note: {row["packet_id"]}')
        return
    question = row["source_question"]
    options = row["source_options"]
    answer = row["canonical_answer_index"]
    minimum = config["question_authoring"]["forced_choice_min_options"]
    maximum = config["question_authoring"]["forced_choice_max_options"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f'Candidate lacks question: {row["packet_id"]}')
    if not isinstance(options, list) or not minimum <= len(options) <= maximum:
        raise ValueError(f'Invalid option count: {row["packet_id"]}')
    if any(not isinstance(option, str) or not option.strip() for option in options):
        raise ValueError(f'Blank option: {row["packet_id"]}')
    if len(set(options)) != len(options):
        raise ValueError(f'Duplicate options: {row["packet_id"]}')
    if not isinstance(answer, int) or not 0 <= answer < len(options):
        raise ValueError(f'Invalid canonical answer: {row["packet_id"]}')
    if not row["evidence_subtypes"] or not set(row["evidence_subtypes"]) <= set(
        config["evidence_subtypes"]
    ):
        raise ValueError(f'Invalid evidence subtypes: {row["packet_id"]}')
    if row["evidence_region"] is None or not str(row["term_or_entity"] or "").strip():
        raise ValueError(f'Candidate lacks evidence localization: {row["packet_id"]}')
    parse_utc(row["question_locked_at_utc"])


def freeze_author_rows(
    author_rows: list[dict], packet_rows: list[dict], config: dict
) -> list[dict]:
    packets = index_packets_for_v2(packet_rows)
    authors = index_unique(author_rows)
    if set(authors) != set(packets):
        raise ValueError("Author and packet ids differ")
    output = []
    for source_row in author_rows:
        packet_id = source_row["packet_id"]
        row = dict(authors[packet_id])
        packet = packets[packet_id]
        expected_frame_path = f"packets/{packet_id}/frame.jpg"
        for key, expected in (
            ("frame_path", expected_frame_path),
            (
                "frame_binding_hmac",
                frame_binding_hmac(packet_id, packet["workspace_frame_sha256"]),
            ),
        ):
            if row[key] != expected:
                raise ValueError(f'Author task field changed ({key}): {packet_id}')
        assert_no_forbidden_keys(row, GLOBAL_FORBIDDEN_KEY_PARTS + ("audio",))
        validate_author_row(row, config)
        row["authoring_lock_sha256"] = canonical_hash(authoring_lock_payload(row))
        if row["authoring_status"] == "candidate":
            row["question_lock_sha256"] = canonical_hash(question_lock_payload(row))
            row.update(
                {
                    "talk_id": packet["talk_id"],
                    "t_evidence_sec": packet["t_evidence_sec"],
                    "frame_sha256": packet["workspace_frame_sha256"],
                    "stage": "author_source_audio_relevance_check",
                    "audio_path": f'packets/{packet_id}/causal_audio.wav',
                    "audio_sha256": None,
                    "causal_audio_end_sec": packet["suggested_audio_window_end_sec"],
                    "speech_relevance_status": "pending",
                    "full_audio_answer_index": None,
                    "frame_current_for_question": None,
                    "speech_reviewed_at_utc": None,
                    "speech_relevance_note": "",
                    "speech_exclusion_labels": [],
                    "author_review_lock_sha256": None,
                }
            )
        else:
            row.update(
                {
                    "talk_id": packet["talk_id"],
                    "t_evidence_sec": packet["t_evidence_sec"],
                    "frame_sha256": packet["workspace_frame_sha256"],
                    "stage": "author_source_audio_relevance_check",
                    "speech_relevance_status": "not_applicable",
                    "speech_reviewed_at_utc": None,
                    "speech_relevance_note": "",
                    "speech_exclusion_labels": [],
                    "author_review_lock_sha256": None,
                }
            )
        output.append(row)
    return output


def validate_authoring_lock(row: dict) -> None:
    if row.get("authoring_lock_sha256") != canonical_hash(authoring_lock_payload(row)):
        raise ValueError(f'Authoring lock mismatch: {row["packet_id"]}')


def freeze_author_audio_reviews(
    rows: list[dict], packet_rows: list[dict], config: dict
) -> list[dict]:
    indexed = index_unique(rows)
    packets = index_packets_for_v2(packet_rows)
    if set(indexed) != set(packets):
        raise ValueError("Author-review and packet ids differ")
    output = []
    for row in rows:
        validate_authoring_lock(row)
        validate_author_audio_review(row, config)
        frozen = {
            **row,
            "author_review_lock_sha256": canonical_hash(row),
        }
        output.append(frozen)
    return output


def validate_frozen_author_review(row: dict, config: dict) -> None:
    validate_authoring_lock(row)
    validate_author_audio_review(row, config)
    if row.get("author_review_lock_sha256") != canonical_hash(row):
        raise ValueError(f'Author review lock mismatch: {row["packet_id"]}')


def materialize_author_audio_review_view(
    rows: list[dict],
    packet_rows: list[dict],
    acl_root: Path,
    output_root: Path,
) -> list[dict]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output root is not empty: {output_root}")
    packets = index_packets_for_v2(packet_rows)
    output = []
    for row in rows:
        if row["authoring_status"] != "candidate":
            output.append(row)
            continue
        if row["question_lock_sha256"] != canonical_hash(question_lock_payload(row)):
            raise ValueError(f'Question lock mismatch: {row["packet_id"]}')
        packet = packets[row["packet_id"]]
        source = acl_root / packet["split"] / "full_wavs" / packet["audio_id"]
        if sha256_file(source) != packet["audio_sha256"]:
            raise ValueError(f'Full source audio hash mismatch: {row["packet_id"]}')
        destination = output_root / row["audio_path"]
        timing = extract_pcm_window(
            source,
            destination,
            start_sec=0.0,
            end_sec=float(row["causal_audio_end_sec"]),
        )
        output.append(
            {
                **row,
                "audio_sha256": sha256_file(destination),
                "audio_duration_sec": timing["clip_duration_sec"],
                "audio_bytes": destination.stat().st_size,
            }
        )
    write_jsonl(output_root / "author_audio_review.jsonl", output)
    return output


def option_order(row: dict, validator_id: str) -> list[int]:
    return sorted(
        range(len(row["source_options"])),
        key=lambda index: hashlib.sha256(
            (
                f'acl6060-v2-option-order:{row["packet_id"]}:'
                f'{row["question_lock_sha256"]}:{validator_id}:{index}'
            ).encode()
        ).hexdigest(),
    )


def blinded_option_id(row: dict, validator_id: str, index: int) -> str:
    digest = hmac.new(
        blinding_secret().encode(),
        (
            f'acl6060-v2-option-id:{row["packet_id"]}:'
            f'{row["question_lock_sha256"]}:{validator_id}:{index}'
        ).encode(),
        hashlib.sha256,
    ).hexdigest()
    return f"VOPT-{digest[:12]}"


def validate_author_audio_review(row: dict, config: dict) -> None:
    if row["authoring_status"] != "candidate":
        if row["speech_relevance_status"] != "not_applicable":
            raise ValueError(f'Non-candidate has audio outcome: {row["packet_id"]}')
        return
    expected_lock = canonical_hash(question_lock_payload(row))
    if row["question_lock_sha256"] != expected_lock:
        raise ValueError(f'Question lock mismatch: {row["packet_id"]}')
    status = row["speech_relevance_status"]
    allowed = set(config["author_audio_review"]["speech_relevance_status"])
    if status not in allowed or status == "pending":
        raise ValueError(f'Incomplete speech relevance: {row["packet_id"]}')
    if not isinstance(row.get("audio_sha256"), str) or len(row["audio_sha256"]) != 64:
        raise ValueError(f'Author audio media hash missing: {row["packet_id"]}')
    reviewed = parse_utc(row["speech_reviewed_at_utc"])
    locked = parse_utc(row["question_locked_at_utc"])
    if reviewed <= locked:
        raise ValueError(f'Audio review did not follow question lock: {row["packet_id"]}')
    if status != "addressed" and not row["speech_relevance_note"].strip():
        raise ValueError(f'Non-addressed candidate lacks note: {row["packet_id"]}')
    if status == "exclude_other" and (
        not row["speech_exclusion_labels"]
        or not set(row["speech_exclusion_labels"]) <= set(config["exclusion_labels"])
    ):
        raise ValueError(f'Invalid speech exclusion labels: {row["packet_id"]}')
    if status != "exclude_other" and row["speech_exclusion_labels"]:
        raise ValueError(f'Unexpected speech exclusion labels: {row["packet_id"]}')
    if status == "addressed":
        answer = row["full_audio_answer_index"]
        if answer != row["canonical_answer_index"]:
            raise ValueError(f'Full-audio and visual gold differ: {row["packet_id"]}')
        if row["frame_current_for_question"] is not True:
            raise ValueError(f'Frame is stale or unverified: {row["packet_id"]}')


def make_audio_validator_rows(
    author_review_rows: list[dict], validator_id: str, config: dict
) -> list[dict]:
    if validator_id in {row["author_id"] for row in author_review_rows}:
        raise ValueError("Question authors and audio validators must be disjoint")
    output = []
    for author in author_review_rows:
        validate_frozen_author_review(author, config)
        if author["authoring_status"] != "candidate" or author["speech_relevance_status"] != "addressed":
            continue
        order = option_order(author, validator_id)
        options = [
            {
                "option_id": blinded_option_id(author, validator_id, index),
                "text": author["source_options"][index],
            }
            for index in order
        ]
        row = {
            "schema_version": author["schema_version"],
            "stage": "validator_audio_only_boundary_pass",
            "packet_id": author["packet_id"],
            "question_lock_sha256": author["question_lock_sha256"],
            "source_question": author["source_question"],
            "source_options": options,
            "option_order_sha256": canonical_hash({"packet_id": author["packet_id"], "order": order}),
            "audio_path": f'packets/{packet_dir_name(author["packet_id"])}/causal_audio.wav',
            "audio_sha256": None,
            "audio_context_start_sec": 0.0,
            "causal_audio_end_sec": author["causal_audio_end_sec"],
            "t_evidence_sec": author["t_evidence_sec"],
            "prefix_step_sec": config["audio_boundary"]["prefix_step_sec"],
            "validator_id": validator_id,
            "question_only_status": "pending",
            "question_only_answer_option_id": None,
            "question_only_submitted_at_utc": None,
            "question_only_lock_sha256": None,
            "audio_annotation_status": "pending",
            "prefix_judgments": [],
            "audio_note": "",
            "audio_submitted_at_utc": None,
            "interaction_log_tail_sha256": None,
            "interaction_log_sha256": None,
            "sequential_delivery_backend": None,
            "audio_annotation_lock_sha256": None,
        }
        assert_no_forbidden_keys(row, GLOBAL_FORBIDDEN_KEY_PARTS + ("frame", "canonical"))
        output.append(row)
    output.sort(
        key=lambda row: hashlib.sha256(
            f'acl6060-v2-validator-order:{validator_id}:{row["packet_id"]}'.encode()
        ).hexdigest()
    )
    return output


def materialize_audio_validator_view(
    rows: list[dict],
    packet_rows: list[dict],
    acl_root: Path,
    output_root: Path,
) -> list[dict]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output root is not empty: {output_root}")
    packets = index_packets_for_v2(packet_rows)
    output = []
    for row in rows:
        packet = packets[row["packet_id"]]
        source = acl_root / packet["split"] / "full_wavs" / packet["audio_id"]
        if sha256_file(source) != packet["audio_sha256"]:
            raise ValueError(f'Full source audio hash mismatch: {row["packet_id"]}')
        destination = output_root / row["audio_path"]
        timing = extract_pcm_window(
            source,
            destination,
            start_sec=0.0,
            end_sec=float(row["causal_audio_end_sec"]),
        )
        materialized = {
            **row,
            "audio_sha256": sha256_file(destination),
            "audio_duration_sec": timing["clip_duration_sec"],
            "audio_bytes": destination.stat().st_size,
        }
        output.append(materialized)
    write_jsonl(output_root / "audio_delivery_private.jsonl", output)
    return output


def grid_index(time_sec: float, start_sec: float, step_sec: float) -> int:
    raw = (float(time_sec) - float(start_sec)) / float(step_sec)
    index = round(raw)
    if abs(raw - index) > 1e-5:
        raise ValueError(f"Boundary is off prefix grid: {time_sec}")
    return index


def expected_prefix_grid(row: dict) -> list[float]:
    start = float(row["t_evidence_sec"])
    end = float(row["causal_audio_end_sec"])
    step = float(row["prefix_step_sec"])
    if end < start:
        raise ValueError(f'Causal audio ends before evidence: {row["packet_id"]}')
    count = int((end - start) // step) + 1
    return [round(start + index * step, 6) for index in range(count)]


def question_only_lock_payload(row: dict) -> dict:
    return {
        "packet_id": row["packet_id"],
        "validator_id": row["validator_id"],
        "question_lock_sha256": row["question_lock_sha256"],
        "source_question": row["source_question"],
        "source_options": row["source_options"],
        "question_only_status": row["question_only_status"],
        "question_only_answer_option_id": row["question_only_answer_option_id"],
        "question_only_submitted_at_utc": row["question_only_submitted_at_utc"],
    }


def validate_audio_annotation(row: dict) -> None:
    option_ids = {option["option_id"] for option in row["source_options"]}
    question_only_status = row["question_only_status"]
    if question_only_status not in {"not_answerable", "answerable", "ambiguous"}:
        raise ValueError(f'Incomplete question-only annotation: {row["packet_id"]}')
    question_only_answer = row["question_only_answer_option_id"]
    if question_only_status == "answerable" and question_only_answer not in option_ids:
        raise ValueError(f'Invalid question-only answer: {row["packet_id"]}')
    if question_only_status != "answerable" and question_only_answer is not None:
        raise ValueError(f'Unexpected question-only answer: {row["packet_id"]}')
    question_only_time = parse_utc(row["question_only_submitted_at_utc"])
    if row["question_only_lock_sha256"] != canonical_hash(question_only_lock_payload(row)):
        raise ValueError(f'Question-only lock mismatch: {row["packet_id"]}')
    if row["audio_annotation_status"] != "complete":
        raise ValueError(f'Incomplete audio annotation: {row["packet_id"]}')
    if not isinstance(row.get("interaction_log_tail_sha256"), str) or len(
        row["interaction_log_tail_sha256"]
    ) != 64:
        raise ValueError(f'Missing sequential interaction log: {row["packet_id"]}')
    if not isinstance(row.get("interaction_log_sha256"), str) or len(
        row["interaction_log_sha256"]
    ) != 64:
        raise ValueError(f'Missing full interaction log hash: {row["packet_id"]}')
    if row.get("sequential_delivery_backend") != "acl6060_audio_gate_v1":
        raise ValueError(f'Invalid sequential delivery backend: {row["packet_id"]}')
    grid = expected_prefix_grid(row)
    judgments = row["prefix_judgments"]
    if len(judgments) != len(grid):
        raise ValueError(f'Incomplete prefix trajectory: {row["packet_id"]}')
    for index, (judgment, expected_time) in enumerate(zip(judgments, grid)):
        if judgment.get("step_index") != index:
            raise ValueError(f'Non-contiguous prefix trajectory: {row["packet_id"]}')
        if abs(float(judgment.get("prefix_end_sec")) - expected_time) > 1e-5:
            raise ValueError(f'Prefix trajectory is off grid: {row["packet_id"]}')
        status = judgment.get("status")
        if status not in {"insufficient", "uncertain", "option"}:
            raise ValueError(f'Invalid prefix response: {row["packet_id"]}')
        option_id = judgment.get("option_id")
        if status == "option" and option_id not in option_ids:
            raise ValueError(f'Invalid option id in trajectory: {row["packet_id"]}')
        if status != "option" and option_id is not None:
            raise ValueError(f'Non-option response carries option id: {row["packet_id"]}')
    if parse_utc(row["audio_submitted_at_utc"]) <= question_only_time:
        raise ValueError(f'Audio annotation preceded question-only lock: {row["packet_id"]}')


def audio_lock_payload(row: dict) -> dict:
    return {
        key: value
        for key, value in row.items()
        if key not in {"audio_annotation_lock_sha256", "frame_path", "frame_sha256"}
    }


def freeze_audio_rows(
    audio_rows: list[dict],
    author_review_rows: list[dict],
    packet_rows: list[dict],
    config: dict,
) -> list[dict]:
    authors = index_unique(author_review_rows)
    packets = index_packets_for_v2(packet_rows)
    if set(authors) != set(packets):
        raise ValueError("Author-review and packet ids differ")
    expected_packet_ids = {
        packet_id
        for packet_id, author in authors.items()
        if author["authoring_status"] == "candidate"
        and author["speech_relevance_status"] == "addressed"
    }
    if {row["packet_id"] for row in audio_rows} != expected_packet_ids:
        raise ValueError("Audio and validated-candidate packet ids differ")
    frozen = []
    for audio in audio_rows:
        validate_audio_annotation(audio)
        author = authors[audio["packet_id"]]
        packet = packets[audio["packet_id"]]
        expected = make_audio_validator_rows([author], audio["validator_id"], config)[0]
        for key in (
            "schema_version",
            "stage",
            "packet_id",
            "question_lock_sha256",
            "source_question",
            "source_options",
            "option_order_sha256",
            "audio_path",
            "audio_context_start_sec",
            "causal_audio_end_sec",
            "t_evidence_sec",
            "prefix_step_sec",
        ):
            if audio[key] != expected[key]:
                raise ValueError(f'Audio task field changed ({key}): {audio["packet_id"]}')
        if not isinstance(audio.get("audio_sha256"), str) or len(audio["audio_sha256"]) != 64:
            raise ValueError(f'Audio media hash missing: {audio["packet_id"]}')
        locked = {**audio, "audio_annotation_lock_sha256": canonical_hash(audio_lock_payload(audio))}
        frozen.append(locked)
    return frozen


def audio_cohort_lock_sha256(audio_a_rows: list[dict], audio_b_rows: list[dict]) -> str:
    for row in audio_a_rows + audio_b_rows:
        validate_frozen_audio_annotation(row)
    validator_ids_a = {row["validator_id"] for row in audio_a_rows}
    validator_ids_b = {row["validator_id"] for row in audio_b_rows}
    if len(validator_ids_a) != 1 or len(validator_ids_b) != 1:
        raise ValueError("Each audio sheet must contain exactly one validator id")
    if validator_ids_a & validator_ids_b:
        raise ValueError("Audio validator ids must be distinct")
    return canonical_hash(
        {
            "audio_locks": sorted(
                row["audio_annotation_lock_sha256"]
                for row in audio_a_rows + audio_b_rows
            )
        }
    )


def frame_task_lock_payload(row: dict) -> dict:
    return {
        key: value
        for key, value in row.items()
        if key
        not in {
            "frame_task_lock_sha256",
            "frame_support_status",
            "frame_answer_option_id",
            "evidence_subtypes",
            "frame_confidence",
            "frame_note",
            "frame_submitted_at_utc",
            "frame_annotation_lock_sha256",
        }
    }


def make_frame_validator_rows(
    author_review_rows: list[dict],
    packet_rows: list[dict],
    validator_id: str,
    prior_audio_cohort_lock_sha256: str,
    config: dict,
) -> list[dict]:
    if validator_id in {row["author_id"] for row in author_review_rows}:
        raise ValueError("Question authors and frame validators must be disjoint")
    packets = index_packets_for_v2(packet_rows)
    output = []
    for author in author_review_rows:
        validate_frozen_author_review(author, config)
        if author["authoring_status"] != "candidate" or author["speech_relevance_status"] != "addressed":
            continue
        packet = packets[author["packet_id"]]
        order = option_order(author, validator_id)
        options = [
            {
                "option_id": blinded_option_id(author, validator_id, index),
                "text": author["source_options"][index],
            }
            for index in order
        ]
        frame = {
            "schema_version": author["schema_version"],
            "stage": "validator_frame_only_answer_pass",
            "packet_id": author["packet_id"],
            "question_lock_sha256": author["question_lock_sha256"],
            "source_question": author["source_question"],
            "source_options": options,
            "option_order_sha256": canonical_hash(
                {"packet_id": author["packet_id"], "order": order}
            ),
            "frame_path": f'packets/{packet_dir_name(author["packet_id"])}/frame.jpg',
            "frame_binding_hmac": frame_binding_hmac(
                author["packet_id"], packet["workspace_frame_sha256"]
            ),
            "validator_id": validator_id,
            "frame_support_status": "pending",
            "frame_answer_option_id": None,
            "evidence_subtypes": [],
            "frame_confidence": None,
            "frame_note": "",
            "frame_submitted_at_utc": None,
            "frame_annotation_lock_sha256": None,
            "prior_stage_cohort_lock_sha256": prior_audio_cohort_lock_sha256,
        }
        frame["frame_task_lock_sha256"] = canonical_hash(frame_task_lock_payload(frame))
        assert_no_forbidden_keys(frame, GLOBAL_FORBIDDEN_KEY_PARTS + ("audio", "canonical"))
        output.append(frame)
    output.sort(
        key=lambda row: hashlib.sha256(
            f'acl6060-v2-validator-order:{validator_id}:{row["packet_id"]}'.encode()
        ).hexdigest()
    )
    return output


def materialize_frame_validator_view(
    rows: list[dict],
    packet_rows: list[dict],
    workspace_root: Path,
    output_root: Path,
) -> list[dict]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output root is not empty: {output_root}")
    packets = index_packets_for_v2(packet_rows)
    for row in rows:
        source = workspace_root / packets[row["packet_id"]]["workspace_frame_path"]
        destination = output_root / row["frame_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        frame_sha256 = sha256_file(destination)
        packet = packets[row["packet_id"]]
        if frame_sha256 != packet["workspace_frame_sha256"]:
            raise ValueError(f'Frame media hash mismatch: {row["packet_id"]}')
        if row["frame_binding_hmac"] != frame_binding_hmac(
            row["packet_id"], frame_sha256
        ):
            raise ValueError(f'Frame media binding mismatch: {row["packet_id"]}')
    write_jsonl(output_root / "frame_validation.jsonl", rows)
    return rows


def validate_frame_annotation(row: dict, config: dict) -> None:
    if row.get("frame_task_lock_sha256") != canonical_hash(frame_task_lock_payload(row)):
        raise ValueError(f'Frame task lock mismatch: {row["packet_id"]}')
    allowed = set(config["frame_validation"]["frame_support_status"])
    status = row["frame_support_status"]
    if status not in allowed or status == "pending":
        raise ValueError(f'Incomplete frame annotation: {row["packet_id"]}')
    if status == "supported":
        answer = row["frame_answer_option_id"]
        option_ids = {option["option_id"] for option in row["source_options"]}
        if answer not in option_ids:
            raise ValueError(f'Invalid frame answer: {row["packet_id"]}')
        if not row["evidence_subtypes"] or not set(row["evidence_subtypes"]) <= set(
            config["evidence_subtypes"]
        ):
            raise ValueError(f'Invalid validator evidence subtype: {row["packet_id"]}')
    elif row["frame_answer_option_id"] is not None:
        raise ValueError(f'Non-supported frame carries answer: {row["packet_id"]}')
    if row["frame_confidence"] not in {"low", "medium", "high"}:
        raise ValueError(f'Invalid frame confidence: {row["packet_id"]}')
    parse_utc(row["frame_submitted_at_utc"])


def freeze_frame_annotation(row: dict, config: dict) -> dict:
    validate_frame_annotation(row, config)
    return {**row, "frame_annotation_lock_sha256": canonical_hash(row)}


def validate_frozen_audio_annotation(row: dict) -> None:
    validate_audio_annotation(row)
    if row.get("audio_annotation_lock_sha256") != canonical_hash(audio_lock_payload(row)):
        raise ValueError(f'Audio annotation lock mismatch: {row["packet_id"]}')


def validate_frozen_frame_annotation(row: dict, config: dict) -> None:
    validate_frame_annotation(row, config)
    if row.get("frame_annotation_lock_sha256") != canonical_hash(row):
        raise ValueError(f'Frame annotation lock mismatch: {row["packet_id"]}')


def option_text(row: dict, option_id: str | None) -> str | None:
    if option_id is None:
        return None
    matches = [option["text"] for option in row["source_options"] if option["option_id"] == option_id]
    if len(matches) != 1:
        raise ValueError(f'Unknown or duplicate option id: {row["packet_id"]}: {option_id}')
    return matches[0]


def canonical_validator_answer(author: dict, validator_row: dict) -> str:
    canonical_text = author["source_options"][author["canonical_answer_index"]]
    matches = [
        option["option_id"]
        for option in validator_row["source_options"]
        if option["text"] == canonical_text
    ]
    if len(matches) != 1:
        raise ValueError(f'Canonical option missing or duplicated: {author["packet_id"]}')
    return matches[0]


def first_stable_correct(
    row: dict, canonical_option_id: str, minimum_steps: int
) -> dict | None:
    judgments = row["prefix_judgments"]
    for index, judgment in enumerate(judgments):
        if judgment["status"] != "option" or judgment["option_id"] != canonical_option_id:
            continue
        if len(judgments) - index >= minimum_steps and all(
            later["status"] == "option" and later["option_id"] == canonical_option_id
            for later in judgments[index:]
        ):
            return judgment
    return None


def first_stable_option(row: dict, minimum_steps: int) -> dict | None:
    judgments = row["prefix_judgments"]
    for index, judgment in enumerate(judgments):
        if judgment["status"] != "option":
            continue
        option_id = judgment["option_id"]
        if len(judgments) - index >= minimum_steps and all(
            later["status"] == "option" and later["option_id"] == option_id
            for later in judgments[index:]
        ):
            return judgment
    return None


def jaccard(left: list[str], right: list[str]) -> float:
    union = set(left) | set(right)
    if not union:
        return 1.0
    return len(set(left) & set(right)) / len(union)


def prepare_adjudication_rows(
    report_rows: list[dict], adjudicator_id: str
) -> list[dict]:
    output = []
    for report in report_rows:
        if not report["requires_adjudication"]:
            continue
        output.append(
            {
                "schema_version": "acl6060_source_event_adjudication_v2",
                "packet_id": report["packet_id"],
                "report_row_lock_sha256": canonical_hash(report),
                "adjudicator_id": adjudicator_id,
                "adjudication_status": "pending",
                "adjudicated_primary_eligible": None,
                "adjudicated_boundary_sec": None,
                "reason_labels": [],
                "note": "",
                "submitted_at_utc": None,
                "adjudication_lock_sha256": None,
            }
        )
    return output


def freeze_adjudication_rows(
    rows: list[dict], report_rows: list[dict], config: dict
) -> list[dict]:
    reports = index_unique(
        [row for row in report_rows if row["requires_adjudication"]]
    )
    annotations = index_unique(rows)
    if set(reports) != set(annotations):
        raise ValueError("Adjudication and flagged report packet ids differ")
    output = []
    for row in rows:
        report = reports[row["packet_id"]]
        if row["report_row_lock_sha256"] != canonical_hash(report):
            raise ValueError(f'Adjudication report lock mismatch: {row["packet_id"]}')
        prior_role_ids = {
            report["question_author_id"],
            *report["audio_validator_ids"],
            *report["frame_validator_ids"],
        }
        if row["adjudicator_id"] in prior_role_ids:
            raise ValueError(f'Adjudicator role overlaps prior role: {row["packet_id"]}')
        status = row["adjudication_status"]
        if status not in set(config["adjudication"]["status"]) or status == "pending":
            raise ValueError(f'Incomplete adjudication: {row["packet_id"]}')
        if not row["reason_labels"] or not set(row["reason_labels"]) <= set(
            config["adjudication"]["reason_labels"]
        ):
            raise ValueError(f'Invalid adjudication reason: {row["packet_id"]}')
        if not row["note"].strip():
            raise ValueError(f'Adjudication lacks note: {row["packet_id"]}')
        parse_utc(row["submitted_at_utc"])
        if status == "resolved" and not isinstance(
            row["adjudicated_primary_eligible"], bool
        ):
            raise ValueError(f'Resolved adjudication lacks decision: {row["packet_id"]}')
        if status == "unresolvable" and row["adjudicated_primary_eligible"] is not None:
            raise ValueError(f'Unresolvable adjudication carries decision: {row["packet_id"]}')
        decision = row["adjudicated_primary_eligible"]
        boundary = row["adjudicated_boundary_sec"]
        if decision is True:
            if not report["adjudication_positive_allowed"]:
                raise ValueError(f'Hard failure cannot be adjudicated positive: {row["packet_id"]}')
            if not isinstance(boundary, (int, float)):
                raise ValueError(f'Positive adjudication lacks boundary: {row["packet_id"]}')
            if not report["t_evidence_sec"] < float(boundary) <= report["causal_audio_end_sec"]:
                raise ValueError(f'Adjudicated boundary is outside causal window: {row["packet_id"]}')
            boundary_index = grid_index(
                float(boundary),
                float(report["t_evidence_sec"]),
                float(report["prefix_step_sec"]),
            )
            minimum_steps = config["audio_boundary"]["minimum_stable_correct_steps"]
            if boundary_index > len(expected_prefix_grid(report)) - minimum_steps:
                raise ValueError(
                    f'Adjudicated boundary lacks stable tail: {row["packet_id"]}'
                )
        elif boundary is not None:
            raise ValueError(f'Non-positive adjudication carries boundary: {row["packet_id"]}')
        frozen = {**row, "adjudication_lock_sha256": canonical_hash(row)}
        output.append(frozen)
    return output


def apply_adjudications(
    report_rows: list[dict], adjudication_rows: list[dict] | None
) -> list[dict]:
    adjudications = index_unique(adjudication_rows or [])
    flagged_ids = {
        row["packet_id"] for row in report_rows if row["requires_adjudication"]
    }
    if adjudications and set(adjudications) != flagged_ids:
        raise ValueError("Adjudication and flagged report packet ids differ")
    output = []
    for report in report_rows:
        row = dict(report)
        row["adjudication_resolved"] = not report["requires_adjudication"]
        if report["packet_id"] in adjudications:
            adjudication = adjudications[report["packet_id"]]
            if adjudication["report_row_lock_sha256"] != canonical_hash(report):
                raise ValueError(f'Adjudication report lock mismatch: {report["packet_id"]}')
            if adjudication.get("adjudication_lock_sha256") != canonical_hash(adjudication):
                raise ValueError(f'Adjudication lock mismatch: {report["packet_id"]}')
            row["adjudication_status"] = adjudication["adjudication_status"]
            row["adjudication_resolved"] = (
                adjudication["adjudication_status"] == "resolved"
            )
            row["primary_eligible"] = adjudication["adjudicated_primary_eligible"]
            row["adjudicated_boundary_sec"] = adjudication[
                "adjudicated_boundary_sec"
            ]
            if adjudication["adjudicated_primary_eligible"] is True:
                row["conservative_first_stable_correct_sec"] = adjudication[
                    "adjudicated_boundary_sec"
                ]
                row["anticipation_lead_sec"] = round(
                    float(adjudication["adjudicated_boundary_sec"])
                    - float(row["t_evidence_sec"]),
                    6,
                )
        output.append(row)
    return output


def validate_frozen_adjudication_rows(
    rows: list[dict], report_rows: list[dict], config: dict
) -> None:
    expected = index_unique(freeze_adjudication_rows(rows, report_rows, config))
    supplied = index_unique(rows)
    for packet_id, row in supplied.items():
        if row.get("adjudication_lock_sha256") != expected[packet_id][
            "adjudication_lock_sha256"
        ]:
            raise ValueError(f"Adjudication lock mismatch: {packet_id}")


def estimate_stratified_prevalence(
    rows: list[dict], sampling_design: dict
) -> dict:
    if any(row["primary_eligible"] is None for row in rows):
        excluded_count = sum(bool(row.get("source_excluded")) for row in rows)
        return {
            "status": "UNRESOLVED_MISSING_OUTCOME",
            "estimate": None,
            "standard_error": None,
            "ci95": None,
            "excluded_count": excluded_count,
            "unresolved_adjudication_count": sum(
                row.get("requires_adjudication", False)
                and not row.get("adjudication_resolved", False)
                for row in rows
            ),
        }
    strata = {
        (entry["talk_id"], entry["selection_stratum"]): entry
        for entry in sampling_design["strata"]
    }
    observed_keys = {(row["talk_id"], row["selection_stratum"]) for row in rows}
    if observed_keys != set(strata):
        raise ValueError("Observed and sampling-design strata differ")
    estimated_total = 0.0
    total_variance = 0.0
    for key, design in strata.items():
        outcomes = [
            float(row["primary_eligible"])
            for row in rows
            if (row["talk_id"], row["selection_stratum"]) == key
        ]
        sample_count = int(design["selected_count"])
        pool_count = int(design["pool_count"])
        if len(outcomes) != sample_count:
            raise ValueError(f"Wrong sampled count for stratum: {key}")
        mean = sum(outcomes) / sample_count
        estimated_total += pool_count * mean
        if sample_count > 1:
            sample_variance = sum((value - mean) ** 2 for value in outcomes) / (
                sample_count - 1
            )
            fraction = sample_count / pool_count
            total_variance += (
                pool_count**2 * (1.0 - fraction) * sample_variance / sample_count
            )
    population_count = int(sampling_design["total_observation_count"])
    if sum(int(entry["pool_count"]) for entry in strata.values()) != population_count:
        raise ValueError("Sampling-design pool counts do not sum to population")
    estimate = estimated_total / population_count
    standard_error = math.sqrt(total_variance) / population_count
    return {
        "status": "ESTIMATED",
        "estimand": "frozen_observation_level_primary_event_prevalence",
        "population_count": population_count,
        "estimate": estimate,
        "standard_error": standard_error,
        "ci95": [
            max(0.0, estimate - 1.96 * standard_error),
            min(1.0, estimate + 1.96 * standard_error),
        ],
        "variance_method": "stratified_srs_without_replacement_normal_ci",
    }


def build_agreement_report(
    author_rows: list[dict],
    packet_rows: list[dict],
    audio_a_rows: list[dict],
    audio_b_rows: list[dict],
    frame_a_rows: list[dict],
    frame_b_rows: list[dict],
    config: dict,
    sampling_design: dict | None = None,
    adjudication_rows: list[dict] | None = None,
) -> tuple[list[dict], dict]:
    authors = index_unique(author_rows)
    packets = index_packets_for_v2(packet_rows)
    audio_a = index_unique(audio_a_rows)
    audio_b = index_unique(audio_b_rows)
    frame_a = index_unique(frame_a_rows)
    frame_b = index_unique(frame_b_rows)
    if set(authors) != set(packets):
        raise ValueError("Author-review and packet ids differ")
    for author in author_rows:
        validate_frozen_author_review(author, config)
    validator_packet_ids = {
        packet_id
        for packet_id, author in authors.items()
        if author["authoring_status"] == "candidate"
        and author["speech_relevance_status"] == "addressed"
    }
    if any(
        set(candidate_rows) != validator_packet_ids
        for candidate_rows in (audio_a, audio_b, frame_a, frame_b)
    ):
        raise ValueError("Validator packet ids differ")
    if {row["validator_id"] for row in audio_a_rows} & {
        row["validator_id"] for row in audio_b_rows
    }:
        raise ValueError("Validator ids must be distinct")
    validator_cohorts = [
        {row["validator_id"] for row in rows}
        for rows in (audio_a_rows, audio_b_rows, frame_a_rows, frame_b_rows)
    ]
    nonempty_cohorts = [cohort for cohort in validator_cohorts if cohort]
    if any(len(cohort) != 1 for cohort in nonempty_cohorts):
        raise ValueError("Each validator sheet must contain one validator id")
    all_validator_ids = set().union(*nonempty_cohorts) if nonempty_cohorts else set()
    if len(all_validator_ids) != len(nonempty_cohorts):
        raise ValueError("Audio and frame validator cohorts must be disjoint")
    if all_validator_ids & {row["author_id"] for row in author_rows}:
        raise ValueError("Question authors and validators must be disjoint")
    expected_audio_cohort_lock = (
        audio_cohort_lock_sha256(audio_a_rows, audio_b_rows)
        if validator_packet_ids
        else None
    )
    max_gap = config["audio_boundary"]["maximum_primary_boundary_disagreement_steps"]
    rows = []
    for packet_id in sorted(authors):
        author = authors[packet_id]
        packet = packets[packet_id]
        if packet_id not in validator_packet_ids:
            source_status = (
                author["authoring_status"]
                if author["authoring_status"] != "candidate"
                else f'speech_{author["speech_relevance_status"]}'
            )
            source_excluded = (
                author["authoring_status"] == "exclude_other"
                or author["speech_relevance_status"] == "exclude_other"
            )
            rows.append(
                {
                    "packet_id": packet_id,
                    "source_packet_id": packet["packet_id"],
                    "talk_id": author["talk_id"],
                    "selection_stratum": packet["selection_stratum"],
                    "negative_labels": author["negative_labels"],
                    "exclusion_labels": author["exclusion_labels"]
                    + author["speech_exclusion_labels"],
                    "source_inventory_status": source_status,
                    "source_excluded": source_excluded,
                    "frame_answer_exact_agreement": None,
                    "frame_support_exact_agreement": None,
                    "audio_stable_answer_exact_agreement": None,
                    "audio_primary_component_agreement": None,
                    "boundary_gap_steps": None,
                    "boundary_interval_overlap": None,
                    "evidence_subtype_jaccard": None,
                    "primary_eligible": None if source_excluded else False,
                    "requires_adjudication": False,
                    "adjudication_positive_allowed": False,
                }
            )
            continue
        a_audio = audio_a[packet_id]
        b_audio = audio_b[packet_id]
        a_frame = frame_a[packet_id]
        b_frame = frame_b[packet_id]
        validate_frozen_frame_annotation(a_frame, config)
        validate_frozen_frame_annotation(b_frame, config)
        validate_frozen_audio_annotation(a_audio)
        validate_frozen_audio_annotation(b_audio)
        if any(
            frame["prior_stage_cohort_lock_sha256"] != expected_audio_cohort_lock
            for frame in (a_frame, b_frame)
        ):
            raise ValueError(f'Frame released against wrong audio cohort: {packet_id}')
        a_audio_canonical = canonical_validator_answer(author, a_audio)
        b_audio_canonical = canonical_validator_answer(author, b_audio)
        a_frame_canonical = canonical_validator_answer(author, a_frame)
        b_frame_canonical = canonical_validator_answer(author, b_frame)
        a_frame_correct = (
            a_frame["frame_support_status"] == "supported"
            and a_frame["frame_answer_option_id"] == a_frame_canonical
        )
        b_frame_correct = (
            b_frame["frame_support_status"] == "supported"
            and b_frame["frame_answer_option_id"] == b_frame_canonical
        )
        minimum_steps = config["audio_boundary"]["minimum_stable_correct_steps"]
        a_stable = first_stable_option(a_audio, minimum_steps)
        b_stable = first_stable_option(b_audio, minimum_steps)
        a_stable_correct = first_stable_correct(
            a_audio, a_audio_canonical, minimum_steps
        )
        b_stable_correct = first_stable_correct(
            b_audio, b_audio_canonical, minimum_steps
        )
        a_audio_insufficient_at_evidence = not (
            a_audio["prefix_judgments"][0]["status"] == "option"
            and a_audio["prefix_judgments"][0]["option_id"] == a_audio_canonical
        )
        b_audio_insufficient_at_evidence = not (
            b_audio["prefix_judgments"][0]["status"] == "option"
            and b_audio["prefix_judgments"][0]["option_id"] == b_audio_canonical
        )
        a_audio_component = (
            a_stable_correct is not None
            and a_audio_insufficient_at_evidence
            and a_audio["question_only_status"] == "not_answerable"
        )
        b_audio_component = (
            b_stable_correct is not None
            and b_audio_insufficient_at_evidence
            and b_audio["question_only_status"] == "not_answerable"
        )
        step = float(a_audio["prefix_step_sec"])
        if a_stable_correct is not None and b_stable_correct is not None:
            gap_steps = abs(a_stable_correct["step_index"] - b_stable_correct["step_index"])
            a_previous = float(a_stable_correct["prefix_end_sec"]) - step
            b_previous = float(b_stable_correct["prefix_end_sec"]) - step
            interval_overlap = max(a_previous, b_previous) <= min(
                float(a_stable_correct["prefix_end_sec"]),
                float(b_stable_correct["prefix_end_sec"]),
            )
            conservative_boundary_sec = min(
                float(a_stable_correct["prefix_end_sec"]),
                float(b_stable_correct["prefix_end_sec"]),
            )
            anticipation_lead_sec = round(
                conservative_boundary_sec - float(author["t_evidence_sec"]), 6
            )
        else:
            gap_steps = None
            interval_overlap = False
            conservative_boundary_sec = None
            anticipation_lead_sec = None
        frame_support_agreement = (
            a_frame["frame_support_status"] == b_frame["frame_support_status"]
        )
        frame_agreement = (
            option_text(a_frame, a_frame["frame_answer_option_id"])
            == option_text(b_frame, b_frame["frame_answer_option_id"])
            if a_frame["frame_answer_option_id"] is not None
            and b_frame["frame_answer_option_id"] is not None
            else None
        )
        a_audio_outcome = (
            option_text(a_audio, a_stable["option_id"]) if a_stable else "right_censored"
        )
        b_audio_outcome = (
            option_text(b_audio, b_stable["option_id"]) if b_stable else "right_censored"
        )
        audio_agreement = a_audio_outcome == b_audio_outcome
        audio_component_agreement = a_audio_component == b_audio_component
        within_gap = gap_steps is not None and gap_steps <= max_gap
        boundary_resolved = (
            within_gap
            if a_audio_component and b_audio_component
            else not a_audio_component and not b_audio_component
        )
        requires_adjudication = not (
            audio_component_agreement
            and frame_support_agreement
            and (frame_agreement is not False)
            and audio_agreement
            and boundary_resolved
        )
        provisional_primary = (
            a_audio_component
            and b_audio_component
            and a_frame_correct
            and b_frame_correct
            and within_gap
        )
        hard_failure_reasons = []
        if any(
            audio["question_only_status"] == "answerable"
            for audio in (a_audio, b_audio)
        ):
            hard_failure_reasons.append("question_only_answerable")
        rows.append(
            {
                "packet_id": packet_id,
                "source_packet_id": packet["packet_id"],
                "talk_id": author["talk_id"],
                "selection_stratum": packet["selection_stratum"],
                "canonical_answer_text": author["source_options"][
                    author["canonical_answer_index"]
                ],
                "source_inventory_status": "validated_candidate",
                "question_author_id": author["author_id"],
                "audio_validator_ids": [
                    a_audio["validator_id"],
                    b_audio["validator_id"],
                ],
                "frame_validator_ids": [
                    a_frame["validator_id"],
                    b_frame["validator_id"],
                ],
                "source_excluded": False,
                "t_evidence_sec": author["t_evidence_sec"],
                "causal_audio_end_sec": min(
                    a_audio["causal_audio_end_sec"], b_audio["causal_audio_end_sec"]
                ),
                "prefix_step_sec": step,
                "frame_support_exact_agreement": frame_support_agreement,
                "frame_answer_exact_agreement": frame_agreement,
                "audio_stable_answer_exact_agreement": audio_agreement,
                "validator_a_first_stable_correct_sec": (
                    a_stable_correct["prefix_end_sec"] if a_stable_correct else None
                ),
                "validator_b_first_stable_correct_sec": (
                    b_stable_correct["prefix_end_sec"] if b_stable_correct else None
                ),
                "conservative_first_stable_correct_sec": conservative_boundary_sec,
                "anticipation_lead_sec": anticipation_lead_sec,
                "audio_validator_a_primary_component": a_audio_component,
                "audio_validator_b_primary_component": b_audio_component,
                "audio_primary_component_agreement": audio_component_agreement,
                "frame_validator_a_correct": a_frame_correct,
                "frame_validator_b_correct": b_frame_correct,
                "boundary_gap_steps": gap_steps,
                "boundary_interval_overlap": interval_overlap,
                "evidence_subtype_jaccard": round(
                    jaccard(a_frame["evidence_subtypes"], b_frame["evidence_subtypes"]), 6
                ),
                "primary_eligible": (
                    None if requires_adjudication else provisional_primary
                ),
                "requires_adjudication": requires_adjudication,
                "hard_failure_reasons": hard_failure_reasons,
                "adjudication_positive_allowed": not hard_failure_reasons,
            }
        )
    if adjudication_rows is not None:
        validate_frozen_adjudication_rows(adjudication_rows, rows, config)
    rows = apply_adjudications(rows, adjudication_rows)
    agreement_rows = [
        row for row in rows if row["source_inventory_status"] == "validated_candidate"
    ]

    def agreement_rate(key: str) -> float | None:
        values = [row[key] for row in agreement_rows if row.get(key) is not None]
        if not values:
            return None
        return sum(bool(value) for value in values) / len(values)

    subtype_values = [
        row["evidence_subtype_jaccard"]
        for row in agreement_rows
        if row["evidence_subtype_jaccard"] is not None
    ]
    summary = {
        "schema_version": "acl6060_source_event_annotation_v2",
        "status": "SOURCE_SIDE_AGREEMENT_ONLY",
        "packet_count": len(rows),
        "validator_packet_count": len(agreement_rows),
        "author_negative_or_excluded_count": len(rows) - len(agreement_rows),
        "primary_eligible_count": sum(row["primary_eligible"] is True for row in rows),
        "resolved_packet_count": sum(
            row["primary_eligible"] is not None for row in rows
        ),
        "source_excluded_count": sum(bool(row.get("source_excluded")) for row in rows),
        "requires_adjudication_count": sum(row["requires_adjudication"] for row in rows),
        "unresolved_adjudication_count": sum(
            not row["adjudication_resolved"] for row in rows
        ),
        "frame_answer_exact_agreement_rate": agreement_rate(
            "frame_answer_exact_agreement"
        ),
        "frame_support_exact_agreement_rate": agreement_rate(
            "frame_support_exact_agreement"
        ),
        "audio_answer_exact_agreement_rate": agreement_rate(
            "audio_stable_answer_exact_agreement"
        ),
        "audio_primary_component_agreement_rate": agreement_rate(
            "audio_primary_component_agreement"
        ),
        "boundary_interval_overlap_rate": agreement_rate(
            "boundary_interval_overlap"
        ),
        "mean_evidence_subtype_jaccard": (
            sum(subtype_values) / len(subtype_values) if subtype_values else None
        ),
        "target_or_reference_consumed": False,
        "stratified_yield": [
            {
                "talk_id": talk_id,
                "selection_stratum": stratum,
                "packet_count": sum(
                    row["talk_id"] == talk_id and row["selection_stratum"] == stratum
                    for row in rows
                ),
                "primary_eligible_count": sum(
                    row["talk_id"] == talk_id
                    and row["selection_stratum"] == stratum
                    and row["primary_eligible"] is True
                    for row in rows
                ),
                "unresolved_count": sum(
                    row["talk_id"] == talk_id
                    and row["selection_stratum"] == stratum
                    and row["primary_eligible"] is None
                    for row in rows
                ),
            }
            for talk_id, stratum in sorted(
                {(row["talk_id"], row["selection_stratum"]) for row in rows}
            )
        ],
        "overall_prevalence": (
            estimate_stratified_prevalence(rows, sampling_design)
            if sampling_design is not None
            else {
                "status": "SAMPLING_DESIGN_NOT_PROVIDED",
                "estimate": None,
            }
        ),
    }
    return rows, summary


def command_prepare_author(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    mapping_out = args.mapping_out.resolve()
    if mapping_out == output_root or output_root in mapping_out.parents:
        raise ValueError("Scorer mapping must be outside the author output root")
    packets = load_jsonl(args.packet_manifest)
    rows = materialize_author_view(
        packets, args.workspace_root, args.output_root, args.author_id
    )
    mapping = [
        {
            "packet_id": blinded_packet_id(packet["packet_id"]),
            "source_packet_id": packet["packet_id"],
            "talk_id": packet["talk_id"],
            "selection_stratum": packet["selection_stratum"],
        }
        for packet in packets
    ]
    mapping.sort(key=lambda row: row["packet_id"])
    write_jsonl(args.mapping_out, mapping)
    print(json.dumps({"authoring_packet_count": len(rows)}))


def command_freeze_author(args: argparse.Namespace) -> None:
    packets = load_jsonl(args.packet_manifest)
    verify_frame_stage_media(
        load_jsonl(args.author_sheet), packets, args.author_frame_root
    )
    rows = freeze_author_rows(
        load_jsonl(args.author_sheet),
        packets,
        load_json(args.config),
    )
    write_jsonl(args.output, rows)
    print(json.dumps({"row_count": len(rows)}))


def command_prepare_author_audio(args: argparse.Namespace) -> None:
    rows = materialize_author_audio_review_view(
        load_jsonl(args.author_review),
        load_jsonl(args.packet_manifest),
        args.acl_root,
        args.output_root,
    )
    print(
        json.dumps(
            {
                "row_count": len(rows),
                "candidate_audio_count": sum(
                    row["authoring_status"] == "candidate" for row in rows
                ),
            }
        )
    )


def command_freeze_author_audio(args: argparse.Namespace) -> None:
    rows = load_jsonl(args.author_review)
    candidate_rows = [row for row in rows if row["authoring_status"] == "candidate"]
    verify_stage_media(candidate_rows, args.author_audio_root, "audio_path", "audio_sha256")
    frozen = freeze_author_audio_reviews(
        rows,
        load_jsonl(args.packet_manifest),
        load_json(args.config),
    )
    write_jsonl(args.output, frozen)
    print(json.dumps({"row_count": len(frozen)}))


def command_prepare_audio(args: argparse.Namespace) -> None:
    config = load_json(args.config)
    reviews = load_jsonl(args.author_review)
    candidate_reviews = [
        row for row in reviews if row["authoring_status"] == "candidate"
    ]
    verify_stage_media(candidate_reviews, args.author_audio_root, "audio_path", "audio_sha256")
    rows_a = make_audio_validator_rows(reviews, args.validator_a, config)
    rows_b = make_audio_validator_rows(reviews, args.validator_b, config)
    packets = load_jsonl(args.packet_manifest)
    materialize_audio_validator_view(rows_a, packets, args.acl_root, args.output_root_a)
    materialize_audio_validator_view(rows_b, packets, args.acl_root, args.output_root_b)
    print(json.dumps({"validator_packet_count": len(rows_a)}))


def command_freeze_audio(args: argparse.Namespace) -> None:
    reviews = load_jsonl(args.author_review)
    packets = load_jsonl(args.packet_manifest)
    config = load_json(args.config)
    audio_a = load_jsonl(args.audio_a)
    audio_b = load_jsonl(args.audio_b)
    verify_stage_media(audio_a, args.audio_root_a, "audio_path", "audio_sha256")
    verify_stage_media(audio_b, args.audio_root_b, "audio_path", "audio_sha256")
    verify_sequential_interaction_log(audio_a, args.event_log_a)
    verify_sequential_interaction_log(audio_b, args.event_log_b)
    frozen_a = freeze_audio_rows(audio_a, reviews, packets, config)
    frozen_b = freeze_audio_rows(audio_b, reviews, packets, config)
    if {row["validator_id"] for row in frozen_a} & {row["validator_id"] for row in frozen_b}:
        raise ValueError("Validator ids must be distinct")
    for path, rows in (
        (args.frozen_audio_a, frozen_a),
        (args.frozen_audio_b, frozen_b),
    ):
        write_jsonl(path, rows)
    cohort_lock = audio_cohort_lock_sha256(frozen_a, frozen_b)
    audio_validator_ids = {
        row["validator_id"] for row in frozen_a + frozen_b
    }
    frame_validator_ids = {args.frame_validator_a, args.frame_validator_b}
    if len(frame_validator_ids) != 2 or audio_validator_ids & frame_validator_ids:
        raise ValueError("Audio and frame validator cohorts must be disjoint")
    frame_a = make_frame_validator_rows(
        reviews, packets, args.frame_validator_a, cohort_lock, config
    )
    frame_b = make_frame_validator_rows(
        reviews, packets, args.frame_validator_b, cohort_lock, config
    )
    materialize_frame_validator_view(frame_a, packets, args.workspace_root, args.frame_root_a)
    materialize_frame_validator_view(frame_b, packets, args.workspace_root, args.frame_root_b)
    print(json.dumps({"validator_packet_count": len(frozen_a)}))


def command_freeze_frame(args: argparse.Namespace) -> None:
    config = load_json(args.config)
    rows = load_jsonl(args.frame_sheet)
    verify_frame_stage_media(rows, load_jsonl(args.packet_manifest), args.frame_root)
    frozen = [freeze_frame_annotation(row, config) for row in rows]
    write_jsonl(args.output, frozen)
    print(json.dumps({"row_count": len(frozen)}))


def command_report(args: argparse.Namespace) -> None:
    frame_a = load_jsonl(args.frame_a)
    frame_b = load_jsonl(args.frame_b)
    packets = load_jsonl(args.packet_manifest)
    verify_frame_stage_media(frame_a, packets, args.frame_root_a)
    verify_frame_stage_media(frame_b, packets, args.frame_root_b)
    rows, summary = build_agreement_report(
        load_jsonl(args.author_review),
        packets,
        load_jsonl(args.audio_a),
        load_jsonl(args.audio_b),
        frame_a,
        frame_b,
        load_json(args.config),
        load_json(args.sampling_design),
        load_jsonl(args.adjudication) if args.adjudication else None,
    )
    write_jsonl(args.output, rows)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


def command_prepare_adjudication(args: argparse.Namespace) -> None:
    rows = prepare_adjudication_rows(
        load_jsonl(args.report_rows), args.adjudicator_id
    )
    write_jsonl(args.output, rows)
    print(json.dumps({"row_count": len(rows)}))


def command_freeze_adjudication(args: argparse.Namespace) -> None:
    rows = freeze_adjudication_rows(
        load_jsonl(args.adjudication),
        load_jsonl(args.report_rows),
        load_json(args.config),
    )
    write_jsonl(args.output, rows)
    print(json.dumps({"row_count": len(rows)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-author")
    prepare.add_argument("--packet-manifest", type=Path, required=True)
    prepare.add_argument("--workspace-root", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--mapping-out", type=Path, required=True)
    prepare.add_argument("--author-id", required=True)
    prepare.set_defaults(handler=command_prepare_author)

    freeze_author = subparsers.add_parser("freeze-author")
    freeze_author.add_argument("--author-sheet", type=Path, required=True)
    freeze_author.add_argument("--author-frame-root", type=Path, required=True)
    freeze_author.add_argument("--packet-manifest", type=Path, required=True)
    freeze_author.add_argument("--config", type=Path, required=True)
    freeze_author.add_argument("--output", type=Path, required=True)
    freeze_author.set_defaults(handler=command_freeze_author)

    prepare_author_audio = subparsers.add_parser("prepare-author-audio")
    prepare_author_audio.add_argument("--author-review", type=Path, required=True)
    prepare_author_audio.add_argument("--packet-manifest", type=Path, required=True)
    prepare_author_audio.add_argument("--acl-root", type=Path, required=True)
    prepare_author_audio.add_argument("--output-root", type=Path, required=True)
    prepare_author_audio.set_defaults(handler=command_prepare_author_audio)

    freeze_author_audio = subparsers.add_parser("freeze-author-audio")
    freeze_author_audio.add_argument("--author-review", type=Path, required=True)
    freeze_author_audio.add_argument("--author-audio-root", type=Path, required=True)
    freeze_author_audio.add_argument("--packet-manifest", type=Path, required=True)
    freeze_author_audio.add_argument("--config", type=Path, required=True)
    freeze_author_audio.add_argument("--output", type=Path, required=True)
    freeze_author_audio.set_defaults(handler=command_freeze_author_audio)

    prepare_audio = subparsers.add_parser("prepare-audio")
    prepare_audio.add_argument("--author-review", type=Path, required=True)
    prepare_audio.add_argument("--author-audio-root", type=Path, required=True)
    prepare_audio.add_argument("--packet-manifest", type=Path, required=True)
    prepare_audio.add_argument("--acl-root", type=Path, required=True)
    prepare_audio.add_argument("--config", type=Path, required=True)
    prepare_audio.add_argument("--validator-a", required=True)
    prepare_audio.add_argument("--validator-b", required=True)
    prepare_audio.add_argument("--output-root-a", type=Path, required=True)
    prepare_audio.add_argument("--output-root-b", type=Path, required=True)
    prepare_audio.set_defaults(handler=command_prepare_audio)

    freeze_audio = subparsers.add_parser("freeze-audio")
    freeze_audio.add_argument("--author-review", type=Path, required=True)
    freeze_audio.add_argument("--packet-manifest", type=Path, required=True)
    freeze_audio.add_argument("--audio-a", type=Path, required=True)
    freeze_audio.add_argument("--audio-b", type=Path, required=True)
    freeze_audio.add_argument("--audio-root-a", type=Path, required=True)
    freeze_audio.add_argument("--audio-root-b", type=Path, required=True)
    freeze_audio.add_argument("--event-log-a", type=Path, required=True)
    freeze_audio.add_argument("--event-log-b", type=Path, required=True)
    freeze_audio.add_argument("--config", type=Path, required=True)
    freeze_audio.add_argument("--frozen-audio-a", type=Path, required=True)
    freeze_audio.add_argument("--frozen-audio-b", type=Path, required=True)
    freeze_audio.add_argument("--workspace-root", type=Path, required=True)
    freeze_audio.add_argument("--frame-root-a", type=Path, required=True)
    freeze_audio.add_argument("--frame-root-b", type=Path, required=True)
    freeze_audio.add_argument("--frame-validator-a", required=True)
    freeze_audio.add_argument("--frame-validator-b", required=True)
    freeze_audio.set_defaults(handler=command_freeze_audio)

    freeze_frame = subparsers.add_parser("freeze-frame")
    freeze_frame.add_argument("--frame-sheet", type=Path, required=True)
    freeze_frame.add_argument("--frame-root", type=Path, required=True)
    freeze_frame.add_argument("--packet-manifest", type=Path, required=True)
    freeze_frame.add_argument("--config", type=Path, required=True)
    freeze_frame.add_argument("--output", type=Path, required=True)
    freeze_frame.set_defaults(handler=command_freeze_frame)

    prepare_adjudication = subparsers.add_parser("prepare-adjudication")
    prepare_adjudication.add_argument("--report-rows", type=Path, required=True)
    prepare_adjudication.add_argument("--adjudicator-id", required=True)
    prepare_adjudication.add_argument("--output", type=Path, required=True)
    prepare_adjudication.set_defaults(handler=command_prepare_adjudication)

    freeze_adjudication = subparsers.add_parser("freeze-adjudication")
    freeze_adjudication.add_argument("--adjudication", type=Path, required=True)
    freeze_adjudication.add_argument("--report-rows", type=Path, required=True)
    freeze_adjudication.add_argument("--config", type=Path, required=True)
    freeze_adjudication.add_argument("--output", type=Path, required=True)
    freeze_adjudication.set_defaults(handler=command_freeze_adjudication)

    report = subparsers.add_parser("report")
    report.add_argument("--author-review", type=Path, required=True)
    report.add_argument("--packet-manifest", type=Path, required=True)
    report.add_argument("--audio-a", type=Path, required=True)
    report.add_argument("--audio-b", type=Path, required=True)
    report.add_argument("--frame-a", type=Path, required=True)
    report.add_argument("--frame-b", type=Path, required=True)
    report.add_argument("--frame-root-a", type=Path, required=True)
    report.add_argument("--frame-root-b", type=Path, required=True)
    report.add_argument("--config", type=Path, required=True)
    report.add_argument("--sampling-design", type=Path, required=True)
    report.add_argument("--adjudication", type=Path)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--summary-out", type=Path, required=True)
    report.set_defaults(handler=command_report)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
