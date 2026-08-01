#!/usr/bin/env python3
"""Build balanced transcript- and target-free ACL60/60 event annotation seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FORBIDDEN_KEYS = ("reference", "transcript", "translation", "target", "sentence")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def assert_source_only(value: object, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            lowered = str(key).lower()
            if any(forbidden in lowered for forbidden in FORBIDDEN_KEYS):
                raise ValueError(f"Forbidden field at {path}.{key}")
            assert_source_only(nested, f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            assert_source_only(nested, f"{path}[{index}]")


def stable_rank(salt: str, talk_id: str, observation_id: str, stratum: str) -> bytes:
    payload = f"{salt}\0{talk_id}\0{observation_id}\0{stratum}".encode()
    return hashlib.sha256(payload).digest()


def select_rows(
    observations: list[dict],
    candidates: list[dict],
    talk: dict,
    contract: dict,
) -> list[dict]:
    talk_id = talk["talk_id"]
    selection = contract["selection"]
    salt = selection["salt"]
    talk_observations = [row for row in observations if row["talk_id"] == talk_id]
    by_id = {row["observation_id"]: row for row in talk_observations}
    candidate_ids = {
        row["current_observation_id"]
        for row in candidates
        if row["talk_id"] == talk_id
    }
    candidate_pool = [by_id[observation_id] for observation_id in candidate_ids]
    random_pool = [row for row in talk_observations if row["observation_id"] not in candidate_ids]
    requested_candidate = int(selection["transition_candidate_states_per_talk"])
    requested_random = int(selection["hash_random_states_per_talk"])
    if len(candidate_pool) < requested_candidate or len(random_pool) < requested_random:
        raise ValueError(f"Insufficient annotation seed pool for {talk_id}")

    selected = []
    for stratum, pool, count in (
        ("transition_candidate", candidate_pool, requested_candidate),
        ("hash_random_observation", random_pool, requested_random),
    ):
        ranked = sorted(
            pool,
            key=lambda row: stable_rank(salt, talk_id, row["observation_id"], stratum),
        )[:count]
        selected.extend((stratum, row) for row in ranked)

    duration_sec = float(talk["duration_sec"])
    before = float(selection["audio_window_before_evidence_sec"])
    after = float(selection["audio_window_after_evidence_sec"])
    output = []
    for index, (stratum, observation) in enumerate(selected, start=1):
        evidence_sec = float(observation["causal_availability_sec"])
        row = {
            "schema_version": contract["schema_version"],
            "packet_id": f"{talk_id}:A{index:03d}",
            "dataset": "acl6060",
            "split": "dev",
            "talk_id": talk_id,
            "selection_stratum": stratum,
            "observation_id": observation["observation_id"],
            "frame_path": observation["frame_path"],
            "frame_sha256": observation["frame_sha256"],
            "t_evidence_sec": evidence_sec,
            "audio_id": f"{talk_id}.wav",
            "audio_sha256": talk["audio_sha256"],
            "suggested_audio_window_start_sec": round(max(0.0, evidence_sec - before), 6),
            "suggested_audio_window_end_sec": round(min(duration_sec, evidence_sec + after), 6),
            "audio_prefix_step_sec": contract["annotation"]["audio_prefix_step_sec"],
            "event_status": "pending",
            "source_question": None,
            "source_options": [],
            "source_answer_index": None,
            "t_last_insufficient_sec": None,
            "t_first_sufficient_sec": None,
            "evidence_subtypes": [],
            "evidence_region": None,
            "term_or_entity": None,
            "negative_labels": [],
            "annotator_id": None,
            "annotation_note": "",
        }
        assert_source_only(row)
        output.append(row)
    return output


def validate_seed(rows: list[dict], talks: list[dict], contract: dict) -> None:
    assert_source_only(rows)
    expected_per_talk = sum(
        int(contract["selection"][key])
        for key in (
            "transition_candidate_states_per_talk",
            "hash_random_states_per_talk",
        )
    )
    if len(rows) != len(talks) * expected_per_talk:
        raise ValueError("Annotation seed row count mismatch")
    if len({row["packet_id"] for row in rows}) != len(rows):
        raise ValueError("Annotation packet ids must be unique")
    if len({row["observation_id"] for row in rows}) != len(rows):
        raise ValueError("Annotation observations must be unique")
    for talk in talks:
        talk_rows = [row for row in rows if row["talk_id"] == talk["talk_id"]]
        counts = {
            stratum: sum(row["selection_stratum"] == stratum for row in talk_rows)
            for stratum in ("transition_candidate", "hash_random_observation")
        }
        expected = {
            "transition_candidate": int(
                contract["selection"]["transition_candidate_states_per_talk"]
            ),
            "hash_random_observation": int(
                contract["selection"]["hash_random_states_per_talk"]
            ),
        }
        if counts != expected:
            raise ValueError(f"Annotation strata mismatch for {talk['talk_id']}: {counts}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-observations", type=Path, required=True)
    parser.add_argument("--transition-candidates", type=Path, required=True)
    parser.add_argument("--talk-manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()

    observations = load_jsonl(args.frame_observations)
    candidates = load_jsonl(args.transition_candidates)
    talks = [row for row in load_jsonl(args.talk_manifest) if row["split"] == "dev"]
    talks.sort(key=lambda row: row["talk_id"])
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    rows = []
    for talk in talks:
        rows.extend(select_rows(observations, candidates, talk, contract))
    validate_seed(rows, talks, contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "schema_version": contract["schema_version"],
        "status": "PENDING_DOUBLE_SOURCE_SIDE_ANNOTATION",
        "talk_count": len(talks),
        "packet_count": len(rows),
        "selection_counts": {
            stratum: sum(row["selection_stratum"] == stratum for row in rows)
            for stratum in ("transition_candidate", "hash_random_observation")
        },
        "source_transcript_consumed": False,
        "target_or_reference_consumed": False,
        "model_output_consumed": False,
        "contract": str(args.contract),
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
