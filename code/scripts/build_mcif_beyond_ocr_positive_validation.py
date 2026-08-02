#!/usr/bin/env python3
"""Build blinded four-role validation packets for robust screen positives."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


EVIDENCE_CONDITIONS = ("audio_only", "ocr", "raw_image", "wrong_image")
ACOUSTIC_CONDITIONS = ("clean", "babble_p5_s0")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


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


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def deterministic_order(values: list[str], seed: str, namespace: str) -> list[str]:
    return sorted(
        values,
        key=lambda value: hashlib.sha256(
            f"{seed}\0{namespace}\0{value}".encode("utf-8")
        ).hexdigest(),
    )


def hashed(row: dict) -> dict:
    output = dict(row)
    output["row_sha256"] = canonical_sha256(output)
    return output


def robust_screens(metric_rows: list[dict], expected: int) -> list[str]:
    passes: dict[str, set[str]] = {}
    for row in metric_rows:
        if row.get("primary_positive") is True:
            passes.setdefault(row["screen_id"], set()).add(row["acoustic_condition"])
    selected = sorted(
        screen_id
        for screen_id, acoustics in passes.items()
        if acoustics == set(ACOUSTIC_CONDITIONS)
    )
    if len(selected) != expected:
        raise ValueError(f"Robust-positive count differs: {len(selected)} != {expected}")
    return selected


def load_result_matrix(run_root: Path) -> dict[tuple[str, str, str], dict]:
    rows = []
    for path in sorted(run_root.glob("runs_shard_*.jsonl")):
        rows.extend(load_jsonl(path))
    matrix = {}
    for row in rows:
        key = (row["screen_id"], row["acoustic_condition"], row["condition"])
        if key in matrix:
            raise ValueError(f"Duplicate result: {key}")
        matrix[key] = row
    return matrix


def pending_visual(item_id: str) -> dict:
    return {
        "item_id": item_id,
        "annotation_status": "pending",
        "ocr_support": None,
        "image_slot_support": {"A": None, "B": None},
        "preferred_image_slot": None,
        "reason_codes": [],
        "annotation_note": "",
        "annotator_id": None,
        "locked_at_utc": None,
    }


def pending_outcome(item_id: str) -> dict:
    return {
        "item_id": item_id,
        "annotation_status": "pending",
        "candidate_is_meaningful": None,
        "target_reference_alignment": None,
        "slot_judgments": [
            {
                "acoustic_condition": acoustic,
                "slot": slot,
                "earliest_candidate_correct_sec": None,
                "final_candidate_judgment": None,
                "unsupported_content": None,
            }
            for acoustic in ACOUSTIC_CONDITIONS
            for slot in ("A", "B", "C", "D")
        ],
        "reason_codes": [],
        "annotation_note": "",
        "annotator_id": None,
        "locked_at_utc": None,
    }


def copy_image(source: Path, destination: Path) -> dict:
    if not source.is_file():
        raise ValueError(f"Image is absent: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if file_sha256(source) != file_sha256(destination):
        raise ValueError(f"Image copy changed bytes: {destination}")
    return {
        "path": destination.name,
        "sha256": file_sha256(destination),
        "bytes": destination.stat().st_size,
    }


def build(args: argparse.Namespace) -> dict:
    if args.output_root.exists():
        raise FileExistsError(f"Output root already exists: {args.output_root}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if file_sha256(args.config) != args.expected_config_sha256:
        raise ValueError("Validation config hash differs")
    seed = config["ordering_seed"]
    for name, value in (
        ("builder Git commit", args.builder_git_commit),
        ("source HF revision", args.source_hf_revision),
    ):
        if len(value) != 40 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"Invalid {name}")
    metric_rows = load_jsonl(args.candidate_metrics)
    selected = robust_screens(metric_rows, int(config["expected_items"]))
    selection = {row["screen_id"]: row for row in load_jsonl(args.selection)}
    inference = {}
    for row in load_jsonl(args.inference_items):
        inference.setdefault(row["screen_id"], row)
    matrix = load_result_matrix(args.run_root)
    for screen_id in selected:
        if screen_id not in selection or screen_id not in inference:
            raise ValueError(f"Selected screen is absent from source inputs: {screen_id}")
        for acoustic in ACOUSTIC_CONDITIONS:
            for condition in EVIDENCE_CONDITIONS:
                if (screen_id, acoustic, condition) not in matrix:
                    raise ValueError(
                        f"Selected result is absent: {screen_id}/{acoustic}/{condition}"
                    )

    visual_roles = ("visual_a", "visual_b")
    outcome_roles = ("outcome_a", "outcome_b")
    mappings = []
    role_counts = {}
    for role in visual_roles:
        role_root = args.output_root / role
        ordered = deterministic_order(selected, seed, role)
        items = []
        working = []
        for index, screen_id in enumerate(ordered, 1):
            item_id = f"MCIF-PV-{role[-1].upper()}{index:03d}"
            source = inference[screen_id]
            image_conditions = deterministic_order(
                ["raw_image", "wrong_image"], seed, f"{role}:{screen_id}:images"
            )
            slot_mapping = dict(zip(("A", "B"), image_conditions, strict=True))
            image_slots = {}
            for slot, condition in slot_mapping.items():
                source_field = "slide_image" if condition == "raw_image" else "wrong_image"
                source_path = (args.inference_items.parent / source[source_field]).resolve()
                destination = role_root / "media" / f"{item_id}_{slot}.png"
                image_slots[slot] = copy_image(source_path, destination)
                image_slots[slot]["path"] = f"media/{destination.name}"
            items.append(
                hashed(
                    {
                        "schema_version": "mcif_positive_visual_validation_item_v1",
                        "role": role,
                        "item_id": item_id,
                        "candidate_source_en": selection[screen_id]["candidate_text"],
                        "flat_ocr_text": source["ocr_text"],
                        "image_slots": image_slots,
                        "instructions": (
                            "Judge OCR support first, then each image independently. "
                            "Do not infer from outside topic knowledge."
                        ),
                    }
                )
            )
            working.append(pending_visual(item_id))
            mappings.append(
                {
                    "screen_id": screen_id,
                    "candidate_id": selection[screen_id]["candidate_id"],
                    "role": role,
                    "item_id": item_id,
                    "image_slot_to_condition": slot_mapping,
                }
            )
        write_jsonl(role_root / "items.jsonl", items)
        write_jsonl(role_root / "working_annotations.jsonl", working)
        role_counts[role] = len(items)

    for role in outcome_roles:
        role_root = args.output_root / role
        ordered = deterministic_order(selected, seed, role)
        items = []
        working = []
        for index, screen_id in enumerate(ordered, 1):
            item_id = f"MCIF-PO-{role[-1].upper()}{index:03d}"
            condition_slots = deterministic_order(
                list(EVIDENCE_CONDITIONS), seed, f"{role}:{screen_id}:conditions"
            )
            slot_mapping = dict(zip(("A", "B", "C", "D"), condition_slots, strict=True))
            trajectories = {}
            for acoustic in ACOUSTIC_CONDITIONS:
                trajectories[acoustic] = []
                for slot, condition in slot_mapping.items():
                    result = matrix[(screen_id, acoustic, condition)]
                    trajectories[acoustic].append(
                        {
                            "slot": slot,
                            "prefix_hypotheses": result["prefix_hypotheses"],
                            "final_hypothesis": result["hypothesis"],
                        }
                    )
            scorer = selection[screen_id]
            items.append(
                hashed(
                    {
                        "schema_version": "mcif_positive_outcome_validation_item_v1",
                        "role": role,
                        "item_id": item_id,
                        "candidate_source_en": scorer["candidate_text"],
                        "source_reference_en": scorer["source_reference_en"],
                        "target_reference_zh": scorer["target_reference_zh"],
                        "trajectories": trajectories,
                        "instructions": (
                            "Judge candidate correctness for every blinded slot and prefix. "
                            "Images, OCR and condition identities are intentionally hidden."
                        ),
                    }
                )
            )
            working.append(pending_outcome(item_id))
            mappings.append(
                {
                    "screen_id": screen_id,
                    "candidate_id": scorer["candidate_id"],
                    "role": role,
                    "item_id": item_id,
                    "condition_slot_to_condition": slot_mapping,
                }
            )
        write_jsonl(role_root / "items.jsonl", items)
        write_jsonl(role_root / "working_annotations.jsonl", working)
        role_counts[role] = len(items)

    private_root = args.output_root / "scorer_private"
    private_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(private_root / "mapping.jsonl", [hashed(row) for row in mappings])
    robust_rows = [
        row for row in metric_rows
        if row["screen_id"] in selected and row["primary_positive"] is True
    ]
    write_jsonl(private_root / "selected_metrics.jsonl", robust_rows)
    shutil.copy2(args.config, private_root / "config.json")
    manifest = {
        "schema_version": "mcif_beyond_ocr_positive_validation_manifest_v1",
        "status": "READY_FOR_FOUR_DISJOINT_HUMAN_VALIDATORS_NO_LABELS",
        "scope": config["scope"],
        "selected_screen_count": len(selected),
        "role_counts": role_counts,
        "roles_must_be_disjoint": config["require_disjoint_humans"],
        "candidate_metrics_sha256": file_sha256(args.candidate_metrics),
        "selection_sha256": file_sha256(args.selection),
        "run_completion_sha256": file_sha256(args.run_root / "completion.json"),
        "config_sha256": args.expected_config_sha256,
        "builder_git_commit": args.builder_git_commit,
        "source_hf_revision": args.source_hf_revision,
        "interpretation": config["interpretation"],
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_root / "README.md").write_text(
        """# MCIF Positive Human Validation V1

This workspace contains six candidates that passed the exploratory sample-level gate under both
clean and +5 dB babble. It has no human labels yet.

Assign `visual_a`, `visual_b`, `outcome_a`, and `outcome_b` to four different people. Give each
person only their role directory. Never distribute `scorer_private` to annotators.

Visual validators judge flat-OCR support before opening the two randomized image slots, then judge
each image independently. Outcome validators see source/target references and randomized model
trajectories, but no OCR, images, or condition identities. They record the earliest prefix where
the candidate is correctly realized, final candidate correctness, and unsupported content for
every slot under both acoustic conditions.

Any disagreement or uncertain judgment requires later adjudication. Passing validates only an
individual sample; it does not establish aggregate raw-image superiority over OCR.
""",
        encoding="utf-8",
    )
    files = sorted(path for path in args.output_root.rglob("*") if path.is_file())
    (args.output_root / "SHA256SUMS").write_text(
        "".join(
            f"{file_sha256(path)}  {path.relative_to(args.output_root).as_posix()}\n"
            for path in files
        ),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--builder-git-commit", required=True)
    parser.add_argument("--source-hf-revision", required=True)
    parser.add_argument("--candidate-metrics", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--inference-items", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
