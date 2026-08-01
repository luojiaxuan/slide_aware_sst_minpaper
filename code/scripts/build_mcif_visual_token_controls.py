#!/usr/bin/env python3
"""Freeze Qwen3-Omni visual-token inventory and wrong-image candidates."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Iterable

import numpy as np
from PIL import Image


PROCESSOR_FILES = (
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
)
VISUAL_PROMPT_TEXT = (
    "Current slide image:\n[image bytes attached]\n"
    "Translate only the causal speech."
)
PAIRING_SEED = "mcif-visual-token-controls-v1"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected JSON objects in {path}")
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_head_clean(repo: Path) -> str:
    resolved = repo.resolve(strict=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=resolved,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=resolved,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("Builder Git revision is not a full commit")
    if status:
        raise ValueError("Visual-token builder requires a clean Git checkout")
    return commit


def unique_by_id(rows: Iterable[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        item_id = row.get("id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(f"{label} contains a missing id")
        if item_id in result:
            raise ValueError(f"{label} contains duplicate id {item_id}")
        result[item_id] = row
    return result


def resolve_regular_file(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("Raw-image path must be canonical and relative")
    if root.is_symlink():
        raise ValueError("Raw-image source root cannot be a symlink")
    resolved_root = root.resolve(strict=True)
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError("Raw-image path traverses a symlink")
    resolved = (root / relative).resolve(strict=True)
    if not resolved.is_file() or not resolved.is_relative_to(resolved_root):
        raise ValueError("Raw-image path escapes the source root")
    return resolved


def processor_file_manifest(root: Path) -> dict[str, Any]:
    files = []
    for relative in PROCESSOR_FILES:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"Missing frozen processor file: {relative}")
        files.append(
            {
                "path": relative,
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "required_files": files,
        "required_file_set_sha256": canonical_sha256(files),
    }


def validate_ladder(
    rows: list[dict[str, Any]],
    *,
    expected_rows: int,
    expected_talks: int,
) -> None:
    if len(rows) != expected_rows or len(unique_by_id(rows, "ladder")) != expected_rows:
        raise ValueError("Ladder row inventory differs from the frozen contract")
    if len({row.get("lecture_id") for row in rows}) != expected_talks:
        raise ValueError("Ladder talk inventory differs from the frozen contract")
    order = [(row.get("lecture_id"), row.get("state_id")) for row in rows]
    if order != sorted(order):
        raise ValueError("Ladder is not in canonical talk/state order")
    by_talk: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_talk[str(row.get("lecture_id"))].append(row)
        if row.get("schema_version") != "mcif_source_evidence_ladder_v1":
            raise ValueError("Unexpected evidence ladder schema")
        if row.get("source_transcript_consumed") is not False or row.get(
            "target_or_reference_consumed"
        ) is not False:
            raise ValueError("Evidence ladder consumed a forbidden source or outcome")
        expected_row_sha256 = canonical_sha256(
            {key: value for key, value in row.items() if key != "row_sha256"}
        )
        if row.get("row_sha256") != expected_row_sha256:
            raise ValueError(f"Evidence ladder row hash mismatch: {row.get('id')}")
    for talk_id, talk_rows in by_talk.items():
        if [row.get("state_id") for row in talk_rows] != list(range(len(talk_rows))):
            raise ValueError(f"Non-contiguous ladder states for {talk_id}")


def qwen_image_messages(path: Path, *, include_audio: bool) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {"type": "image", "image": str(path)},
        {"type": "text", "text": VISUAL_PROMPT_TEXT},
    ]
    if include_audio:
        content.append({"type": "audio", "audio": np.zeros(16_000, dtype=np.float32)})
    return [{"role": "user", "content": content}]


def row_values(value: Any) -> list[int]:
    return [int(item) for item in (value.tolist() if hasattr(value, "tolist") else value)]


def build_visual_token_inventory(
    ladder_rows: list[dict[str, Any]],
    *,
    source_root: Path,
    processor: Any,
    processor_binding_sha256: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    if batch_size <= 0:
        raise ValueError("Processor batch size must be positive")
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    if not isinstance(image_token_id, int) or image_token_id < 0:
        raise ValueError("Processor has no valid image token id")
    inventory = []
    for offset in range(0, len(ladder_rows), batch_size):
        batch = ladder_rows[offset : offset + batch_size]
        paths = []
        images = []
        texts = []
        for row in batch:
            raw = row.get("r2_raw_image") or {}
            path = resolve_regular_file(source_root, str(raw.get("source_media_path", "")))
            if file_sha256(path) != raw.get("source_media_sha256"):
                raise ValueError(f"Raw-image bytes changed for {row.get('id')}")
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                converted = image.convert("RGB")
            if converted.size != (raw.get("width"), raw.get("height")):
                converted.close()
                raise ValueError(f"Raw-image dimensions changed for {row.get('id')}")
            paths.append(path)
            images.append(converted)
            texts.append(
                processor.apply_chat_template(
                    qwen_image_messages(path, include_audio=False),
                    add_generation_prompt=True,
                    tokenize=False,
                )
            )
        try:
            encoded = processor(
                text=texts,
                images=images,
                videos=None,
                return_tensors="pt",
                padding=True,
            )
        finally:
            for image in images:
                image.close()
        input_ids = encoded["input_ids"]
        grids = encoded["image_grid_thw"]
        if len(input_ids) != len(batch) or len(grids) != len(batch):
            raise ValueError("Processor returned an unexpected image batch size")
        for row, path, token_row, grid_row in zip(
            batch, paths, input_ids, grids, strict=True
        ):
            token_ids = row_values(token_row)
            grid = row_values(grid_row)
            count = sum(token_id == image_token_id for token_id in token_ids)
            if count <= 0 or len(grid) != 3 or any(value <= 0 for value in grid):
                raise ValueError(f"Processor produced invalid visual tokens for {row['id']}")
            raw = row["r2_raw_image"]
            result = {
                "schema_version": "mcif_qwen3_omni_visual_token_inventory_v1",
                "id": row["id"],
                "lecture_id": row["lecture_id"],
                "state_id": row["state_id"],
                "availability_start_sec": row["availability_start_sec"],
                "source_media_path": raw["source_media_path"],
                "source_media_sha256": raw["source_media_sha256"],
                "width": raw["width"],
                "height": raw["height"],
                "image_grid_thw": grid,
                "visual_token_count": count,
                "image_token": processor.image_token,
                "image_token_id": image_token_id,
                "processor_binding_sha256": processor_binding_sha256,
                "source_ladder_row_sha256": row["row_sha256"],
                "source_transcript_consumed": False,
                "target_or_reference_consumed": False,
            }
            result["row_sha256"] = canonical_sha256(result)
            inventory.append(result)
    return inventory


def verify_audio_invariance(
    inventory: list[dict[str, Any]],
    *,
    source_root: Path,
    processor: Any,
) -> list[dict[str, Any]]:
    representatives = {}
    for row in inventory:
        key = (
            row["visual_token_count"],
            tuple(row["image_grid_thw"]),
            row["width"],
            row["height"],
        )
        representatives.setdefault(key, row)
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    audit = []
    for key, row in sorted(representatives.items()):
        path = resolve_regular_file(source_root, row["source_media_path"])
        with Image.open(path) as image:
            converted = image.convert("RGB")
        audio = np.zeros(16_000, dtype=np.float32)
        messages = qwen_image_messages(path, include_audio=True)
        text = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        try:
            encoded = processor(
                text=[text],
                images=[converted],
                audio=[audio],
                videos=None,
                return_tensors="pt",
                padding=True,
                sampling_rate=16_000,
                use_audio_in_video=False,
            )
        finally:
            converted.close()
        observed = sum(
            token_id == image_token_id for token_id in row_values(encoded["input_ids"][0])
        )
        if observed != row["visual_token_count"]:
            raise ValueError(f"Audio changes visual token count for {row['id']}")
        audit.append(
            {
                "representative_id": row["id"],
                "visual_token_count": observed,
                "image_grid_thw": row["image_grid_thw"],
                "width": row["width"],
                "height": row["height"],
                "dummy_audio_sample_rate": 16_000,
                "dummy_audio_sample_count": 16_000,
            }
        )
    return audit


def seeded_rank(seed: str, source_id: str, candidate_id: str) -> str:
    return hashlib.sha256(
        f"{seed}\0{source_id}\0{candidate_id}".encode("utf-8")
    ).hexdigest()


def candidate_record(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": candidate["id"],
        "lecture_id": candidate["lecture_id"],
        "state_id": candidate["state_id"],
        "source_media_path": candidate["source_media_path"],
        "source_media_sha256": candidate["source_media_sha256"],
        "width": candidate["width"],
        "height": candidate["height"],
        "image_grid_thw": candidate["image_grid_thw"],
        "visual_token_count": candidate["visual_token_count"],
        "inventory_row_sha256": candidate["row_sha256"],
    }


def build_wrong_image_candidates(
    inventory: list[dict[str, Any]],
    *,
    seed: str,
) -> list[dict[str, Any]]:
    by_visual_tokens: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in inventory:
        by_visual_tokens[row["visual_token_count"]].append(row)
    results = []
    for row in inventory:
        candidates = [
            candidate
            for candidate in by_visual_tokens[row["visual_token_count"]]
            if candidate["id"] != row["id"]
            and candidate["source_media_sha256"] != row["source_media_sha256"]
        ]
        stale = [
            candidate
            for candidate in candidates
            if candidate["lecture_id"] == row["lecture_id"]
            and candidate["state_id"] < row["state_id"]
            and candidate["availability_start_sec"] <= row["availability_start_sec"]
        ]
        stale_same_grid = [
            candidate
            for candidate in stale
            if candidate["image_grid_thw"] == row["image_grid_thw"]
        ]
        stale_pool = stale_same_grid or stale
        stale_pool.sort(key=lambda candidate: (-candidate["state_id"], candidate["id"]))
        cross_talk = [
            candidate
            for candidate in candidates
            if candidate["lecture_id"] != row["lecture_id"]
        ]
        same_grid = [
            candidate
            for candidate in cross_talk
            if candidate["image_grid_thw"] == row["image_grid_thw"]
        ]
        same_dimensions = [
            candidate
            for candidate in cross_talk
            if (candidate["width"], candidate["height"])
            == (row["width"], row["height"])
        ]
        cross_pool = same_dimensions or same_grid or cross_talk
        if not cross_pool:
            raise ValueError(f"No cross-talk exact-token control for {row['id']}")
        cross = min(
            cross_pool,
            key=lambda candidate: seeded_rank(seed, row["id"], candidate["id"]),
        )
        if same_dimensions:
            cross_match_level = "same_dimensions"
        elif same_grid:
            cross_match_level = "same_grid"
        else:
            cross_match_level = "same_visual_token_count"
        result = {
            "schema_version": "mcif_qwen3_omni_wrong_image_candidates_v1",
            "id": row["id"],
            "lecture_id": row["lecture_id"],
            "state_id": row["state_id"],
            "visual_token_count": row["visual_token_count"],
            "inventory_row_sha256": row["row_sha256"],
            "same_talk_stale": (
                None if not stale_pool else candidate_record(stale_pool[0])
            ),
            "same_talk_stale_same_grid": bool(stale_same_grid),
            "cross_talk_wrong": candidate_record(cross),
            "cross_talk_match_level": cross_match_level,
            "cross_talk_same_grid": bool(same_grid),
            "cross_talk_same_dimensions": bool(same_dimensions),
            "pairing_seed": seed,
            "source_transcript_consumed": False,
            "target_or_reference_consumed": False,
        }
        result["row_sha256"] = canonical_sha256(result)
        results.append(result)
    return results


def verify_frozen_inputs(
    inventory: list[dict[str, Any]],
    *,
    source_root: Path,
    processor_root: Path,
    expected_processor_files: dict[str, Any],
) -> None:
    if processor_file_manifest(processor_root) != expected_processor_files:
        raise ValueError("Processor files changed during visual-token materialization")
    for row in inventory:
        path = resolve_regular_file(source_root, row["source_media_path"])
        if file_sha256(path) != row["source_media_sha256"]:
            raise ValueError(f"Raw-image bytes changed after processing: {row['id']}")
        with Image.open(path) as image:
            if image.size != (row["width"], row["height"]):
                raise ValueError(f"Raw-image dimensions changed after processing: {row['id']}")


def summarize(
    inventory: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    audio_audit: list[dict[str, Any]],
) -> dict[str, Any]:
    count_distribution = Counter(row["visual_token_count"] for row in inventory)
    grid_distribution = Counter(tuple(row["image_grid_thw"]) for row in inventory)
    return {
        "rows": len(inventory),
        "talks": len({row["lecture_id"] for row in inventory}),
        "visual_token_count_distribution": {
            str(key): value for key, value in sorted(count_distribution.items())
        },
        "image_grid_distribution": {
            "x".join(str(value) for value in key): count
            for key, count in sorted(grid_distribution.items())
        },
        "same_talk_stale_coverage": sum(
            row["same_talk_stale"] is not None for row in controls
        ),
        "same_talk_stale_same_grid_coverage": sum(
            row["same_talk_stale_same_grid"] for row in controls
        ),
        "cross_talk_wrong_coverage": sum(
            row["cross_talk_wrong"] is not None for row in controls
        ),
        "cross_talk_same_grid_coverage": sum(
            row["cross_talk_same_grid"] for row in controls
        ),
        "cross_talk_same_dimensions_coverage": sum(
            row["cross_talk_same_dimensions"] for row in controls
        ),
        "cross_talk_match_level_distribution": dict(
            sorted(Counter(row["cross_talk_match_level"] for row in controls).items())
        ),
        "audio_invariance_representatives": len(audio_audit),
    }


def write_bundle(
    output_root: Path,
    *,
    inventory: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    processor_manifest: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError("Visual-token output root must not already exist")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    try:
        inventory_path = temporary / "visual_token_inventory.jsonl"
        controls_path = temporary / "matched_wrong_candidates.jsonl"
        processor_path = temporary / "processor_manifest.json"
        inventory_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in inventory
            ),
            encoding="utf-8",
        )
        controls_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in controls
            ),
            encoding="utf-8",
        )
        processor_path.write_text(
            json.dumps(processor_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        final_report = {
            **report,
            "visual_token_inventory_sha256": file_sha256(inventory_path),
            "matched_wrong_candidates_sha256": file_sha256(controls_path),
            "processor_manifest_sha256": file_sha256(processor_path),
        }
        report_path = temporary / "report.json"
        report_path.write_text(
            json.dumps(final_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        readme_path = temporary / "README.md"
        readme_path.write_text(
            "# MCIF Qwen3-Omni Visual Token Controls V1\n\n"
            "Private source-only processor inventory for the 304-state MCIF evidence "
            "ladder. Every raw image is bound to its Qwen3-Omni image grid and visual "
            "token count. Candidate controls contain both the nearest causally prior "
            "same-talk state when available and a deterministic cross-talk visual-token-"
            "matched image, preferring equal dimensions and then equal processor grid. "
            "No target, reference, transcript, audio content, annotation, or ST "
            "output is included.\n\n"
            f"- model: `{final_report['model_id']}@{final_report['model_revision']}`\n"
            f"- rows / talks: {final_report['rows']} / {final_report['talks']}\n"
            f"- inventory SHA256: `{final_report['visual_token_inventory_sha256']}`\n"
            f"- controls SHA256: `{final_report['matched_wrong_candidates_sha256']}`\n",
            encoding="utf-8",
        )
        checksum_paths = sorted(
            path for path in temporary.iterdir() if path.is_file() and path.name != "SHA256SUMS"
        )
        checksum_path = temporary / "SHA256SUMS"
        checksum_path.write_text(
            "".join(
                f"{file_sha256(path)}  {path.name}\n" for path in checksum_paths
            ),
            encoding="utf-8",
        )
        os.rename(temporary, output_root)
        return {
            **final_report,
            "checksum_manifest_sha256": file_sha256(output_root / "SHA256SUMS"),
            "checksum_entries": len(checksum_paths),
        }
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ladder", type=Path, required=True)
    parser.add_argument("--expected-ladder-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--processor-root", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--code-repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--expected-rows", type=int, default=304)
    parser.add_argument("--expected-talks", type=int, default=21)
    args = parser.parse_args()
    if (
        not args.model_id
        or len(args.model_revision) != 40
        or any(
            character not in "0123456789abcdef"
            for character in args.model_revision
        )
    ):
        raise ValueError("Model identity must include a full immutable revision")
    if file_sha256(args.ladder) != args.expected_ladder_sha256:
        raise ValueError("Evidence ladder hash differs from the frozen input")
    builder_git_commit = git_head_clean(args.code_repo)
    rows = load_jsonl(args.ladder)
    validate_ladder(
        rows,
        expected_rows=args.expected_rows,
        expected_talks=args.expected_talks,
    )
    processor_files = processor_file_manifest(args.processor_root)
    processor_contract = {
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "transformers_version": None,
        "processor_files": processor_files,
        "visual_prompt_text": VISUAL_PROMPT_TEXT,
        "message_order": ["image", "text", "audio"],
        "pairing_seed": PAIRING_SEED,
    }
    from transformers import Qwen3OmniMoeProcessor, __version__ as transformers_version

    processor_contract["transformers_version"] = transformers_version
    processor_binding_sha256 = canonical_sha256(processor_contract)
    processor = Qwen3OmniMoeProcessor.from_pretrained(
        args.processor_root,
        local_files_only=True,
        trust_remote_code=True,
    )
    inventory = build_visual_token_inventory(
        rows,
        source_root=args.source_root,
        processor=processor,
        processor_binding_sha256=processor_binding_sha256,
        batch_size=args.batch_size,
    )
    audio_audit = verify_audio_invariance(
        inventory,
        source_root=args.source_root,
        processor=processor,
    )
    controls = build_wrong_image_candidates(inventory, seed=PAIRING_SEED)
    verify_frozen_inputs(
        inventory,
        source_root=args.source_root,
        processor_root=args.processor_root,
        expected_processor_files=processor_files,
    )
    summary = summarize(inventory, controls, audio_audit)
    report = {
        "schema_version": "mcif_qwen3_omni_visual_token_controls_report_v1",
        "status": "SOURCE_ONLY_PROCESSOR_INVENTORY_NOT_ANNOTATION_OR_ST_RESULT",
        "builder_git_commit": builder_git_commit,
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "transformers_version": transformers_version,
        "processor_binding_sha256": processor_binding_sha256,
        "processor_required_file_set_sha256": processor_files[
            "required_file_set_sha256"
        ],
        "source_ladder_sha256": args.expected_ladder_sha256,
        "pairing_seed": PAIRING_SEED,
        "prompt_order": ["image", "text", "audio"],
        "audio_invariance_audit": audio_audit,
        **summary,
        "source_transcript_consumed": False,
        "target_or_reference_consumed": False,
        "interpretation": (
            "The inventory freezes processor-dependent image token budgets. It offers "
            "two source-only wrong-image candidates per state where available; it does "
            "not choose the final paper control, label image necessity, or measure ST."
        ),
    }
    processor_manifest = {
        "schema_version": "qwen3_omni_processor_manifest_v1",
        **processor_contract,
        "processor_binding_sha256": processor_binding_sha256,
    }
    final_report = write_bundle(
        args.output_root,
        inventory=inventory,
        controls=controls,
        processor_manifest=processor_manifest,
        report=report,
    )
    print(json.dumps(final_report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
