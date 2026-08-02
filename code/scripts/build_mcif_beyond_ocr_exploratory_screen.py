#!/usr/bin/env python3
"""Build a reference-isolated MCIF raw-image versus OCR screen."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any

from PIL import Image

from scripts.materialize_full_talk_corruptions import (
    build_noise_bed,
    energy_vad_v1,
    mix_at_snr,
    read_pcm16_wav,
    select_sources,
    sha256_file,
    stable_rng,
    write_pcm16_wav,
)


FORBIDDEN_INFERENCE_KEYS = (
    "reference",
    "transcript",
    "translation",
    "target_text",
    "candidate",
)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def assert_inference_safe(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            lowered = str(key).lower()
            if any(forbidden in lowered for forbidden in FORBIDDEN_INFERENCE_KEYS):
                raise ValueError(f"Forbidden inference field: {path}.{key}")
            assert_inference_safe(nested, f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            assert_inference_safe(nested, f"{path}[{index}]")


def normalized_words(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def validate_selection(candidates: list[dict], config: dict) -> list[dict]:
    by_id = {row["candidate_id"]: row for row in candidates}
    requested = config["selection"]["candidate_ids"]
    if len(requested) != len(set(requested)):
        raise ValueError("Selected candidate ids are not unique")
    missing = sorted(set(requested) - set(by_id))
    if missing:
        raise ValueError(f"Selected candidates are absent: {missing}")
    selected = [by_id[candidate_id] for candidate_id in requested]
    rules = config["selection"]
    if not rules["minimum_candidates"] <= len(selected) <= rules["maximum_candidates"]:
        raise ValueError("Selected candidate count is outside the frozen range")
    talk_counts = Counter(row["talk_id"] for row in selected)
    segment_counts = Counter(row["segment_id"] for row in selected)
    if max(talk_counts.values()) > rules["maximum_candidates_per_talk"]:
        raise ValueError("A talk exceeds the selected-candidate cap")
    if max(segment_counts.values()) > rules["maximum_candidates_per_segment"]:
        raise ValueError("A segment exceeds the selected-candidate cap")
    for row in selected:
        duration = row["source_segment_end_sec"] - row["source_segment_offset_sec"]
        if row["lead_lower_bound_sec"] < rules["minimum_lead_sec"]:
            raise ValueError(f"Insufficient visual lead: {row['candidate_id']}")
        if not rules["minimum_segment_duration_sec"] <= duration <= rules[
            "maximum_segment_duration_sec"
        ]:
            raise ValueError(f"Segment duration outside contract: {row['candidate_id']}")
        if row.get("current_r0_candidate_absent") is not True:
            raise ValueError(f"Candidate is already in flat OCR: {row['candidate_id']}")
        if normalized_words(row["normalized_source_candidate"]) not in normalized_words(
            row["source_reference_en"]
        ):
            raise ValueError(f"Candidate is absent from source reference: {row['candidate_id']}")
    return selected


def resolve_control_media(
    media: dict,
    evidence_root: Path,
    control_media_root: Path,
) -> Path:
    relative = Path(media["source_media_path"])
    if media["location"] == "canonical_native_source":
        path = evidence_root / relative
    elif media["location"] == "control_media_bundle":
        if not relative.parts or relative.parts[0] != "visual_control_media_v1":
            raise ValueError(f"Unexpected control bundle path: {relative}")
        path = control_media_root.joinpath(*relative.parts[1:])
    else:
        raise ValueError(f"Unknown media location: {media['location']}")
    resolved = path.resolve()
    allowed = (
        evidence_root.resolve()
        if media["location"] == "canonical_native_source"
        else control_media_root.resolve()
    )
    if not resolved.is_relative_to(allowed) or not resolved.is_file():
        raise ValueError(f"Unsafe or absent media path: {resolved}")
    if sha256_file(resolved) != media["source_media_sha256"]:
        raise ValueError(f"Media hash mismatch: {resolved}")
    return resolved


def copy_bound_image(source: Path, destination: Path, media: dict) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    digest = sha256_file(destination)
    if digest != media["source_media_sha256"]:
        raise ValueError(f"Copied image hash mismatch: {destination}")
    with Image.open(destination) as image:
        image.verify()
    with Image.open(destination) as image:
        if image.size != (media["width"], media["height"]):
            raise ValueError(f"Copied image dimensions differ: {destination}")
    return {
        "sha256": digest,
        "width": media["width"],
        "height": media["height"],
        "visual_token_count": media["visual_token_count"],
    }


def clip_audio(source: Path, start_sec: float, end_sec: float) -> tuple[Any, int]:
    samples, sample_rate = read_pcm16_wav(source)
    start = max(0, int(round(start_sec * sample_rate)))
    end = min(len(samples), int(round(end_sec * sample_rate)))
    if end <= start:
        raise ValueError(f"Empty segment interval: {source}:{start_sec}-{end_sec}")
    return samples[start:end].copy(), sample_rate


def materialize_noisy(
    clean,
    sample_rate: int,
    screen_id: str,
    noise_config: dict,
    source_pool: list[dict],
) -> tuple[Any, dict]:
    rng = stable_rng(
        int(noise_config["global_seed"]),
        screen_id,
        noise_config["condition_id"],
    )
    sources = select_sources(
        source_pool,
        "babble_speech" if noise_config["kind"] == "babble" else noise_config["kind"],
        noise_config["source_split"],
        int(noise_config["source_count"]),
        rng,
    )
    for source in sources:
        path = Path(source["path"])
        if not path.is_file() or sha256_file(path) != source["sha256"]:
            raise ValueError(f"Noise source is absent or changed: {path}")
    noise, provenance = build_noise_bed(sources, len(clean), sample_rate, rng)
    active, vad = energy_vad_v1(clean, sample_rate)
    mixed, mixing = mix_at_snr(
        clean,
        noise,
        active,
        float(noise_config["snr_db"]),
    )
    return mixed, {"sources": provenance, "activity": vad, "mixing": mixing}


def build(args: argparse.Namespace) -> dict:
    if args.output_root.exists():
        raise FileExistsError(f"Output root already exists: {args.output_root}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if sha256_file(args.config) != args.expected_config_sha256:
        raise ValueError("Config hash differs from the expected frozen hash")
    candidates = load_jsonl(args.r1_candidates) + load_jsonl(args.r2_candidates)
    selected = validate_selection(candidates, config)
    ladder = {row["id"]: row for row in load_jsonl(args.ladder)}
    controls = {row["id"]: row for row in load_jsonl(args.control_media)}
    talks = {row["audio"]["path"].split("/")[-1].removesuffix(".wav"): row
             for row in load_jsonl(args.inference_manifest)}
    source_pool = load_jsonl(args.noise_source_pool)
    noise_config = next(
        row for row in config["acoustic_conditions"] if row["condition_id"] != "clean"
    )

    inference_root = args.output_root / "inference_bundle"
    scorer_root = args.output_root / "scorer_private"
    inference_rows: list[dict] = []
    scorer_rows: list[dict] = []
    for candidate in selected:
        state_id = candidate["current_state_id"]
        if state_id not in ladder or state_id not in controls:
            raise ValueError(f"Candidate state is absent: {state_id}")
        state = ladder[state_id]
        control = controls[state_id]
        if state["row_sha256"] != candidate["current_state_row_sha256"]:
            raise ValueError(f"Candidate state binding changed: {candidate['candidate_id']}")
        if control.get("target_or_reference_consumed") is not False:
            raise ValueError(f"Wrong-image control consumed target data: {state_id}")
        correct_media = control["correct_image"]
        wrong_media = control["cross_talk_wrong"]["final_media"]
        if correct_media["visual_token_count"] != wrong_media["visual_token_count"]:
            raise ValueError(f"Wrong image is not token matched: {state_id}")
        screen_id = "MCIF-E" + hashlib.sha256(
            candidate["candidate_id"].encode("utf-8")
        ).hexdigest()[:12].upper()
        correct_source = resolve_control_media(
            correct_media, args.evidence_root, args.control_media_root
        )
        wrong_source = resolve_control_media(
            wrong_media, args.evidence_root, args.control_media_root
        )
        correct_relative = Path("media/images/correct") / f"{screen_id}.png"
        wrong_relative = Path("media/images/wrong") / f"{screen_id}.png"
        correct_info = copy_bound_image(
            correct_source, inference_root / correct_relative, correct_media
        )
        wrong_info = copy_bound_image(
            wrong_source, inference_root / wrong_relative, wrong_media
        )

        talk = talks[candidate["talk_id"]]
        source_audio = Path(talk["audio"]["path"])
        if sha256_file(source_audio) != talk["audio"]["sha256"]:
            raise ValueError(f"Clean talk audio changed: {candidate['talk_id']}")
        clean, sample_rate = clip_audio(
            source_audio,
            candidate["source_segment_offset_sec"],
            candidate["source_segment_end_sec"],
        )
        noisy, noise_provenance = materialize_noisy(
            clean, sample_rate, screen_id, noise_config, source_pool
        )
        acoustic_manifest = {}
        for acoustic_condition, samples in (
            ("clean", clean),
            (noise_config["condition_id"], noisy),
        ):
            audio_relative = Path("media/audio") / acoustic_condition / f"{screen_id}.wav"
            audio_path = inference_root / audio_relative
            write_pcm16_wav(audio_path, samples, sample_rate)
            acoustic_manifest[acoustic_condition] = {
                "path": str(audio_relative),
                "sha256": sha256_file(audio_path),
                "sample_rate_hz": sample_rate,
                "frames": len(samples),
            }
            item = {
                "id": f"{screen_id}-{acoustic_condition}",
                "screen_id": screen_id,
                "acoustic_condition": acoustic_condition,
                "audio": str(audio_relative),
                "slide_image": str(correct_relative),
                "wrong_image": str(wrong_relative),
                "ocr_text": state["r0_flat_ocr"]["model_input_text"],
                "src_lang": "English",
                "tgt_lang": "Chinese",
                "segment_duration_sec": round(len(samples) / sample_rate, 6),
            }
            assert_inference_safe(item)
            item["input_row_sha256"] = canonical_sha256(item)
            inference_rows.append(item)
        scorer_rows.append(
            {
                "schema_version": "mcif_beyond_ocr_exploratory_selection_v1",
                "screen_id": screen_id,
                "candidate_id": candidate["candidate_id"],
                "candidate_inventory_row_sha256": candidate["row_sha256"],
                "candidate_text": candidate["normalized_source_candidate"],
                "candidate_kind": candidate["candidate_kind"],
                "evidence_tier": candidate["evidence_tier"],
                "talk_id": candidate["talk_id"],
                "segment_id": candidate["segment_id"],
                "state_id": state_id,
                "lead_lower_bound_sec": candidate["lead_lower_bound_sec"],
                "source_reference_en": candidate["source_reference_en"],
                "target_reference_zh": candidate["target_reference_zh"],
                "correct_image": correct_info,
                "wrong_image": wrong_info,
                "acoustic_inputs": acoustic_manifest,
                "noise_provenance": noise_provenance,
                "formal_human_validation_status": "NOT_STARTED",
            }
        )

    inference_rows.sort(key=lambda row: (row["screen_id"], row["acoustic_condition"]))
    scorer_rows.sort(key=lambda row: row["screen_id"])
    for row in inference_rows:
        assert_inference_safe(row)
    serialized_inference = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in inference_rows
    )
    for scorer in scorer_rows:
        target = scorer["target_reference_zh"].strip()
        if target and target in serialized_inference:
            raise ValueError(
                f"Target reference leaked into inference bundle: {scorer['screen_id']}"
            )
    write_jsonl(inference_root / "items.jsonl", inference_rows)
    write_jsonl(scorer_root / "selection.jsonl", scorer_rows)
    shutil.copy2(args.config, scorer_root / "config.json")

    report = {
        "schema_version": "mcif_beyond_ocr_exploratory_bundle_report_v1",
        "scope": config["scope"],
        "candidate_count": len(scorer_rows),
        "talk_count": len({row["talk_id"] for row in scorer_rows}),
        "inference_item_count": len(inference_rows),
        "expected_result_count": len(inference_rows)
        * len(config["evidence_conditions"]),
        "evidence_conditions": config["evidence_conditions"],
        "acoustic_conditions": [
            row["condition_id"] for row in config["acoustic_conditions"]
        ],
        "config_sha256": args.expected_config_sha256,
        "reference_boundary": (
            "Only inference_bundle may be transferred to the model worker; "
            "scorer_private contains references and remains scorer-only."
        ),
    }
    (args.output_root / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    files = sorted(path for path in args.output_root.rglob("*") if path.is_file())
    checksums = [
        f"{sha256_file(path)}  {path.relative_to(args.output_root)}"
        for path in files
    ]
    (args.output_root / "SHA256SUMS").write_text("\n".join(checksums) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--r1-candidates", type=Path, required=True)
    parser.add_argument("--r2-candidates", type=Path, required=True)
    parser.add_argument("--ladder", type=Path, required=True)
    parser.add_argument("--control-media", type=Path, required=True)
    parser.add_argument("--inference-manifest", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--control-media-root", type=Path, required=True)
    parser.add_argument("--noise-source-pool", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
