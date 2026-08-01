"""Validation helpers for Phase-A long-form context packets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable


CONDITIONS = ("C0", "C1", "C2", "C3")
REQUIRED_COMPILER_KEYS = {
    "implementation",
    "revision",
    "command",
    "input_sha256",
    "asr_tokenizer_revision",
    "mt_tokenizer_revision",
}
FORBIDDEN_KEY_MARKERS = {
    "reference",
    "source_transcript",
    "tagged_terminology",
}


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        keys = {str(key) for key in value}
        for item in value.values():
            keys.update(_all_keys(item))
        return keys
    if isinstance(value, list):
        keys = set()
        for item in value:
            keys.update(_all_keys(item))
        return keys
    return set()


def read_wav_order(path: Path) -> list[Path]:
    return [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_context_bundle(path: Path, wav_list_path: Path) -> dict:
    bundle = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(bundle, dict) or set(bundle) != {"schema_version", "compiler", "packets"}:
        raise ValueError("Context bundle must define schema_version, compiler, and packets")
    if bundle["schema_version"] != "phase_a_context_v1":
        raise ValueError(f"Unsupported context schema: {bundle['schema_version']}")
    compiler = bundle["compiler"]
    if not isinstance(compiler, dict) or set(compiler) != REQUIRED_COMPILER_KEYS:
        raise ValueError("Context bundle has an invalid compiler provenance block")
    if any(not isinstance(compiler[key], str) or not compiler[key] for key in REQUIRED_COMPILER_KEYS):
        raise ValueError("Context compiler provenance values must be non-empty strings")
    packets = bundle["packets"]
    if not isinstance(packets, list) or not packets:
        raise ValueError("Context bundle must contain a non-empty packet list")
    wavs = read_wav_order(wav_list_path)
    if len(packets) != len(wavs):
        raise ValueError("Context packet count does not match WAV list count")
    for index, (packet, wav) in enumerate(zip(packets, wavs, strict=True)):
        leaked = {
            key
            for key in _all_keys(packet)
            if any(marker in key.lower() for marker in FORBIDDEN_KEY_MARKERS)
        }
        if leaked:
            raise ValueError(f"Context packet {index} leaks scoring-only keys: {sorted(leaked)}")
        talk_id = packet.get("talk_id")
        if talk_id != wav.stem or packet.get("audio_basename") != wav.name:
            raise ValueError(f"Context packet {index} does not match WAV order: {wav}")
        conditions = packet.get("conditions")
        if not isinstance(conditions, dict) or set(conditions) != set(CONDITIONS):
            raise ValueError(f"Context packet {index} must define exactly C0-C3")
        for condition in CONDITIONS:
            row = conditions[condition]
            if not isinstance(row, dict):
                raise ValueError(f"Context packet {index}/{condition} must be an object")
            if set(row) != {
                "asr_context",
                "mt_context",
                "asr_token_count",
                "mt_token_count",
                "source_ids",
            }:
                raise ValueError(f"Context packet {index}/{condition} has an invalid schema")
            if not isinstance(row["asr_context"], str) or not isinstance(row["mt_context"], str):
                raise ValueError(f"Context packet {index}/{condition} contexts must be strings")
            if not isinstance(row["source_ids"], list):
                raise ValueError(f"Context packet {index}/{condition} source_ids must be a list")
            if any(not isinstance(source_id, str) or not source_id for source_id in row["source_ids"]):
                raise ValueError(f"Context packet {index}/{condition} source_ids must be non-empty strings")
            for side in ("asr_token_count", "mt_token_count"):
                if not isinstance(row[side], int) or row[side] < 0:
                    raise ValueError(f"Context packet {index}/{condition}/{side} must be non-negative")
            for side in ("asr", "mt"):
                if bool(row[f"{side}_context"]) != (row[f"{side}_token_count"] > 0):
                    raise ValueError(
                        f"Context packet {index}/{condition}/{side} text/count emptiness differs"
                    )
        c0 = conditions["C0"]
        if (
            c0["asr_context"]
            or c0["mt_context"]
            or c0["source_ids"]
            or c0["asr_token_count"]
            or c0["mt_token_count"]
        ):
            raise ValueError(f"Context packet {index}/C0 must be audio-only")
        for condition in CONDITIONS[1:]:
            row = conditions[condition]
            if not row["source_ids"] or not (row["asr_context"] or row["mt_context"]):
                raise ValueError(f"Context packet {index}/{condition} has no compiled evidence")
    return bundle


def load_context_packets(path: Path, wav_list_path: Path) -> list[dict]:
    return load_context_bundle(path, wav_list_path)["packets"]


def condition_contexts(
    packets: list[dict],
    condition: str,
    max_tokens: int,
    mt_token_counter: Callable[[str], int] | None = None,
) -> tuple[list[str], list[str]]:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    asr_contexts = []
    mt_contexts = []
    for index, packet in enumerate(packets):
        row = packet["conditions"][condition]
        for side in ("asr", "mt"):
            count = row[f"{side}_token_count"]
            if count > max_tokens:
                raise ValueError(
                    f"Context packet {index}/{condition}/{side}_context uses {count} tokens; "
                    f"limit is {max_tokens}"
                )
        if mt_token_counter is not None:
            actual_mt_tokens = mt_token_counter(row["mt_context"])
            if actual_mt_tokens != row["mt_token_count"]:
                raise ValueError(
                    f"Context packet {index}/{condition}/mt token count changed: "
                    f"stored={row['mt_token_count']} actual={actual_mt_tokens}"
                )
        asr_contexts.append(row["asr_context"])
        mt_contexts.append(row["mt_context"])
    return asr_contexts, mt_contexts
