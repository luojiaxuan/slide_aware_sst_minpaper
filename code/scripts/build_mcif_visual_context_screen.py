#!/usr/bin/env python3
"""Build a reference-free MCIF causal-state manifest for private VLM screening."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def resolve_frame(path_value: object, state_root: Path) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("MCIF causal state lacks an evidence frame path")
    path = Path(path_value)
    if not path.is_absolute():
        raise ValueError("MCIF evidence frame path must be absolute")
    if state_root.is_symlink():
        raise ValueError("MCIF state root cannot be a symlink")
    root = state_root.resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("MCIF evidence frame escapes the state root")
    component = root
    for part in path.relative_to(root).parts:
        component = component / part
        if component.is_symlink():
            raise ValueError("MCIF evidence frame path cannot traverse a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or not resolved.is_relative_to(root):
        raise ValueError("MCIF evidence frame escapes the state root")
    return resolved


def validate_talk_states(
    talk_id: str,
    states: list[dict],
    *,
    duration_sec: float,
    state_root: Path,
) -> list[Path]:
    ordered = sorted(states, key=lambda row: row.get("state_id", -1))
    if [row.get("state_id") for row in ordered] != list(range(len(ordered))):
        raise ValueError(f"Non-contiguous MCIF state ids for {talk_id}")
    if not ordered or float(ordered[0].get("availability_start_sec", -1)) != 0.0:
        raise ValueError(f"MCIF state timeline does not start at zero for {talk_id}")
    frames = []
    for index, state in enumerate(ordered):
        if state.get("talk_id") != talk_id:
            raise ValueError(f"MCIF state talk mismatch for {talk_id}")
        start = float(state["availability_start_sec"])
        end = float(state["availability_end_sec"])
        if end <= start:
            raise ValueError(f"Non-positive MCIF state interval for {talk_id}")
        if index and abs(
            float(ordered[index - 1]["availability_end_sec"]) - start
        ) > 1e-6:
            raise ValueError(f"Non-contiguous MCIF state intervals for {talk_id}")
        frame = resolve_frame(state.get("evidence_frame_path"), state_root)
        expected_hash = state.get("evidence_frame_sha256")
        if not isinstance(expected_hash, str) or sha256_file(frame) != expected_hash:
            raise ValueError(f"MCIF evidence frame hash mismatch for {talk_id}")
        frames.append(frame)
    if abs(float(ordered[-1]["availability_end_sec"]) - duration_sec) > 1e-3:
        raise ValueError(f"MCIF state timeline duration mismatch for {talk_id}")
    return frames


def build_rows(
    states: list[dict], source_manifest: dict, state_root: Path
) -> list[dict]:
    if source_manifest.get("dataset") != "mcif":
        raise ValueError("Source manifest is not MCIF")
    if source_manifest.get("reference_files_extracted") is not False:
        raise ValueError("MCIF reference files must remain unextracted")
    if source_manifest.get("reference_content_inspected") is not False:
        raise ValueError("MCIF reference content must remain uninspected")
    talks = {row["talk_id"]: row for row in source_manifest.get("talks", [])}
    if len(talks) != source_manifest.get("talk_count"):
        raise ValueError("MCIF source manifest talk ids are not unique")
    state_talks = {row.get("talk_id") for row in states}
    if state_talks != set(talks):
        raise ValueError("MCIF causal-state and source-manifest talk ids differ")
    if len({(row.get("talk_id"), row.get("state_id")) for row in states}) != len(
        states
    ):
        raise ValueError("MCIF causal states contain duplicate identifiers")

    rows = []
    resolved_state_root = state_root.resolve(strict=True)
    for talk_id in sorted(talks):
        talk_states = [row for row in states if row["talk_id"] == talk_id]
        frames = validate_talk_states(
            talk_id,
            talk_states,
            duration_sec=float(talks[talk_id]["video_duration_sec"]),
            state_root=state_root,
        )
        for state, frame in zip(
            sorted(talk_states, key=lambda row: row["state_id"]), frames, strict=True
        ):
            state_id = int(state["state_id"])
            item_id = f"mcif:{talk_id}:S{state_id:03d}"
            rows.append(
                {
                    "id": item_id,
                    "lecture_id": talk_id,
                    "source_lang": "en",
                    "target_lang": "zh",
                    "source_transcript": "",
                    "video": {
                        "start_sec": float(state["availability_start_sec"]),
                        "end_sec": float(state["availability_end_sec"]),
                        "frame_paths": [
                            frame.relative_to(resolved_state_root).as_posix()
                        ],
                    },
                    "visual_context": {
                        "video_id": talk_id,
                        "clip_id": item_id,
                        "metadata": {
                            "screen_role": "private_source_only_prescreen_not_annotation",
                            "state_id": state_id,
                            "availability_start_sec": float(
                                state["availability_start_sec"]
                            ),
                            "availability_end_sec": float(
                                state["availability_end_sec"]
                            ),
                            "evidence_frame_sha256": state[
                                "evidence_frame_sha256"
                            ],
                        },
                    },
                }
            )
    return rows


def validate_source_only_rows(rows: list[dict]) -> None:
    for row in rows:
        if row.get("source_transcript") != "":
            raise ValueError("MCIF screen input contains source transcript text")
        forbidden = {
            "audio",
            "reference",
            "reference_translation",
            "target_translation",
            "model_output",
        }
        if forbidden & set(row):
            raise ValueError("MCIF screen input contains forbidden outcome fields")
        frame_paths = row.get("video", {}).get("frame_paths", [])
        if len(frame_paths) != 1 or Path(frame_paths[0]).is_absolute():
            raise ValueError("MCIF screen frame path must be one portable relative path")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--causal-states", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--portable-output-label", required=True)
    args = parser.parse_args()
    if args.output.exists() or args.summary_out.exists():
        raise FileExistsError("MCIF VLM screen outputs must be created once")

    source_manifest = load_json(args.source_manifest)
    states = load_jsonl(args.causal_states)
    rows = build_rows(states, source_manifest, args.state_root)
    validate_source_only_rows(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    summary = {
        "dataset": "mcif",
        "subset": source_manifest["subset"],
        "artifact": "reference_free_visual_context_screen_input_v1",
        "status": "READY_FOR_PRIVATE_VLM_SCREEN_NOT_ANNOTATION",
        "upstream_revision": source_manifest["upstream"]["revision"],
        "talk_count": len({row["lecture_id"] for row in rows}),
        "causal_state_count": len(rows),
        "portable_output_label": args.portable_output_label,
        "screen_input_sha256": sha256_file(args.output),
        "causal_state_rows_sha256": canonical_hash(states),
        "frame_binding_set_sha256": canonical_hash(
            [
                {
                    "id": row["id"],
                    "sha256": row["visual_context"]["metadata"][
                        "evidence_frame_sha256"
                    ],
                }
                for row in rows
            ]
        ),
        "source_transcript_consumed": any(
            bool(row["source_transcript"]) for row in rows
        ),
        "target_or_reference_consumed": any(
            {"reference", "reference_translation", "target_translation"} & set(row)
            for row in rows
        ),
        "model_output_consumed": any("model_output" in row for row in rows),
        "interpretation": (
            "The manifest is a private source-only VLM prescreen over every causal "
            "state. It is not a human label, eligibility decision, or paper result, "
            "and no state may be dropped based on its output."
        ),
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
