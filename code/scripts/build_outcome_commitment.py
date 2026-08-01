#!/usr/bin/env python3
"""Freeze raw annotation/adjudication artifacts before inference outputs exist."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from slidesst.eval.event_timing import OutcomeCommitment, directory_tree_sha256, file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-events", type=Path, required=True)
    parser.add_argument("--target-scores", type=Path, required=True)
    parser.add_argument("--source-annotation-report", type=Path, required=True)
    parser.add_argument("--source-adjudication", type=Path, required=True)
    parser.add_argument("--target-annotation-report", type=Path, required=True)
    parser.add_argument("--target-adjudication", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.artifact_root.exists():
        raise FileExistsError(f"outcome artifact root already exists: {args.artifact_root}")
    args.artifact_root.mkdir(parents=True)
    role_paths = {
        "source_annotation_report": args.source_annotation_report,
        "source_adjudication": args.source_adjudication,
        "target_annotation_report": args.target_annotation_report,
        "target_adjudication": args.target_adjudication,
    }
    artifacts = []
    for role, source in role_paths.items():
        suffix = "".join(source.suffixes) or ".bin"
        destination = args.artifact_root / f"{role}{suffix}"
        shutil.copyfile(source, destination)
        artifacts.append(
            {
                "role": role,
                "relative_path": destination.name,
                "sha256": file_sha256(destination),
            }
        )
    commitment = OutcomeCommitment(
        schema_version="acl6060_outcome_commitment_v1",
        created_at_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        source_events_sha256=file_sha256(args.source_events),
        target_scores_sha256=file_sha256(args.target_scores),
        artifacts=artifacts,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        output.write(commitment.model_dump_json(indent=2) + "\n")
    print(
        json.dumps(
            {
                "commitment_sha256": file_sha256(args.output),
                "artifact_tree_sha256": directory_tree_sha256(args.artifact_root),
            }
        )
    )


if __name__ == "__main__":
    main()
