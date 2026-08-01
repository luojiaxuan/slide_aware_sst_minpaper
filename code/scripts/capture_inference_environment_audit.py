#!/usr/bin/env python3
"""Capture Docker mounts and worker open files for an inference run."""

from __future__ import annotations

import argparse
from pathlib import Path

from slidesst.eval.inference_audit import capture_inference_environment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--worker-command-match", required=True)
    parser.add_argument("--inference-repo", required=True)
    parser.add_argument("--capture-phase", choices=("workers_start", "workers_end"), required=True)
    parser.add_argument("--forbidden-container-artifact-root", action="append", required=True)
    parser.add_argument("--forbidden-host-mount-source-root", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    audit = capture_inference_environment(
        run_id=args.run_id,
        container_name=args.container,
        worker_command_match=args.worker_command_match,
        inference_repo_path=args.inference_repo,
        forbidden_container_artifact_roots=args.forbidden_container_artifact_root,
        forbidden_host_mount_source_roots=args.forbidden_host_mount_source_root,
        capture_phase=args.capture_phase,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(audit.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
