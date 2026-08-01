#!/usr/bin/env python3
"""Serve frozen causal PCM prefixes to isolated inference workers."""

from __future__ import annotations

import argparse
import hashlib
import os
import shlex
import signal
import sys
import uuid
from pathlib import Path

from slidesst.eval.causal_audio import build_broker_audit, serve_causal_audio_broker
from slidesst.eval.event_timing import CausalAudioSchedule


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--release-events", type=Path, required=True)
    parser.add_argument("--broker-audit", type=Path, required=True)
    args = parser.parse_args()

    for path in (args.socket, args.release_events, args.broker_audit):
        if not path.is_absolute():
            raise ValueError(f"broker output path must be absolute: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    schedule_bytes = args.schedule.read_bytes()
    schedule = CausalAudioSchedule.model_validate_json(schedule_bytes)
    if schedule.run_id != args.run_id:
        raise ValueError("broker run id differs from causal audio schedule")
    repo_path = Path(__file__).resolve().parents[2]
    audit = build_broker_audit(
        schedule,
        schedule_sha256=hashlib.sha256(schedule_bytes).hexdigest(),
        repo_path=repo_path,
        entrypoint_path=Path(__file__),
        broker_command=shlex.join(sys.argv),
        socket_path=args.socket,
        release_events_path=args.release_events,
    )
    audit_temporary = args.broker_audit.parent / (
        f".{args.broker_audit.name}.{uuid.uuid4().hex}.tmp"
    )
    with audit_temporary.open("x", encoding="utf-8") as output:
        output.write(audit.model_dump_json(indent=2) + "\n")
        output.flush()
        os.fsync(output.fileno())

    def publish_broker_ready() -> None:
        os.link(audit_temporary, args.broker_audit)
        directory_fd = os.open(args.broker_audit.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        audit_temporary.unlink()

    def stop(_signum, _frame) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        serve_causal_audio_broker(
            schedule,
            socket_path=args.socket,
            release_events_path=args.release_events,
            ready_callback=publish_broker_ready,
        )
    except KeyboardInterrupt:
        pass
    finally:
        audit_temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
