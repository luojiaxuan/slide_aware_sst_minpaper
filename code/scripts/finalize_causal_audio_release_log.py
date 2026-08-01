#!/usr/bin/env python3
"""Seal append-only causal audio releases into the scorer input artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from slidesst.eval.causal_audio import finalize_release_log


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--broker-audit", type=Path, required=True)
    parser.add_argument("--release-events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    finalize_release_log(
        schedule_path=args.schedule,
        broker_audit_path=args.broker_audit,
        release_events_path=args.release_events,
        output_path=args.output,
    )
    print(args.output)


if __name__ == "__main__":
    main()
