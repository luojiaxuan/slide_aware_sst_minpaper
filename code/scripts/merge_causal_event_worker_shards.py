#!/usr/bin/env python3
"""Validate and merge completed causal inference worker trajectory shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pydantic import TypeAdapter

from slidesst.eval.causal_worker import merge_trajectory_shards
from slidesst.eval.event_timing import (
    CausalAudioSchedule,
    EvidencePacketSpec,
    InferenceEnvironmentAudit,
)
from slidesst.eval.inference_contract import load_inference_contract_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-contract", type=Path, required=True)
    parser.add_argument("--inference-contract-ready-file", type=Path, required=True)
    parser.add_argument("--inference-environment-start-audit", type=Path, required=True)
    parser.add_argument("--causal-audio-schedule", type=Path, required=True)
    parser.add_argument("--evidence-packets", type=Path, required=True)
    parser.add_argument("--worker-output", type=Path, action="append", required=True)
    parser.add_argument("--worker-done", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    contract, contract_sha256 = load_inference_contract_snapshot(
        run_id=None,
        inference_contract=args.inference_contract,
        ready_file=args.inference_contract_ready_file,
        require_worker_paths=False,
    )
    start_audit_bytes = args.inference_environment_start_audit.read_bytes()
    if (
        hashlib.sha256(start_audit_bytes).hexdigest()
        != contract.environment_start_audit_sha256
    ):
        raise ValueError("merge start-audit hash differs from contract")
    start_audit = InferenceEnvironmentAudit.model_validate_json(start_audit_bytes)
    schedule_bytes = args.causal_audio_schedule.read_bytes()
    evidence_bytes = args.evidence_packets.read_bytes()
    if hashlib.sha256(schedule_bytes).hexdigest() != contract.causal_audio_schedule_sha256:
        raise ValueError("merge schedule hash differs from contract")
    if hashlib.sha256(evidence_bytes).hexdigest() != contract.evidence_packets_sha256:
        raise ValueError("merge evidence-packet hash differs from contract")
    schedule = CausalAudioSchedule.model_validate_json(schedule_bytes)
    packet_adapter = TypeAdapter(EvidencePacketSpec)
    packets = [
        packet_adapter.validate_json(line)
        for line in evidence_bytes.splitlines()
        if line.strip()
    ]
    merged = merge_trajectory_shards(
        contract=contract,
        inference_contract_sha256=contract_sha256,
        schedule=schedule,
        packets=packets,
        environment_start_audit=start_audit,
        shard_paths=args.worker_output,
        done_paths=args.worker_done,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "trajectory_count": len(merged),
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
