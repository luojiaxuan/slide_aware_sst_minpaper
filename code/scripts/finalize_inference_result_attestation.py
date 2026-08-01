#!/usr/bin/env python3
"""Seal trajectories and runtime audits into a post-run result attestation."""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from slidesst.eval.event_timing import (
    CausalAudioReleaseLog,
    EventTrajectory,
    InferenceContract,
    InferenceEnvironmentAudit,
    InferenceResultAttestation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-contract", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--causal-audio-release-log", type=Path, required=True)
    parser.add_argument("--environment-start-audit", type=Path, required=True)
    parser.add_argument("--environment-end-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    input_paths = {
        "inference_contract": args.inference_contract,
        "trajectories": args.trajectories,
        "causal_audio_release_log": args.causal_audio_release_log,
        "environment_start_audit": args.environment_start_audit,
        "environment_end_audit": args.environment_end_audit,
    }
    input_bytes = {name: path.read_bytes() for name, path in input_paths.items()}
    input_sha256 = {
        name: hashlib.sha256(payload).hexdigest() for name, payload in input_bytes.items()
    }
    contract = InferenceContract.model_validate_json(input_bytes["inference_contract"])
    release_log = CausalAudioReleaseLog.model_validate_json(
        input_bytes["causal_audio_release_log"]
    )
    start_audit = InferenceEnvironmentAudit.model_validate_json(
        input_bytes["environment_start_audit"]
    )
    end_audit = InferenceEnvironmentAudit.model_validate_json(
        input_bytes["environment_end_audit"]
    )
    contract_sha256 = input_sha256["inference_contract"]
    trajectories = [
        EventTrajectory.model_validate_json(line)
        for line in input_bytes["trajectories"].splitlines()
        if line.strip()
    ]
    if not trajectories:
        raise ValueError("cannot attest an empty trajectory file")
    if any(
        row.inference_run_id != contract.run_id
        or row.inference_contract_sha256 != contract_sha256
        for row in trajectories
    ):
        raise ValueError("trajectory does not bind the frozen inference contract")
    if release_log.run_id != contract.run_id:
        raise ValueError("causal audio release log run id differs from contract")
    if start_audit.run_id != contract.run_id or start_audit.capture_phase != "workers_start":
        raise ValueError("invalid inference start audit")
    if end_audit.run_id != contract.run_id or end_audit.capture_phase != "workers_end":
        raise ValueError("invalid inference end audit")
    attestation = InferenceResultAttestation(
        schema_version="acl6060_event_inference_result_attestation_v1",
        run_id=contract.run_id,
        created_at_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        inference_contract_sha256=contract_sha256,
        trajectories_sha256=input_sha256["trajectories"],
        causal_audio_release_log_sha256=input_sha256["causal_audio_release_log"],
        environment_start_audit_sha256=input_sha256["environment_start_audit"],
        environment_end_audit_sha256=input_sha256["environment_end_audit"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        output.write(attestation.model_dump_json(indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
