#!/usr/bin/env python3
"""Freeze the pre-run inference contract and release its generation barrier."""

from __future__ import annotations

import argparse
from pathlib import Path

from slidesst.eval.inference_contract import build_inference_contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-events", type=Path, required=True)
    parser.add_argument("--source-artifact-root", type=Path, required=True)
    parser.add_argument("--evidence-packets", type=Path, required=True)
    parser.add_argument("--control-pairs", type=Path, required=True)
    parser.add_argument("--scientific-config", type=Path, required=True)
    parser.add_argument("--scoring-config", type=Path, required=True)
    parser.add_argument("--model-artifact-root", type=Path, required=True)
    parser.add_argument("--target-scores", type=Path, required=True)
    parser.add_argument("--outcome-commitment", type=Path, required=True)
    parser.add_argument("--outcome-artifact-root", type=Path, required=True)
    parser.add_argument("--causal-audio-schedule", type=Path, required=True)
    parser.add_argument("--causal-audio-broker-audit", type=Path, required=True)
    parser.add_argument("--tokenizer-artifact-root", type=Path, required=True)
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--environment-start-audit", type=Path, required=True)
    parser.add_argument("--worker-inference-contract-path", required=True)
    parser.add_argument("--worker-contract-ready-file-path", required=True)
    parser.add_argument("--worker-scientific-config-path", required=True)
    parser.add_argument("--worker-model-artifact-root-path", required=True)
    parser.add_argument("--worker-tokenizer-artifact-root-path", required=True)
    parser.add_argument("--scoring-protected-artifact-root", action="append", required=True)
    parser.add_argument("--code-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    args = parser.parse_args()

    build_inference_contract(
        run_id=args.run_id,
        source_events=args.source_events,
        source_artifact_root=args.source_artifact_root,
        evidence_packets=args.evidence_packets,
        control_pairs=args.control_pairs,
        scientific_config=args.scientific_config,
        scoring_config=args.scoring_config,
        model_artifact_root=args.model_artifact_root,
        target_scores=args.target_scores,
        outcome_commitment=args.outcome_commitment,
        outcome_artifact_root=args.outcome_artifact_root,
        causal_audio_schedule=args.causal_audio_schedule,
        causal_audio_broker_audit=args.causal_audio_broker_audit,
        tokenizer_artifact_root=args.tokenizer_artifact_root,
        tokenizer_model=args.tokenizer_model,
        tokenizer_revision=args.tokenizer_revision,
        environment_start_audit=args.environment_start_audit,
        worker_inference_contract_path=args.worker_inference_contract_path,
        worker_contract_ready_file_path=args.worker_contract_ready_file_path,
        worker_scientific_config_path=args.worker_scientific_config_path,
        worker_model_artifact_root_path=args.worker_model_artifact_root_path,
        worker_tokenizer_artifact_root_path=args.worker_tokenizer_artifact_root_path,
        scoring_protected_artifact_roots=args.scoring_protected_artifact_root,
        code_repo=args.code_repo,
        output=args.output,
        ready_file=args.ready_file,
    )
    print(args.output)
    print(args.ready_file)


if __name__ == "__main__":
    main()
