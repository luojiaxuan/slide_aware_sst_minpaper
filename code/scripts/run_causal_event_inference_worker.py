#!/usr/bin/env python3
"""Run one frozen, broker-mediated causal Qwen3-Omni inference worker."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from slidesst.eval.causal_worker import (
    Qwen3OmniBatchGenerator,
    load_worker_inputs,
    process_start_time_ticks,
    run_causal_event_worker,
    verify_evidence_packet_tokenization,
    wait_for_shutdown_file,
    worker_talk_partition,
    write_trajectory_shard,
)
from slidesst.eval.event_timing import directory_tree_sha256
from slidesst.eval.inference_contract import (
    load_frozen_scientific_config,
    wait_for_inference_contract_snapshot,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--worker-count", type=int, required=True)
    parser.add_argument("--inference-contract", type=Path, required=True)
    parser.add_argument("--inference-contract-ready-file", type=Path, required=True)
    parser.add_argument("--scientific-config", type=Path, required=True)
    parser.add_argument("--model-artifact-root", type=Path, required=True)
    parser.add_argument("--tokenizer-artifact-root", type=Path, required=True)
    parser.add_argument("--source-artifact-root", type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--causal-audio-schedule", type=Path, required=True)
    parser.add_argument("--causal-audio-broker-audit", type=Path, required=True)
    parser.add_argument("--evidence-packets", type=Path, required=True)
    parser.add_argument("--broker-socket", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--barrier-waiting-file", type=Path, required=True)
    parser.add_argument("--done-file", type=Path, required=True)
    parser.add_argument("--shutdown-file", type=Path, required=True)
    parser.add_argument("--barrier-timeout-sec", type=float, default=3600.0)
    parser.add_argument("--shutdown-timeout-sec", type=float, default=3600.0)
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--attention-implementation", default="sdpa")
    args = parser.parse_args()
    worker_id = f"worker-{args.worker_index:02d}-of-{args.worker_count:02d}"
    worker_pid = os.getpid()
    worker_start_time_ticks = process_start_time_ticks(worker_pid)

    for path, label in (
        (args.inference_contract, "inference contract"),
        (args.inference_contract_ready_file, "contract ready marker"),
        (args.output, "trajectory shard"),
        (args.barrier_waiting_file, "barrier-waiting marker"),
        (args.done_file, "worker done marker"),
        (args.shutdown_file, "post-audit shutdown marker"),
    ):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"stale {label} exists before worker start: {path}")
    args.barrier_waiting_file.parent.mkdir(parents=True, exist_ok=True)
    with args.barrier_waiting_file.open("x", encoding="utf-8") as output:
        output.write(
            json.dumps(
                {
                    "schema_version": "acl6060_worker_barrier_waiting_v1",
                    "run_id": args.run_id,
                    "worker_id": worker_id,
                    "pid": worker_pid,
                    "process_start_time_ticks": worker_start_time_ticks,
                },
                sort_keys=True,
            )
            + "\n"
        )
        output.flush()
        os.fsync(output.fileno())

    contract, contract_sha256 = wait_for_inference_contract_snapshot(
        run_id=args.run_id,
        inference_contract=args.inference_contract,
        ready_file=args.inference_contract_ready_file,
        timeout_sec=args.barrier_timeout_sec,
    )
    scientific = load_frozen_scientific_config(contract, args.scientific_config)
    if args.worker_count != contract.expected_worker_count:
        raise ValueError("worker count differs from frozen environment audit")
    if (args.model_id, args.model_revision) != (
        contract.model_id,
        contract.model_revision,
    ):
        raise ValueError("worker model identity differs from contract")
    if args.model_artifact_root.as_posix() != contract.worker_model_artifact_root_path:
        raise ValueError("worker model root differs from contract")
    if (
        args.tokenizer_artifact_root.as_posix()
        != contract.worker_tokenizer_artifact_root_path
    ):
        raise ValueError("worker tokenizer root differs from contract")
    observed_source_root = (
        None
        if args.source_artifact_root is None
        else args.source_artifact_root.as_posix()
    )
    if observed_source_root != contract.worker_source_artifact_root_path:
        raise ValueError("worker source artifact root differs from contract")
    if directory_tree_sha256(args.model_artifact_root) != contract.model_artifact_tree_sha256:
        raise ValueError("worker model artifact tree changed before model load")
    if (
        directory_tree_sha256(args.tokenizer_artifact_root)
        != contract.tokenizer_artifact_sha256
    ):
        raise ValueError("worker tokenizer artifact tree changed before model load")
    if args.source_artifact_root is not None and (
        directory_tree_sha256(args.source_artifact_root)
        != contract.source_artifact_tree_sha256
    ):
        raise ValueError("worker source artifact tree changed before model load")
    schedule, broker_audit, packets = load_worker_inputs(
        contract=contract,
        schedule_path=args.causal_audio_schedule,
        broker_audit_path=args.causal_audio_broker_audit,
        evidence_packets_path=args.evidence_packets,
    )
    if args.broker_socket.as_posix() != broker_audit.socket_path:
        raise ValueError("worker broker socket differs from frozen broker audit")
    talk_ids = worker_talk_partition(
        schedule,
        worker_index=args.worker_index,
        worker_count=args.worker_count,
    )
    generator = Qwen3OmniBatchGenerator(
        model_artifact_root=args.model_artifact_root,
        max_new_tokens=scientific.decoding.max_new_tokens,
        device_map=args.device_map,
        attention_implementation=args.attention_implementation,
    )
    verify_evidence_packet_tokenization(packets, generator.tokenize_evidence)
    trajectories = run_causal_event_worker(
        run_id=args.run_id,
        worker_id=worker_id,
        inference_contract_sha256=contract_sha256,
        schedule=schedule,
        packets=packets,
        selected_talk_ids=talk_ids,
        broker_socket=args.broker_socket,
        prompt_template=scientific.prompt_template,
        generate_batch=generator,
        tokenizer_model=contract.tokenizer_model,
        tokenizer_revision=contract.tokenizer_revision,
        tokenizer_artifact_sha256=contract.tokenizer_artifact_sha256,
        source_artifact_root=args.source_artifact_root,
    )
    if directory_tree_sha256(args.model_artifact_root) != contract.model_artifact_tree_sha256:
        raise ValueError("worker model artifact tree changed during generation")
    if (
        directory_tree_sha256(args.tokenizer_artifact_root)
        != contract.tokenizer_artifact_sha256
    ):
        raise ValueError("worker tokenizer artifact tree changed during generation")
    if args.source_artifact_root is not None and (
        directory_tree_sha256(args.source_artifact_root)
        != contract.source_artifact_tree_sha256
    ):
        raise ValueError("worker source artifact tree changed during generation")
    write_trajectory_shard(
        args.output,
        args.done_file,
        run_id=args.run_id,
        worker_id=worker_id,
        worker_index=args.worker_index,
        worker_count=args.worker_count,
        pid=worker_pid,
        process_start_time_ticks=worker_start_time_ticks,
        worker_process_identity_tree_sha256=(
            contract.worker_process_identity_tree_sha256
        ),
        inference_contract_sha256=contract_sha256,
        causal_audio_schedule_sha256=contract.causal_audio_schedule_sha256,
        evidence_packets_sha256=contract.evidence_packets_sha256,
        talk_ids=talk_ids,
        trajectories=trajectories,
    )
    wait_for_shutdown_file(
        args.shutdown_file,
        timeout_sec=args.shutdown_timeout_sec,
    )


if __name__ == "__main__":
    main()
