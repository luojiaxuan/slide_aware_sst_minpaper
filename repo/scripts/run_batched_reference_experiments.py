#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from collections.abc import Iterator
from pathlib import Path

import yaml
from tqdm import tqdm

from slidesst.context.policy import EvidencePolicy
from slidesst.context.retriever import EvidenceRetriever
from slidesst.data.io import read_jsonl, write_jsonl
from slidesst.data.schema import ChallengeItem, EvidenceItem, ModelOutput
from slidesst.streaming.simulator import StreamState
from slidesst.translation.adapters import build_translator
from slidesst.vision.evidence_builder import build_visual_evidence
from generate_references import _translation_config
from run_stream_translate import evidence_for_item, select_condition_packet


DEFAULT_CONDITIONS = (
    "V0_no_context",
    "V2_ocr_only",
    "V3_visual_caption_only",
    "V4_ocr_plus_visual",
    "V5_naive_all_visual",
    "V6_policy_visual",
    "V8_wrong_visual",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", default=None)
    parser.add_argument("--evidence-index", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--mismatch", default="matched")
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--prompt-version", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    translation_cfg = _translation_config(
        cfg,
        args.provider,
        args.model,
        args.device,
        args.prompt_version,
    )
    batch_size = args.batch_size or int(translation_cfg.get("batch_size", 1))
    conditions = args.conditions or cfg.get("conditions") or list(DEFAULT_CONDITIONS)
    items = read_jsonl(args.input or cfg["paths"]["challenge_jsonl"], ChallengeItem)
    if args.limit:
        items = items[: args.limit]
    evidence = read_jsonl(args.evidence_index or cfg["paths"]["context_index_jsonl"], EvidenceItem)
    evidence_by_item: dict[str, list[EvidenceItem]] = {}
    for ev in evidence:
        evidence_by_item.setdefault(ev.item_id or "", []).append(ev)

    translator = build_translator(translation_cfg)
    policy = _policy(cfg)
    run_dir = Path(args.run_dir or cfg["paths"]["run_dir"])
    for condition in conditions:
        outputs = _run_condition(
            items,
            evidence_by_item,
            translator,
            translation_cfg,
            policy,
            cfg,
            condition,
            args.mismatch,
            batch_size,
        )
        out_dir = run_dir / args.mismatch / condition
        out_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(out_dir / "outputs.jsonl", outputs)
        print(f"Wrote {out_dir / 'outputs.jsonl'}")


def _run_condition(
    items: list[ChallengeItem],
    evidence_by_item: dict[str, list[EvidenceItem]],
    translator,
    translation_cfg: dict,
    policy: EvidencePolicy,
    cfg: dict,
    condition: str,
    mismatch: str,
    batch_size: int,
) -> list[ModelOutput]:
    outputs: list[ModelOutput] = []
    for batch in tqdm(list(_chunks(items, batch_size)), desc=f"Running {condition}"):
        states: list[StreamState] = []
        packets: list[list[EvidenceItem]] = []
        policy_logs: list[list[dict]] = []
        start = time.time()
        for item in batch:
            item.evidence = _condition_evidence(item, evidence_by_item, condition, mismatch)
            state = _final_state(item)
            states.append(state)
            packet, policy_log = _packet_for_condition(item, state, condition, policy, cfg)
            packets.append(packet)
            policy_logs.append(policy_log)
        results = _translate_batch(translator, states, packets, condition)
        elapsed = time.time() - start
        per_item_latency = elapsed / max(1, len(batch))
        for item, result, packet, policy_log in zip(batch, results, packets, policy_logs, strict=True):
            output = ModelOutput(
                id=item.id,
                condition=condition,
                hypothesis=result.text,
                used_evidence_ids=result.used_evidence_ids,
                evidence_packet=packet,
                prompt=result.prompt,
                latency={"wall_time_sec": per_item_latency},
                metadata={"mismatch": mismatch, "policy_log": policy_log},
            )
            output.metadata.update(
                {
                    "model": translation_cfg.get("model"),
                    "prompt_version": translation_cfg.get("prompt_version"),
                    "seed": cfg.get("seed"),
                    "batch_size": batch_size,
                }
            )
            outputs.append(output)
    return outputs


def _translate_batch(translator, states: list[StreamState], packets: list[list[EvidenceItem]], condition: str):
    batch_translate = getattr(translator, "translate_batch", None)
    if callable(batch_translate):
        return batch_translate(states, packets, condition)
    return [translator.translate(state, packet, condition) for state, packet in zip(states, packets, strict=True)]


def _condition_evidence(
    item: ChallengeItem,
    evidence_by_item: dict[str, list[EvidenceItem]],
    condition: str,
    mismatch: str,
) -> list[EvidenceItem]:
    evidence = evidence_for_item(item, evidence_by_item.get(item.id, []))
    if condition == "V8_wrong_visual":
        wrong_mismatch = mismatch if mismatch != "matched" else "negative_visual"
        wrong = [
            ev
            for ev in build_visual_evidence(item, mismatch=wrong_mismatch)
            if ev.source_type in {"wrong_video", "wrong_clip", "negative_visual"}
        ]
        return [*evidence, *wrong]
    return evidence


def _packet_for_condition(
    item: ChallengeItem,
    state: StreamState,
    condition: str,
    policy: EvidencePolicy,
    cfg: dict,
) -> tuple[list[EvidenceItem], list[dict]]:
    if condition in {"policy", "V6_policy_visual"}:
        retriever = EvidenceRetriever(item.evidence, weights=cfg["policy"]["weights"])
        retrieved = retriever.retrieve(
            state.partial_transcript,
            state.current_slide_id,
            cfg["context"]["evidence_top_m"],
            current_time_sec=state.video_time_sec,
            visual_availability=cfg.get("streaming", {}).get("visual_availability", "offline_context"),
            allowed_lookahead_sec=float(cfg.get("streaming", {}).get("allowed_lookahead_sec", 0.0)),
        )
        packet, decisions = policy.select(retrieved)
        return packet, [
            {
                "t": state.t,
                "partial_transcript": state.partial_transcript,
                "decisions": [decision.__dict__ for decision in decisions],
            }
        ]
    return select_condition_packet(item, condition, cfg["context"]["packet_top_k"]), []


def _final_state(item: ChallengeItem) -> StreamState:
    unit = item.streaming_units[-1] if item.streaming_units else None
    transcript = unit.partial_transcript if unit else item.source_transcript
    t = unit.t if unit else (item.video.end_sec if item.video else 0.0)
    return StreamState(
        item_id=item.id,
        t=t,
        partial_transcript=transcript,
        current_slide_id=item.slides.matched_slide_id,
        video_time_sec=t,
    )


def _policy(cfg: dict) -> EvidencePolicy:
    return EvidencePolicy(
        use_threshold=cfg["policy"]["use_threshold"],
        delay_threshold=cfg["policy"]["delay_threshold"],
        top_k=cfg["context"]["packet_top_k"],
        penalties=cfg["policy"].get("penalties"),
    )


def _chunks(items: list[ChallengeItem], batch_size: int) -> Iterator[list[ChallengeItem]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


if __name__ == "__main__":
    main()
