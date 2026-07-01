#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import time
import yaml

from slidesst.data.io import read_jsonl, write_jsonl
from slidesst.data.schema import ChallengeItem, ModelOutput, EvidenceItem
from slidesst.context.retriever import EvidenceRetriever
from slidesst.context.policy import EvidencePolicy
from slidesst.streaming.simulator import TranscriptOracleStreamer
from slidesst.translation.adapters import build_translator


def select_condition_packet(item: ChallengeItem, condition: str, top_k: int) -> list[EvidenceItem]:
    evidence = item.evidence
    if condition == "no_context":
        return []
    if condition == "naive_all_context":
        return evidence[:top_k]
    if condition == "glossary_only":
        return [e for e in evidence if e.source_type == "glossary"][:top_k]
    if condition == "slide_ocr_current":
        return [e for e in evidence if e.source_type == "slide_ocr"][:top_k]
    if condition == "background_only":
        return [e for e in evidence if e.source_type == "background"][:top_k]
    if condition == "oracle":
        return [e for e in evidence if e.is_supporting is True][:top_k]
    return []


def evidence_for_item(item: ChallengeItem, evidence_index: list[EvidenceItem]) -> list[EvidenceItem]:
    if not evidence_index:
        return item.evidence
    specific = [e for e in evidence_index if e.item_id == item.id]
    if specific:
        return specific
    unscoped = [e for e in evidence_index if e.item_id is None]
    return unscoped or item.evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--mismatch", default="matched")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    condition = args.condition

    items = read_jsonl(cfg["paths"]["challenge_jsonl"], ChallengeItem)
    if args.limit:
        items = items[: args.limit]
    evidence_path = Path(cfg["paths"]["context_index_jsonl"])
    evidence = read_jsonl(evidence_path, EvidenceItem) if evidence_path.exists() else []
    policy = EvidencePolicy(
        use_threshold=cfg["policy"]["use_threshold"],
        delay_threshold=cfg["policy"]["delay_threshold"],
        top_k=cfg["context"]["packet_top_k"],
        penalties=cfg["policy"].get("penalties"),
    )
    streamer = TranscriptOracleStreamer()
    translator = build_translator(cfg["translation"])

    outputs = []
    for item in items:
        item.evidence = evidence_for_item(item, evidence)
        retriever = EvidenceRetriever(item.evidence, weights=cfg["policy"]["weights"])
        final_text = ""
        used_ids = []
        final_prompt = None
        final_packet = []
        policy_log = []
        start = time.time()
        for state in streamer.stream(item):
            if condition == "policy":
                retrieved = retriever.retrieve(state.partial_transcript, state.current_slide_id, cfg["context"]["evidence_top_m"])
                packet, decisions = policy.select(retrieved)
                policy_log.append(
                    {
                        "t": state.t,
                        "partial_transcript": state.partial_transcript,
                        "decisions": [decision.__dict__ for decision in decisions],
                    }
                )
            else:
                packet = select_condition_packet(item, condition, cfg["context"]["packet_top_k"])
            result = translator.translate(state, packet, condition)
            final_text = result.text
            used_ids.extend(result.used_evidence_ids)
            final_prompt = result.prompt
            final_packet = packet
        outputs.append(ModelOutput(
            id=item.id,
            condition=condition,
            hypothesis=final_text,
            used_evidence_ids=sorted(set(used_ids)),
            evidence_packet=final_packet,
            prompt=final_prompt,
            latency={"wall_time_sec": time.time() - start},
            metadata={"mismatch": args.mismatch, "policy_log": policy_log},
        ))

    out_dir = Path(cfg["paths"]["run_dir"]) / args.mismatch / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "outputs.jsonl", outputs)
    print(f"Wrote {out_dir / 'outputs.jsonl'}")


if __name__ == "__main__":
    main()
