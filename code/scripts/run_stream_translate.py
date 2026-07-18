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
    if condition in {"no_context", "V0_no_context"}:
        return []
    if condition in {"naive_all_context", "V5_naive_all_visual"}:
        return evidence[:top_k]
    if condition == "V1_text_context":
        return [e for e in evidence if e.source_type in {"glossary", "background", "history"}][:top_k]
    if condition == "V2_ocr_only":
        return [e for e in evidence if e.source_type in {"slide_ocr", "video_ocr"}][:top_k]
    if condition == "V3_visual_caption_only":
        return [e for e in evidence if e.source_type in {"slide_vlm", "video_scene", "video_object", "video_action", "video_spatial", "video_vlm_frame", "video_vlm_clip"}][:top_k]
    if condition == "V4_ocr_plus_visual":
        return [e for e in evidence if e.modality in {"image", "video", "mixed"} or e.source_type in {"slide_ocr", "video_ocr", "slide_vlm"}][:top_k]
    if condition == "glossary_only":
        return [e for e in evidence if e.source_type == "glossary"][:top_k]
    if condition == "slide_ocr_current":
        return [e for e in evidence if e.source_type == "slide_ocr"][:top_k]
    if condition == "background_only":
        return [e for e in evidence if e.source_type == "background"][:top_k]
    if condition in {"oracle", "V7_oracle_supporting"}:
        return [e for e in evidence if e.is_supporting is True][:top_k]
    if condition == "V8_wrong_visual":
        return [e for e in evidence if e.source_type in {"wrong_video", "wrong_clip", "negative_visual"}][:top_k]
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
    streaming_cfg = cfg.get("streaming", {})
    streamer = TranscriptOracleStreamer(
        visual_availability=streaming_cfg.get("visual_availability", "offline_context"),
        allowed_lookahead_sec=float(streaming_cfg.get("allowed_lookahead_sec", 0.0)),
        visual_window_sec=float(streaming_cfg.get("visual_window_sec", 2.0)),
    )
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
            if condition in {"policy", "V6_policy_visual"}:
                retrieved = retriever.retrieve(
                    state.partial_transcript,
                    state.current_slide_id,
                    cfg["context"]["evidence_top_m"],
                    current_time_sec=state.video_time_sec,
                    visual_availability=streaming_cfg.get("visual_availability", "offline_context"),
                    allowed_lookahead_sec=float(streaming_cfg.get("allowed_lookahead_sec", 0.0)),
                )
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
        outputs[-1].metadata.update({
            "model": cfg["translation"].get("model"),
            "prompt_version": cfg["translation"].get("prompt_version"),
            "seed": cfg.get("seed"),
        })

    out_dir = Path(cfg["paths"]["run_dir"]) / args.mismatch / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "outputs.jsonl", outputs)
    print(f"Wrote {out_dir / 'outputs.jsonl'}")


if __name__ == "__main__":
    main()
