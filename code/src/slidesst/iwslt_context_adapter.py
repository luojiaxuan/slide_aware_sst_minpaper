"""Thin Phase-A adapter for the pinned IWSLT 2026 SimulStream baseline."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace

from agent_simulstream import CascadeSpeechProcessor

from slidesst.phase_a_contract import condition_contexts, load_context_bundle


LOGGER = logging.getLogger(__name__)


class PhaseAContextSpeechProcessor(CascadeSpeechProcessor):
    """Inject matched, budgeted context into both ASR and MT without vendoring upstream code."""

    def __init__(self, config: SimpleNamespace):
        condition = config.context_condition
        max_tokens = int(getattr(config, "max_context_tokens", 256))
        bundle = load_context_bundle(Path(config.context_packet_path), Path(config.wav_list_file))
        packets = bundle["packets"]

        if getattr(config, "ner_results_path", None) is not None:
            raise ValueError("Use context_packet_path instead of upstream ner_results_path")
        if getattr(config, "abstract_results_path", None) is not None:
            raise ValueError("Use context_packet_path instead of upstream abstract_results_path")

        super().__init__(config)
        token_counter = lambda text: len(self.tokenizer.encode(text, add_special_tokens=False))
        asr_contexts, mt_contexts = condition_contexts(
            packets,
            condition,
            max_tokens,
            mt_token_counter=token_counter,
        )
        self.ner_results = asr_contexts if any(asr_contexts) else None
        self._phase_a_mt_contexts = mt_contexts
        self._phase_a_talk_ids = [packet["talk_id"] for packet in packets]
        for packet in packets:
            row = packet["conditions"][condition]
            packet_sha256 = hashlib.sha256(
                json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            LOGGER.info(
                "phase_a_context=%s",
                json.dumps(
                    {
                        "talk_id": packet["talk_id"],
                        "condition": condition,
                        "asr_token_count": row["asr_token_count"],
                        "mt_token_count": row["mt_token_count"],
                        "source_ids": row["source_ids"],
                        "packet_sha256": packet_sha256,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            )

    def _transcribe_audio(self, state):
        asr_segment, utterance_finished = super()._transcribe_audio(state)
        if asr_segment is None:
            talk_id = self._phase_a_talk_ids[state.speech_id]
            raise RuntimeError(f"ASR timestamp/text integrity failure for talk_id={talk_id}")
        return asr_segment, utterance_finished

    def _prepare_llm_inputs(self, asr_segment: str, prev_translation: str, context: str) -> str:
        speech_id = self._state.speech_id
        if speech_id >= len(self._phase_a_mt_contexts):
            raise IndexError(f"Missing MT context for speech_id={speech_id}")
        return super()._prepare_llm_inputs(
            asr_segment,
            prev_translation,
            self._phase_a_mt_contexts[speech_id],
        )

    def end_of_stream(self):
        speech_id = self._state.speech_id
        talk_id = self._phase_a_talk_ids[speech_id]
        result = super().end_of_stream()
        LOGGER.info("phase_a_talk_complete=%s", json.dumps({"speech_id": speech_id, "talk_id": talk_id}))
        return result
