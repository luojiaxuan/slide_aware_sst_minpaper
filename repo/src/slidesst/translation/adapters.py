from __future__ import annotations

from dataclasses import dataclass
import json
import os
import urllib.error
import urllib.request
from typing import Protocol

from slidesst.data.schema import EvidenceItem
from slidesst.streaming.simulator import StreamState


@dataclass
class TranslationOutput:
    text: str
    used_evidence_ids: list[str]
    prompt: str | None = None
    raw_response: dict | None = None


class Translator(Protocol):
    def translate(self, state: StreamState, evidence_packet: list[EvidenceItem], condition: str) -> TranslationOutput: ...


class MockTranslator:
    """Deterministic placeholder used in tests.

    Replace with OpenAI-compatible, local HF, or vLLM adapters.
    """

    def translate(self, state: StreamState, evidence_packet: list[EvidenceItem], condition: str) -> TranslationOutput:
        hints = [e.target_hint or e.text for e in evidence_packet]
        text = f"[MOCK {condition}] {state.partial_transcript}"
        if hints:
            text += " | hints: " + "; ".join(hints[:3])
        return TranslationOutput(text=text, used_evidence_ids=[e.evidence_id for e in evidence_packet], prompt=build_prompt(state, evidence_packet, condition))


class OpenAICompatibleTranslator:
    def __init__(self, cfg: dict):
        self.model = cfg["model"]
        self.base_url = cfg.get("base_url", "http://127.0.0.1:8000/v1").rstrip("/")
        self.api_key = os.environ.get(cfg.get("api_key_env", "OPENAI_API_KEY"), "EMPTY")
        self.temperature = float(cfg.get("temperature", 0.0))
        self.max_new_tokens = int(cfg.get("max_new_tokens", 128))
        self.timeout = float(cfg.get("timeout_sec", 120))

    def translate(self, state: StreamState, evidence_packet: list[EvidenceItem], condition: str) -> TranslationOutput:
        prompt = build_prompt(state, evidence_packet, condition)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
        }
        request = urllib.request.Request(
            url=self._chat_url(),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenAI-compatible translation request failed: {exc}") from exc
        text = raw["choices"][0]["message"]["content"].strip()
        return TranslationOutput(
            text=text,
            used_evidence_ids=[e.evidence_id for e in evidence_packet],
            prompt=prompt,
            raw_response=raw,
        )

    def _chat_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"


def build_translator(cfg: dict) -> Translator:
    provider = cfg.get("provider", "mock")
    if provider == "mock":
        return MockTranslator()
    if provider in {"openai_compatible", "vllm"}:
        return OpenAICompatibleTranslator(cfg)
    raise ValueError(f"Unsupported translation provider: {provider}")


def build_prompt(state: StreamState, evidence_packet: list[EvidenceItem], condition: str) -> str:
    evidence_lines = []
    for e in evidence_packet:
        evidence_lines.append(
            f"- id={e.evidence_id}; source={e.source_type}; text={e.text}; "
            f"target_hint={e.target_hint}; modality={e.modality}; anchor={e.spoken_anchor}; "
            f"confidence={e.confidence}; visual_only={e.visual_only}; supporting={e.is_supporting}; label={e.support_label}"
        )
    evidence = "\n".join(evidence_lines) if evidence_lines else "(none)"
    return f"""You are doing low-latency Chinese-to-English streaming speech translation.
Translate only what has been spoken in the current partial transcript.
Visual evidence may help disambiguate spoken words, deixis, objects, actions, or on-screen text.
Do not add any object, action, label, or fact that is visible but not spoken.
If visual evidence conflicts with speech, prefer speech.

Condition: {condition}
Current partial transcript: {state.partial_transcript}
Previous translation: {state.previous_translation}
Video time: {state.video_time_sec}
Evidence:
{evidence}

Return only the English translation:"""
