from __future__ import annotations

from dataclasses import dataclass
import json
import os
import urllib.error
import urllib.request
from typing import Protocol, Sequence

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


class HuggingFaceTransformersTranslator:
    def __init__(self, cfg: dict):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Install torch and transformers to use provider=hf_transformers") from exc

        self.torch = torch
        self.model_name = cfg["model"]
        self.temperature = float(cfg.get("temperature", 0.0))
        self.max_new_tokens = int(cfg.get("max_new_tokens", 128))
        self.device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.chat_template_kwargs = _chat_template_kwargs(cfg)
        self.system_prompt = cfg.get("system_prompt")
        dtype_name = cfg.get("torch_dtype") or ("bfloat16" if self.device.startswith("cuda") else "float32")
        dtype = getattr(torch, dtype_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=bool(cfg.get("trust_remote_code", False)),
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = cfg.get("padding_side", "left")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            trust_remote_code=bool(cfg.get("trust_remote_code", False)),
        ).to(self.device)
        self.model.eval()

    def translate(self, state: StreamState, evidence_packet: list[EvidenceItem], condition: str) -> TranslationOutput:
        return self.translate_batch([state], [evidence_packet], condition)[0]

    def translate_batch(
        self,
        states: Sequence[StreamState],
        evidence_packets: Sequence[list[EvidenceItem]],
        condition: str,
    ) -> list[TranslationOutput]:
        prompts = [build_prompt(state, evidence, condition) for state, evidence in zip(states, evidence_packets, strict=True)]
        texts = self.complete_prompts(prompts)
        return [
            TranslationOutput(
                text=text.strip(),
                used_evidence_ids=[e.evidence_id for e in evidence],
                prompt=prompt,
            )
            for text, evidence, prompt in zip(texts, evidence_packets, prompts, strict=True)
        ]

    def complete_prompts(self, prompts: Sequence[str]) -> list[str]:
        encoded = self._encode_prompts(prompts)
        model_inputs, input_length = self._model_inputs(encoded)
        kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            kwargs["temperature"] = self.temperature
        with self.torch.inference_mode():
            output_ids = self.model.generate(**model_inputs, **kwargs)
        new_token_batches = output_ids[:, input_length:]
        return [text.strip() for text in self.tokenizer.batch_decode(new_token_batches, skip_special_tokens=True)]

    def _encode_prompts(self, prompts: Sequence[str]):
        conversations = [_build_messages(prompt, self.system_prompt) for prompt in prompts]
        if hasattr(self.tokenizer, "apply_chat_template"):
            chat_input = conversations[0] if len(conversations) == 1 else conversations
            return self.tokenizer.apply_chat_template(
                chat_input,
                add_generation_prompt=True,
                return_tensors="pt",
                padding=len(conversations) > 1,
                **self.chat_template_kwargs,
            )
        return self.tokenizer(list(prompts), return_tensors="pt", padding=True)

    def _model_inputs(self, encoded):
        if hasattr(encoded, "shape"):
            input_ids = encoded.to(self.device)
            return {"input_ids": input_ids}, input_ids.shape[-1]
        if hasattr(encoded, "to"):
            encoded = encoded.to(self.device)
        else:
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
        input_ids = encoded["input_ids"]
        return dict(encoded), input_ids.shape[-1]


def build_translator(cfg: dict) -> Translator:
    provider = cfg.get("provider", "mock")
    if provider == "mock":
        return MockTranslator()
    if provider in {"openai_compatible", "vllm"}:
        return OpenAICompatibleTranslator(cfg)
    if provider == "hf_transformers":
        return HuggingFaceTransformersTranslator(cfg)
    raise ValueError(f"Unsupported translation provider: {provider}")


def _chat_template_kwargs(cfg: dict) -> dict:
    kwargs = dict(cfg.get("chat_template_kwargs") or {})
    if "enable_thinking" in cfg:
        kwargs["enable_thinking"] = bool(cfg["enable_thinking"])
    return kwargs


def _build_messages(prompt: str, system_prompt: str | None = None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


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
