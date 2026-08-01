from __future__ import annotations

import hashlib
import json
import math
import posixpath
import random
import re
import shlex
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from string import Formatter
from typing import Annotated, Callable, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


SHA256_PATTERN = r"^[0-9a-f]{64}$"
GIT_SHA_PATTERN = r"^[0-9a-f]{40}$"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
EXPECTED_ACOUSTIC_GROUPS_V1 = {
    "native": ("native",),
    "babble_10db": ("babble_p10_s0", "babble_p10_s1", "babble_p10_s2"),
    "babble_5db": ("babble_p5_s0", "babble_p5_s1", "babble_p5_s2"),
    "babble_0db": ("babble_0_s0", "babble_0_s1", "babble_0_s2"),
    "babble_minus5db": ("babble_m5_s0", "babble_m5_s1", "babble_m5_s2"),
    "generic_noise_0db": ("generic_noise_0_s0",),
    "music_0db": ("music_0_s0",),
    "rir_medium_near": ("rir_medium_near",),
}
EXPECTED_CONDITIONS_V1 = (
    "audio_only",
    "document_only",
    "empty",
    "ocr",
    "matched_wrong_ocr",
    "correct_semantic",
    "matched_wrong_semantic",
    "correct_relation",
    "matched_wrong_relation",
)
EXPECTED_PACKET_META_V1 = {
    "audio_only": ("none", "baseline"),
    "document_only": ("document", "baseline"),
    "empty": ("empty", "baseline"),
    "ocr": ("ocr", "correct"),
    "matched_wrong_ocr": ("ocr", "matched_wrong"),
    "correct_semantic": ("semantic", "correct"),
    "matched_wrong_semantic": ("semantic", "matched_wrong"),
    "correct_relation": ("relation", "correct"),
    "matched_wrong_relation": ("relation", "matched_wrong"),
}
EXPECTED_CONTEXT_KIND_V1 = {
    "audio_only": "none",
    "document_only": "document",
    "empty": "empty",
    "ocr": "ocr",
    "matched_wrong_ocr": "ocr",
    "correct_semantic": "semantic",
    "matched_wrong_semantic": "semantic",
    "correct_relation": "relation",
    "matched_wrong_relation": "relation",
}
EXPECTED_CONTRASTS_V1 = (
    ("semantic_content_specificity", "correct_semantic", "matched_wrong_semantic", True, "semantic"),
    ("relation_content_specificity", "correct_relation", "matched_wrong_relation", True, "relation"),
    ("ocr_content_specificity", "ocr", "matched_wrong_ocr", True, "ocr"),
    ("ocr_over_audio_only", "ocr", "audio_only", False, None),
)
EXPECTED_CONDITIONS_V2 = EXPECTED_CONDITIONS_V1 + (
    "correct_image",
    "matched_wrong_image",
)
EXPECTED_PACKET_META_V2 = {
    **EXPECTED_PACKET_META_V1,
    "correct_image": ("image", "correct"),
    "matched_wrong_image": ("image", "matched_wrong"),
}
EXPECTED_CONTEXT_KIND_V2 = {
    **EXPECTED_CONTEXT_KIND_V1,
    "correct_image": "image",
    "matched_wrong_image": "image",
}
EXPECTED_CONTRASTS_V2 = EXPECTED_CONTRASTS_V1 + (
    (
        "image_content_specificity",
        "correct_image",
        "matched_wrong_image",
        True,
        "image",
    ),
    ("image_over_relation", "correct_image", "correct_relation", False, None),
    ("relation_over_ocr", "correct_relation", "ocr", False, None),
)
EXPECTED_BABBLE_ORDER_V1 = (
    "native",
    "babble_10db",
    "babble_5db",
    "babble_0db",
    "babble_minus5db",
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class InferenceDecodingConfig(StrictModel):
    max_new_tokens: int = Field(gt=0)
    do_sample: Literal[False]
    num_beams: Literal[1]


class InferenceScientificConfig(StrictModel):
    schema_version: Literal[
        "acl6060_event_inference_scientific_config_v1",
        "acl6060_event_inference_scientific_config_v2",
    ]
    model_id: str = Field(min_length=1)
    model_revision: str = Field(pattern=GIT_SHA_PATTERN)
    model_artifact_tree_sha256: str = Field(pattern=SHA256_PATTERN)
    execution_backend: Literal["in_process_transformers"]
    source_language: Literal["en"]
    target_language: Literal["zh"]
    chunk_policy: Literal["external_causal_audio_prefix_broker_v1"]
    prompt_template: str = Field(min_length=1)
    expected_conditions: list[str] = Field(min_length=1)
    decoding: InferenceDecodingConfig

    @model_validator(mode="after")
    def validate_scientific_config(self) -> "InferenceScientificConfig":
        expected = (
            EXPECTED_CONDITIONS_V1
            if self.schema_version.endswith("_v1")
            else EXPECTED_CONDITIONS_V2
        )
        if tuple(self.expected_conditions) != expected:
            raise ValueError("scientific config condition matrix differs from its schema")
        fields = [
            field_name
            for _, field_name, _, _ in Formatter().parse(self.prompt_template)
            if field_name is not None
        ]
        if fields.count("evidence") != 1 or set(fields) - {"evidence"}:
            raise ValueError("prompt template must contain only one {evidence} placeholder")
        forbidden_fragments = (
            "target_scores",
            "acceptable_realizations",
            "forbidden_realizations",
            "reference_text",
            "gold_target",
        )
        normalized_prompt = self.prompt_template.casefold()
        if any(fragment in normalized_prompt for fragment in forbidden_fragments):
            raise ValueError("scientific prompt contains a forbidden target/reference field")
        return self


class ExpectedEvidenceSource(StrictModel):
    condition: str = Field(min_length=1)
    context_kind: Literal["document", "ocr", "semantic", "relation", "image"]
    source_media_kind: Literal["source_document", "slide_image"]
    source_media_path: str = Field(min_length=1)
    source_media_sha256: str = Field(pattern=SHA256_PATTERN)
    extractor: str = Field(min_length=1)
    extractor_revision: str = Field(pattern=GIT_SHA_PATTERN)

    @model_validator(mode="after")
    def validate_identity(self) -> "ExpectedEvidenceSource":
        expected_context_kind = EXPECTED_CONTEXT_KIND_V2.get(self.condition)
        if expected_context_kind not in {
            "document",
            "ocr",
            "semantic",
            "relation",
            "image",
        }:
            raise ValueError("expected evidence source has a baseline or unknown condition")
        if self.context_kind != expected_context_kind:
            raise ValueError("expected evidence source context differs from condition")
        expected_media_kind = "source_document" if self.context_kind == "document" else "slide_image"
        if self.source_media_kind != expected_media_kind:
            raise ValueError("expected evidence source media kind differs from context")
        return self


class SourceEventTiming(StrictModel):
    condition_matrix_version: Literal["v1", "v2"] = "v1"
    event_id: str
    talk_id: str
    primary_eligible: bool
    evidence_available_sec: float = Field(ge=0, allow_inf_nan=False)
    audio_insufficient_until_sec: float = Field(ge=0, allow_inf_nan=False)
    audio_endpoint_sec: float = Field(gt=0, allow_inf_nan=False)
    expected_evidence_sources: list[ExpectedEvidenceSource] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_interval(self) -> "SourceEventTiming":
        if self.evidence_available_sec > self.audio_insufficient_until_sec:
            raise ValueError("evidence becomes available after audio-insufficient boundary")
        if self.audio_insufficient_until_sec > self.audio_endpoint_sec:
            raise ValueError("audio-insufficient boundary exceeds endpoint")
        by_condition = {source.condition: source for source in self.expected_evidence_sources}
        if len(by_condition) != len(self.expected_evidence_sources):
            raise ValueError("expected evidence source conditions must be unique")
        context_schema = (
            EXPECTED_CONTEXT_KIND_V1
            if self.condition_matrix_version == "v1"
            else EXPECTED_CONTEXT_KIND_V2
        )
        expected_conditions = {
            condition
            for condition, context_kind in context_schema.items()
            if context_kind not in {"none", "empty"}
        }
        if set(by_condition) != expected_conditions:
            raise ValueError("expected evidence sources do not cover the frozen condition set")
        return self


class TargetEventSpec(StrictModel):
    event_id: str
    acceptable_realizations: list[str] = Field(min_length=1)
    forbidden_realizations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_realizations(self) -> "TargetEventSpec":
        acceptable = {_tokens(value) for value in self.acceptable_realizations}
        forbidden = {_tokens(value) for value in self.forbidden_realizations}
        if () in acceptable:
            raise ValueError("acceptable realization cannot be empty")
        if () in forbidden:
            raise ValueError("forbidden realization cannot be empty")
        if acceptable & forbidden:
            raise ValueError("acceptable and forbidden realizations overlap")
        return self


class TrajectoryObservation(StrictModel):
    audio_time_sec: float = Field(ge=0, allow_inf_nan=False)
    causal_audio_prefix_id: str = Field(min_length=1)
    causal_audio_prefix_sha256: str = Field(pattern=SHA256_PATTERN)
    hypothesis: str


class CausalAudioSourceSpec(StrictModel):
    source_id: str = Field(min_length=1)
    talk_id: str = Field(min_length=1)
    acoustic_condition: str = Field(min_length=1)
    source_pcm_path: str = Field(min_length=1)
    source_pcm_sha256: str = Field(pattern=SHA256_PATTERN)
    pcm_format: Literal["float32le_mono"]
    sample_rate: int = Field(gt=0)
    total_sample_count: int = Field(gt=0)
    materialization_kind: Literal["native", "babble", "generic_noise", "music", "rir"]
    upstream_audio_sha256: str = Field(pattern=SHA256_PATTERN)
    materializer_git_commit: str = Field(pattern=GIT_SHA_PATTERN)
    materializer_entrypoint_sha256: str = Field(pattern=SHA256_PATTERN)
    source_provenance_path: str = Field(min_length=1)
    source_provenance_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_source(self) -> "CausalAudioSourceSpec":
        if not canonical_absolute_posix_path(self.source_pcm_path):
            raise ValueError("causal audio source path must be canonical and absolute")
        if not canonical_absolute_posix_path(self.source_provenance_path):
            raise ValueError("causal audio source provenance path must be canonical and absolute")
        return self


class CausalAudioPrefixSpec(StrictModel):
    source_id: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    acoustic_condition: str = Field(min_length=1)
    sequence_index: int = Field(ge=0)
    audio_time_sec: float = Field(gt=0, allow_inf_nan=False)
    prefix_id: str = Field(min_length=1)
    prefix_pcm_sha256: str = Field(pattern=SHA256_PATTERN)
    sample_rate: int = Field(gt=0)
    sample_count: int = Field(gt=0)


class CausalAudioSchedule(StrictModel):
    schema_version: Literal["acl6060_causal_audio_schedule_v3"]
    run_id: str = Field(min_length=1)
    expected_conditions: list[str] = Field(min_length=1)
    source_audio_roots: list[str] = Field(min_length=1)
    sources: list[CausalAudioSourceSpec] = Field(min_length=1)
    prefixes: list[CausalAudioPrefixSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_schedule(self) -> "CausalAudioSchedule":
        if len(self.expected_conditions) != len(set(self.expected_conditions)):
            raise ValueError("causal audio expected conditions must be unique")
        if len(self.source_audio_roots) != len(set(self.source_audio_roots)):
            raise ValueError("causal audio source roots must be unique")
        if any(not canonical_absolute_posix_path(root) for root in self.source_audio_roots):
            raise ValueError("causal audio source roots must be canonical absolute paths")
        source_by_id = {source.source_id: source for source in self.sources}
        if len(source_by_id) != len(self.sources):
            raise ValueError("causal audio source ids must be globally unique")
        source_streams = {
            (source.talk_id, source.acoustic_condition) for source in self.sources
        }
        if len(source_streams) != len(self.sources):
            raise ValueError("causal audio sources must uniquely identify event/acoustic streams")
        source_paths = {source.source_pcm_path for source in self.sources}
        if len(source_paths) != len(self.sources):
            raise ValueError("causal audio source paths must be globally unique")
        for source in self.sources:
            for path in (source.source_pcm_path, source.source_provenance_path):
                if not any(path_is_within(path, root) for root in self.source_audio_roots):
                    raise ValueError("causal audio source path is outside frozen source roots")
        by_stream: dict[tuple[str, str], list[CausalAudioPrefixSpec]] = defaultdict(list)
        prefix_ids: set[str] = set()
        referenced_source_ids: set[str] = set()
        for prefix in self.prefixes:
            if prefix.prefix_id in prefix_ids:
                raise ValueError("causal audio prefix ids must be globally unique")
            prefix_ids.add(prefix.prefix_id)
            source = source_by_id.get(prefix.source_id)
            if source is None:
                raise ValueError("causal audio prefix references an unknown source")
            if prefix.acoustic_condition != source.acoustic_condition:
                raise ValueError("causal audio prefix acoustic condition differs from source")
            if prefix.sample_rate != source.sample_rate:
                raise ValueError("causal audio prefix sample rate differs from source")
            if prefix.sample_count > source.total_sample_count:
                raise ValueError("causal audio prefix exceeds the frozen source")
            expected_time = prefix.sample_count / prefix.sample_rate
            if not math.isclose(
                prefix.audio_time_sec,
                expected_time,
                rel_tol=0.0,
                abs_tol=0.5 / prefix.sample_rate,
            ):
                raise ValueError("causal audio prefix time differs from its sample boundary")
            referenced_source_ids.add(prefix.source_id)
            by_stream[(prefix.event_id, prefix.acoustic_condition)].append(prefix)
        if referenced_source_ids != set(source_by_id):
            raise ValueError("causal audio schedule contains an unreferenced source")
        for stream, prefixes in by_stream.items():
            ordered = sorted(prefixes, key=lambda value: value.sequence_index)
            if [value.sequence_index for value in ordered] != list(range(len(ordered))):
                raise ValueError(f"causal audio prefix sequence is not contiguous: {stream}")
            times = [value.audio_time_sec for value in ordered]
            samples = [value.sample_count for value in ordered]
            if any(right <= left for left, right in zip(times, times[1:])):
                raise ValueError(f"causal audio prefix times are not increasing: {stream}")
            if any(right <= left for left, right in zip(samples, samples[1:])):
                raise ValueError(f"causal audio prefix sample counts are not increasing: {stream}")
            if len({value.sample_rate for value in ordered}) != 1:
                raise ValueError(f"causal audio prefix sample rate changes within stream: {stream}")
        return self


class CausalAudioReleaseRecord(StrictModel):
    record_type: Literal["prefix_release"]
    source_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    server_ordinal: int = Field(ge=0)
    event_id: str = Field(min_length=1)
    condition: str = Field(min_length=1)
    acoustic_condition: str = Field(min_length=1)
    sequence_index: int = Field(ge=0)
    audio_time_sec: float = Field(gt=0, allow_inf_nan=False)
    prefix_id: str = Field(min_length=1)
    prefix_pcm_sha256: str = Field(pattern=SHA256_PATTERN)
    sample_count: int = Field(gt=0)
    request_id: str = Field(min_length=1)
    granted_monotonic_ns: int = Field(gt=0)
    granted_at_utc: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
    previous_record_sha256: str = Field(pattern=SHA256_PATTERN)
    record_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_record_hash(self) -> "CausalAudioReleaseRecord":
        expected = canonical_json_sha256(self.model_dump(exclude={"record_sha256"}))
        if self.record_sha256 != expected:
            raise ValueError("causal audio release record hash mismatch")
        return self


class CausalAudioObservationCommitRecord(StrictModel):
    record_type: Literal["observation_commit"]
    source_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    server_ordinal: int = Field(ge=0)
    event_id: str = Field(min_length=1)
    condition: str = Field(min_length=1)
    acoustic_condition: str = Field(min_length=1)
    sequence_index: int = Field(ge=0)
    prefix_id: str = Field(min_length=1)
    prefix_pcm_sha256: str = Field(pattern=SHA256_PATTERN)
    observation_sha256: str = Field(pattern=SHA256_PATTERN)
    request_id: str = Field(min_length=1)
    committed_monotonic_ns: int = Field(gt=0)
    committed_at_utc: str = Field(
        pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$"
    )
    previous_record_sha256: str = Field(pattern=SHA256_PATTERN)
    record_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_record_hash(self) -> "CausalAudioObservationCommitRecord":
        expected = canonical_json_sha256(self.model_dump(exclude={"record_sha256"}))
        if self.record_sha256 != expected:
            raise ValueError("causal audio observation-commit hash mismatch")
        return self


CausalAudioInteractionRecord = Annotated[
    CausalAudioReleaseRecord | CausalAudioObservationCommitRecord,
    Field(discriminator="record_type"),
]


class CausalAudioReleaseLog(StrictModel):
    schema_version: Literal["acl6060_causal_audio_release_log_v3"]
    run_id: str = Field(min_length=1)
    schedule_sha256: str = Field(pattern=SHA256_PATTERN)
    broker_audit_sha256: str = Field(pattern=SHA256_PATTERN)
    interactions: list[CausalAudioInteractionRecord] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_interactions(self) -> "CausalAudioReleaseLog":
        keys = [
            (
                interaction.record_type,
                interaction.event_id,
                interaction.condition,
                interaction.acoustic_condition,
                interaction.sequence_index,
            )
            for interaction in self.interactions
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("causal audio interaction keys must be unique")
        request_ids = [interaction.request_id for interaction in self.interactions]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("causal audio interaction request ids must be unique")
        expected_previous = EMPTY_SHA256
        previous_monotonic_ns = -1
        stream_state: dict[tuple[str, str, str, str], tuple[int, int]] = {}
        stream_by_session: dict[str, tuple[str, str, str]] = {}
        for ordinal, interaction in enumerate(self.interactions):
            if interaction.server_ordinal != ordinal:
                raise ValueError("causal audio interaction ordinals are not contiguous")
            if interaction.previous_record_sha256 != expected_previous:
                raise ValueError("causal audio interaction hash chain is broken")
            monotonic_ns = (
                interaction.granted_monotonic_ns
                if isinstance(interaction, CausalAudioReleaseRecord)
                else interaction.committed_monotonic_ns
            )
            if monotonic_ns <= previous_monotonic_ns:
                raise ValueError("causal audio interaction timestamps are not increasing")
            previous_monotonic_ns = monotonic_ns
            expected_previous = interaction.record_sha256
            stream_key = (
                interaction.session_id,
                interaction.event_id,
                interaction.condition,
                interaction.acoustic_condition,
            )
            stream_identity = (
                interaction.event_id,
                interaction.condition,
                interaction.acoustic_condition,
            )
            previous_stream = stream_by_session.setdefault(
                interaction.session_id,
                stream_identity,
            )
            if previous_stream != stream_identity:
                raise ValueError("causal audio session was reused across inference streams")
            last_released, last_committed = stream_state.get(stream_key, (-1, -1))
            if isinstance(interaction, CausalAudioReleaseRecord):
                if (
                    interaction.sequence_index != last_released + 1
                    or last_committed != last_released
                ):
                    raise ValueError("causal audio prefix was released before prior commit")
                last_released = interaction.sequence_index
            else:
                if (
                    interaction.sequence_index != last_committed + 1
                    or interaction.sequence_index != last_released
                ):
                    raise ValueError("causal observation commit is out of sequence")
                last_committed = interaction.sequence_index
            stream_state[stream_key] = (last_released, last_committed)
        if any(released != committed for released, committed in stream_state.values()):
            raise ValueError("causal audio release log has an uncommitted final observation")
        return self

    @property
    def releases(self) -> list[CausalAudioReleaseRecord]:
        return [
            interaction
            for interaction in self.interactions
            if isinstance(interaction, CausalAudioReleaseRecord)
        ]

    @property
    def observation_commits(self) -> list[CausalAudioObservationCommitRecord]:
        return [
            interaction
            for interaction in self.interactions
            if isinstance(interaction, CausalAudioObservationCommitRecord)
        ]


class CausalAudioBrokerAudit(StrictModel):
    schema_version: Literal["acl6060_causal_audio_broker_audit_v2"]
    run_id: str = Field(min_length=1)
    schedule_sha256: str = Field(pattern=SHA256_PATTERN)
    broker_git_commit: str = Field(pattern=GIT_SHA_PATTERN)
    broker_repo_path: str = Field(min_length=1)
    broker_entrypoint_path: str = Field(min_length=1)
    broker_entrypoint_sha256: str = Field(pattern=SHA256_PATTERN)
    broker_command: str = Field(min_length=1)
    broker_pid: int = Field(gt=0)
    socket_path: str = Field(min_length=1)
    release_events_path: str = Field(min_length=1)
    source_audio_roots: list[str] = Field(min_length=1)
    delivery_protocol: Literal["length_prefixed_unix_socket_v1"]
    captured_at_utc: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")

    @model_validator(mode="after")
    def validate_broker(self) -> "CausalAudioBrokerAudit":
        if not canonical_absolute_posix_path(self.broker_repo_path):
            raise ValueError("broker repo path must be canonical and absolute")
        if not path_is_within(self.broker_entrypoint_path, self.broker_repo_path):
            raise ValueError("broker entrypoint is outside the broker repo")
        if not canonical_absolute_posix_path(self.socket_path):
            raise ValueError("broker socket path must be canonical and absolute")
        if not canonical_absolute_posix_path(self.release_events_path):
            raise ValueError("broker release-events path must be canonical and absolute")
        if len(self.source_audio_roots) != len(set(self.source_audio_roots)):
            raise ValueError("broker source audio roots must be unique")
        if any(not canonical_absolute_posix_path(root) for root in self.source_audio_roots):
            raise ValueError("broker source audio roots must be canonical absolute paths")
        return self


class EventTrajectory(StrictModel):
    event_id: str = Field(min_length=1)
    talk_id: str = Field(min_length=1)
    condition: str = Field(min_length=1)
    acoustic_condition: str = Field(min_length=1)
    inference_run_id: str = Field(min_length=1)
    inference_contract_sha256: str = Field(pattern=SHA256_PATTERN)
    evidence_packet_id: str = Field(min_length=1)
    evidence_packet_sha256: str = Field(pattern=SHA256_PATTERN)
    observations: list[TrajectoryObservation] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_observations(self) -> "EventTrajectory":
        times = [observation.audio_time_sec for observation in self.observations]
        if any(right <= left for left, right in zip(times, times[1:])):
            raise ValueError("trajectory observation times must be strictly increasing")
        return self


def causal_observation_sha256(
    *,
    run_id: str,
    inference_contract_sha256: str,
    event_id: str,
    condition: str,
    acoustic_condition: str,
    sequence_index: int,
    observation: TrajectoryObservation,
) -> str:
    return canonical_json_sha256(
        {
            "schema_version": "acl6060_causal_observation_commitment_v1",
            "run_id": run_id,
            "inference_contract_sha256": inference_contract_sha256,
            "event_id": event_id,
            "condition": condition,
            "acoustic_condition": acoustic_condition,
            "sequence_index": sequence_index,
            "audio_time_sec": observation.audio_time_sec,
            "causal_audio_prefix_id": observation.causal_audio_prefix_id,
            "causal_audio_prefix_sha256": observation.causal_audio_prefix_sha256,
            "hypothesis": observation.hypothesis,
        }
    )


class ControlPairSpec(StrictModel):
    event_id: str
    contrast_id: str
    control_pair_id: str
    evidence_type: Literal["ocr", "semantic", "relation", "image"]
    first_condition: str
    second_condition: str
    first_packet_id: str
    first_packet_sha256: str = Field(pattern=SHA256_PATTERN)
    second_packet_id: str
    second_packet_sha256: str = Field(pattern=SHA256_PATTERN)
    first_available_sec: float = Field(ge=0, allow_inf_nan=False)
    second_available_sec: float = Field(ge=0, allow_inf_nan=False)
    first_token_count: int = Field(ge=0)
    second_token_count: int = Field(ge=0)
    first_visual_token_count: int = Field(default=0, ge=0)
    second_visual_token_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_match(self) -> "ControlPairSpec":
        if self.first_condition == self.second_condition:
            raise ValueError("control pair conditions must differ")
        if self.first_packet_id == self.second_packet_id:
            raise ValueError("correct and control packet ids must differ")
        if self.first_packet_sha256 == self.second_packet_sha256:
            raise ValueError("correct and control packet hashes must differ")
        if not math.isclose(
            self.first_available_sec,
            self.second_available_sec,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError("control pair availability times differ")
        if self.first_token_count != self.second_token_count:
            raise ValueError("control pair token counts differ")
        if self.first_visual_token_count != self.second_visual_token_count:
            raise ValueError("control pair visual token counts differ")
        if self.evidence_type == "image" and self.first_visual_token_count == 0:
            raise ValueError("image control pair requires visual tokens")
        if self.evidence_type != "image" and self.first_visual_token_count != 0:
            raise ValueError("text control pair cannot declare visual tokens")
        return self


class SourceEvidenceArtifact(StrictModel):
    schema_version: Literal[
        "acl6060_source_evidence_artifact_v1",
        "acl6060_source_evidence_artifact_v2",
    ]
    event_id: str = Field(min_length=1)
    context_kind: Literal["document", "ocr", "semantic", "relation", "image"]
    source_media_kind: Literal["source_document", "slide_image"]
    source_media_path: str = Field(min_length=1)
    source_media_sha256: str = Field(pattern=SHA256_PATTERN)
    extractor: str = Field(min_length=1)
    extractor_revision: str = Field(pattern=GIT_SHA_PATTERN)
    items: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_items(self) -> "SourceEvidenceArtifact":
        if any(not item.strip() for item in self.items):
            raise ValueError("source evidence artifact items must be non-empty")
        if self.schema_version.endswith("_v1") and self.context_kind == "image":
            raise ValueError("v1 source evidence artifact cannot contain raw image context")
        if self.context_kind == "image" and self.items:
            raise ValueError("raw image source evidence artifact cannot contain text items")
        if self.context_kind != "image" and not self.items:
            raise ValueError("text source evidence artifact requires items")
        expected_media_kind = "source_document" if self.context_kind == "document" else "slide_image"
        if self.source_media_kind != expected_media_kind:
            raise ValueError("source evidence artifact media kind differs from context kind")
        suffix = Path(self.source_media_path).suffix.casefold()
        if self.source_media_kind == "slide_image" and suffix not in {".jpg", ".jpeg", ".png"}:
            raise ValueError("slide evidence source media must be an image")
        return self


class SourceEvidenceReference(StrictModel):
    text: str = Field(min_length=1)
    artifact_path: str = Field(min_length=1)
    artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    item_index: int = Field(ge=0)


class SourceImageReference(StrictModel):
    artifact_path: str = Field(min_length=1)
    artifact_sha256: str = Field(pattern=SHA256_PATTERN)


class EvidencePacketPayload(StrictModel):
    schema_version: Literal[
        "acl6060_source_evidence_packet_v1",
        "acl6060_source_evidence_packet_v2",
    ]
    context_kind: Literal[
        "none",
        "empty",
        "document",
        "ocr",
        "semantic",
        "relation",
        "image",
    ]
    context_items: list[SourceEvidenceReference]
    image_reference: SourceImageReference | None = None

    @model_validator(mode="after")
    def validate_content(self) -> "EvidencePacketPayload":
        if self.schema_version.endswith("_v1") and (
            self.context_kind == "image" or self.image_reference is not None
        ):
            raise ValueError("v1 evidence packet cannot contain raw image context")
        if self.context_kind in {"none", "empty"} and self.context_items:
            raise ValueError("none/empty evidence packets cannot contain context items")
        if self.context_kind == "image":
            if self.context_items or self.image_reference is None:
                raise ValueError("image evidence packet requires only an image reference")
        elif self.image_reference is not None:
            raise ValueError("non-image evidence packet cannot contain an image reference")
        elif self.context_kind not in {"none", "empty"} and not self.context_items:
            raise ValueError("non-empty evidence packet requires context items")
        return self


class EvidencePacketSpec(StrictModel):
    event_id: str = Field(min_length=1)
    condition: str = Field(min_length=1)
    packet_id: str = Field(min_length=1)
    packet_sha256: str = Field(pattern=SHA256_PATTERN)
    evidence_type: Literal[
        "none",
        "document",
        "empty",
        "ocr",
        "semantic",
        "relation",
        "image",
    ]
    evidence_role: Literal["baseline", "correct", "matched_wrong"]
    available_sec: float = Field(ge=0, allow_inf_nan=False)
    tokenizer_model: str = Field(min_length=1)
    tokenizer_revision: str = Field(pattern=GIT_SHA_PATTERN)
    tokenizer_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    token_ids: list[Annotated[int, Field(ge=0)]]
    token_ids_sha256: str = Field(pattern=SHA256_PATTERN)
    rendered_text_sha256: str = Field(pattern=SHA256_PATTERN)
    visual_token_count: int = Field(default=0, ge=0)
    packet_payload: EvidencePacketPayload

    @model_validator(mode="after")
    def validate_bytes(self) -> "EvidencePacketSpec":
        payload_value = self.packet_payload.model_dump(mode="json", exclude_none=True)
        if self.packet_sha256 != canonical_json_sha256(payload_value):
            raise ValueError("evidence packet payload hash mismatch")
        if self.token_ids_sha256 != canonical_json_sha256(self.token_ids):
            raise ValueError("evidence packet token-id hash mismatch")
        if self.rendered_text_sha256 != text_sha256(render_evidence_packet(self.packet_payload)):
            raise ValueError("evidence packet rendered-text hash mismatch")
        if self.evidence_type == "image" and self.visual_token_count == 0:
            raise ValueError("image evidence packet requires visual tokens")
        if self.evidence_type != "image" and self.visual_token_count != 0:
            raise ValueError("text evidence packet cannot declare visual tokens")
        return self


class OutcomeArtifactSpec(StrictModel):
    role: Literal[
        "source_annotation_report",
        "source_adjudication",
        "target_annotation_report",
        "target_adjudication",
    ]
    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_path(self) -> "OutcomeArtifactSpec":
        path = PurePosixPath(self.relative_path)
        if path.is_absolute() or ".." in path.parts or str(path) != self.relative_path:
            raise ValueError("outcome artifact path must be canonical and relative")
        return self


class OutcomeCommitment(StrictModel):
    schema_version: Literal["acl6060_outcome_commitment_v1"]
    created_at_utc: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
    source_events_sha256: str = Field(pattern=SHA256_PATTERN)
    target_scores_sha256: str = Field(pattern=SHA256_PATTERN)
    artifacts: list[OutcomeArtifactSpec] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def validate_artifacts(self) -> "OutcomeCommitment":
        expected_roles = {
            "source_annotation_report",
            "source_adjudication",
            "target_annotation_report",
            "target_adjudication",
        }
        roles = {artifact.role for artifact in self.artifacts}
        paths = {artifact.relative_path for artifact in self.artifacts}
        if roles != expected_roles or len(paths) != len(self.artifacts):
            raise ValueError("outcome commitment must contain four unique frozen artifact roles")
        return self


class InferenceContract(StrictModel):
    schema_version: Literal["acl6060_event_inference_contract_v1"]
    run_id: str = Field(min_length=1)
    created_at_utc: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
    git_commit: str = Field(pattern=GIT_SHA_PATTERN)
    scientific_config_sha256: str = Field(pattern=SHA256_PATTERN)
    scoring_config_sha256: str = Field(pattern=SHA256_PATTERN)
    model_id: str = Field(min_length=1)
    model_revision: str = Field(pattern=GIT_SHA_PATTERN)
    model_artifact_tree_sha256: str = Field(pattern=SHA256_PATTERN)
    tokenizer_model: str = Field(min_length=1)
    tokenizer_revision: str = Field(pattern=GIT_SHA_PATTERN)
    tokenizer_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    source_artifact_tree_sha256: str = Field(pattern=SHA256_PATTERN)
    source_events_sha256: str = Field(pattern=SHA256_PATTERN)
    evidence_packets_sha256: str = Field(pattern=SHA256_PATTERN)
    control_pairs_sha256: str = Field(pattern=SHA256_PATTERN)
    target_scores_sha256: str = Field(pattern=SHA256_PATTERN)
    outcome_commitment_sha256: str = Field(pattern=SHA256_PATTERN)
    outcome_artifact_tree_sha256: str = Field(pattern=SHA256_PATTERN)
    causal_audio_schedule_sha256: str = Field(pattern=SHA256_PATTERN)
    causal_audio_broker_audit_sha256: str = Field(pattern=SHA256_PATTERN)
    causal_audio_protocol: Literal["external_talk_synchronized_prefix_broker_v2"]
    causal_audio_broker_entrypoint_sha256: str = Field(pattern=SHA256_PATTERN)
    expected_conditions: list[str] = Field(min_length=1)
    expected_acoustic_conditions: list[str] = Field(min_length=1)
    target_artifact_mounted: Literal[False]
    reference_artifact_mounted: Literal[False]
    future_audio_access: Literal[False]
    forbidden_container_artifact_roots: list[str] = Field(min_length=1)
    forbidden_host_mount_source_roots: list[str] = Field(min_length=1)
    scoring_protected_artifact_roots: list[str] = Field(min_length=1)
    worker_command_match: str = Field(min_length=1)
    worker_inference_contract_path: str = Field(min_length=1)
    worker_contract_ready_file_path: str = Field(min_length=1)
    scientific_config_host_path: str = Field(min_length=1)
    worker_scientific_config_path: str = Field(min_length=1)
    model_artifact_host_root_path: str = Field(min_length=1)
    worker_model_artifact_root_path: str = Field(min_length=1)
    tokenizer_artifact_host_root_path: str = Field(min_length=1)
    worker_tokenizer_artifact_root_path: str = Field(min_length=1)
    source_artifact_host_root_path: str | None = None
    worker_source_artifact_root_path: str | None = None
    expected_worker_count: int = Field(gt=0)
    inference_repo_path: str = Field(min_length=1)
    container_image_id: str = Field(pattern=SHA256_PATTERN)
    environment_start_audit_sha256: str = Field(pattern=SHA256_PATTERN)
    worker_process_identity_tree_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_isolation_identity(self) -> "InferenceContract":
        try:
            marker_tokens = shlex.split(self.worker_command_match)
        except ValueError as exc:
            raise ValueError("worker command match is not valid shell-token syntax") from exc
        if marker_tokens.count(self.run_id) != 1:
            raise ValueError("worker command match must contain the exact run id token once")
        source_paths = (
            self.source_artifact_host_root_path,
            self.worker_source_artifact_root_path,
        )
        if (source_paths[0] is None) != (source_paths[1] is None):
            raise ValueError("worker source artifact host/container paths must be paired")
        if "correct_image" in self.expected_conditions and source_paths[0] is None:
            raise ValueError("raw image inference contract requires a worker source artifact mount")
        for path in (
            self.worker_inference_contract_path,
            self.worker_contract_ready_file_path,
            self.scientific_config_host_path,
            self.worker_scientific_config_path,
            self.model_artifact_host_root_path,
            self.worker_model_artifact_root_path,
            self.tokenizer_artifact_host_root_path,
            self.worker_tokenizer_artifact_root_path,
            *(value for value in source_paths if value is not None),
        ):
            if not canonical_absolute_posix_path(path):
                raise ValueError("worker contract barrier paths must be canonical and absolute")
        if self.worker_inference_contract_path == self.worker_contract_ready_file_path:
            raise ValueError("worker contract and ready-file paths must differ")
        for label, roots in (
            ("manifest forbidden container artifact", self.forbidden_container_artifact_roots),
            ("manifest forbidden host mount source", self.forbidden_host_mount_source_roots),
            ("scoring protected artifact", self.scoring_protected_artifact_roots),
        ):
            if len(roots) != len(set(roots)):
                raise ValueError(f"{label} roots must be unique")
            if any(not canonical_absolute_posix_path(root) for root in roots):
                raise ValueError(f"{label} roots must be canonical absolute paths")
        if not canonical_absolute_posix_path(self.inference_repo_path):
            raise ValueError("inference repo path must be a canonical absolute path")
        for scoring_root in self.scoring_protected_artifact_roots:
            if not any(
                path_is_within(scoring_root, forbidden_root)
                for forbidden_root in self.forbidden_host_mount_source_roots
            ):
                raise ValueError(
                    "scoring protected root is not forbidden as a host mount source"
                )
        return self


class InferenceResultAttestation(StrictModel):
    schema_version: Literal["acl6060_event_inference_result_attestation_v1"]
    run_id: str = Field(min_length=1)
    created_at_utc: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
    inference_contract_sha256: str = Field(pattern=SHA256_PATTERN)
    trajectories_sha256: str = Field(pattern=SHA256_PATTERN)
    causal_audio_release_log_sha256: str = Field(pattern=SHA256_PATTERN)
    environment_start_audit_sha256: str = Field(pattern=SHA256_PATTERN)
    environment_end_audit_sha256: str = Field(pattern=SHA256_PATTERN)


class MountRecord(StrictModel):
    source: str
    destination: str
    read_only: bool


class WorkerProcessRecord(StrictModel):
    pid: int = Field(gt=0)
    parent_pid: int = Field(ge=0)
    process_start_time_ticks: int = Field(gt=0)
    command: str = Field(min_length=1)
    marker_process: bool
    executable_path: str = Field(min_length=1)
    executable_sha256: str = Field(pattern=SHA256_PATTERN)
    working_directory: str = Field(min_length=1)
    entrypoint_path: str | None
    entrypoint_sha256: str | None
    environment_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_entrypoint(self) -> "WorkerProcessRecord":
        if (self.entrypoint_path is None) != (self.entrypoint_sha256 is None):
            raise ValueError("worker entrypoint path/hash must be present together")
        try:
            has_python_script = any(token.endswith(".py") for token in shlex.split(self.command))
        except ValueError as exc:
            raise ValueError("worker command is not valid shell-token syntax") from exc
        if (self.marker_process or has_python_script) and self.entrypoint_path is None:
            raise ValueError("Python worker must bind its entrypoint")
        return self


def worker_process_identity_tree_sha256(processes: Iterable[WorkerProcessRecord | dict]) -> str:
    rows = []
    for process in processes:
        row = process.model_dump() if isinstance(process, WorkerProcessRecord) else dict(process)
        rows.append(
            {
                key: row.get(key)
                for key in (
                    "pid",
                    "parent_pid",
                    "process_start_time_ticks",
                    "command",
                    "marker_process",
                    "executable_path",
                    "executable_sha256",
                    "working_directory",
                    "entrypoint_path",
                    "entrypoint_sha256",
                    "environment_sha256",
                )
            }
        )
    payload = json.dumps(
        sorted(rows, key=lambda row: json.dumps(row, sort_keys=True)),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class InferenceEnvironmentAudit(StrictModel):
    schema_version: Literal["acl6060_event_inference_environment_audit_v5"]
    run_id: str = Field(min_length=1)
    container_name: str = Field(min_length=1)
    container_id: str = Field(pattern=SHA256_PATTERN)
    container_image_id: str = Field(pattern=SHA256_PATTERN)
    container_read_only_rootfs: Literal[True]
    container_network_mode: Literal["none"]
    capture_host: str = Field(min_length=1)
    captured_at_utc: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
    capture_command: Literal["docker_inspect_proc_tree_worker_discovery_git_v5"]
    capture_phase: Literal["workers_start", "workers_end"]
    worker_command_match: str = Field(min_length=1)
    worker_processes: list[WorkerProcessRecord] = Field(min_length=1)
    docker_inspect_sha256: str = Field(pattern=SHA256_PATTERN)
    process_listing_sha256: str = Field(pattern=SHA256_PATTERN)
    proc_open_files_sha256: str = Field(pattern=SHA256_PATTERN)
    process_identity_tree_sha256: str = Field(pattern=SHA256_PATTERN)
    inference_repo_path: str = Field(min_length=1)
    inference_git_commit: str = Field(pattern=GIT_SHA_PATTERN)
    inference_git_status_sha256: str = Field(pattern=SHA256_PATTERN)
    forbidden_container_artifact_roots: list[str] = Field(min_length=1)
    forbidden_host_mount_source_roots: list[str] = Field(min_length=1)
    observed_mounts: list[MountRecord]
    process_open_file_paths: list[str]
    forbidden_artifact_exposure_detected: Literal[False]

    @model_validator(mode="after")
    def validate_paths(self) -> "InferenceEnvironmentAudit":
        pids = [process.pid for process in self.worker_processes]
        if len(pids) != len(set(pids)):
            raise ValueError("inference environment audit worker pids must be unique")
        marker_processes = [process for process in self.worker_processes if process.marker_process]
        if not marker_processes:
            raise ValueError("inference environment audit has no exact marker worker")
        if self.process_identity_tree_sha256 != worker_process_identity_tree_sha256(
            self.worker_processes
        ):
            raise ValueError("inference process identity tree hash mismatch")
        if any(
            not command_contains_exact_marker(process.command, self.worker_command_match)
            for process in marker_processes
        ):
            raise ValueError("inference environment audit worker command mismatch")
        if self.inference_git_status_sha256 != EMPTY_SHA256:
            raise ValueError("inference environment audit captured a dirty checkout")
        if not canonical_absolute_posix_path(self.inference_repo_path):
            raise ValueError("audit inference repo path must be canonical and absolute")
        for process in self.worker_processes:
            if not canonical_absolute_posix_path(process.working_directory):
                raise ValueError("worker cwd must be canonical and absolute")
            if process.marker_process and not path_is_within(
                process.working_directory, self.inference_repo_path
            ):
                raise ValueError("marker worker cwd is outside the inference repo")
            if process.entrypoint_path is not None and not path_is_within(
                process.entrypoint_path, self.inference_repo_path
            ):
                raise ValueError("worker entrypoint is outside the inference repo")
        container_roots = list(self.forbidden_container_artifact_roots)
        host_roots = list(self.forbidden_host_mount_source_roots)
        for label, roots in (
            ("forbidden container artifact", container_roots),
            ("forbidden host mount source", host_roots),
        ):
            if len(roots) != len(set(roots)):
                raise ValueError(f"{label} roots must be unique")
            if any(not canonical_absolute_posix_path(root) for root in roots):
                raise ValueError(f"{label} roots must be canonical and absolute")
        for path in self.process_open_file_paths:
            normalized = posixpath.normpath(path)
            for root in container_roots:
                if path_is_within(normalized, root):
                    raise ValueError(f"inference environment exposes forbidden artifact root: {root}")
        for mount in self.observed_mounts:
            normalized_source = posixpath.normpath(mount.source)
            for root in host_roots:
                if (
                    path_is_within(normalized_source, root)
                    or path_is_within(root, normalized_source)
                ):
                    raise ValueError(
                        f"inference environment host mount exposes forbidden artifact root: {root}"
                    )
            normalized_destination = posixpath.normpath(mount.destination)
            for root in container_roots:
                if (
                    path_is_within(normalized_destination, root)
                    or path_is_within(root, normalized_destination)
                ):
                    raise ValueError(
                        f"inference environment container mount exposes forbidden artifact root: {root}"
                    )
        return self


def canonical_absolute_posix_path(path: str) -> bool:
    return path.startswith("/") and posixpath.normpath(path) == path


def path_is_within(path: str, root: str) -> bool:
    if not canonical_absolute_posix_path(path) or not canonical_absolute_posix_path(root):
        return False
    return path == root or path.startswith(root.rstrip("/") + "/")


def require_read_only_mount(
    audit: InferenceEnvironmentAudit,
    *,
    source: str,
    destination: str,
    label: str,
) -> None:
    matches = [
        mount
        for mount in audit.observed_mounts
        if posixpath.normpath(mount.source) == source
        and posixpath.normpath(mount.destination) == destination
    ]
    if len(matches) != 1 or not matches[0].read_only:
        raise ValueError(f"{label} is not an exact read-only worker mount")


def command_contains_exact_marker(command: str, marker: str) -> bool:
    try:
        command_tokens = shlex.split(command)
        marker_tokens = shlex.split(marker)
    except ValueError:
        return False
    if not marker_tokens or len(marker_tokens) > len(command_tokens):
        return False
    width = len(marker_tokens)
    return any(
        command_tokens[index:index + width] == marker_tokens
        for index in range(len(command_tokens) - width + 1)
    )


class AcousticGroupSpec(StrictModel):
    id: str
    members: list[str] = Field(min_length=1)


class ContrastSpec(StrictModel):
    id: str
    first: str
    second: str
    requires_matched_control: bool
    evidence_type: Literal["ocr", "semantic", "relation", "image"] | None

    @model_validator(mode="after")
    def validate_evidence_type(self) -> "ContrastSpec":
        if self.requires_matched_control != (self.evidence_type is not None):
            raise ValueError("matched-control contrast must declare one evidence_type")
        return self


class DevelopmentSignalSpec(StrictModel):
    early_risk_difference_percentage_points: float
    final_correctness_point_estimate_floor_percentage_points: float
    directionally_consistent_talks: int
    minimum_talks: int
    forbidden_adoption_point_estimate_ceiling_percentage_points: float
    overcommit_point_estimate_ceiling_percentage_points: float

    @model_validator(mode="after")
    def validate_frozen_values(self) -> "DevelopmentSignalSpec":
        expected = (5.0, -1.0, 3, 5, 1.0, 1.0)
        observed = (
            self.early_risk_difference_percentage_points,
            self.final_correctness_point_estimate_floor_percentage_points,
            self.directionally_consistent_talks,
            self.minimum_talks,
            self.forbidden_adoption_point_estimate_ceiling_percentage_points,
            self.overcommit_point_estimate_ceiling_percentage_points,
        )
        if observed != expected:
            raise ValueError(f"development signal differs from frozen v1 values: {observed}")
        return self


class EventScoringConfig(StrictModel):
    schema_version: Literal[
        "acl6060_event_trajectory_scoring_v1",
        "acl6060_event_trajectory_scoring_v2",
    ]
    scope: Literal["exploratory_acl_dev_not_confirmatory"]
    primary_development_estimand: Literal[
        "talk_equal_risk_difference_of_first_stable_correct_target_decision_by_conservative_audio_insufficient_boundary"
    ]
    min_stability_observations: int = Field(ge=2)
    native_acoustic_group: Literal["native"]
    expected_acoustic_conditions: list[str] = Field(min_length=1)
    acoustic_groups: list[AcousticGroupSpec] = Field(min_length=1)
    expected_conditions: list[str] = Field(min_length=1)
    contrasts: list[ContrastSpec] = Field(min_length=1)
    development_signal: DevelopmentSignalSpec
    bootstrap_samples: int = Field(ge=1000)
    bootstrap_seed: int = Field(ge=0)
    babble_severity_order: list[str] = Field(min_length=5, max_length=5)
    interpretation: str

    @model_validator(mode="after")
    def validate_partition(self) -> "EventScoringConfig":
        if len(self.expected_conditions) != len(set(self.expected_conditions)):
            raise ValueError("expected_conditions must be unique")
        if len(self.expected_acoustic_conditions) != len(set(self.expected_acoustic_conditions)):
            raise ValueError("expected_acoustic_conditions must be unique")
        group_ids = [group.id for group in self.acoustic_groups]
        members = [member for group in self.acoustic_groups for member in group.members]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("acoustic group ids must be unique")
        if len(members) != len(set(members)):
            raise ValueError("each acoustic condition must belong to exactly one group")
        if set(members) != set(self.expected_acoustic_conditions):
            raise ValueError("acoustic groups must partition expected_acoustic_conditions")
        native = next((group for group in self.acoustic_groups if group.id == "native"), None)
        if native is None or native.members != ["native"]:
            raise ValueError("native acoustic group must contain only native")
        contrast_ids = [contrast.id for contrast in self.contrasts]
        if len(contrast_ids) != len(set(contrast_ids)):
            raise ValueError("contrast ids must be unique")
        condition_set = set(self.expected_conditions)
        for contrast in self.contrasts:
            if contrast.first == contrast.second:
                raise ValueError(f"contrast compares a condition with itself: {contrast.id}")
            if {contrast.first, contrast.second} - condition_set:
                raise ValueError(f"contrast has an unknown condition: {contrast.id}")
        if set(self.babble_severity_order) - set(group_ids):
            raise ValueError("babble severity order has unknown groups")
        observed_groups = {group.id: tuple(group.members) for group in self.acoustic_groups}
        observed_contrasts = tuple(
            (
                contrast.id,
                contrast.first,
                contrast.second,
                contrast.requires_matched_control,
                contrast.evidence_type,
            )
            for contrast in self.contrasts
        )
        expected_acoustic = tuple(
            member
            for members_for_group in EXPECTED_ACOUSTIC_GROUPS_V1.values()
            for member in members_for_group
        )
        schema_label = "v1" if self.schema_version.endswith("_v1") else "v2"
        if self.min_stability_observations != 2:
            raise ValueError(f"{schema_label} min_stability_observations must equal 2")
        if tuple(self.expected_acoustic_conditions) != expected_acoustic:
            raise ValueError(f"{schema_label} acoustic condition matrix differs from frozen order")
        if observed_groups != EXPECTED_ACOUSTIC_GROUPS_V1:
            raise ValueError(f"{schema_label} acoustic groups differ from frozen mapping")
        if tuple(group_ids) != tuple(EXPECTED_ACOUSTIC_GROUPS_V1):
            raise ValueError(f"{schema_label} acoustic group order differs from frozen order")
        expected_conditions = (
            EXPECTED_CONDITIONS_V1
            if self.schema_version.endswith("_v1")
            else EXPECTED_CONDITIONS_V2
        )
        expected_contrasts = (
            EXPECTED_CONTRASTS_V1
            if self.schema_version.endswith("_v1")
            else EXPECTED_CONTRASTS_V2
        )
        if tuple(self.expected_conditions) != expected_conditions:
            raise ValueError(f"{schema_label} model condition matrix differs from frozen order")
        if observed_contrasts != expected_contrasts:
            raise ValueError(f"{schema_label} contrasts differ from frozen definitions")
        if self.bootstrap_samples != 10000 or self.bootstrap_seed != 20260801:
            raise ValueError(f"{schema_label} bootstrap count/seed differ from frozen values")
        if tuple(self.babble_severity_order) != EXPECTED_BABBLE_ORDER_V1:
            raise ValueError(f"{schema_label} babble severity order differs from frozen order")
        return self


@dataclass(frozen=True)
class EventTimingScore:
    event_id: str
    talk_id: str
    condition: str
    acoustic_condition: str
    inference_run_id: str
    inference_contract_sha256: str
    evidence_packet_id: str
    evidence_packet_sha256: str
    observation_count: int
    first_correct_sec: float | None
    first_stable_correct_sec: float | None
    stable_correct_before_audio_sufficient: bool
    final_correct: bool
    ever_forbidden: bool
    final_forbidden: bool
    correctness_retractions: int
    overcommit: bool
    stable_right_censored: bool

    def to_dict(self) -> dict:
        return asdict(self)


def tokenize_realization(text: str) -> tuple[str, ...]:
    return _tokens(text)


def canonical_json_sha256(value) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", exclude_none=True)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_evidence_packet(payload: EvidencePacketPayload) -> str:
    if payload.context_kind == "none":
        return ""
    if payload.context_kind == "empty":
        return "Advance source context:\n[no context available]"
    if payload.context_kind == "image":
        return "Current slide image:\n[image bytes attached]"
    headings = {
        "document": "Document context:",
        "ocr": "Current slide text:",
        "semantic": "Current slide semantic facts:",
        "relation": "Current slide relations:",
    }
    items = "\n".join(f"- {item.text}" for item in payload.context_items)
    return f"{headings[payload.context_kind]}\n{items}"


def directory_tree_sha256(root: Path) -> str:
    if root.is_symlink():
        raise ValueError(f"artifact root is a mutable symlink: {root}")
    if not root.is_dir():
        raise ValueError(f"artifact root is not a directory: {root}")
    paths = sorted(path for path in root.rglob("*") if path.is_file() or path.is_symlink())
    if not paths:
        raise ValueError(f"artifact root has no files: {root}")
    digest = hashlib.sha256()
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"artifact root contains a mutable symlink: {path}")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(16, "big"))
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def validate_outcome_commitment(
    commitment: OutcomeCommitment,
    *,
    artifact_root: Path,
    commitment_sha256: str,
    artifact_tree_sha256: str,
    contract: InferenceContract,
) -> None:
    if contract.outcome_commitment_sha256 != commitment_sha256:
        raise ValueError("inference contract outcome-commitment hash mismatch")
    if contract.outcome_artifact_tree_sha256 != artifact_tree_sha256:
        raise ValueError("inference contract outcome artifact tree hash mismatch")
    if commitment.source_events_sha256 != contract.source_events_sha256:
        raise ValueError("outcome commitment source-events hash differs from contract")
    if commitment.target_scores_sha256 != contract.target_scores_sha256:
        raise ValueError("outcome commitment target-scores hash differs from contract")
    resolved_root = artifact_root.resolve(strict=True)
    for artifact in commitment.artifacts:
        path = resolved_root / artifact.relative_path
        if path.is_symlink() or path.resolve(strict=True) != path:
            raise ValueError("outcome commitment artifact is symlinked or non-canonical")
        if file_sha256(path) != artifact.sha256:
            raise ValueError(f"outcome commitment artifact hash mismatch: {artifact.role}")


def realization_present(text: str, realization: str) -> bool:
    haystack = _tokens(text)
    needle = _tokens(realization)
    if not needle or len(needle) > len(haystack):
        return False
    width = len(needle)
    return any(haystack[index:index + width] == needle for index in range(len(haystack) - width + 1))


def score_trajectory(
    source: SourceEventTiming,
    target: TargetEventSpec,
    trajectory: EventTrajectory,
    *,
    min_stability_observations: int = 2,
) -> EventTimingScore:
    if min_stability_observations < 2:
        raise ValueError("min_stability_observations must be at least 2")
    if source.event_id != target.event_id or source.event_id != trajectory.event_id:
        raise ValueError("source, target, and trajectory event ids differ")
    if source.talk_id != trajectory.talk_id:
        raise ValueError("source and trajectory talk ids differ")
    if not math.isclose(
        trajectory.observations[-1].audio_time_sec,
        source.audio_endpoint_sec,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError("trajectory must end at the frozen causal endpoint")

    correct: list[bool] = []
    forbidden: list[bool] = []
    for observation in trajectory.observations:
        has_acceptable = any(
            realization_present(observation.hypothesis, value)
            for value in target.acceptable_realizations
        )
        has_forbidden = any(
            realization_present(observation.hypothesis, value)
            for value in target.forbidden_realizations
        )
        forbidden.append(has_forbidden)
        correct.append(has_acceptable and not has_forbidden)

    first_correct_index = next((index for index, value in enumerate(correct) if value), None)
    stable_index = _first_stable_index(correct, min_stability_observations)
    first_correct_sec = _time_at(trajectory, first_correct_index)
    first_stable_sec = _time_at(trajectory, stable_index)
    retractions = sum(left and not right for left, right in zip(correct, correct[1:]))
    final_correct = correct[-1]
    return EventTimingScore(
        event_id=source.event_id,
        talk_id=source.talk_id,
        condition=trajectory.condition,
        acoustic_condition=trajectory.acoustic_condition,
        inference_run_id=trajectory.inference_run_id,
        inference_contract_sha256=trajectory.inference_contract_sha256,
        evidence_packet_id=trajectory.evidence_packet_id,
        evidence_packet_sha256=trajectory.evidence_packet_sha256,
        observation_count=len(trajectory.observations),
        first_correct_sec=first_correct_sec,
        first_stable_correct_sec=first_stable_sec,
        stable_correct_before_audio_sufficient=(
            first_stable_sec is not None
            and first_stable_sec <= source.audio_insufficient_until_sec + 1e-9
        ),
        final_correct=final_correct,
        ever_forbidden=any(forbidden),
        final_forbidden=forbidden[-1],
        correctness_retractions=retractions,
        overcommit=any(correct[:-1]) and not final_correct,
        stable_right_censored=final_correct and stable_index is None,
    )


def validate_complete_matrix(
    sources: Iterable[SourceEventTiming],
    targets: Iterable[TargetEventSpec],
    trajectories: Iterable[EventTrajectory],
    *,
    expected_conditions: Iterable[str],
    expected_acoustic_conditions: Iterable[str],
) -> tuple[dict[str, SourceEventTiming], dict[str, TargetEventSpec], list[EventTrajectory]]:
    source_by_id = _unique_by_event_id(source for source in sources if source.primary_eligible)
    target_by_id = _unique_by_event_id(targets)
    if set(source_by_id) != set(target_by_id):
        missing_target = sorted(set(source_by_id) - set(target_by_id))
        extra_target = sorted(set(target_by_id) - set(source_by_id))
        raise ValueError(
            f"source/target event mismatch: missing_target={missing_target}, extra_target={extra_target}"
        )
    conditions = tuple(expected_conditions)
    acoustic_conditions = tuple(expected_acoustic_conditions)
    if not conditions or len(conditions) != len(set(conditions)):
        raise ValueError("expected conditions must be non-empty and unique")
    if not acoustic_conditions or len(acoustic_conditions) != len(set(acoustic_conditions)):
        raise ValueError("expected acoustic conditions must be non-empty and unique")

    trajectory_rows = list(trajectories)
    by_key: dict[tuple[str, str, str], EventTrajectory] = {}
    for trajectory in trajectory_rows:
        key = (trajectory.event_id, trajectory.condition, trajectory.acoustic_condition)
        if key in by_key:
            raise ValueError(f"duplicate trajectory: {key}")
        if trajectory.event_id not in source_by_id:
            raise ValueError(f"trajectory has unknown or ineligible event: {trajectory.event_id}")
        if trajectory.condition not in conditions:
            raise ValueError(f"unexpected condition: {trajectory.condition}")
        if trajectory.acoustic_condition not in acoustic_conditions:
            raise ValueError(f"unexpected acoustic condition: {trajectory.acoustic_condition}")
        by_key[key] = trajectory
    expected = {
        (event_id, condition, acoustic_condition)
        for event_id in source_by_id
        for condition in conditions
        for acoustic_condition in acoustic_conditions
    }
    missing = sorted(expected - set(by_key))
    if missing:
        raise ValueError(f"incomplete trajectory matrix: missing={missing[:10]} total={len(missing)}")
    for event_id in source_by_id:
        schedules = {
            tuple(
                observation.audio_time_sec
                for observation in by_key[(event_id, condition, acoustic_condition)].observations
            )
            for condition in conditions
            for acoustic_condition in acoustic_conditions
        }
        if len(schedules) != 1:
            raise ValueError(
                "trajectories use different audio-time grids across model or acoustic conditions: "
                f"event={event_id}"
            )
    return source_by_id, target_by_id, trajectory_rows


def validate_inference_provenance(
    contract: InferenceContract,
    attestation: InferenceResultAttestation,
    environment_start_audit: InferenceEnvironmentAudit,
    environment_end_audit: InferenceEnvironmentAudit,
    *,
    contract_sha256: str,
    trajectories_sha256: str,
    source_events_sha256: str,
    evidence_packets_sha256: str,
    control_pairs_sha256: str,
    scientific_config_sha256: str,
    scoring_config_sha256: str,
    target_scores_sha256: str,
    environment_start_audit_sha256: str,
    environment_end_audit_sha256: str,
    config: EventScoringConfig,
    scientific_config: InferenceScientificConfig,
    model_artifact_tree_sha256: str,
    trajectories: Iterable[EventTrajectory],
    expected_git_commit: str,
) -> None:
    if contract.source_events_sha256 != source_events_sha256:
        raise ValueError("inference contract source-events hash mismatch")
    if contract.control_pairs_sha256 != control_pairs_sha256:
        raise ValueError("inference contract control-pairs hash mismatch")
    if contract.evidence_packets_sha256 != evidence_packets_sha256:
        raise ValueError("inference contract evidence-packets hash mismatch")
    if contract.scientific_config_sha256 != scientific_config_sha256:
        raise ValueError("inference contract scientific-config hash mismatch")
    if (
        contract.model_id,
        contract.model_revision,
        contract.model_artifact_tree_sha256,
    ) != (
        scientific_config.model_id,
        scientific_config.model_revision,
        scientific_config.model_artifact_tree_sha256,
    ):
        raise ValueError("inference contract model identity differs from scientific config")
    if contract.model_artifact_tree_sha256 != model_artifact_tree_sha256:
        raise ValueError("inference contract model artifact tree hash mismatch")
    if contract.scoring_config_sha256 != scoring_config_sha256:
        raise ValueError("inference contract scoring-config hash mismatch")
    if contract.target_scores_sha256 != target_scores_sha256:
        raise ValueError("inference contract target-scores hash mismatch")
    if contract.expected_conditions != config.expected_conditions:
        raise ValueError("inference contract model conditions differ from scoring config")
    if contract.expected_acoustic_conditions != config.expected_acoustic_conditions:
        raise ValueError("inference contract acoustic conditions differ from scoring config")
    if contract.git_commit != expected_git_commit:
        raise ValueError("inference contract Git commit differs from frozen checkout")
    if attestation.run_id != contract.run_id:
        raise ValueError("inference result attestation run id differs from contract")
    if attestation.inference_contract_sha256 != contract_sha256:
        raise ValueError("inference result attestation does not bind the contract")
    if attestation.trajectories_sha256 != trajectories_sha256:
        raise ValueError("inference result attestation does not bind trajectories")
    if attestation.environment_start_audit_sha256 != environment_start_audit_sha256:
        raise ValueError("inference result attestation does not bind the start audit")
    if contract.environment_start_audit_sha256 != environment_start_audit_sha256:
        raise ValueError("inference contract does not bind the pre-run start audit")
    if (
        contract.worker_process_identity_tree_sha256
        != environment_start_audit.process_identity_tree_sha256
    ):
        raise ValueError("inference contract process identity tree differs from start audit")
    if attestation.environment_end_audit_sha256 != environment_end_audit_sha256:
        raise ValueError("inference result attestation does not bind the end audit")
    if (
        environment_start_audit.process_identity_tree_sha256
        != environment_end_audit.process_identity_tree_sha256
    ):
        raise ValueError("inference process identity tree changed during generation")
    if (
        environment_start_audit.container_id,
        environment_start_audit.container_name,
        environment_start_audit.capture_host,
        environment_start_audit.observed_mounts,
    ) != (
        environment_end_audit.container_id,
        environment_end_audit.container_name,
        environment_end_audit.capture_host,
        environment_end_audit.observed_mounts,
    ):
        raise ValueError("inference container or mount topology changed during generation")
    for phase, environment_audit in (
        ("workers_start", environment_start_audit),
        ("workers_end", environment_end_audit),
    ):
        if environment_audit.capture_phase != phase:
            raise ValueError("inference environment audit phase mismatch")
        if environment_audit.run_id != contract.run_id:
            raise ValueError("inference environment audit run id differs from contract")
        if environment_audit.inference_git_commit != contract.git_commit:
            raise ValueError("inference-time Git commit differs from contract")
        if (
            environment_audit.forbidden_container_artifact_roots
            != contract.forbidden_container_artifact_roots
        ):
            raise ValueError("inference forbidden container roots differ from contract")
        if (
            environment_audit.forbidden_host_mount_source_roots
            != contract.forbidden_host_mount_source_roots
        ):
            raise ValueError("inference forbidden host mount roots differ from contract")
        if environment_audit.worker_command_match != contract.worker_command_match:
            raise ValueError("inference worker command match differs from contract")
        if environment_audit.inference_repo_path != contract.inference_repo_path:
            raise ValueError("inference audit repo path differs from contract")
        if environment_audit.container_image_id != contract.container_image_id:
            raise ValueError("inference container image differs from contract")
        marker_processes = [
            process for process in environment_audit.worker_processes if process.marker_process
        ]
        if len(marker_processes) != contract.expected_worker_count:
            raise ValueError("inference marker worker count differs from contract")
        required_worker_arguments = [
            ("--inference-contract", contract.worker_inference_contract_path),
            ("--inference-contract-ready-file", contract.worker_contract_ready_file_path),
            ("--scientific-config", contract.worker_scientific_config_path),
            ("--model-artifact-root", contract.worker_model_artifact_root_path),
            (
                "--tokenizer-artifact-root",
                contract.worker_tokenizer_artifact_root_path,
            ),
            ("--model-id", contract.model_id),
            ("--model-revision", contract.model_revision),
        ]
        if contract.worker_source_artifact_root_path is not None:
            required_worker_arguments.append(
                ("--source-artifact-root", contract.worker_source_artifact_root_path)
            )
        if any(
            not all(
                command_contains_exact_marker(process.command, shlex.join(argument_pair))
                for argument_pair in required_worker_arguments
            )
            for process in marker_processes
        ):
            raise ValueError("inference worker command does not bind the contract barrier")
        require_read_only_mount(
            environment_audit,
            source=contract.scientific_config_host_path,
            destination=contract.worker_scientific_config_path,
            label="scientific config",
        )
        require_read_only_mount(
            environment_audit,
            source=contract.model_artifact_host_root_path,
            destination=contract.worker_model_artifact_root_path,
            label="model artifact tree",
        )
        require_read_only_mount(
            environment_audit,
            source=contract.tokenizer_artifact_host_root_path,
            destination=contract.worker_tokenizer_artifact_root_path,
            label="tokenizer artifact tree",
        )
        if contract.source_artifact_host_root_path is not None:
            require_read_only_mount(
                environment_audit,
                source=contract.source_artifact_host_root_path,
                destination=contract.worker_source_artifact_root_path,
                label="source artifact tree",
            )
    for trajectory in trajectories:
        if trajectory.inference_run_id != contract.run_id:
            raise ValueError(f"trajectory inference run mismatch: {trajectory.event_id}")
        if trajectory.inference_contract_sha256 != contract_sha256:
            raise ValueError(f"trajectory inference contract hash mismatch: {trajectory.event_id}")


def validate_causal_audio_provenance(
    contract: InferenceContract,
    attestation: InferenceResultAttestation,
    schedule: CausalAudioSchedule,
    release_log: CausalAudioReleaseLog,
    broker_audit: CausalAudioBrokerAudit,
    *,
    schedule_sha256: str,
    release_log_sha256: str,
    broker_audit_sha256: str,
    expected_broker_git_commit: str,
    observed_broker_entrypoint_sha256: str,
    config: EventScoringConfig,
    source_by_id: dict[str, SourceEventTiming],
    trajectories: Iterable[EventTrajectory],
) -> None:
    trajectory_rows = list(trajectories)
    if contract.causal_audio_schedule_sha256 != schedule_sha256:
        raise ValueError("inference contract causal audio schedule hash mismatch")
    if attestation.causal_audio_release_log_sha256 != release_log_sha256:
        raise ValueError("inference result attestation causal audio release-log hash mismatch")
    if contract.causal_audio_broker_audit_sha256 != broker_audit_sha256:
        raise ValueError("inference contract causal audio broker-audit hash mismatch")
    if not (schedule.run_id == release_log.run_id == broker_audit.run_id == contract.run_id):
        raise ValueError("causal audio artifacts use different run ids")
    if release_log.schedule_sha256 != schedule_sha256:
        raise ValueError("causal audio release log differs from schedule")
    if release_log.broker_audit_sha256 != broker_audit_sha256:
        raise ValueError("causal audio release log differs from broker audit")
    if broker_audit.schedule_sha256 != schedule_sha256:
        raise ValueError("causal audio broker audit differs from schedule")
    if broker_audit.broker_git_commit != contract.git_commit:
        raise ValueError("causal audio broker Git commit differs from inference contract")
    if broker_audit.broker_git_commit != expected_broker_git_commit:
        raise ValueError("causal audio broker Git commit differs from frozen checkout")
    if broker_audit.broker_entrypoint_sha256 != observed_broker_entrypoint_sha256:
        raise ValueError("causal audio broker entrypoint bytes changed after capture")
    if (
        broker_audit.broker_entrypoint_sha256
        != contract.causal_audio_broker_entrypoint_sha256
    ):
        raise ValueError("causal audio broker entrypoint differs from inference manifest")
    if broker_audit.source_audio_roots != schedule.source_audio_roots:
        raise ValueError("causal audio broker source roots differ from schedule")
    if schedule.expected_conditions != config.expected_conditions:
        raise ValueError("causal audio schedule conditions differ from scoring config")
    for source_root in schedule.source_audio_roots:
        if not any(
            path_is_within(source_root, forbidden_root)
            for forbidden_root in contract.forbidden_host_mount_source_roots
        ):
            raise ValueError("causal audio source root is not forbidden as a host mount source")

    source_spec_by_id = {source.source_id: source for source in schedule.sources}
    schedule_streams = {
        (source.talk_id, source.acoustic_condition) for source in schedule.sources
    }
    expected_streams = {
        (source_by_id[trajectory.event_id].talk_id, trajectory.acoustic_condition)
        for trajectory in trajectory_rows
    }
    if schedule_streams != expected_streams:
        raise ValueError("causal audio sources do not exactly cover frozen event/acoustic streams")

    schedule_by_key = {
        (prefix.event_id, prefix.acoustic_condition, prefix.sequence_index): prefix
        for prefix in schedule.prefixes
    }
    if len(schedule_by_key) != len(schedule.prefixes):
        raise ValueError("causal audio schedule keys must be unique")
    if any(
        prefix.acoustic_condition not in config.expected_acoustic_conditions
        for prefix in schedule.prefixes
    ):
        raise ValueError("causal audio schedule has an unexpected acoustic condition")
    release_by_key = {
        (
            release.event_id,
            release.condition,
            release.acoustic_condition,
            release.sequence_index,
        ): release
        for release in release_log.releases
    }
    commit_by_key = {
        (
            commit.event_id,
            commit.condition,
            commit.acoustic_condition,
            commit.sequence_index,
        ): commit
        for commit in release_log.observation_commits
    }
    expected_release_keys: set[tuple[str, str, str, int]] = set()
    observed_schedule_keys: set[tuple[str, str, int]] = set()
    for trajectory in trajectory_rows:
        if trajectory.condition not in config.expected_conditions:
            raise ValueError("trajectory has an unexpected causal audio condition")
        for sequence_index, observation in enumerate(trajectory.observations):
            schedule_key = (
                trajectory.event_id,
                trajectory.acoustic_condition,
                sequence_index,
            )
            release_key = (
                trajectory.event_id,
                trajectory.condition,
                trajectory.acoustic_condition,
                sequence_index,
            )
            prefix = schedule_by_key.get(schedule_key)
            release = release_by_key.get(release_key)
            commit = commit_by_key.get(release_key)
            if prefix is None or release is None or commit is None:
                raise ValueError(f"missing causal audio evidence: {release_key}")
            if not math.isclose(
                observation.audio_time_sec,
                prefix.audio_time_sec,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ValueError(f"trajectory time differs from causal audio schedule: {release_key}")
            observation_identity = (
                observation.causal_audio_prefix_id,
                observation.causal_audio_prefix_sha256,
            )
            prefix_identity = (prefix.prefix_id, prefix.prefix_pcm_sha256)
            release_identity = (release.prefix_id, release.prefix_pcm_sha256)
            if observation_identity != prefix_identity or release_identity != prefix_identity:
                raise ValueError(f"causal audio prefix identity mismatch: {release_key}")
            if release.source_id != prefix.source_id:
                raise ValueError(f"causal audio release source mismatch: {release_key}")
            commit_identity = (
                commit.source_id,
                commit.prefix_id,
                commit.prefix_pcm_sha256,
                commit.session_id,
            )
            release_commit_identity = (
                release.source_id,
                release.prefix_id,
                release.prefix_pcm_sha256,
                release.session_id,
            )
            if commit_identity != release_commit_identity:
                raise ValueError(f"causal observation commit identity mismatch: {release_key}")
            expected_observation_sha256 = causal_observation_sha256(
                run_id=trajectory.inference_run_id,
                inference_contract_sha256=trajectory.inference_contract_sha256,
                event_id=trajectory.event_id,
                condition=trajectory.condition,
                acoustic_condition=trajectory.acoustic_condition,
                sequence_index=sequence_index,
                observation=observation,
            )
            if commit.observation_sha256 != expected_observation_sha256:
                raise ValueError(f"causal observation commit hash mismatch: {release_key}")
            if release.server_ordinal >= commit.server_ordinal:
                raise ValueError(f"observation committed before prefix release: {release_key}")
            next_release = release_by_key.get(
                (
                    trajectory.event_id,
                    trajectory.condition,
                    trajectory.acoustic_condition,
                    sequence_index + 1,
                )
            )
            if next_release is not None and commit.server_ordinal >= next_release.server_ordinal:
                raise ValueError(f"next prefix released before observation commit: {release_key}")
            if release.sample_count != prefix.sample_count:
                raise ValueError(f"causal audio release sample boundary mismatch: {release_key}")
            source_spec = source_spec_by_id[prefix.source_id]
            event_source = source_by_id[trajectory.event_id]
            if (source_spec.talk_id, source_spec.acoustic_condition) != (
                event_source.talk_id,
                trajectory.acoustic_condition,
            ):
                raise ValueError(f"causal audio source stream mismatch: {release_key}")
            if not math.isclose(
                release.audio_time_sec,
                prefix.audio_time_sec,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ValueError(f"causal audio release time mismatch: {release_key}")
            expected_release_keys.add(release_key)
            observed_schedule_keys.add(schedule_key)
    if set(release_by_key) != expected_release_keys:
        raise ValueError("causal audio release log does not exactly cover trajectory observations")
    if set(commit_by_key) != expected_release_keys:
        raise ValueError("causal observation commits do not exactly cover trajectory observations")
    if set(schedule_by_key) != observed_schedule_keys:
        raise ValueError("causal audio schedule does not exactly cover trajectory time grids")

    source_by_source_id = {source.source_id: source for source in schedule.sources}
    interactions_by_talk_time: dict[
        tuple[str, float],
        list[tuple[CausalAudioReleaseRecord, CausalAudioObservationCommitRecord]],
    ] = defaultdict(list)
    for release_key, release in release_by_key.items():
        event_id, condition, acoustic_condition, sequence_index = release_key
        prefix = schedule_by_key[(event_id, acoustic_condition, sequence_index)]
        talk_id = source_by_source_id[prefix.source_id].talk_id
        interactions_by_talk_time[(talk_id, prefix.audio_time_sec)].append(
            (release, commit_by_key[release_key])
        )
    times_by_talk: dict[str, list[float]] = defaultdict(list)
    for talk_id, audio_time_sec in interactions_by_talk_time:
        times_by_talk[talk_id].append(audio_time_sec)
    for talk_id, times in times_by_talk.items():
        ordered_times = sorted(set(times))
        for audio_time_sec in ordered_times:
            interactions = sorted(
                interactions_by_talk_time[(talk_id, audio_time_sec)],
                key=lambda pair: pair[0].server_ordinal,
            )
            for (_, commit), (next_release, _) in zip(
                interactions,
                interactions[1:],
            ):
                if commit.server_ordinal >= next_release.server_ordinal:
                    raise ValueError(
                        "causal audio released concurrent same-time talk streams"
                    )
        for current_time, next_time in zip(ordered_times, ordered_times[1:]):
            current_commits = interactions_by_talk_time[(talk_id, current_time)]
            next_releases = interactions_by_talk_time[(talk_id, next_time)]
            if max(commit.server_ordinal for _, commit in current_commits) >= min(
                release.server_ordinal for release, _ in next_releases
            ):
                raise ValueError(
                    "causal audio advanced a talk before all current-time observations committed"
                )


def validate_evidence_packets(
    evidence_packets: Iterable[EvidencePacketSpec],
    *,
    source_by_id: dict[str, SourceEventTiming],
    trajectories: Iterable[EventTrajectory],
    config: EventScoringConfig,
    expected_tokenizer_model: str,
    expected_tokenizer_revision: str,
    expected_tokenizer_artifact_sha256: str,
    expected_source_artifact_tree_sha256: str,
    source_artifact_root: Path,
    tokenize: Callable[[str], list[int]],
) -> dict[tuple[str, str], EvidencePacketSpec]:
    observed_source_tree_sha256 = directory_tree_sha256(source_artifact_root)
    if observed_source_tree_sha256 != expected_source_artifact_tree_sha256:
        raise ValueError("source evidence artifact tree hash mismatch")
    by_key: dict[tuple[str, str], EvidencePacketSpec] = {}
    packet_ids: set[str] = set()
    artifact_cache: dict[str, SourceEvidenceArtifact] = {}
    packet_meta = (
        EXPECTED_PACKET_META_V1
        if config.schema_version.endswith("_v1")
        else EXPECTED_PACKET_META_V2
    )
    context_schema = (
        EXPECTED_CONTEXT_KIND_V1
        if config.schema_version.endswith("_v1")
        else EXPECTED_CONTEXT_KIND_V2
    )
    expected_packet_schema = (
        "acl6060_source_evidence_packet_v1"
        if config.schema_version.endswith("_v1")
        else "acl6060_source_evidence_packet_v2"
    )
    for packet in evidence_packets:
        key = (packet.event_id, packet.condition)
        if key in by_key:
            raise ValueError(f"duplicate evidence packet row: {key}")
        if packet.packet_id in packet_ids:
            raise ValueError(f"duplicate evidence packet id: {packet.packet_id}")
        if packet.event_id not in source_by_id:
            raise ValueError(f"evidence packet has unknown event: {packet.event_id}")
        expected_meta = packet_meta.get(packet.condition)
        if packet.condition not in config.expected_conditions or expected_meta is None:
            raise ValueError(f"evidence packet has unexpected condition: {packet.condition}")
        if (packet.evidence_type, packet.evidence_role) != expected_meta:
            raise ValueError(f"evidence packet type/role mismatch: {key}")
        if packet.tokenizer_model != expected_tokenizer_model:
            raise ValueError(f"evidence packet tokenizer model mismatch: {key}")
        if packet.tokenizer_revision != expected_tokenizer_revision:
            raise ValueError(f"evidence packet tokenizer revision mismatch: {key}")
        if packet.tokenizer_artifact_sha256 != expected_tokenizer_artifact_sha256:
            raise ValueError(f"evidence packet tokenizer artifact mismatch: {key}")
        expected_context_kind = context_schema[packet.condition]
        if packet.packet_payload.context_kind != expected_context_kind:
            raise ValueError(f"evidence packet context kind mismatch: {key}")
        if packet.packet_payload.schema_version != expected_packet_schema:
            raise ValueError(f"evidence packet schema version mismatch: {key}")
        expected_source_version = "v1" if config.schema_version.endswith("_v1") else "v2"
        if source_by_id[packet.event_id].condition_matrix_version != expected_source_version:
            raise ValueError(f"source event condition matrix version mismatch: {key}")
        expected_source = {
            value.condition: value
            for value in source_by_id[packet.event_id].expected_evidence_sources
        }.get(packet.condition)

        def load_artifact(reference_path: str, reference_sha256: str) -> SourceEvidenceArtifact:
            artifact_path = _safe_artifact_path(source_artifact_root, reference_path)
            if file_sha256(artifact_path) != reference_sha256:
                raise ValueError(f"source evidence artifact hash mismatch: {key}")
            artifact = artifact_cache.get(reference_path)
            if artifact is None:
                artifact = SourceEvidenceArtifact.model_validate_json(
                    artifact_path.read_text(encoding="utf-8")
                )
                artifact_cache[reference_path] = artifact
            if artifact.event_id != packet.event_id:
                raise ValueError(f"source evidence artifact event mismatch: {key}")
            if artifact.context_kind != packet.packet_payload.context_kind:
                raise ValueError(f"source evidence artifact context kind mismatch: {key}")
            if expected_source is None:
                raise ValueError(f"source evidence artifact has no frozen source identity: {key}")
            observed_source_identity = (
                artifact.context_kind,
                artifact.source_media_kind,
                artifact.source_media_path,
                artifact.source_media_sha256,
                artifact.extractor,
                artifact.extractor_revision,
            )
            frozen_source_identity = (
                expected_source.context_kind,
                expected_source.source_media_kind,
                expected_source.source_media_path,
                expected_source.source_media_sha256,
                expected_source.extractor,
                expected_source.extractor_revision,
            )
            if observed_source_identity != frozen_source_identity:
                raise ValueError(f"source evidence artifact differs from event source identity: {key}")
            media_path = _safe_artifact_path(source_artifact_root, artifact.source_media_path)
            if file_sha256(media_path) != artifact.source_media_sha256:
                raise ValueError(f"source media hash mismatch: {key}")
            return artifact

        for reference in packet.packet_payload.context_items:
            artifact = load_artifact(reference.artifact_path, reference.artifact_sha256)
            if reference.item_index >= len(artifact.items):
                raise ValueError(f"source evidence artifact item index is out of range: {key}")
            if artifact.items[reference.item_index] != reference.text:
                raise ValueError(f"source evidence text differs from frozen artifact: {key}")
        image_reference = packet.packet_payload.image_reference
        if image_reference is not None:
            artifact = load_artifact(
                image_reference.artifact_path,
                image_reference.artifact_sha256,
            )
            if artifact.context_kind != "image" or artifact.items:
                raise ValueError(f"raw image artifact contains text context: {key}")
        replayed_token_ids = tokenize(render_evidence_packet(packet.packet_payload))
        if replayed_token_ids != packet.token_ids:
            raise ValueError(f"evidence packet token ids differ from tokenizer replay: {key}")
        source = source_by_id[packet.event_id]
        if packet.condition in {"audio_only", "document_only"}:
            if packet.available_sec > source.evidence_available_sec + 1e-6:
                raise ValueError(f"baseline packet is not pre-available: {key}")
        elif not math.isclose(
            packet.available_sec,
            source.evidence_available_sec,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(f"slide packet availability differs from source event: {key}")
        by_key[key] = packet
        packet_ids.add(packet.packet_id)
    expected = {
        (event_id, condition)
        for event_id in source_by_id
        for condition in config.expected_conditions
    }
    missing = sorted(expected - set(by_key))
    extra = sorted(set(by_key) - expected)
    if missing or extra:
        raise ValueError(f"evidence-packet matrix mismatch: missing={missing[:10]}, extra={extra[:10]}")
    for trajectory in trajectories:
        packet = by_key[(trajectory.event_id, trajectory.condition)]
        if (trajectory.evidence_packet_id, trajectory.evidence_packet_sha256) != (
            packet.packet_id,
            packet.packet_sha256,
        ):
            raise ValueError(
                f"trajectory packet differs from evidence-packet manifest: {trajectory.event_id}"
            )
    if directory_tree_sha256(source_artifact_root) != observed_source_tree_sha256:
        raise ValueError("source evidence artifact tree changed during validation")
    return by_key


def validate_control_pairs(
    control_pairs: Iterable[ControlPairSpec],
    *,
    source_by_id: dict[str, SourceEventTiming],
    trajectories: Iterable[EventTrajectory],
    config: EventScoringConfig,
    evidence_packet_by_key: dict[tuple[str, str], EvidencePacketSpec],
) -> None:
    required_contrasts = {
        contrast.id: contrast
        for contrast in config.contrasts
        if contrast.requires_matched_control
    }
    expected = {
        (event_id, contrast_id)
        for event_id in source_by_id
        for contrast_id in required_contrasts
    }
    by_key: dict[tuple[str, str], ControlPairSpec] = {}
    control_pair_ids: set[str] = set()
    for pair in control_pairs:
        key = (pair.event_id, pair.contrast_id)
        if key in by_key:
            raise ValueError(f"duplicate control pair: {key}")
        if pair.control_pair_id in control_pair_ids:
            raise ValueError(f"duplicate control_pair_id: {pair.control_pair_id}")
        if pair.event_id not in source_by_id:
            raise ValueError(f"control pair has unknown event: {pair.event_id}")
        contrast = required_contrasts.get(pair.contrast_id)
        if contrast is None:
            raise ValueError(f"control pair has unexpected contrast: {pair.contrast_id}")
        if (pair.first_condition, pair.second_condition) != (contrast.first, contrast.second):
            raise ValueError(f"control pair condition mismatch: {key}")
        if pair.evidence_type != contrast.evidence_type:
            raise ValueError(f"control pair evidence type mismatch: {key}")
        source = source_by_id[pair.event_id]
        if not math.isclose(
            pair.first_available_sec,
            source.evidence_available_sec,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(f"control pair availability differs from source event: {key}")
        first_packet = evidence_packet_by_key[(pair.event_id, pair.first_condition)]
        second_packet = evidence_packet_by_key[(pair.event_id, pair.second_condition)]
        if (
            pair.first_packet_id,
            pair.first_packet_sha256,
            pair.first_available_sec,
            pair.first_token_count,
            pair.first_visual_token_count,
        ) != (
            first_packet.packet_id,
            first_packet.packet_sha256,
            first_packet.available_sec,
            len(first_packet.token_ids),
            first_packet.visual_token_count,
        ):
            raise ValueError(f"correct control-pair metadata differs from packet bytes: {key}")
        if (
            pair.second_packet_id,
            pair.second_packet_sha256,
            pair.second_available_sec,
            pair.second_token_count,
            pair.second_visual_token_count,
        ) != (
            second_packet.packet_id,
            second_packet.packet_sha256,
            second_packet.available_sec,
            len(second_packet.token_ids),
            second_packet.visual_token_count,
        ):
            raise ValueError(f"wrong control-pair metadata differs from packet bytes: {key}")
        by_key[key] = pair
        control_pair_ids.add(pair.control_pair_id)
    missing = sorted(expected - set(by_key))
    extra = sorted(set(by_key) - expected)
    if missing or extra:
        raise ValueError(f"control-pair matrix mismatch: missing={missing[:10]}, extra={extra[:10]}")

    trajectory_by_key = {
        (trajectory.event_id, trajectory.condition, trajectory.acoustic_condition): trajectory
        for trajectory in trajectories
    }
    for (event_id, contrast_id), pair in by_key.items():
        contrast = required_contrasts[contrast_id]
        for acoustic_condition in config.expected_acoustic_conditions:
            first = trajectory_by_key[(event_id, contrast.first, acoustic_condition)]
            second = trajectory_by_key[(event_id, contrast.second, acoustic_condition)]
            if (first.evidence_packet_id, first.evidence_packet_sha256) != (
                pair.first_packet_id,
                pair.first_packet_sha256,
            ):
                raise ValueError(f"correct trajectory packet differs from control-pair manifest: {event_id}")
            if (second.evidence_packet_id, second.evidence_packet_sha256) != (
                pair.second_packet_id,
                pair.second_packet_sha256,
            ):
                raise ValueError(f"wrong trajectory packet differs from control-pair manifest: {event_id}")


def summarize_contrast(
    scores: Iterable[EventTimingScore],
    *,
    first: str,
    second: str,
    acoustic_group: str,
    acoustic_conditions: Iterable[str],
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 0,
) -> dict:
    members = tuple(acoustic_conditions)
    if not members or len(members) != len(set(members)):
        raise ValueError("acoustic group members must be non-empty and unique")
    selected = [score for score in scores if score.acoustic_condition in members]
    by_key = {
        (score.event_id, score.condition, score.acoustic_condition): score
        for score in selected
    }
    event_ids = sorted({score.event_id for score in selected})
    pairs: list[tuple[EventTimingScore, EventTimingScore]] = []
    for event_id in event_ids:
        for acoustic_condition in members:
            try:
                pairs.append(
                    (
                        by_key[(event_id, first, acoustic_condition)],
                        by_key[(event_id, second, acoustic_condition)],
                    )
                )
            except KeyError as error:
                raise ValueError(f"missing contrast member for {event_id}: {error}") from error
    if not pairs:
        raise ValueError(f"no scores for acoustic group {acoustic_group}")

    pooled_early_delta = _mean(
        float(left.stable_correct_before_audio_sufficient)
        - float(right.stable_correct_before_audio_sufficient)
        for left, right in pairs
    )
    pooled_final_delta = _mean(
        float(left.final_correct) - float(right.final_correct)
        for left, right in pairs
    )
    talk_deltas: dict[str, list[float]] = defaultdict(list)
    final_talk_deltas: dict[str, list[float]] = defaultdict(list)
    forbidden_talk_deltas: dict[str, list[float]] = defaultdict(list)
    overcommit_talk_deltas: dict[str, list[float]] = defaultdict(list)
    for left, right in pairs:
        if left.talk_id != right.talk_id:
            raise ValueError(f"paired scores have different talks: {left.event_id}")
        talk_deltas[left.talk_id].append(
            float(left.stable_correct_before_audio_sufficient)
            - float(right.stable_correct_before_audio_sufficient)
        )
        final_talk_deltas[left.talk_id].append(
            float(left.final_correct) - float(right.final_correct)
        )
        forbidden_talk_deltas[left.talk_id].append(
            float(left.ever_forbidden) - float(right.ever_forbidden)
        )
        overcommit_talk_deltas[left.talk_id].append(
            float(left.overcommit) - float(right.overcommit)
        )
    per_talk_early = {talk: _mean(values) for talk, values in sorted(talk_deltas.items())}
    per_talk_final = {talk: _mean(values) for talk, values in sorted(final_talk_deltas.items())}
    per_talk_forbidden = {
        talk: _mean(values) for talk, values in sorted(forbidden_talk_deltas.items())
    }
    per_talk_overcommit = {
        talk: _mean(values) for talk, values in sorted(overcommit_talk_deltas.items())
    }

    advances = [
        right.first_stable_correct_sec - left.first_stable_correct_sec
        for left, right in pairs
        if left.final_correct
        and right.final_correct
        and left.first_stable_correct_sec is not None
        and right.first_stable_correct_sec is not None
    ]
    summary = {
        "first": first,
        "second": second,
        "acoustic_group": acoustic_group,
        "acoustic_conditions": list(members),
        "event_count": len(event_ids),
        "paired_trajectory_count": len(pairs),
        "replicates_per_event": len(members),
        "talk_count": len(talk_deltas),
        "early_stable_correct_rate_first": _mean(
            float(left.stable_correct_before_audio_sufficient) for left, _ in pairs
        ),
        "early_stable_correct_rate_second": _mean(
            float(right.stable_correct_before_audio_sufficient) for _, right in pairs
        ),
        "pooled_early_risk_difference": pooled_early_delta,
        "talk_equal_early_risk_difference": _mean(per_talk_early.values()),
        "per_talk_early_risk_difference": per_talk_early,
        "pooled_final_correct_risk_difference": pooled_final_delta,
        "talk_equal_final_correct_risk_difference": _mean(per_talk_final.values()),
        "per_talk_final_correct_risk_difference": per_talk_final,
        "talk_equal_forbidden_adoption_risk_difference": _mean(per_talk_forbidden.values()),
        "per_talk_forbidden_adoption_risk_difference": per_talk_forbidden,
        "talk_equal_overcommit_risk_difference": _mean(per_talk_overcommit.values()),
        "per_talk_overcommit_risk_difference": per_talk_overcommit,
        "both_final_correct_and_stable_count": len(advances),
        "mean_commit_advance_sec": _mean(advances) if advances else None,
        "median_commit_advance_sec": _median(advances) if advances else None,
        "ever_forbidden_rate_first": _mean(float(left.ever_forbidden) for left, _ in pairs),
        "ever_forbidden_rate_second": _mean(float(right.ever_forbidden) for _, right in pairs),
        "overcommit_rate_first": _mean(float(left.overcommit) for left, _ in pairs),
        "overcommit_rate_second": _mean(float(right.overcommit) for _, right in pairs),
        "inference_unit": "talk",
    }
    if bootstrap_samples:
        summary["talk_cluster_bootstrap_samples"] = bootstrap_samples
        summary["talk_cluster_bootstrap_seed"] = bootstrap_seed
        summary["talk_equal_early_risk_difference_ci95"] = _cluster_bootstrap_ci(
            per_talk_early,
            bootstrap_samples,
            bootstrap_seed,
        )
        summary["talk_equal_final_correct_risk_difference_ci95"] = _cluster_bootstrap_ci(
            per_talk_final,
            bootstrap_samples,
            bootstrap_seed + 1,
        )
        summary["talk_equal_forbidden_adoption_risk_difference_ci95"] = _cluster_bootstrap_ci(
            per_talk_forbidden,
            bootstrap_samples,
            bootstrap_seed + 2,
        )
        summary["talk_equal_overcommit_risk_difference_ci95"] = _cluster_bootstrap_ci(
            per_talk_overcommit,
            bootstrap_samples,
            bootstrap_seed + 3,
        )
    return summary


def apply_development_gate(summary: dict, signal: DevelopmentSignalSpec) -> None:
    positive_talks = sum(
        value > 0 for value in summary["per_talk_early_risk_difference"].values()
    )
    early_pass = summary["talk_equal_early_risk_difference"] >= (
        signal.early_risk_difference_percentage_points / 100
    )
    final_pass = summary["talk_equal_final_correct_risk_difference"] >= (
        signal.final_correctness_point_estimate_floor_percentage_points / 100
    )
    direction_pass = positive_talks >= signal.directionally_consistent_talks
    talk_count_pass = summary["talk_count"] >= signal.minimum_talks
    forbidden_pass = summary["talk_equal_forbidden_adoption_risk_difference"] <= (
        signal.forbidden_adoption_point_estimate_ceiling_percentage_points / 100
    )
    overcommit_pass = summary["talk_equal_overcommit_risk_difference"] <= (
        signal.overcommit_point_estimate_ceiling_percentage_points / 100
    )
    summary["development_gate"] = {
        "early_practical_signal_pass": early_pass,
        "final_correctness_point_estimate_pass": final_pass,
        "directionally_positive_talk_count": positive_talks,
        "directional_consistency_pass": direction_pass,
        "minimum_talk_count_pass": talk_count_pass,
        "forbidden_adoption_point_estimate_pass": forbidden_pass,
        "overcommit_point_estimate_pass": overcommit_pass,
        "all_components_pass": (
            early_pass
            and final_pass
            and direction_pass
            and talk_count_pass
            and forbidden_pass
            and overcommit_pass
        ),
        "exploratory_only": True,
    }


def summarize_acoustic_interaction(native: dict, comparison: dict) -> dict:
    if native["first"] != comparison["first"] or native["second"] != comparison["second"]:
        raise ValueError("noise interaction contrasts differ")
    if native["event_count"] != comparison["event_count"]:
        raise ValueError("noise interaction event/talk sets differ")
    native_talks = native["per_talk_early_risk_difference"]
    comparison_talks = comparison["per_talk_early_risk_difference"]
    if set(native_talks) != set(comparison_talks):
        raise ValueError("noise interaction event/talk sets differ")
    per_talk_interaction = {
        talk_id: comparison_talks[talk_id] - native_talks[talk_id]
        for talk_id in sorted(native_talks)
    }
    if native["acoustic_group"] == comparison["acoustic_group"]:
        raise ValueError("noise interaction requires distinct conditions")
    return {
        "first": native["first"],
        "second": native["second"],
        "native_acoustic_group": native["acoustic_group"],
        "comparison_acoustic_group": comparison["acoustic_group"],
        "talk_count": len(per_talk_interaction),
        "per_talk_early_risk_difference_interaction": per_talk_interaction,
        "directionally_positive_talk_count": sum(
            value > 0 for value in per_talk_interaction.values()
        ),
        "talk_equal_early_risk_difference_interaction": (
            comparison["talk_equal_early_risk_difference"]
            - native["talk_equal_early_risk_difference"]
        ),
        "pooled_early_risk_difference_interaction": (
            comparison["pooled_early_risk_difference"]
            - native["pooled_early_risk_difference"]
        ),
    }


def summarize_babble_severity_curve(by_group: dict[str, dict], order: list[str]) -> dict:
    values = [by_group[group_id]["talk_equal_early_risk_difference"] for group_id in order]
    return {
        "acoustic_groups_in_increasing_severity": order,
        "talk_equal_early_risk_differences": values,
        "pearson_correlation_severity_vs_effect": _linear_correlation(values),
        "monotonic_non_decreasing": all(
            right >= left for left, right in zip(values, values[1:])
        ),
    }


def joint_talk_cluster_bootstrap(
    by_group: dict[str, dict],
    *,
    native_group: str,
    severity_order: list[str],
    samples: int,
    seed: int,
) -> dict:
    talk_sets = {
        tuple(sorted(summary["per_talk_early_risk_difference"]))
        for summary in by_group.values()
    }
    if len(talk_sets) != 1:
        raise ValueError("joint bootstrap acoustic groups have different talk sets")
    talks = list(next(iter(talk_sets)))
    if not talks:
        raise ValueError("joint bootstrap has no talks")
    rng = random.Random(seed)
    interaction_samples = {
        group_id: [] for group_id in by_group if group_id != native_group
    }
    correlation_samples: list[float] = []
    monotonic_count = 0
    for _ in range(samples):
        draw = rng.choices(talks, k=len(talks))
        means = {
            group_id: _mean(
                summary["per_talk_early_risk_difference"][talk_id]
                for talk_id in draw
            )
            for group_id, summary in by_group.items()
        }
        for group_id, values in interaction_samples.items():
            values.append(means[group_id] - means[native_group])
        severity_values = [means[group_id] for group_id in severity_order]
        correlation = _linear_correlation(severity_values)
        if correlation is not None:
            correlation_samples.append(correlation)
        monotonic_count += all(
            right >= left
            for left, right in zip(severity_values, severity_values[1:])
        )
    return {
        "samples": samples,
        "seed": seed,
        "interaction_ci95_by_acoustic_group": {
            group_id: _sorted_ci(values)
            for group_id, values in interaction_samples.items()
        },
        "severity_correlation_defined_samples": len(correlation_samples),
        "severity_correlation_undefined_samples": samples - len(correlation_samples),
        "severity_correlation_ci95": (
            _sorted_ci(correlation_samples)
            if len(correlation_samples) == samples
            else None
        ),
        "severity_correlation_interval_status": (
            "COMPLETE" if len(correlation_samples) == samples else "UNDEFINED_DRAWS_PRESENT"
        ),
        "severity_monotonic_bootstrap_probability": monotonic_count / samples,
    }


def _tokens(text: str) -> tuple[str, ...]:
    normalized = (
        unicodedata.normalize("NFKC", text)
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .casefold()
    )
    tokens: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            raw = "".join(buffer).rstrip(".")
            buffer.clear()
            if re.fullmatch(r"(?:[a-z]\.)+[a-z]", raw):
                raw = raw.replace(".", "")
            raw = re.sub(r"(?<=[a-z])-(?=\d)", "", raw)
            parts = [raw] if raw.startswith(("-", "+")) else raw.split("-")
            tokens.extend(part for part in parts if part)

    def boundary() -> None:
        if tokens and tokens[-1] != "\0":
            tokens.append("\0")

    for character in normalized:
        codepoint = ord(character)
        if (
            0x3400 <= codepoint <= 0x4DBF
            or 0x4E00 <= codepoint <= 0x9FFF
            or 0xF900 <= codepoint <= 0xFAFF
        ):
            flush()
            tokens.append(character)
        elif unicodedata.category(character)[0] in {"L", "N"} or character in ".+#-%°":
            buffer.append(character)
        elif character.isspace():
            flush()
        else:
            flush()
            boundary()
    flush()
    while tokens and tokens[-1] == "\0":
        tokens.pop()
    return tuple(tokens)


def _safe_artifact_path(root: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"artifact path must be relative and traversal-free: {relative}")
    resolved_root = root.resolve()
    resolved = (root / relative_path).resolve()
    if resolved.parent != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"artifact path escapes root: {relative}")
    if not resolved.is_file():
        raise ValueError(f"artifact path is not a file: {relative}")
    return resolved


def _first_stable_index(correct: list[bool], minimum: int) -> int | None:
    if len(correct) < minimum or not correct[-1]:
        return None
    index = len(correct) - 1
    while index > 0 and correct[index - 1]:
        index -= 1
    return index if len(correct) - index >= minimum else None


def _time_at(trajectory: EventTrajectory, index: int | None) -> float | None:
    return None if index is None else trajectory.observations[index].audio_time_sec


def _unique_by_event_id(rows: Iterable[BaseModel]) -> dict[str, BaseModel]:
    result: dict[str, BaseModel] = {}
    for row in rows:
        event_id = getattr(row, "event_id")
        if event_id in result:
            raise ValueError(f"duplicate event id: {event_id}")
        result[event_id] = row
    if not result:
        raise ValueError("no eligible events")
    return result


def _cluster_bootstrap_ci(
    per_talk_values: dict[str, float],
    samples: int,
    seed: int,
) -> list[float]:
    values = list(per_talk_values.values())
    if not values:
        raise ValueError("cannot bootstrap empty talk values")
    rng = random.Random(seed)
    estimates = sorted(
        _mean(rng.choices(values, k=len(values)))
        for _ in range(samples)
    )
    return _sorted_ci(estimates)


def _sorted_ci(values: Iterable[float]) -> list[float]:
    estimates = sorted(values)
    if not estimates:
        raise ValueError("cannot calculate interval from empty values")
    last = len(estimates) - 1
    return [estimates[int(0.025 * last)], estimates[int(0.975 * last)]]


def _linear_correlation(values: list[float]) -> float | None:
    levels = list(range(len(values)))
    mean_level = _mean(levels)
    mean_value = _mean(values)
    numerator = math.fsum(
        (level - mean_level) * (value - mean_value)
        for level, value in zip(levels, values)
    )
    denominator = math.sqrt(
        math.fsum((level - mean_level) ** 2 for level in levels)
        * math.fsum((value - mean_value) ** 2 for value in values)
    )
    return numerator / denominator if denominator else None


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise ValueError("cannot calculate mean of empty values")
    return math.fsum(materialized) / len(materialized)


def _median(values: Iterable[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2
