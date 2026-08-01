import hashlib
import json
import os
import struct
import threading
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from slidesst.eval.causal_audio import CausalAudioBroker, ThreadedUnixAudioServer
from slidesst.eval.causal_worker import (
    CausalGenerationInput,
    Qwen3OmniBatchGenerator,
    qwen_multimodal_content,
    resolve_packet_image,
    run_causal_event_worker,
    validate_visual_token_counts,
)
from slidesst.eval.event_timing import (
    CausalAudioSchedule,
    EvidencePacketPayload,
    EvidencePacketSpec,
    EventScoringConfig,
    SourceEvidenceArtifact,
    SourceImageReference,
    canonical_json_sha256,
    file_sha256,
    render_evidence_packet,
    text_sha256,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0c"
    b"IDAT\x08\xd7c\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb1"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _image_packet(root: Path) -> tuple[EvidencePacketSpec, Path, Path]:
    media_path = root / "media" / "slide.png"
    media_path.parent.mkdir(parents=True)
    media_path.write_bytes(PNG_BYTES)
    artifact = SourceEvidenceArtifact(
        schema_version="acl6060_source_evidence_artifact_v2",
        event_id="event-image",
        context_kind="image",
        source_media_kind="slide_image",
        source_media_path="media/slide.png",
        source_media_sha256=file_sha256(media_path),
        extractor="source-image-materializer",
        extractor_revision="1" * 40,
        items=[],
    )
    artifact_path = root / "evidence" / "event-image.json"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text(artifact.model_dump_json() + "\n", encoding="utf-8")
    payload = EvidencePacketPayload(
        schema_version="acl6060_source_evidence_packet_v2",
        context_kind="image",
        context_items=[],
        image_reference=SourceImageReference(
            artifact_path="evidence/event-image.json",
            artifact_sha256=file_sha256(artifact_path),
        ),
    )
    token_ids = [101, 102]
    packet = EvidencePacketSpec(
        event_id="event-image",
        condition="correct_image",
        packet_id="packet:event-image:correct-image",
        packet_sha256=canonical_json_sha256(payload),
        evidence_type="image",
        evidence_role="correct",
        available_sec=0.0,
        tokenizer_model="fixture/tokenizer",
        tokenizer_revision="2" * 40,
        tokenizer_artifact_sha256="3" * 64,
        token_ids=token_ids,
        token_ids_sha256=canonical_json_sha256(token_ids),
        rendered_text_sha256=text_sha256(render_evidence_packet(payload)),
        visual_token_count=4,
        packet_payload=payload,
    )
    return packet, artifact_path, media_path


def _generation_input(
    condition: str,
    *,
    image_path: Path | None = None,
    visual_token_count: int = 0,
) -> CausalGenerationInput:
    return CausalGenerationInput(
        event_id=f"event-{condition}",
        talk_id="talk-1",
        condition=condition,
        acoustic_condition="native",
        sequence_index=0,
        sample_rate=16_000,
        audio=np.zeros(16, dtype=np.float32),
        evidence_text="evidence",
        prompt_text="Translate the speech.",
        evidence_image_path=image_path,
        expected_visual_token_count=visual_token_count,
    )


def _single_image_schedule(tmp_path: Path) -> CausalAudioSchedule:
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    pcm = struct.pack("<4f", 0.0, 0.25, -0.25, 0.0)
    pcm_path = audio_root / "talk-1.f32le"
    pcm_path.write_bytes(pcm)
    provenance_path = audio_root / "talk-1.json"
    provenance_path.write_text('{"talk_id":"talk-1"}\n', encoding="utf-8")
    return CausalAudioSchedule.model_validate(
        {
            "schema_version": "acl6060_causal_audio_schedule_v3",
            "run_id": "run-image-test",
            "expected_conditions": ["correct_image"],
            "source_audio_roots": [str(audio_root)],
            "sources": [
                {
                    "source_id": "source:talk-1:native",
                    "talk_id": "talk-1",
                    "acoustic_condition": "native",
                    "source_pcm_path": str(pcm_path),
                    "source_pcm_sha256": hashlib.sha256(pcm).hexdigest(),
                    "pcm_format": "float32le_mono",
                    "sample_rate": 4,
                    "total_sample_count": 4,
                    "materialization_kind": "native",
                    "upstream_audio_sha256": "4" * 64,
                    "materializer_git_commit": "5" * 40,
                    "materializer_entrypoint_sha256": "6" * 64,
                    "source_provenance_path": str(provenance_path),
                    "source_provenance_sha256": file_sha256(provenance_path),
                }
            ],
            "prefixes": [
                {
                    "source_id": "source:talk-1:native",
                    "event_id": "event-image",
                    "acoustic_condition": "native",
                    "sequence_index": 0,
                    "audio_time_sec": 1.0,
                    "prefix_id": "prefix:event-image:0",
                    "prefix_pcm_sha256": hashlib.sha256(pcm).hexdigest(),
                    "sample_rate": 4,
                    "sample_count": 4,
                }
            ],
        }
    )


def test_v2_scoring_config_freezes_raw_image_contrasts():
    config = EventScoringConfig.model_validate_json(
        (REPO_ROOT / "code/configs/acl6060_event_trajectory_scoring_v2.json").read_bytes()
    )
    assert config.expected_conditions[-2:] == [
        "correct_image",
        "matched_wrong_image",
    ]
    assert [contrast.id for contrast in config.contrasts[-3:]] == [
        "image_content_specificity",
        "image_over_relation",
        "relation_over_ocr",
    ]


def test_v1_models_reject_raw_image_evidence():
    with pytest.raises(ValidationError, match="v1 source evidence artifact"):
        SourceEvidenceArtifact(
            schema_version="acl6060_source_evidence_artifact_v1",
            event_id="event-image",
            context_kind="image",
            source_media_kind="slide_image",
            source_media_path="slide.png",
            source_media_sha256="1" * 64,
            extractor="fixture",
            extractor_revision="2" * 40,
            items=[],
        )
    with pytest.raises(ValidationError, match="v1 evidence packet"):
        EvidencePacketPayload(
            schema_version="acl6060_source_evidence_packet_v1",
            context_kind="image",
            context_items=[],
            image_reference=SourceImageReference(
                artifact_path="image.json",
                artifact_sha256="3" * 64,
            ),
        )


def test_resolve_packet_image_binds_artifact_and_media_bytes(tmp_path):
    packet, artifact_path, media_path = _image_packet(tmp_path)
    assert resolve_packet_image(packet, tmp_path) == media_path

    media_path.write_bytes(PNG_BYTES + b"drift")
    with pytest.raises(ValueError, match="image evidence bytes changed"):
        resolve_packet_image(packet, tmp_path)

    media_path.write_bytes(PNG_BYTES)
    artifact_path.write_text(artifact_path.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        resolve_packet_image(packet, tmp_path)


def test_resolve_packet_image_rejects_path_escape(tmp_path):
    packet, artifact_path, _ = _image_packet(tmp_path)
    escaped_path = tmp_path.parent / f"{tmp_path.name}-escaped.png"
    escaped_path.write_bytes(PNG_BYTES)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["source_media_path"] = f"../{escaped_path.name}"
    artifact["source_media_sha256"] = file_sha256(escaped_path)
    artifact_path.write_text(json.dumps(artifact) + "\n", encoding="utf-8")
    payload = packet.packet_payload.model_copy(
        update={
            "image_reference": SourceImageReference(
                artifact_path="evidence/event-image.json",
                artifact_sha256=file_sha256(artifact_path),
            )
        }
    )
    escaped_packet = packet.model_copy(
        update={
            "packet_payload": payload,
            "packet_sha256": canonical_json_sha256(payload),
            "rendered_text_sha256": text_sha256(render_evidence_packet(payload)),
        }
    )
    try:
        with pytest.raises(ValueError, match="canonical and relative"):
            resolve_packet_image(escaped_packet, tmp_path)
    finally:
        escaped_path.unlink()


def test_qwen_content_orders_image_before_prompt_and_audio(tmp_path):
    item = _generation_input(
        "correct_image",
        image_path=tmp_path / "slide.png",
        visual_token_count=4,
    )
    content = qwen_multimodal_content(item)
    assert [part["type"] for part in content] == ["image", "text", "audio"]
    assert content[0]["image"] == str(item.evidence_image_path)


def test_visual_token_count_must_match_each_frozen_packet(tmp_path):
    image_item = _generation_input(
        "correct_image",
        image_path=tmp_path / "slide.png",
        visual_token_count=2,
    )
    text_item = _generation_input("audio_only")
    validate_visual_token_counts(
        [[7, 7, 0], [0, 0, 0]],
        image_token_id=7,
        batch=[image_item, text_item],
    )
    with pytest.raises(ValueError, match="visual token count differs"):
        validate_visual_token_counts(
            [[7, 0, 0], [0, 0, 0]],
            image_token_id=7,
            batch=[image_item, text_item],
        )


def test_qwen_generator_partitions_modalities_and_restores_order(tmp_path):
    batch = [
        _generation_input("audio_only"),
        _generation_input(
            "correct_image",
            image_path=tmp_path / "slide.png",
            visual_token_count=4,
        ),
        _generation_input("ocr"),
    ]
    generator = object.__new__(Qwen3OmniBatchGenerator)
    observed_groups = []

    def generate_group(group):
        observed_groups.append([item.condition for item in group])
        return [f"hypothesis:{item.condition}" for item in group]

    generator._generate_homogeneous = generate_group
    assert generator(batch) == [
        "hypothesis:audio_only",
        "hypothesis:correct_image",
        "hypothesis:ocr",
    ]
    assert observed_groups == [["audio_only", "ocr"], ["correct_image"]]


def test_causal_worker_delivers_resolved_image_to_generation(tmp_path):
    source_root = tmp_path / "source"
    packet, _, media_path = _image_packet(source_root)
    schedule = _single_image_schedule(tmp_path)
    release_path = tmp_path / "release.jsonl"
    socket_path = Path("/tmp") / f"slidesst-image-{os.getpid()}-{tmp_path.name[-6:]}.sock"
    broker = CausalAudioBroker(schedule, release_path)
    server = ThreadedUnixAudioServer(str(socket_path), broker)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    observed = []

    def generate(batch):
        observed.extend(batch)
        return ["译文" for _ in batch]

    try:
        trajectories = run_causal_event_worker(
            run_id=schedule.run_id,
            worker_id="worker-00-of-01",
            inference_contract_sha256="7" * 64,
            schedule=schedule,
            packets=[packet],
            selected_talk_ids=["talk-1"],
            broker_socket=socket_path,
            prompt_template="{evidence}\nTranslate only the speech.",
            generate_batch=generate,
            source_artifact_root=source_root,
        )
    finally:
        server.shutdown()
        server.server_close()
        broker.close()
        thread.join(timeout=2)
        socket_path.unlink(missing_ok=True)

    assert len(trajectories) == 1
    assert len(observed) == 1
    assert observed[0].evidence_image_path == media_path
    assert observed[0].expected_visual_token_count == packet.visual_token_count
