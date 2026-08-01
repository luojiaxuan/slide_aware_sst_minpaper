import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from scripts.build_mcif_visual_token_controls import (
    PAIRING_SEED,
    PROCESSOR_FILES,
    build_visual_token_inventory,
    build_wrong_image_candidates,
    canonical_sha256,
    file_sha256,
    processor_file_manifest,
    summarize,
    validate_ladder,
    verify_audio_invariance,
    verify_control_processing,
    verify_frozen_inputs,
    write_bundle,
)


class FakeProcessor:
    image_token = "<image>"

    def __init__(self):
        self.tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda _: 99)

    def apply_chat_template(self, messages, **kwargs):
        return "template"

    def __call__(self, *, images, **kwargs):
        counts = [image.width * image.height // 8 for image in images]
        width = max(counts) + 2
        input_ids = np.zeros((len(images), width), dtype=np.int64)
        grids = []
        for index, (image, count) in enumerate(zip(images, counts, strict=True)):
            input_ids[index, :count] = 99
            grids.append([1, image.height, image.width])
        return {
            "input_ids": input_ids,
            "image_grid_thw": np.asarray(grids, dtype=np.int64),
        }


def write_image(path: Path, *, size=(16, 8), color=(0, 0, 0)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)


def ladder_row(source_root: Path, talk: str, state_id: int, *, size=(16, 8)) -> dict:
    relative = f"native_causal_v1/frames/talks/{talk}/frames/state_{state_id:03d}.png"
    path = source_root / relative
    write_image(path, size=size, color=(state_id * 30, len(talk) * 10, 20))
    row = {
        "schema_version": "mcif_source_evidence_ladder_v1",
        "id": f"mcif:{talk}:S{state_id:03d}",
        "lecture_id": talk,
        "state_id": state_id,
        "availability_start_sec": state_id + 0.5,
        "availability_end_sec": state_id + 1.5,
        "evidence_timestamp_sec": state_id + 0.5,
        "r0_flat_ocr": {"model_input_text": "text"},
        "r1_structured_text": {"model_input_text": "structure"},
        "r2_raw_image": {
            "source_media_path": relative,
            "source_media_sha256": file_sha256(path),
            "width": size[0],
            "height": size[1],
        },
        "source_transcript_consumed": False,
        "target_or_reference_consumed": False,
    }
    row["row_sha256"] = canonical_sha256(row)
    return row


def sample_rows(source_root: Path) -> list[dict]:
    return [
        ladder_row(source_root, talk, state_id)
        for talk in ("talk-1", "talk-2")
        for state_id in range(2)
    ]


def test_visual_inventory_and_both_control_families_are_exact_token_matched(tmp_path):
    rows = sample_rows(tmp_path)
    validate_ladder(rows, expected_rows=4, expected_talks=2)
    processor = FakeProcessor()
    inventory = build_visual_token_inventory(
        rows,
        source_root=tmp_path,
        processor=processor,
        processor_binding_sha256="1" * 64,
        batch_size=3,
    )
    assert [row["visual_token_count"] for row in inventory] == [16, 16, 16, 16]
    assert all(row["image_grid_thw"] == [1, 8, 16] for row in inventory)
    audio_audit = verify_audio_invariance(
        inventory,
        source_root=tmp_path,
        processor=processor,
    )
    assert len(audio_audit) == 1

    controls = build_wrong_image_candidates(inventory, seed=PAIRING_SEED)
    assert controls[0]["same_talk_stale"] is None
    assert controls[1]["same_talk_stale"]["id"] == inventory[0]["id"]
    assert controls[2]["same_talk_stale"] is None
    assert controls[3]["same_talk_stale"]["id"] == inventory[2]["id"]
    for source, control in zip(inventory, controls, strict=True):
        cross = control["cross_talk_wrong"]
        assert cross["lecture_id"] != source["lecture_id"]
        assert cross["visual_token_count"] == source["visual_token_count"]
        assert control["cross_talk_match_level"] == "same_dimensions"
        assert cross["processor_input"]["mode"] == "identity"
    processing_audit = verify_control_processing(
        inventory,
        controls,
        source_root=tmp_path,
        processor=processor,
        batch_size=3,
    )
    assert processing_audit == {"records_verified": 6, "records_transformed": 0}
    report = summarize(inventory, controls, audio_audit, processing_audit)
    assert report["same_talk_stale_coverage"] == 2
    assert report["cross_talk_wrong_coverage"] == 4


def test_cross_talk_control_is_deterministic(tmp_path):
    rows = sample_rows(tmp_path)
    inventory = build_visual_token_inventory(
        rows,
        source_root=tmp_path,
        processor=FakeProcessor(),
        processor_binding_sha256="1" * 64,
        batch_size=4,
    )
    first = build_wrong_image_candidates(inventory, seed=PAIRING_SEED)
    second = build_wrong_image_candidates(inventory, seed=PAIRING_SEED)
    assert first == second


def test_cross_talk_control_falls_back_to_same_token_with_different_grid(tmp_path):
    rows = [
        ladder_row(tmp_path, "talk-1", 0, size=(16, 8)),
        ladder_row(tmp_path, "talk-2", 0, size=(8, 16)),
    ]
    inventory = build_visual_token_inventory(
        rows,
        source_root=tmp_path,
        processor=FakeProcessor(),
        processor_binding_sha256="1" * 64,
        batch_size=2,
    )
    controls = build_wrong_image_candidates(inventory, seed=PAIRING_SEED)
    assert all(
        row["cross_talk_match_level"] == "same_visual_token_count"
        for row in controls
    )
    assert all(
        row["cross_talk_wrong"]["visual_token_count"] == row["visual_token_count"]
        for row in controls
    )
    assert all(
        row["cross_talk_wrong"]["processor_input"]["mode"]
        == "fit_pad_to_source_canvas"
        for row in controls
    )
    audit = verify_control_processing(
        inventory,
        controls,
        source_root=tmp_path,
        processor=FakeProcessor(),
        batch_size=2,
    )
    assert audit == {"records_verified": 2, "records_transformed": 2}


def test_cross_talk_control_transforms_when_natural_token_bucket_is_singleton(tmp_path):
    rows = [
        ladder_row(tmp_path, "talk-1", 0, size=(16, 8)),
        ladder_row(tmp_path, "talk-2", 0, size=(16, 16)),
    ]
    inventory = build_visual_token_inventory(
        rows,
        source_root=tmp_path,
        processor=FakeProcessor(),
        processor_binding_sha256="1" * 64,
        batch_size=2,
    )
    controls = build_wrong_image_candidates(inventory, seed=PAIRING_SEED)
    assert all(
        row["cross_talk_match_level"] == "fit_pad_to_source_canvas"
        for row in controls
    )
    audit = verify_control_processing(
        inventory,
        controls,
        source_root=tmp_path,
        processor=FakeProcessor(),
        batch_size=2,
    )
    assert audit == {"records_verified": 2, "records_transformed": 2}


def test_control_builder_rejects_visual_token_group_with_only_one_talk(tmp_path):
    rows = [ladder_row(tmp_path, "talk-1", state_id) for state_id in range(2)]
    inventory = build_visual_token_inventory(
        rows,
        source_root=tmp_path,
        processor=FakeProcessor(),
        processor_binding_sha256="1" * 64,
        batch_size=2,
    )
    with pytest.raises(ValueError, match="No cross-talk wrong-image control"):
        build_wrong_image_candidates(inventory, seed=PAIRING_SEED)


@pytest.mark.parametrize("mutation", ["hash", "dimensions", "row_hash", "duplicate"])
def test_visual_inventory_fails_closed(tmp_path, mutation):
    rows = sample_rows(tmp_path)
    if mutation == "hash":
        rows[0]["r2_raw_image"]["source_media_sha256"] = "0" * 64
    elif mutation == "dimensions":
        rows[0]["r2_raw_image"]["width"] = 99
    elif mutation == "row_hash":
        rows[0]["row_sha256"] = "0" * 64
    elif mutation == "duplicate":
        rows.append(rows[0])
    if mutation in {"row_hash", "duplicate"}:
        with pytest.raises(ValueError):
            validate_ladder(rows, expected_rows=4, expected_talks=2)
    else:
        with pytest.raises(ValueError):
            build_visual_token_inventory(
                rows,
                source_root=tmp_path,
                processor=FakeProcessor(),
                processor_binding_sha256="1" * 64,
                batch_size=2,
            )


def test_visual_inventory_rejects_symlinked_image(tmp_path):
    rows = sample_rows(tmp_path)
    image_path = tmp_path / rows[0]["r2_raw_image"]["source_media_path"]
    outside = tmp_path / "outside.png"
    image_path.rename(outside)
    image_path.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        build_visual_token_inventory(
            rows,
            source_root=tmp_path,
            processor=FakeProcessor(),
            processor_binding_sha256="1" * 64,
            batch_size=2,
        )


def test_processor_manifest_binds_required_bytes(tmp_path):
    for index, name in enumerate(PROCESSOR_FILES):
        (tmp_path / name).write_text(f"file-{index}\n", encoding="utf-8")
    manifest = processor_file_manifest(tmp_path)
    assert len(manifest["required_files"]) == len(PROCESSOR_FILES)
    first = manifest["required_file_set_sha256"]
    (tmp_path / PROCESSOR_FILES[0]).write_text("drift\n", encoding="utf-8")
    assert processor_file_manifest(tmp_path)["required_file_set_sha256"] != first


def test_post_processing_revalidation_rejects_media_or_processor_drift(tmp_path):
    source_root = tmp_path / "source"
    rows = sample_rows(source_root)
    inventory = build_visual_token_inventory(
        rows,
        source_root=source_root,
        processor=FakeProcessor(),
        processor_binding_sha256="1" * 64,
        batch_size=4,
    )
    processor_root = tmp_path / "processor"
    processor_root.mkdir()
    for index, name in enumerate(PROCESSOR_FILES):
        (processor_root / name).write_text(f"file-{index}\n", encoding="utf-8")
    manifest = processor_file_manifest(processor_root)
    verify_frozen_inputs(
        inventory,
        source_root=source_root,
        processor_root=processor_root,
        expected_processor_files=manifest,
    )
    (source_root / inventory[0]["source_media_path"]).write_bytes(b"drift")
    with pytest.raises(ValueError, match="bytes changed after processing"):
        verify_frozen_inputs(
            inventory,
            source_root=source_root,
            processor_root=processor_root,
            expected_processor_files=manifest,
        )


def test_write_bundle_is_create_once_and_checksum_bound(tmp_path):
    source_root = tmp_path / "source"
    rows = sample_rows(source_root)
    processor = FakeProcessor()
    inventory = build_visual_token_inventory(
        rows,
        source_root=source_root,
        processor=processor,
        processor_binding_sha256="1" * 64,
        batch_size=4,
    )
    audio_audit = verify_audio_invariance(
        inventory,
        source_root=source_root,
        processor=processor,
    )
    controls = build_wrong_image_candidates(inventory, seed=PAIRING_SEED)
    output_root = tmp_path / "bundle"
    final = write_bundle(
        output_root,
        inventory=inventory,
        controls=controls,
        processor_manifest={"processor": "fixture"},
        report={
            "model_id": "fixture/model",
            "model_revision": "2" * 40,
            **summarize(
                inventory,
                controls,
                audio_audit,
                {"records_verified": 6, "records_transformed": 0},
            ),
        },
    )
    assert final["checksum_entries"] == 5
    assert len(json.loads((output_root / "report.json").read_text())) > 5
    for line in (output_root / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        assert file_sha256(output_root / relative) == expected
    with pytest.raises(FileExistsError):
        write_bundle(
            output_root,
            inventory=inventory,
            controls=controls,
            processor_manifest={"processor": "fixture"},
            report={"model_id": "fixture/model", "model_revision": "2" * 40},
        )
