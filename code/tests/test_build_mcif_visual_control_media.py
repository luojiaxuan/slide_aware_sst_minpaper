import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from scripts.build_mcif_visual_control_media import (
    BUNDLE_PATH_PREFIX,
    build_bundle,
    media_path_for,
    validate_control_inputs,
)
from scripts.build_mcif_visual_token_controls import (
    PAIRING_SEED,
    build_visual_token_inventory,
    build_wrong_image_candidates,
    canonical_sha256,
    file_sha256,
)


class FakeProcessor:
    image_token = "<image>"

    def __init__(self):
        self.tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda _: 99)

    def apply_chat_template(self, messages, **kwargs):
        return "template"

    def __call__(self, *, images, **kwargs):
        counts = [image.width * image.height // 8 for image in images]
        input_ids = np.zeros((len(images), max(counts) + 2), dtype=np.int64)
        grids = []
        for index, (image, count) in enumerate(zip(images, counts, strict=True)):
            input_ids[index, :count] = 99
            grids.append([1, image.height, image.width])
        return {
            "input_ids": input_ids,
            "image_grid_thw": np.asarray(grids, dtype=np.int64),
        }


def write_image(path: Path, *, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)


def ladder_row(
    source_root: Path,
    talk: str,
    state_id: int,
    *,
    size: tuple[int, int],
) -> dict:
    relative = f"native_causal_v1/frames/talks/{talk}/frames/state_{state_id:03d}.png"
    path = source_root / relative
    write_image(
        path,
        size=size,
        color=(state_id * 40, len(talk) * 20, size[0]),
    )
    row = {
        "schema_version": "mcif_source_evidence_ladder_v1",
        "id": f"mcif:{talk}:S{state_id:03d}",
        "lecture_id": talk,
        "state_id": state_id,
        "availability_start_sec": state_id + 0.5,
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


def inputs(source_root: Path, *, same_shape: bool = False):
    sizes = {"talk-1": (16, 8), "talk-2": (16, 8) if same_shape else (16, 16)}
    ladder = [
        ladder_row(source_root, talk, state_id, size=sizes[talk])
        for talk in ("talk-1", "talk-2")
        for state_id in range(2)
    ]
    processor = FakeProcessor()
    inventory = build_visual_token_inventory(
        ladder,
        source_root=source_root,
        processor=processor,
        processor_binding_sha256="1" * 64,
        batch_size=4,
    )
    controls = build_wrong_image_candidates(inventory, seed=PAIRING_SEED)
    processor_manifest = {
        "schema_version": "qwen3_omni_processor_manifest_v1",
        "model_id": "fixture/model",
        "model_revision": "2" * 40,
        "transformers_version": "fixture",
        "processor_binding_sha256": "1" * 64,
    }
    return processor, inventory, controls, processor_manifest


def materialize(tmp_path: Path, *, output_name: str, same_shape: bool = False):
    source_root = tmp_path / "source"
    processor, inventory, controls, processor_manifest = inputs(
        source_root,
        same_shape=same_shape,
    )
    output_root = tmp_path / output_name
    report = build_bundle(
        output_root,
        inventory=inventory,
        controls=controls,
        processor_manifest=processor_manifest,
        source_root=source_root,
        processor=processor,
        builder_git_commit="3" * 40,
        source_inventory_sha256="4" * 64,
        source_controls_sha256="5" * 64,
        source_processor_manifest_sha256="6" * 64,
        source_controls_hf_revision="7" * 40,
        batch_size=3,
        expected_rows=4,
        expected_talks=2,
    )
    return source_root, inventory, controls, processor_manifest, output_root, report


def test_materializes_only_transformed_controls_and_replays_final_media(tmp_path):
    _, _, _, _, output_root, report = materialize(tmp_path, output_name="bundle")
    assert report["rows"] == 4
    assert report["same_talk_stale_coverage"] == 2
    assert report["cross_talk_wrong_coverage"] == 4
    assert report["materialized_media_files"] == 4
    assert report["processor_audit"] == {
        "records_verified": 6,
        "bundle_media_verified": 4,
    }
    rows = [
        json.loads(line)
        for line in (output_root / "control_media_manifest.jsonl").read_text().splitlines()
    ]
    assert all(
        row["cross_talk_wrong"]["final_media"]["source_media_path"].startswith(
            f"{BUNDLE_PATH_PREFIX}/media/cross_talk_wrong/"
        )
        for row in rows
    )
    assert all(
        row["same_talk_stale"] is None
        or row["same_talk_stale"]["final_media"]["location"]
        == "canonical_native_source"
        for row in rows
    )
    for line in (output_root / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        assert file_sha256(output_root / relative) == expected


def test_natural_processor_shape_controls_reference_canonical_media(tmp_path):
    _, _, _, _, output_root, report = materialize(
        tmp_path,
        output_name="bundle",
        same_shape=True,
    )
    assert report["materialized_media_files"] == 0
    assert not (output_root / "media").exists()
    rows = [
        json.loads(line)
        for line in (output_root / "control_media_manifest.jsonl").read_text().splitlines()
    ]
    assert all(
        row["cross_talk_wrong"]["final_media"]["location"]
        == "canonical_native_source"
        for row in rows
    )


def test_independent_build_is_byte_identical(tmp_path):
    first = materialize(tmp_path / "first", output_name="bundle")[4]
    second = materialize(tmp_path / "second", output_name="bundle")[4]
    first_files = {
        path.relative_to(first): path.read_bytes()
        for path in first.rglob("*")
        if path.is_file()
    }
    second_files = {
        path.relative_to(second): path.read_bytes()
        for path in second.rglob("*")
        if path.is_file()
    }
    assert first_files == second_files


def test_validation_rejects_candidate_identity_drift(tmp_path):
    _, inventory, controls, processor_manifest = inputs(tmp_path / "source")
    controls[0]["cross_talk_wrong"]["source_media_sha256"] = "0" * 64
    controls[0]["row_sha256"] = canonical_sha256(
        {key: value for key, value in controls[0].items() if key != "row_sha256"}
    )
    with pytest.raises(ValueError, match="candidate identity drift"):
        validate_control_inputs(
            inventory,
            controls,
            processor_manifest,
            expected_rows=4,
            expected_talks=2,
        )


def test_validation_rejects_forbidden_data_or_row_drift(tmp_path):
    _, inventory, controls, processor_manifest = inputs(tmp_path / "source")
    inventory[0]["target_or_reference_consumed"] = True
    inventory[0]["row_sha256"] = canonical_sha256(
        {key: value for key, value in inventory[0].items() if key != "row_sha256"}
    )
    controls[0]["inventory_row_sha256"] = inventory[0]["row_sha256"]
    controls[0]["row_sha256"] = canonical_sha256(
        {key: value for key, value in controls[0].items() if key != "row_sha256"}
    )
    with pytest.raises(ValueError, match="forbidden data"):
        validate_control_inputs(
            inventory,
            controls,
            processor_manifest,
            expected_rows=4,
            expected_talks=2,
        )


def test_bundle_is_create_once(tmp_path):
    source_root, inventory, controls, processor_manifest, output_root, _ = materialize(
        tmp_path,
        output_name="bundle",
    )
    with pytest.raises(FileExistsError):
        build_bundle(
            output_root,
            inventory=inventory,
            controls=controls,
            processor_manifest=processor_manifest,
            source_root=source_root,
            processor=FakeProcessor(),
            builder_git_commit="3" * 40,
            source_inventory_sha256="4" * 64,
            source_controls_sha256="5" * 64,
            source_processor_manifest_sha256="6" * 64,
            source_controls_hf_revision="7" * 40,
            batch_size=3,
            expected_rows=4,
            expected_talks=2,
        )


def test_media_path_rejects_unsafe_source_identity():
    with pytest.raises(ValueError, match="lecture id"):
        media_path_for({"lecture_id": "../escape", "state_id": 0}, "cross_talk_wrong")
    with pytest.raises(ValueError, match="state id"):
        media_path_for({"lecture_id": "talk-1", "state_id": -1}, "cross_talk_wrong")


def test_failed_final_revalidation_leaves_no_bundle(tmp_path):
    source_root = tmp_path / "source"
    processor, inventory, controls, processor_manifest = inputs(source_root)
    output_root = tmp_path / "bundle"

    def reject() -> None:
        raise ValueError("frozen input drift")

    with pytest.raises(ValueError, match="frozen input drift"):
        build_bundle(
            output_root,
            inventory=inventory,
            controls=controls,
            processor_manifest=processor_manifest,
            source_root=source_root,
            processor=processor,
            builder_git_commit="3" * 40,
            source_inventory_sha256="4" * 64,
            source_controls_sha256="5" * 64,
            source_processor_manifest_sha256="6" * 64,
            source_controls_hf_revision="7" * 40,
            batch_size=3,
            expected_rows=4,
            expected_talks=2,
            frozen_input_revalidator=reject,
        )
    assert not output_root.exists()
