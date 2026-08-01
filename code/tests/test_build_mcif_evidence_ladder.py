import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest

from scripts.build_mcif_evidence_ladder import (
    build_ladder,
    canonical_sha256,
    file_sha256,
    write_bundle,
)


def write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 8), color=color).save(path)


def native_row(root: Path, state_id: int) -> dict:
    relative = f"talks/talk-1/frames/state_{state_id:03d}.png"
    frame = root / relative
    write_image(frame, (state_id * 20, 10, 30))
    return {
        "id": f"mcif:talk-1:S{state_id:03d}",
        "lecture_id": "talk-1",
        "state_id": state_id,
        "availability_start_sec": state_id + 0.5,
        "availability_end_sec": state_id + 1.5,
        "evidence_timestamp_sec": state_id + 0.5,
        "frame_path": relative,
        "frame_sha256": file_sha256(frame),
        "frame_width": 16,
        "frame_height": 8,
        "source_transcript_consumed": False,
        "target_or_reference_consumed": False,
    }


def ppstructure_row(native: dict, native_manifest_sha256: str) -> dict:
    state_id = native["state_id"]
    blocks = [
        {
            "block_id": 0,
            "provider_index": 0,
            "reading_order": 1,
            "label": "doc_title",
            "bbox_norm": [0.1, 0.1, 0.9, 0.2],
            "content": "# Slide title",
        },
        {
            "block_id": 1,
            "provider_index": 1,
            "reading_order": None,
            "label": "image",
            "bbox_norm": [0.1, 0.3, 0.4, 0.8],
            "content": '<img src="private/crop.jpg" />',
        },
    ]
    if state_id == 1:
        blocks.append(
            {
                "block_id": 2,
                "provider_index": 2,
                "reading_order": 2,
                "label": "chart",
                "bbox_norm": [0.5, 0.3, 0.9, 0.8],
                "content": "| x | y |\n|---|---|\n|a|1|",
            }
        )
    items = [{"text": f"Text {state_id}", "bbox_norm": [0.1, 0.1, 0.2, 0.2]}]
    return {
        "id": native["id"],
        "lecture_id": native["lecture_id"],
        "state_id": native["state_id"],
        "availability_start_sec": native["availability_start_sec"],
        "availability_end_sec": native["availability_end_sec"],
        "evidence_timestamp_sec": native["evidence_timestamp_sec"],
        "frame": {
            "path": native["frame_path"],
            "sha256": native["frame_sha256"],
            "width": native["frame_width"],
            "height": native["frame_height"],
        },
        "flat_ocr": {
            "item_count": len(items),
            "items": items,
            "text": f"Text {state_id}",
            "ordering": "paddleocr_provider_order",
        },
        "structured_text": {
            "block_count": len(blocks),
            "blocks": blocks,
            "ordering": "ppstructurev3_reading_order_then_provider_order",
        },
        "provenance": {
            "provider": "PaddleOCR.PPStructureV3",
            "input_manifest_sha256": native_manifest_sha256,
        },
        "source_transcript_consumed": False,
        "target_or_reference_consumed": False,
    }


def provenance(native_sha: str, pp_sha: str) -> tuple[dict, dict]:
    native = {
        "inventory": {
            "source_transcript_consumed": False,
            "target_or_reference_consumed": False,
        },
        "quality_audit": {"manifest_sha256": native_sha},
        "hugging_face": {
            "repo": "owner/private-mcif",
            "repo_type": "dataset",
            "path": "native_causal_v1",
            "private_verified": True,
            "revision": "1" * 40,
        },
        "upstream": {
            "revision": "2" * 40,
            "license": "cc-by-4.0",
        },
    }
    ppstructure = {
        "inventory": {
            "source_transcript_consumed": False,
            "target_or_reference_consumed": False,
        },
        "checksums": {"output_sha256": pp_sha},
        "hugging_face": {
            "repo": "owner/private-mcif",
            "repo_type": "dataset",
            "path": "ppstructurev3_source_screen_v1",
            "private_verified": True,
            "revision": "3" * 40,
        },
        "upstream": {
            "revision": "2" * 40,
            "native_evidence_manifest_sha256": native_sha,
        },
    }
    return native, ppstructure


def build_fixture(tmp_path: Path):
    native_root = tmp_path / "native"
    native_rows = [native_row(native_root, state_id) for state_id in range(2)]
    native_sha = "a" * 64
    pp_rows = [ppstructure_row(row, native_sha) for row in native_rows]
    pp_sha = "b" * 64
    native_provenance, pp_provenance = provenance(native_sha, pp_sha)
    kwargs = {
        "native_root": native_root,
        "native_provenance": native_provenance,
        "ppstructure_provenance": pp_provenance,
        "native_manifest_sha256": native_sha,
        "ppstructure_output_sha256": pp_sha,
        "native_provenance_sha256": "c" * 64,
        "ppstructure_provenance_sha256": "d" * 64,
        "builder_git_commit": "e" * 40,
        "expected_rows": 2,
        "expected_talks": 1,
    }
    return native_rows, pp_rows, kwargs


def test_build_ladder_preserves_matched_r0_r1_r2_and_sanitizes_images(tmp_path):
    native_rows, pp_rows, kwargs = build_fixture(tmp_path)
    rows, report = build_ladder(native_rows, pp_rows, **kwargs)

    assert [row["id"] for row in rows] == [row["id"] for row in native_rows]
    assert rows[0]["r0_flat_ocr"]["model_input_text"] == "Text 0"
    assert "bbox_norm" not in rows[0]["r0_flat_ocr"]
    assert rows[0]["r1_structured_text"]["blocks"][1]["content"] == (
        "[non-text visual region]"
    )
    assert "private/crop.jpg" not in rows[0]["r1_structured_text"]["model_input_text"]
    assert rows[1]["r1_structured_text"]["blocks"][1]["content_kind"] == (
        "chart_markdown"
    )
    assert rows[0]["r2_raw_image"]["source_media_path"] == (
        "native_causal_v1/frames/talks/talk-1/frames/state_000.png"
    )
    assert all(row["row_sha256"] == canonical_sha256({
        key: value for key, value in row.items() if key != "row_sha256"
    }) for row in rows)
    assert report["rows"] == 2
    assert report["r1_content_kind_counts"] == {
        "chart_markdown": 1,
        "text": 2,
        "visual_placeholder": 2,
    }
    assert report["target_or_reference_consumed"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "missing",
        "timing",
        "frame_binding",
        "frame_bytes",
        "reference",
        "native_provenance",
        "pp_provenance",
        "state_gap",
    ],
)
def test_build_ladder_fails_closed(tmp_path, mutation):
    native_rows, pp_rows, kwargs = build_fixture(tmp_path)
    if mutation == "duplicate":
        pp_rows.append(copy.deepcopy(pp_rows[0]))
    elif mutation == "missing":
        pp_rows.pop()
    elif mutation == "timing":
        pp_rows[0]["availability_start_sec"] = 0.0
    elif mutation == "frame_binding":
        pp_rows[0]["frame"]["sha256"] = "wrong"
    elif mutation == "frame_bytes":
        (kwargs["native_root"] / native_rows[0]["frame_path"]).write_bytes(b"drift")
    elif mutation == "reference":
        pp_rows[0]["reference_translation"] = "target"
    elif mutation == "native_provenance":
        kwargs["native_provenance"]["quality_audit"]["manifest_sha256"] = "0" * 64
    elif mutation == "pp_provenance":
        kwargs["ppstructure_provenance"]["checksums"]["output_sha256"] = "0" * 64
    elif mutation == "state_gap":
        native_rows[1]["state_id"] = 2
    with pytest.raises(ValueError):
        build_ladder(native_rows, pp_rows, **kwargs)


def test_build_ladder_rejects_symlinked_frame(tmp_path):
    native_rows, pp_rows, kwargs = build_fixture(tmp_path)
    real_frame = kwargs["native_root"] / native_rows[0]["frame_path"]
    target = tmp_path / "outside.png"
    real_frame.rename(target)
    real_frame.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        build_ladder(native_rows, pp_rows, **kwargs)


def test_write_bundle_is_create_once_and_checksum_bound(tmp_path):
    native_rows, pp_rows, kwargs = build_fixture(tmp_path)
    rows, report = build_ladder(native_rows, pp_rows, **kwargs)
    native_provenance_path = tmp_path / "native-provenance.json"
    pp_provenance_path = tmp_path / "pp-provenance.json"
    native_provenance_path.write_text(
        json.dumps(kwargs["native_provenance"]) + "\n", encoding="utf-8"
    )
    pp_provenance_path.write_text(
        json.dumps(kwargs["ppstructure_provenance"]) + "\n", encoding="utf-8"
    )
    output_root = tmp_path / "bundle"
    final_report = write_bundle(
        output_root,
        rows=rows,
        report=report,
        native_provenance_path=native_provenance_path,
        ppstructure_provenance_path=pp_provenance_path,
    )
    assert final_report["checksum_entries"] == 5
    for line in (output_root / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        assert file_sha256(output_root / relative) == expected
    with pytest.raises(FileExistsError):
        write_bundle(
            output_root,
            rows=rows,
            report=report,
            native_provenance_path=native_provenance_path,
            ppstructure_provenance_path=pp_provenance_path,
        )


def test_cli_binds_clean_git_commit_and_input_hashes(tmp_path):
    native_root = tmp_path / "native"
    native_rows = [native_row(native_root, state_id) for state_id in range(2)]
    native_manifest = tmp_path / "native.jsonl"
    native_manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in native_rows),
        encoding="utf-8",
    )
    native_sha = file_sha256(native_manifest)
    pp_rows = [ppstructure_row(row, native_sha) for row in native_rows]
    pp_output = tmp_path / "pp.jsonl"
    pp_output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in pp_rows),
        encoding="utf-8",
    )
    native_provenance, pp_provenance = provenance(native_sha, file_sha256(pp_output))
    native_provenance_path = tmp_path / "native-provenance.json"
    pp_provenance_path = tmp_path / "pp-provenance.json"
    native_provenance_path.write_text(json.dumps(native_provenance), encoding="utf-8")
    pp_provenance_path.write_text(json.dumps(pp_provenance), encoding="utf-8")

    code_repo = tmp_path / "repo"
    code_repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=code_repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=code_repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=code_repo, check=True)
    (code_repo / "README").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "add", "README"], cwd=code_repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "fixture"], cwd=code_repo, check=True)

    output_root = tmp_path / "cli-bundle"
    script = Path(__file__).parents[1] / "scripts" / "build_mcif_evidence_ladder.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--native-manifest",
            str(native_manifest),
            "--native-root",
            str(native_root),
            "--native-provenance",
            str(native_provenance_path),
            "--ppstructure-output",
            str(pp_output),
            "--ppstructure-provenance",
            str(pp_provenance_path),
            "--code-repo",
            str(code_repo),
            "--output-root",
            str(output_root),
            "--expected-rows",
            "2",
            "--expected-talks",
            "1",
        ],
        check=True,
    )
    report = json.loads((output_root / "report.json").read_text(encoding="utf-8"))
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=code_repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert report["builder_git_commit"] == commit
    assert report["source_binding"]["native_manifest_sha256"] == native_sha
