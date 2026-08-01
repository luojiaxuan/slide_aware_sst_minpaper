import hashlib
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest

from scripts.render_mcif_ppstructure_qa_sheets import resolve_frame


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory_row(item_id: str, stratum: str, frame_path: str, frame_sha: str) -> dict:
    return {
        "qa_stratum": stratum,
        "id": item_id,
        "frame": {
            "path": frame_path,
            "sha256": frame_sha,
            "width": 32,
            "height": 18,
        },
        "inference_fallback": None,
    }


def test_renderer_creates_one_hash_bound_sheet_per_stratum(tmp_path):
    frame_root = tmp_path / "frames"
    first = frame_root / "talks" / "a" / "frames" / "state_0.png"
    second = frame_root / "talks" / "b" / "frames" / "state_0.png"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    Image.new("RGB", (32, 18), "red").save(first)
    Image.new("RGB", (32, 18), "blue").save(second)
    rows = [
        inventory_row("a0", "chart", "talks/a/frames/state_0.png", sha256(first)),
        inventory_row("b0", "table", "talks/b/frames/state_0.png", sha256(second)),
    ]
    inventory = tmp_path / "inventory.jsonl"
    inventory.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    script = Path(__file__).parents[1] / "scripts" / "render_mcif_ppstructure_qa_sheets.py"
    output_dir = tmp_path / "sheets"
    manifest = tmp_path / "manifest.json"
    command = [
        sys.executable,
        str(script),
        "--inventory",
        str(inventory),
        "--frame-root",
        str(frame_root),
        "--output-dir",
        str(output_dir),
        "--manifest",
        str(manifest),
        "--expected-inventory-sha256",
        sha256(inventory),
    ]
    subprocess.run(command, check=True)
    result = json.loads(manifest.read_text(encoding="utf-8"))
    assert result["sheet_count"] == 2
    assert result["row_count"] == 2
    assert {path.name for path in output_dir.iterdir()} == {"chart.png", "table.png"}
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(command, check=True)


def test_resolve_frame_rejects_hash_drift(tmp_path):
    frame_root = tmp_path / "frames"
    frame = frame_root / "talks" / "a.png"
    frame.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), "white").save(frame)
    row = inventory_row("a", "chart", "talks/a.png", "wrong")
    with pytest.raises(ValueError, match="SHA256"):
        resolve_frame(frame_root, row)
