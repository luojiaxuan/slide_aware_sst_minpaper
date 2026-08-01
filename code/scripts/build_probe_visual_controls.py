#!/usr/bin/env python3
"""Attach deterministic cross-talk and blank-image controls to probe items."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_media(row: dict, input_root: Path) -> dict:
    output = dict(row)
    for key in ("audio", "slide_image", "wrong_image"):
        path = Path(output[key])
        output[key] = str(path if path.is_absolute() else input_root / path)
    return output


def build_controls(
    rows: list[dict], input_root: Path, cross_image_root: Path, output_root: Path
) -> tuple[list[dict], dict]:
    cross_images = sorted(cross_image_root.rglob("*.jpg"))
    if not cross_images:
        raise ValueError("No cross-talk control images found")
    output_root.mkdir(parents=True, exist_ok=True)
    blank_path = output_root / "blank_control.png"
    Image.new("RGB", (1280, 720), color=(255, 255, 255)).save(blank_path)
    output = []
    assignments = []
    for row in rows:
        resolved = resolve_media(row, input_root)
        index = int.from_bytes(
            hashlib.sha256(f'cross-talk-control-v1:{row["id"]}'.encode()).digest()[:8],
            "big",
        ) % len(cross_images)
        cross_image = cross_images[index]
        resolved.update(
            {
                "cross_talk_image": str(cross_image),
                "blank_image": str(blank_path),
            }
        )
        output.append(resolved)
        assignments.append(
            {
                "id": row["id"],
                "cross_talk_image": str(cross_image),
                "cross_talk_image_sha256": sha256_file(cross_image),
            }
        )
    manifest = {
        "schema_version": "speech_vision_probe_visual_controls_v1",
        "item_count": len(output),
        "cross_talk_image_pool_count": len(cross_images),
        "assignment_salt": "cross-talk-control-v1",
        "blank_image": str(blank_path),
        "blank_image_sha256": sha256_file(blank_path),
        "conditions": ["cross_talk", "blank"],
        "assignments": assignments,
    }
    return output, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--cross-image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.items.read_text(encoding="utf-8").splitlines()
        if line
    ]
    output, manifest = build_controls(
        rows, args.input_root, args.cross_image_root, args.output_root
    )
    item_path = args.output_root / "items_visual_controls_v1.jsonl"
    item_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output),
        encoding="utf-8",
    )
    manifest["items_sha256"] = sha256_file(item_path)
    (args.output_root / "visual_controls_manifest_v1.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"item_count": len(output), "items": str(item_path)}))


if __name__ == "__main__":
    main()
