#!/usr/bin/env python3
"""Render hash-bound MCIF PP-Structure QA contact sheets."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def resolve_frame(frame_root: Path, row: dict[str, Any]) -> Path:
    frame = row.get("frame") or {}
    relative = Path(frame.get("path") or "")
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Invalid portable frame path for {row.get('id')}")
    root = frame_root.resolve(strict=True)
    path = frame_root / relative
    cursor = frame_root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"Frame path traverses a symlink for {row.get('id')}")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError(f"Frame path escapes the root for {row.get('id')}")
    if sha256_file(resolved) != frame.get("sha256"):
        raise ValueError(f"Frame SHA256 changed for {row.get('id')}")
    return resolved


def render_sheet(
    rows: list[dict[str, Any]],
    frame_root: Path,
    output: Path,
    *,
    columns: int,
    image_size: tuple[int, int],
) -> None:
    label_height = 44
    cell_width, cell_height = image_size
    row_count = (len(rows) + columns - 1) // columns
    sheet = Image.new(
        "RGB", (columns * cell_width, row_count * (cell_height + label_height)), "white"
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=16)
    for index, row in enumerate(rows):
        x = (index % columns) * cell_width
        y = (index // columns) * (cell_height + label_height)
        with Image.open(resolve_frame(frame_root, row)) as source:
            image = ImageOps.contain(source.convert("RGB"), image_size)
        image_x = x + (cell_width - image.width) // 2
        image_y = y + (cell_height - image.height) // 2
        sheet.paste(image, (image_x, image_y))
        draw.rectangle((x, y, x + cell_width - 1, y + cell_height - 1), outline="#666666")
        draw.text((x + 6, y + cell_height + 4), row["id"], fill="black", font=font)
        fallback = row.get("inference_fallback")
        if fallback:
            draw.text(
                (x + 6, y + cell_height + 23),
                f"fallback: {fallback['strategy']}",
                fill="#9a3412",
                font=font,
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="PNG", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-inventory-sha256", required=True)
    parser.add_argument("--columns", type=int, default=4)
    args = parser.parse_args()
    if args.output_dir.exists() or args.manifest.exists():
        raise FileExistsError("PP-Structure QA sheets must be created once")
    if args.columns < 1:
        raise ValueError("Contact-sheet column count must be positive")
    inventory_sha256 = sha256_file(args.inventory)
    if inventory_sha256 != args.expected_inventory_sha256:
        raise ValueError("QA inventory SHA256 differs from the frozen input")

    rows = load_jsonl(args.inventory)
    keys = [(row.get("qa_stratum"), row.get("id")) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("QA inventory contains duplicate stratum/id pairs")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        stratum = row.get("qa_stratum")
        if not isinstance(stratum, str) or not stratum:
            raise ValueError("QA inventory row is missing a stratum")
        grouped[stratum].append(row)

    sheets = []
    for stratum, stratum_rows in sorted(grouped.items()):
        output = args.output_dir / f"{stratum}.png"
        render_sheet(
            stratum_rows,
            args.frame_root,
            output,
            columns=args.columns,
            image_size=(480, 270),
        )
        sheets.append(
            {
                "qa_stratum": stratum,
                "rows": len(stratum_rows),
                "path": output.name,
                "sha256": sha256_file(output),
            }
        )
    manifest = {
        "artifact": "mcif_ppstructurev3_source_screen_qa_sheets_v1",
        "status": "SOURCE_ONLY_AUTOMATIC_QA_NOT_ANNOTATION",
        "inventory_sha256": inventory_sha256,
        "sheet_count": len(sheets),
        "row_count": len(rows),
        "sheets": sheets,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
