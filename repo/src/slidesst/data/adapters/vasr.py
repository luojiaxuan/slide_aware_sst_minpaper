from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from slidesst.data.schema import AudioSpan, ChallengeItem, VideoSpan, VisualContext


def load_vasr_manifest(path: str | Path, *, validate_paths: bool = False) -> list[ChallengeItem]:
    manifest = Path(path).expanduser()
    rows = _read_rows(manifest)
    root = manifest.parent
    return [row_to_item(row, root=root, idx=idx, validate_paths=validate_paths) for idx, row in enumerate(rows)]


def row_to_item(row: dict[str, Any], *, root: Path, idx: int, validate_paths: bool = False) -> ChallengeItem:
    clip_id = str(row.get("id") or row.get("clip_id") or f"clip_{idx:06d}")
    video_id = str(row.get("video_id") or row.get("lecture_id") or row.get("video_path") or "video")
    start_sec = float(row.get("start_sec", 0.0) or 0.0)
    end_sec = float(row.get("end_sec", start_sec) or start_sec)
    video_path = _resolve(row.get("video_path"), root)
    audio_path = _resolve(row.get("audio_path"), root)
    frame_paths = [_resolve(path, root) for path in row.get("frame_paths", [])]

    if validate_paths:
        for path in [video_path, audio_path, *frame_paths]:
            if path and not Path(path).exists():
                raise FileNotFoundError(path)

    transcript = row.get("zh_transcript") or row.get("source_transcript") or row.get("transcript") or ""
    metadata = row.get("metadata") or {}
    visual = row.get("visual_context") or {}
    visual_context = VisualContext(
        video_id=video_id,
        clip_id=clip_id,
        scene_summary=visual.get("scene_summary") or row.get("scene_summary"),
        ocr_text=list(visual.get("ocr_text") or row.get("ocr_text") or []),
        objects=list(visual.get("objects") or row.get("objects") or []),
        actions=list(visual.get("actions") or row.get("actions") or []),
        spatial_relations=list(visual.get("spatial_relations") or row.get("spatial_relations") or []),
        frame_ids=list(visual.get("frame_ids") or row.get("frame_ids") or []),
        metadata=metadata,
    )
    return ChallengeItem(
        id=clip_id,
        lecture_id=video_id,
        source_lang="zh",
        target_lang="en",
        audio=AudioSpan(path=audio_path, start_sec=start_sec, end_sec=end_sec) if audio_path else None,
        video=VideoSpan(path=video_path, start_sec=start_sec, end_sec=end_sec, frame_paths=frame_paths, fps=_float_or_none(row.get("fps"))),
        source_transcript=str(transcript),
        visual_context=visual_context,
    )


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    for key in ("clips", "items", "data", "records"):
        if isinstance(data.get(key), list):
            return data[key]
    return [data]


def _resolve(value: Any, root: Path) -> str | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = root / path
    return str(path)


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)
