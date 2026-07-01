from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from slidesst.data.io import read_jsonl
from slidesst.data.schema import (
    AudioSpan,
    BackgroundDoc,
    ChallengeItem,
    GlossaryEntry,
    SlideInfo,
)


def load_challenge_items_from_config(cfg: dict) -> list[ChallengeItem]:
    dataset_cfg = cfg.get("dataset", {})
    paths = cfg.get("paths", {})
    raw_root = Path(paths.get("raw_data_root", ".")).expanduser()
    metadata_glob = dataset_cfg.get("metadata_glob")

    if metadata_glob:
        files = sorted(raw_root.glob(metadata_glob))
        if files:
            records: list[dict[str, Any]] = []
            for path in files:
                records.extend(load_metadata_records(path))
            return [
                record_to_challenge_item(record, raw_root, dataset_cfg, idx)
                for idx, record in enumerate(records)
            ]

    if dataset_cfg.get("allow_toy_fallback", True):
        toy_path = Path(dataset_cfg.get("toy_challenge_path", "examples/toy_challenge.jsonl"))
        if not toy_path.is_absolute():
            toy_path = Path.cwd() / toy_path
        if toy_path.exists():
            return read_jsonl(toy_path, ChallengeItem)

    raise FileNotFoundError(
        f"No metadata files found under {raw_root} with glob={metadata_glob!r}; "
        "set dataset.metadata_glob or enable dataset.allow_toy_fallback."
    )


def load_metadata_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        for key in ("items", "segments", "data", "records"):
            if isinstance(data.get(key), list):
                return data[key]
        return [data]
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported metadata file type: {path}")


def record_to_challenge_item(record: dict[str, Any], raw_root: Path, dataset_cfg: dict, idx: int) -> ChallengeItem:
    mapping = dataset_cfg.get("field_mappings", {})

    lecture_id = str(_field(record, mapping.get("lecture_id"), ("lecture_id", "video_id", "talk_id", "course_id")) or "lecture")
    item_id = str(_field(record, mapping.get("id"), ("id", "segment_id", "clip_id")) or f"{lecture_id}_seg{idx:06d}")
    transcript = str(_field(record, mapping.get("source_transcript"), ("source_transcript", "transcript", "text", "sentence", "asr_text")) or "")
    reference = _field(record, mapping.get("reference_translation"), ("reference_translation", "translation_en", "target_text", "en"))

    start_sec = _float_or_none(_field(record, mapping.get("start_sec"), ("start_sec", "start", "begin", "begin_sec")))
    end_sec = _float_or_none(_field(record, mapping.get("end_sec"), ("end_sec", "end", "stop", "end_time")))
    audio_path = _field(record, mapping.get("audio_path"), ("audio_path", "wav", "audio", "path"))
    audio = None
    if audio_path or start_sec is not None or end_sec is not None:
        audio = AudioSpan(
            path=_resolve_path(audio_path, raw_root) if audio_path else None,
            start_sec=start_sec or 0.0,
            end_sec=end_sec or 0.0,
        )

    slide_id = _field(record, mapping.get("slide_id"), ("slide_id", "page_id", "matched_slide_id"))
    slide_image = _field(record, mapping.get("slide_image"), ("slide_image", "slide_path", "frame_path", "slide_frame"))
    slides = SlideInfo(
        matched_slide_id=str(slide_id) if slide_id else None,
        previous_slide_id=_string_or_none(_field(record, mapping.get("previous_slide_id"), ("previous_slide_id", "prev_slide_id"))),
        next_slide_id=_string_or_none(_field(record, mapping.get("next_slide_id"), ("next_slide_id", "next_slide"))),
        matched_slide_text=_string_or_none(_field(record, mapping.get("slide_text"), ("slide_text", "slide_ocr", "ocr", "ocr_text"))),
        matched_slide_image=_resolve_path(slide_image, raw_root) if slide_image else None,
    )

    return ChallengeItem(
        id=item_id,
        lecture_id=lecture_id,
        source_lang=dataset_cfg.get("source_lang", "zh"),
        target_lang=dataset_cfg.get("target_lang", "en"),
        audio=audio,
        source_transcript=transcript,
        reference_translation=str(reference) if reference else None,
        slides=slides,
        glossary=_parse_glossary(_field(record, mapping.get("glossary"), ("glossary", "terms"))),
        background_docs=_parse_background(_field(record, mapping.get("background_docs"), ("background_docs", "background", "abstract"))),
    )


def _field(record: dict[str, Any], configured: Any, fallbacks: tuple[str, ...]) -> Any:
    candidates: list[str] = []
    if isinstance(configured, str):
        candidates.append(configured)
    elif isinstance(configured, list):
        candidates.extend(str(x) for x in configured)
    candidates.extend(fallbacks)
    for key in candidates:
        value = _get_path(record, key)
        if value not in (None, ""):
            return value
    return None


def _get_path(record: dict[str, Any], path: str) -> Any:
    cur: Any = record
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _resolve_path(value: Any, raw_root: Path) -> str:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = raw_root / path
    return str(path)


def _string_or_none(value: Any) -> str | None:
    return str(value) if value not in (None, "") else None


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _parse_glossary(value: Any) -> list[GlossaryEntry]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            entries = []
            for chunk in value.split(";"):
                if not chunk.strip():
                    continue
                if "=>" in chunk:
                    src, tgt = chunk.split("=>", 1)
                elif ":" in chunk:
                    src, tgt = chunk.split(":", 1)
                else:
                    continue
                entries.append(GlossaryEntry(src=src.strip(), tgt=tgt.strip()))
            return entries
    if isinstance(value, dict):
        value = [value]
    entries: list[GlossaryEntry] = []
    for item in value:
        if isinstance(item, dict):
            src = item.get("src") or item.get("source") or item.get("zh") or item.get("term")
            tgt = item.get("tgt") or item.get("target") or item.get("en") or item.get("translation")
            if src and tgt:
                entries.append(GlossaryEntry(src=str(src), tgt=str(tgt), desc=item.get("desc"), source=item.get("source", "metadata")))
    return entries


def _parse_background(value: Any) -> list[BackgroundDoc]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return [BackgroundDoc(doc_id="background", text=value)]
    if isinstance(value, dict):
        value = [value]
    docs: list[BackgroundDoc] = []
    for idx, item in enumerate(value):
        if isinstance(item, dict):
            text = item.get("text") or item.get("content") or item.get("abstract")
            if text:
                docs.append(BackgroundDoc(doc_id=str(item.get("doc_id", f"background_{idx}")), text=str(text)))
        elif item:
            docs.append(BackgroundDoc(doc_id=f"background_{idx}", text=str(item)))
    return docs
