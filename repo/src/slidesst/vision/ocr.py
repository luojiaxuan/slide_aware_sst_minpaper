from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class OCREntry:
    frame_id: str
    text: str
    confidence: float = 1.0


class OCRExtractor(Protocol):
    def extract(self, frame_paths: list[str]) -> list[OCREntry]: ...


class MockOCRExtractor:
    def __init__(self, text_by_frame: dict[str, str] | None = None):
        self.text_by_frame = text_by_frame or {}

    def extract(self, frame_paths: list[str]) -> list[OCREntry]:
        entries = []
        for path in frame_paths:
            frame_id = _frame_id(path)
            text = self.text_by_frame.get(frame_id, "")
            if text:
                entries.append(OCREntry(frame_id=frame_id, text=text, confidence=0.99))
        return entries


class EmptyOCRExtractor:
    def extract(self, frame_paths: list[str]) -> list[OCREntry]:
        return []


def build_ocr_extractor(provider: str = "mock", text_by_frame: dict[str, str] | None = None) -> OCRExtractor:
    if provider == "mock":
        return MockOCRExtractor(text_by_frame=text_by_frame)
    return EmptyOCRExtractor()


def _frame_id(path: str) -> str:
    return path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
