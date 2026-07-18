from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class VisualDescription:
    frame_id: str | None
    text: str
    objects: list[str]
    actions: list[str]
    spatial_relations: list[str]
    confidence: float = 1.0


class VLMDescriber(Protocol):
    def describe_frames(self, frame_paths: list[str], prompt: str) -> list[VisualDescription]: ...
    def describe_clip(self, frame_paths: list[str], prompt: str) -> VisualDescription: ...


class MockVLMDescriber:
    def __init__(self, descriptions_by_frame: dict[str, VisualDescription] | None = None):
        self.descriptions_by_frame = descriptions_by_frame or {}

    def describe_frames(self, frame_paths: list[str], prompt: str) -> list[VisualDescription]:
        descriptions = []
        for path in frame_paths:
            frame_id = _frame_id(path)
            descriptions.append(
                self.descriptions_by_frame.get(
                    frame_id,
                    VisualDescription(frame_id=frame_id, text="", objects=[], actions=[], spatial_relations=[], confidence=0.0),
                )
            )
        return descriptions

    def describe_clip(self, frame_paths: list[str], prompt: str) -> VisualDescription:
        frames = self.describe_frames(frame_paths, prompt)
        text = " ".join(desc.text for desc in frames if desc.text).strip()
        return VisualDescription(
            frame_id=None,
            text=text,
            objects=_unique(term for desc in frames for term in desc.objects),
            actions=_unique(term for desc in frames for term in desc.actions),
            spatial_relations=_unique(term for desc in frames for term in desc.spatial_relations),
            confidence=max([desc.confidence for desc in frames] or [0.0]),
        )


class EmptyVLMDescriber(MockVLMDescriber):
    pass


def build_vlm_describer(provider: str = "mock", descriptions_by_frame: dict[str, VisualDescription] | None = None) -> VLMDescriber:
    if provider == "mock":
        return MockVLMDescriber(descriptions_by_frame=descriptions_by_frame)
    return EmptyVLMDescriber()


def _frame_id(path: str) -> str:
    return path.rsplit("/", 1)[-1].rsplit(".", 1)[0]


def _unique(values) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out
