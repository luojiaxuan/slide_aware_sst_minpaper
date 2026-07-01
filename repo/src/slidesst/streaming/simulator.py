from __future__ import annotations

from dataclasses import dataclass

from slidesst.data.schema import ChallengeItem, StreamingUnit


@dataclass
class StreamState:
    item_id: str
    t: float
    partial_transcript: str
    previous_translation: str = ""
    current_slide_id: str | None = None


class TranscriptOracleStreamer:
    """Reveal transcript prefixes over time.

    If `streaming_units` already exist in the data, use them. Otherwise create rough
    character-based chunks. This is not a real ASR model; it isolates context effects.
    """

    def __init__(self, chars_per_step: int = 12):
        self.chars_per_step = chars_per_step

    def stream(self, item: ChallengeItem):
        units = item.streaming_units or self._make_units(item.source_transcript)
        for unit in units:
            yield StreamState(
                item_id=item.id,
                t=unit.t,
                partial_transcript=unit.partial_transcript,
                previous_translation="",
                current_slide_id=item.slides.matched_slide_id,
            )

    def _make_units(self, transcript: str) -> list[StreamingUnit]:
        units = []
        for i in range(self.chars_per_step, len(transcript) + self.chars_per_step, self.chars_per_step):
            units.append(StreamingUnit(t=float(len(units) + 1), partial_transcript=transcript[:i]))
        return units
