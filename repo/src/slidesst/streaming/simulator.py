from __future__ import annotations

from dataclasses import dataclass, field

from slidesst.data.schema import ChallengeItem, StreamingUnit


@dataclass
class StreamState:
    item_id: str
    t: float
    partial_transcript: str
    previous_translation: str = ""
    current_slide_id: str | None = None
    video_time_sec: float | None = None
    available_frame_ids: list[str] = field(default_factory=list)
    visual_window_start_sec: float | None = None
    visual_window_end_sec: float | None = None


class TranscriptOracleStreamer:
    """Reveal transcript prefixes over time.

    If `streaming_units` already exist in the data, use them. Otherwise create rough
    character-based chunks. This is not a real ASR model; it isolates context effects.
    """

    def __init__(self, chars_per_step: int = 12, visual_availability: str = "offline_context", allowed_lookahead_sec: float = 0.0, visual_window_sec: float = 2.0):
        self.chars_per_step = chars_per_step
        self.visual_availability = visual_availability
        self.allowed_lookahead_sec = allowed_lookahead_sec
        self.visual_window_sec = visual_window_sec

    def stream(self, item: ChallengeItem):
        units = item.streaming_units or self._make_units(item.source_transcript)
        for unit in units:
            visual_start, visual_end = self._visual_window(unit.t, item)
            yield StreamState(
                item_id=item.id,
                t=unit.t,
                partial_transcript=unit.partial_transcript,
                previous_translation="",
                current_slide_id=item.slides.matched_slide_id,
                video_time_sec=unit.t,
                available_frame_ids=self._available_frame_ids(item, visual_end),
                visual_window_start_sec=visual_start,
                visual_window_end_sec=visual_end,
            )

    def _make_units(self, transcript: str) -> list[StreamingUnit]:
        units = []
        for i in range(self.chars_per_step, len(transcript) + self.chars_per_step, self.chars_per_step):
            units.append(StreamingUnit(t=float(len(units) + 1), partial_transcript=transcript[:i]))
        return units

    def _visual_window(self, t: float, item: ChallengeItem) -> tuple[float | None, float | None]:
        if self.visual_availability == "offline_context":
            if item.video:
                return item.video.start_sec, item.video.end_sec
            return None, None
        end = t + self.allowed_lookahead_sec
        if self.visual_availability == "past_only":
            start = item.video.start_sec if item.video else 0.0
            return start, end
        return max(0.0, t - self.visual_window_sec), end

    def _available_frame_ids(self, item: ChallengeItem, visual_end: float | None) -> list[str]:
        if not item.visual_context:
            return []
        if self.visual_availability == "offline_context":
            return list(item.visual_context.frame_ids)
        return list(item.visual_context.frame_ids) if visual_end is not None else []
