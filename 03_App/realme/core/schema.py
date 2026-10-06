"""Data contracts. Everything crossing a subsystem boundary is validated here."""
from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator

Layout = Literal["slide_only", "pip", "side_by_side"]


class Prosody(BaseModel):
    pace: float = Field(1.0, ge=0.5, le=2.0)
    #: Silence after the slide appears, before the first word. Without it the
    #: slide cut and the voice onset land on the same frame and it reads as a
    #: burst. Three seconds also gives a viewer time to take the slide in.
    lead_in_s: float = Field(3.0, ge=0.0, le=60.0)
    #: Silence after the last word, before the slide changes. Speech ending and
    #: the slide flipping at the same instant feels rushed; the pause is where
    #: a real lecturer lets a point land. Override per slide, or with
    #: [[pause:400ms]] inside the narration for finer control.
    pause_after_s: float = Field(3.0, ge=0.0, le=60.0)


class Segment(BaseModel):
    segment_id: int
    slide_index: int
    spoken_text: str
    layout: Layout = "slide_only"
    prosody: Prosody = Prosody()
    # Filled in ONLY after synthesis, from a real ffprobe measurement.
    measured_audio_s: Optional[float] = None
    #: (offset_from_speech_start, duration, text) per synthesized utterance.
    #: Captions are built from THIS, not from spoken_text, so that control
    #: markers never appear on screen and masked words never leak into the
    #: subtitles — masking the audio while captioning the words would defeat
    #: the whole point of it.
    spoken_cues: list[tuple[float, float, str]] = []
    #: Phrases the AUTHOR wants pointed at on this slide, in their own words.
    #: Empty means "derive them", which is what always happened before. A cue
    #: needs a place on the slide and a moment in the narration; these are
    #: checked when the deck renders, and a phrase that fails either is
    #: reported rather than silently dropped.
    cues: list[str] = []
    #: Content hash of the slide image this was rendered against. Recorded so
    #: a later version of the deck can be compared with this one, and so the
    #: video cache can tell "same words, different picture" from "same slide".
    slide_digest: str = ""
    #: How this slide compares with the version last rendered: same, moved,
    #: edited or added. Set by a revision, shown in the editor, and cleared by
    #: the next render.
    revision: str = ""
    #: Whether this slide's frame should be made again. A revision sets it
    #: from what actually differs; the editor lets a person overrule that,
    #: because "the logo moved" and "the figure is wrong" look identical to a
    #: comparison and not at all alike to the person who made the change.
    #:
    #: Unticked with a recorded `slide_digest`, the render deliberately keys
    #: the cache on the OLD picture, so the previous frame and its audio come
    #: back untouched.
    rerender: bool = True

    @field_validator("spoken_text")
    @classmethod
    def _nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("spoken_text must not be empty")
        return v


class Manifest(BaseModel):
    schema_version: Literal["2.0"] = "2.0"
    project_id: str
    title: str
    mode: Literal["lecture", "podcast", "debate"] = "lecture"
    # Provenance: which model wrote this script, or whether a human/placeholder did.
    script_source: str = "unknown"
    segments: list[Segment]

    def total_measured_s(self) -> Optional[float]:
        if any(s.measured_audio_s is None for s in self.segments):
            return None
        return sum(s.measured_audio_s + s.prosody.lead_in_s + s.prosody.pause_after_s
                   for s in self.segments)


class Turn(BaseModel):
    turn_id: int
    speaker_id: str
    speaker_name: str
    voice: str
    stance: str = "statement"
    spoken_text: str
    measured_audio_s: Optional[float] = None


class DialogueScript(BaseModel):
    schema_version: Literal["2.0"] = "2.0"
    session_id: str
    topic: str
    mode: Literal["socratic", "debate", "interview"] = "debate"
    script_source: str = "unknown"
    turns: list[Turn]
