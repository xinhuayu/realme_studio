"""
Captions built from MEASURED segment durations.

v1 estimated every timestamp at 140 wpm. Over a 30-minute lecture a 10% rate
error accumulates to ~3 minutes of caption drift by the end. Here, each
segment's window is its real ffprobe'd duration, and sentences are distributed
inside that window proportionally to character count. That is accurate to the
segment boundary and typically within ~0.3s inside it, which is fine for
sentence-level captions. For true word-level timing, run forced alignment
(WhisperX / stable-ts) over the synthesized audio -- do NOT re-transcribe with
plain Whisper, since you already know the exact text.
"""
from __future__ import annotations
import re
from pathlib import Path


def _ts(t: float, sep: str = ",") -> str:
    h, rem = divmod(max(t, 0.0), 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}{sep}{int(round((s - int(s)) * 1000)):03d}"


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p] or [text.strip()]


def build_cues(segments) -> list[tuple[float, float, str]]:
    cues, clock = [], 0.0
    for seg in segments:
        clock += seg.prosody.lead_in_s      # captions start when speech does
        if seg.spoken_cues:
            # Real measured utterances: exact timing, markers already stripped,
            # masked spans absent entirely.
            for offset, dur, text in seg.spoken_cues:
                if text.strip():
                    cues.append((clock + offset, clock + offset + dur, text.strip()))
            clock += (seg.measured_audio_s or 0.0) + seg.prosody.pause_after_s
            continue
        dur = seg.measured_audio_s
        if dur is None:
            raise ValueError(
                f"Segment {seg.segment_id} has no measured duration. "
                "Captions must be built after synthesis, never from estimates.")
        sents = _sentences(seg.spoken_text)
        total_chars = sum(len(s) for s in sents) or 1
        t = clock
        for s in sents:
            share = dur * (len(s) / total_chars)
            cues.append((t, t + share, s))
            t += share
        clock += dur + seg.prosody.pause_after_s
    return cues


def write_srt(segments, out: Path) -> Path:
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, (a, b, txt) in enumerate(build_cues(segments), 1):
        lines += [str(i), f"{_ts(a)} --> {_ts(b)}", txt, ""]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_vtt(segments, out: Path) -> Path:
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["WEBVTT", ""]
    for a, b, txt in build_cues(segments):
        lines += [f"{_ts(a, '.')} --> {_ts(b, '.')}", txt, ""]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_chapters(segments, out: Path, titles=None) -> Path:
    """YouTube-style chapter list. First chapter MUST be 00:00 or YouTube ignores all."""
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    lines, clock = [], 0.0
    for i, seg in enumerate(segments):
        # Chapters mark when the SLIDE appears, not when speech starts.
        h, rem = divmod(clock, 3600); m, s = divmod(rem, 60)
        stamp = (f"{int(h)}:{int(m):02d}:{int(s):02d}" if h >= 1 else f"{int(m)}:{int(s):02d}")
        title = (titles[i] if titles and i < len(titles)
                 else f"Slide {seg.slide_index + 1}")
        lines.append(f"{stamp} {title}")
        clock += (seg.prosody.lead_in_s + seg.measured_audio_s
                  + seg.prosody.pause_after_s)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
