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
    """HH:MM:SS,mmm. Rounded to whole milliseconds FIRST and then split, so
    59.9996 s is 00:01:00,000 rather than the invalid 00:00:59,1000."""
    total_ms = int(round(max(t, 0.0) * 1000))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p] or [text.strip()]


def build_cues(segments, inserts=None) -> list[tuple[float, float, str]]:
    """`inserts`: videos played between segments (see pipeline/inserts.py),
    each with `.after_index`, `.duration` and its own `.cues`. Everything after
    one moves by its length, and its own captions are carried in."""
    cues, clock = [], 0.0
    inserts = list(inserts or [])

    def play_inserts(after_index):
        nonlocal clock
        for ins in inserts:
            if ins.after_index == after_index:
                cues.extend((clock + a, clock + b, t) for a, b, t in ins.cues)
                clock += ins.duration

    play_inserts(-1)
    for i, seg in enumerate(segments):
        if i:
            play_inserts(i - 1)
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
    if segments:
        play_inserts(len(segments) - 1)
    return cues


def write_srt(segments, out: Path, inserts=None) -> Path:
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, (a, b, txt) in enumerate(build_cues(segments, inserts), 1):
        lines += [str(i), f"{_ts(a)} --> {_ts(b)}", txt, ""]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def shift_srt(src: Path, out: Path, by: float) -> Path:
    """Copy a caption file with every time moved later by `by` seconds.

    Captions are timed against the narration, which starts at zero. A video
    holds a lead-in before the first word, so the same file sitting beside the
    video is early by exactly that lead-in -- 0.6 s, which is enough to read
    as a mistake. A lecture already accounts for this when it builds its cues;
    anything that narrates first and muxes afterwards has to shift the file.
    """
    src, out = Path(src), Path(out)
    lines = []
    for line in src.read_text(encoding="utf-8").splitlines():
        if "-->" in line:
            a, _, b = line.partition("-->")
            lines.append(f"{_ts(_secs(a) + by)} --> {_ts(_secs(b) + by)}")
        else:
            lines.append(line)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def _secs(stamp: str) -> float:
    h, m, s = stamp.strip().replace(",", ".").split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def write_vtt(segments, out: Path, inserts=None) -> Path:
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["WEBVTT", ""]
    for a, b, txt in build_cues(segments, inserts):
        lines += [f"{_ts(a, '.')} --> {_ts(b, '.')}", txt, ""]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_chapters(segments, out: Path, titles=None, inserts=None) -> Path:
    """YouTube-style chapter list. First chapter MUST be 00:00 or YouTube ignores all.

    An inserted video is a chapter of its own, under its title."""
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    lines, clock = [], 0.0
    inserts = list(inserts or [])

    def stamp_of(t):
        h, rem = divmod(t, 3600); m, s = divmod(rem, 60)
        return (f"{int(h)}:{int(m):02d}:{int(s):02d}" if h >= 1
                else f"{int(m)}:{int(s):02d}")

    def play_inserts(after_index):
        nonlocal clock
        for ins in inserts:
            if ins.after_index == after_index:
                lines.append(f"{stamp_of(clock)} {ins.title}")
                clock += ins.duration

    play_inserts(-1)
    for i, seg in enumerate(segments):
        if i:
            play_inserts(i - 1)
        # Chapters mark when the SLIDE appears, not when speech starts.
        title = (titles[i] if titles and i < len(titles)
                 else f"Slide {seg.slide_index + 1}")
        lines.append(f"{stamp_of(clock)} {title}")
        clock += (seg.prosody.lead_in_s + seg.measured_audio_s
                  + seg.prosody.pause_after_s)
    if segments:
        play_inserts(len(segments) - 1)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
