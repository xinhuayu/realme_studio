"""
Rebuilding a lecture video from an old one plus new material.

The same-project path in `revise.py` needs nothing but the new deck, because
the project folder still holds every per-slide video and the ledger that keys
them. This module is for when it does not: the video was made on another
machine, the project folder is gone, or the lecture was rendered before any of
this existed.

What it needs, and why each:

  the old video        the audio and frames being kept
  <deck>_segments.json where each slide starts and ends, to three decimals
  the new deck         to see which slides changed
  narration            only for the slides that changed

What it does NOT need is the old deck. The old video contains the old slides:
one frame per segment is enough to compare against. Matching across that
boundary cannot use an exact hash -- a frame recovered from a compressed video
never matches a freshly rasterised PNG bit for bit -- so comparison is
perceptual.

**What perceptual comparison can and cannot see.** Measured on a rendered
1920x1080 slide, re-encoded at CRF 18 and compared with the original:

    the same slide, re-encoded          0.015
    one number changed on a dense slide 0.040
    an entirely different slide       126

A replaced figure, a new slide, a reordering: seen easily, by two orders of
magnitude. A single corrected number on a text-heavy slide: 0.04 against a
noise floor of 0.015, which is under three times the noise and not a margin
anyone should trust across encoders and scaling.

So detection is tuned to catch substantial changes reliably and is NOT relied
on for small edits. A slide can always be named explicitly -- the author knows
which line they corrected, and asking is cheaper than a heuristic that is
right most of the time.

`_chapters.txt` is deliberately not accepted as a timing source. It rounds to
whole seconds because YouTube wants "12:34", and a second of error per cut
lands mid-word and accumulates down the file.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

from realme.core.media import require, run, duration_of, MediaError
from realme.core.textio import read_text

#: Side of the grid a frame is reduced to, in colour. Larger grids were
#: measured and do not help: the signal from an edited line of text stays near
#: 0.04 whether the grid is 24 or 96, because the change is small relative to
#: the whole slide either way.
THUMB = 32

#: Mean absolute difference, 0-255, above which two slides are called
#: different. Sits far above the re-encoding noise floor (0.015) and far below
#: a genuinely different slide (126). Deliberately NOT set down at 0.03 to
#: catch edited text: that would be inside the noise of a different encoder.
DIFF_TOLERANCE = 2.0


@dataclass
class OldSegment:
    slide: int
    start_s: float
    end_s: float
    digest: str = ""


def read_segments(path: Path) -> tuple[list[OldSegment], dict]:
    """The cut list written beside a rendered lecture."""
    data = json.loads(read_text(path))
    if "segments" not in data:
        raise ValueError(
            f"{Path(path).name} is not a RealMe cut list. The file needed is "
            f"<deck>_segments.json, written beside the video.")
    segs = [OldSegment(slide=s["slide"], start_s=float(s["start_s"]),
                       end_s=float(s["end_s"]),
                       digest=s.get("slide_digest", ""))
            for s in data["segments"]]
    return segs, data.get("render", {})


def frame_at(video: Path, when_s: float, out_png: Path) -> Path:
    """One frame, a moment after a slide appears.

    Half a second in, not at the boundary: a cut frame can catch the tail of
    the previous slide, and any transition the encoder smeared across the
    join.
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)
    run([require("ffmpeg"), "-y", "-loglevel", "error",
         "-ss", f"{max(0.0, when_s) + 0.5:.3f}", "-i", str(video),
         "-frames:v", "1", str(out_png)], f"frame at {when_s:.1f}s")
    if not out_png.is_file():
        raise MediaError(f"Could not read a frame at {when_s:.1f}s from "
                         f"{video.name}")
    return out_png


def thumb(png: Path) -> list[int]:
    """A tiny COLOUR fingerprint, via ffmpeg so there is no new dependency.

    Colour, not grayscale: a red title slide and a green one reduce to nearly
    the same luma, and the first version of this called them identical.
    """
    import subprocess
    proc = subprocess.run(
        [require("ffmpeg"), "-loglevel", "error", "-i", str(png),
         "-vf", f"scale={THUMB}:{THUMB},format=rgb24", "-f", "rawvideo", "-"],
        capture_output=True)
    need = THUMB * THUMB * 3
    if proc.returncode or len(proc.stdout) < need:
        raise MediaError(f"Could not fingerprint {png.name}")
    return list(proc.stdout[:need])


def difference(a: list[int], b: list[int]) -> float:
    """Mean absolute difference between two fingerprints, 0-255."""
    if len(a) != len(b) or not a:
        return 255.0
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def looks_same(a: list[int], b: list[int],
               tolerance: float = DIFF_TOLERANCE) -> bool:
    """
    Same slide, allowing for compression and rescaling.

    A plain mean absolute difference, deliberately. The first version
    subtracted each image's mean first, reasoning that a rescaled slide shifts
    every value slightly -- which is true, and which also erases the
    difference between one flat colour and another. It reported a red slide
    and a green slide as the same slide.
    """
    return difference(a, b) <= tolerance


def cut(video: Path, start_s: float, end_s: float, out: Path) -> Path:
    """One slide's span, re-encoded.

    Re-encoded rather than stream-copied on purpose: a copy can only cut at a
    keyframe, so it silently moves the boundary to wherever the last one was
    -- which is how a spliced lecture gains half of the previous slide.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    run([require("ffmpeg"), "-y", "-loglevel", "error",
         "-ss", f"{start_s:.3f}", "-to", f"{end_s:.3f}", "-i", str(video),
         "-c:v", "libx264", "-preset", "medium", "-crf", "18",
         "-c:a", "aac", "-b:a", "192k", str(out)],
        f"cut {start_s:.1f}-{end_s:.1f}s")
    return out


def plan_from_video(video: Path, segments_json: Path, new_slides: list[Path],
                    *, also_rerecord: list[int] | None = None,
                    workdir: Path | None = None, log=print) -> dict:
    """
    Which slides of the new deck can be reused from the old video.

    `also_rerecord` is the author's own list, in NEW slide numbers, and it is
    additive: perceptual comparison catches replaced figures and new slides
    reliably and small text edits not at all, so the person who made the edit
    says so. Nothing here guesses on their behalf.
    """
    from realme.pipeline.revise import compare
    video, segments_json = Path(video), Path(segments_json)
    workdir = Path(workdir or video.parent / "_splice")
    workdir.mkdir(parents=True, exist_ok=True)

    old_segs, render = read_segments(segments_json)
    if not old_segs:
        raise ValueError(f"{segments_json.name} lists no segments.")
    total = duration_of(video)
    if old_segs[-1].end_s > total + 1.0:
        raise ValueError(
            f"{segments_json.name} describes {old_segs[-1].end_s:.1f}s of "
            f"video but {video.name} is {total:.1f}s. These are not the same "
            f"render.")

    log(f"  reading {len(old_segs)} slides out of {video.name}")
    old_prints = []
    for seg in old_segs:
        f = frame_at(video, seg.start_s, workdir / f"old_{seg.slide:03d}.png")
        old_prints.append(thumb(f))
    new_prints = [thumb(p) for p in new_slides]

    # Reuse the same content-matching as the same-project path, by handing it
    # identity strings instead of digests: two slides get the same string when
    # they look the same. One comparison rule, whether the old slides came
    # from a deck or out of a video.
    ident: list[list[int]] = []
    def key_for(pr) -> str:
        for i, seen in enumerate(ident):
            if looks_same(seen, pr):
                return f"s{i}"
        ident.append(pr)
        return f"s{len(ident) - 1}"

    old_keys = [key_for(p) for p in old_prints]
    new_keys = [key_for(p) for p in new_prints]
    plan = compare(old_keys, new_keys)

    forced = sorted(set(also_rerecord or []))
    to_render = sorted(set(plan.to_render) | set(forced))
    reuse = {}
    for c in plan.changes:
        if c.slide_number in to_render or c.old_slide_number is None:
            continue
        old = old_segs[c.old_slide_number - 1]
        reuse[c.slide_number] = (old.start_s, old.end_s)
    log(f"  {plan.summary()}")
    if forced:
        log(f"  also re-recording, as asked: {', '.join(map(str, forced))}")
    return {"plan": plan, "reuse": reuse, "to_render": to_render,
            "render": render, "old_segments": old_segs}
