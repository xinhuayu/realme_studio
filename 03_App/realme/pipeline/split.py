"""
Cutting a finished lecture into parts, at slide boundaries.

A thirty-slide lecture is one file because that is how it was rendered, not
because that is how anyone wants to watch it. Splitting it into sections is a
five-minute job in a video editor and a lifetime of nudging the playhead,
because the thing you want -- "start part two where slide twenty-one appears"
-- is not visible in the waveform.

It does not have to be. Every render already writes `<name>_segments.json`
beside the video, and every slide in it carries a `start_s` measured by ffprobe
at the moment it was assembled. The beginning of slide 21 is a lookup. Nothing
here estimates anything.

COPY WHERE IT IS EXACT, RE-ENCODE WHERE IT IS NOT.

`-c copy` can only cut at a keyframe. Cutting a general video that way
silently moves the boundary back to wherever the last keyframe happened to be,
which is how a "precise" split gains the tail of the previous section -- so the
obvious rule is to re-encode everything, and that was this module's first
version.

Measuring it said otherwise. A RealMe master is assembled by
`compose.concat`, which is a `-c copy` concat of one mp4 per slide, so every
slide boundary in it IS a keyframe. At exactly the points this tool cuts, a
stream copy is not approximate at all: it is exact, lossless, and roughly two
orders of magnitude faster than re-encoding a forty-minute lecture.

So the default is `auto`: copy, measure where it actually landed, and re-encode
only that part if the boundary moved. A video that has been through another
editor, or re-encoded elsewhere, falls back automatically and says so. Neither
speed nor precision is assumed -- the fast path is taken and then checked.

MEASURED AFTERWARDS, ALWAYS. ffmpeg is asked for a span and reports success;
what lands on disk is checked with ffprobe against what was planned. A cut that
came out 0.4s off is reported, not assumed away.
"""
from __future__ import annotations
import json, re
from dataclasses import dataclass, field, asdict
from pathlib import Path

from realme.core.textio import read_text
from realme.core.media import duration_of

#: How far a finished part may sit from its planned length before it is
#: called out, in seconds. Re-encoding lands within a frame or two; a third of
#: a second means something moved that should not have.
#:
#: This does NOT decide whether a stream copy landed correctly, though an
#: earlier version tried to use it that way. Asked for a start and a duration,
#: ffmpeg snaps the start back to the previous keyframe and still honours the
#: duration -- so a copy that begins in the wrong slide comes out exactly the
#: right length. The test passed and the part opened on the previous slide.
#: Where a copy may be used is settled by `keyframe_times` before cutting,
#: not inferred from the result afterwards.
DRIFT_TOLERANCE_S = 0.35

#: How far from a cut point a keyframe may sit and still be used, in seconds:
#: (earliest, latest) relative to the requested time.
#:
#: Asymmetric on purpose, and this is the whole reason the window is not a
#: single tolerance. A keyframe slightly LATE costs a few milliseconds off the
#: front of the slide's own lead-in pause, which nobody can see. A keyframe
#: slightly EARLY shows a flash of the PREVIOUS slide, which everybody can. So
#: late is cheap and early is nearly forbidden.
#:
#: The late figure is not arbitrary: a master concatenated by `compose.concat`
#: carries a constant ~23 ms offset from the audio encoder's priming delay, so
#: its slide boundaries land just after the times the sidecar records. Without
#: room for that, every part in a real lecture would fall back to re-encoding.
KEYFRAME_WINDOW_S = (-0.05, 0.20)

#: How the parts are cut.
#:   auto    copy, measure, re-encode only what the copy got wrong (default)
#:   copy    stream copy always: instant and lossless, wrong off a keyframe
#:   encode  re-encode always: exact anywhere, slow, one generation of loss
MODES = ("auto", "copy", "encode")

#: Characters Windows will not accept in a file name, plus the ones that make
#: a shell line ambiguous. Replaced rather than rejected: a section called
#: "Bias: what to do" is a perfectly good title and a terrible file name.
_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


@dataclass
class Part:
    """One output file: a run of consecutive slides."""
    index: int                    # 1-based, in output order
    title: str
    first_slide: int
    last_slide: int
    start_s: float
    end_s: float
    filename: str = ""
    #: Filled in after cutting, by measurement.
    how: str = ""                 # "copy" or "encode": what actually happened
    measured_s: float | None = None
    drift_s: float | None = None
    captions: str = ""
    cut_list: str = ""
    note: str = ""

    @property
    def planned_s(self) -> float:
        return self.end_s - self.start_s

    def as_dict(self) -> dict:
        d = asdict(self)
        d["planned_s"] = round(self.planned_s, 3)
        return d


@dataclass
class SplitPlan:
    parts: list[Part] = field(default_factory=list)
    total_s: float = 0.0
    slides: int = 0

    def summary(self) -> str:
        return (f"{len(self.parts)} part(s) from {self.slides} slides, "
                f"{self.total_s / 60:.1f} minutes total")

    def as_dict(self) -> dict:
        return {"parts": [p.as_dict() for p in self.parts],
                "total_s": round(self.total_s, 3), "slides": self.slides}


def safe_name(title: str, fallback: str) -> str:
    """A file name from a typed title, without losing the title's meaning."""
    cleaned = _UNSAFE.sub("-", (title or "").strip())
    cleaned = re.sub(r"\s+", "_", cleaned).strip("._-")
    # Windows also refuses a trailing dot and reserves a handful of names.
    if cleaned.upper().split(".")[0] in {
            "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4",
            "LPT1", "LPT2", "LPT3"}:
        cleaned = f"_{cleaned}"
    return cleaned[:80] or fallback


def load_cuts(segments_json: Path) -> tuple[list[dict], dict]:
    """The per-slide cut list, and the render settings beside it.

    Cut times come from the same sidecar `splice` reads, and its guard is
    reused so an unrelated JSON gets one explanation rather than two different
    ones depending on which feature you reached first.
    """
    from realme.pipeline.splice import read_segments
    segs, render = read_segments(Path(segments_json))       # validates
    raw = json.loads(read_text(segments_json))
    cuts = []
    for s, r in zip(segs, raw.get("segments", [])):
        cuts.append({"slide": s.slide, "start_s": s.start_s, "end_s": s.end_s,
                     "words": r.get("words", 0),
                     "speech_start_s": r.get("speech_start_s", s.start_s)})
    return cuts, {**render, "total_s": float(raw.get("total_s") or 0.0),
                  "master": raw.get("master", "")}


def plan(cuts: list[dict], starts: list[int], titles=None,
         *, stem: str = "lecture") -> SplitPlan:
    """
    Turn "start a new part at slide 9 and slide 21" into spans.

    `starts` are slide numbers where a NEW part begins. Slide 1 is always the
    beginning of the first part whether or not it is listed, because a part
    that started at slide 9 and left slides 1-8 in no file at all would lose
    them silently.
    """
    if not cuts:
        raise ValueError("The cut list is empty; there is nothing to split.")
    by_slide = {c["slide"]: c for c in cuts}
    ordered = sorted(by_slide)
    first, last = ordered[0], ordered[-1]

    wanted = sorted({int(s) for s in (starts or [])} | {first})
    unknown = [s for s in wanted if s not in by_slide]
    if unknown:
        raise ValueError(
            f"This video has slides {first}-{last}; cannot start a part at "
            f"{', '.join(map(str, unknown))}.")

    titles = list(titles or [])
    p = SplitPlan(slides=len(ordered))
    for i, begin in enumerate(wanted):
        end_slide = (wanted[i + 1] - 1) if i + 1 < len(wanted) else last
        # The slide before the next split point may not exist if the deck
        # numbering has gaps; take the last one that does.
        end_slide = max([s for s in ordered if s <= end_slide] or [begin])
        title = titles[i].strip() if i < len(titles) and titles[i] else ""
        default = (f"slides{begin:02d}-{end_slide:02d}")
        part = Part(index=i + 1, title=title or default,
                    first_slide=begin, last_slide=end_slide,
                    start_s=by_slide[begin]["start_s"],
                    end_s=by_slide[end_slide]["end_s"])
        part.filename = f"{stem}_{i + 1:02d}_{safe_name(title, default)}.mp4"
        p.parts.append(part)
    p.total_s = p.parts[-1].end_s - p.parts[0].start_s if p.parts else 0.0
    return p


def to_projects(video: Path, p: SplitPlan, outdir: Path, source_project: Path,
                projects_root: Path, *, log=print) -> list[dict]:
    """Make each part a project of its own: slides, notes, captions, a deck.

    The counterpart of what a join does. A part that is only a video and a
    caption file cannot be opened, corrected or re-recorded -- and splitting a
    lecture into sections is usually the first step of working on the sections,
    not the last step of anything.

    Each part moves out of the parts folder into its own project directory,
    taking its captions and cut list with it, and is then assembled from the
    slides it covers. A part that cannot be assembled keeps its files and is
    reported; the others are unaffected, because one awkward section should not
    cost you the rest.
    """
    import shutil
    from realme.pipeline.assemble import assemble, Source
    made = []
    # The record of the split belongs to the lecture that was split, not to a
    # folder that is about to disappear.
    leftover = Path(outdir) / "parts.json"
    if leftover.is_file():
        target = Path(source_project) / "_work" / f"{Path(video).stem}_parts.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(leftover), str(target))
    for part in p.parts:
        name = Path(part.filename).stem
        dest_dir = Path(projects_root) / name
        if dest_dir.exists() and any(dest_dir.iterdir()):
            log(f"  ! a project called '{name}' already exists; part "
                f"{part.index} was left in the parts folder")
            made.append({"part": part.index, "error": f"'{name}' already exists"})
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        src = Path(outdir) / part.filename
        moved = dest_dir / part.filename
        shutil.move(str(src), str(moved))
        for extra in (".srt",):
            f = src.with_suffix(extra)
            if f.is_file():
                shutil.move(str(f), str(moved.with_suffix(extra)))
        cut = src.with_name(src.stem + "_segments.json")
        if cut.is_file():
            dest_cut = moved.with_name(moved.stem + "_segments.json")
            shutil.move(str(cut), str(dest_cut))
            # Renumber the cut list to this project's OWN slide numbers.
            #
            # A part keeps the original numbers while it is a fragment of its
            # parent -- that is how the assembly knows which slides it covers,
            # and it is what makes a rejoin restore the original numbering. But
            # the moment it becomes a project in its own right, its manifest
            # and its slide images are renumbered from 1, and a cut list still
            # saying "slides 3 and 4" contradicts them. It did: merging a part
            # back reported its own slides missing, because the cut list asked
            # for pictures numbered 3 and 4 in a project whose pictures are 1
            # and 2. The original number is kept as provenance, not as an
            # index.
            try:
                data = json.loads(read_text(dest_cut))
                for i, seg in enumerate(data.get("segments", [])):
                    seg["source_slide"] = seg.get("slide")
                    seg["slide"] = i + 1
                    seg["segment_id"] = i + 1
                data["project_id"] = name
                dest_cut.write_text(json.dumps(data, indent=2), encoding="utf-8")
            except (ValueError, OSError) as e:
                log(f"  ! could not renumber {dest_cut.name}: {e}")
        try:
            a = assemble([Source(video=moved, project=Path(source_project),
                                 slides=list(range(part.first_slide,
                                                   part.last_slide + 1)))],
                         dest_dir, name, title=part.title, log=log)
            made.append({"part": part.index, "project_id": a.project_id,
                         "slides": a.slides, "deck": a.deck})
        except ValueError as e:
            log(f"  ! part {part.index} is a video but not a lecture: {e}")
            made.append({"part": part.index, "error": str(e)})
    return made


# ------------------------------------------------------------------ captions

_SRT_TIME = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*"
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{1,3})")


def _secs(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000.0


def _stamp(t: float) -> str:
    h, rem = divmod(max(t, 0.0), 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s - int(s)) * 1000)):03d}"


def slice_srt(srt: Path, start_s: float, end_s: float, out: Path) -> Path | None:
    """The captions belonging to one part, rebased to start at zero.

    Sliced from the finished .srt rather than rebuilt from the manifest, on
    purpose: someone splitting a video a year later has the file and not
    necessarily the project it came from.

    A cue straddling a boundary is kept by the part that holds its START and
    clipped there, so a sentence spoken across the join appears once rather
    than twice or not at all.
    """
    srt = Path(srt)
    if not srt.is_file():
        return None
    blocks = re.split(r"\n\s*\n", srt.read_text(encoding="utf-8", errors="replace"))
    kept, n = [], 0
    for block in blocks:
        m = _SRT_TIME.search(block)
        if not m:
            continue
        a = _secs(*m.group(1, 2, 3, 4))
        b = _secs(*m.group(5, 6, 7, 8))
        if a < start_s or a >= end_s:
            continue
        text = block[m.end():].strip("\n")
        n += 1
        kept.append(f"{n}\n{_stamp(a - start_s)} --> "
                    f"{_stamp(min(b, end_s) - start_s)}\n{text}")
    if not kept:
        return None
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n\n".join(kept) + "\n", encoding="utf-8")
    return out


# ------------------------------------------------------------------- cutting

def keyframe_times(video: Path) -> list[float]:
    """Every keyframe in the video stream, in seconds.

    One ffprobe pass with `-skip_frame nokey`, which demuxes without decoding
    the frames in between, so it is quick even on a long lecture. Asking the
    file directly is the only way to know where a stream copy can begin;
    everything else is a guess that is right until it is not.
    """
    import subprocess
    from realme.core.media import require
    out = subprocess.run(
        [require("ffprobe"), "-v", "error", "-select_streams", "v",
         "-skip_frame", "nokey", "-show_entries", "frame=pts_time",
         "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    times = []
    for line in out.stdout.splitlines():
        token = line.strip().rstrip(",")
        if not token:
            continue
        try:
            times.append(float(token))
        except ValueError:
            continue
    return sorted(times)


def snap_to_keyframe(t: float, keys: list[float]) -> float | None:
    """The keyframe a copy may start from for a cut at `t`, or None.

    Returns the keyframe itself rather than `t`, so the cut is made exactly
    where the copy would land anyway -- asking ffmpeg for a time it cannot
    honour and letting it decide is how the boundary moves without anyone
    being told.
    """
    lo, hi = KEYFRAME_WINDOW_S
    inside = [k for k in keys if lo <= (k - t) <= hi]
    return min(inside, key=lambda k: abs(k - t)) if inside else None


def copy_cut(video: Path, start_s: float, duration_s: float, out: Path) -> Path:
    """One span, stream-copied: no re-encoding, no generation loss.

    `-t` rather than `-to`, because with `-ss` placed before `-i` the meaning
    of `-to` has changed between ffmpeg versions while a duration never has.
    `-avoid_negative_ts make_zero` is what makes the part start at 00:00
    instead of carrying the original timeline's timestamps into a file that
    some players then refuse to seek in.
    """
    from realme.core.media import require, run
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    run([require("ffmpeg"), "-y", "-loglevel", "error",
         "-ss", f"{start_s:.3f}", "-i", str(video), "-t", f"{duration_s:.3f}",
         "-c", "copy", "-avoid_negative_ts", "make_zero",
         "-movflags", "+faststart", str(out)],
        f"copy {start_s:.1f}s +{duration_s:.1f}s")
    return out


def _measure(part: Part, dest: Path) -> float | None:
    """Measured length of what landed, or None when it cannot be read."""
    try:
        part.measured_s = round(float(duration_of(dest)), 3)
        part.drift_s = round(part.measured_s - part.planned_s, 3)
        return part.drift_s
    except Exception as e:
        part.note = f"could not measure the finished file: {e}"
        return None


def execute(video: Path, p: SplitPlan, outdir: Path, *, srt: Path | None = None,
            mode: str = "auto", cuts: list[dict] | None = None,
            render: dict | None = None, log=print) -> SplitPlan:
    """Cut every part, then MEASURE what landed rather than trusting ffmpeg.

    `cuts` and `render` are the sidecar the plan was made from. Passing them
    lets each part carry its own cut list; without them the parts are still
    correct video, just no longer splittable themselves.
    """
    cuts = cuts or []
    render = render or {}
    from realme.pipeline.splice import cut
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}, not {mode!r}")
    video, outdir = Path(video), Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    keys = [] if mode == "encode" else keyframe_times(video)
    if mode != "encode":
        log(f"  {len(keys)} keyframe(s) in the source")
    for part in p.parts:
        dest = outdir / part.filename
        log(f"  part {part.index}: slides {part.first_slide}-{part.last_slide}, "
            f"{part.start_s:.1f}s to {part.end_s:.1f}s -> {dest.name}")
        snap = snap_to_keyframe(part.start_s, keys) if mode != "encode" else None
        if snap is not None:
            # Only the START of a copied span must be a keyframe; the end can
            # fall anywhere, because truncating a stream needs no reference
            # frame. So each part is decided on its own and a single awkward
            # boundary does not force the whole lecture to be re-encoded.
            copy_cut(video, snap, part.end_s - snap, dest)
            part.how = "copy"
            drift = _measure(part, dest)
        elif mode == "copy":
            # Asked for, so done -- and said plainly, because the boundary will
            # be wrong and nothing else would reveal it.
            copy_cut(video, part.start_s, part.planned_s, dest)
            part.how = "copy"
            drift = _measure(part, dest)
            part.note = ("no keyframe at this cut point, so the copy begins "
                         "in the previous slide; use auto or encode for an "
                         "exact boundary")
        else:
            if mode == "auto":
                log("    no keyframe at this cut point; re-encoding this part "
                    "so the boundary is exact")
            cut(video, part.start_s, part.end_s, dest)
            part.how = "encode"
            drift = _measure(part, dest)
        if drift is not None and abs(drift) > DRIFT_TOLERANCE_S and not part.note:
            part.note = (f"came out {drift:+.2f}s from the "
                         f"{part.planned_s:.2f}s asked for")
        if part.note:
            log(f"    ! {part.note}")
        if srt:
            sliced = slice_srt(Path(srt), part.start_s, part.end_s,
                               dest.with_suffix(".srt"))
            if sliced:
                part.captions = sliced.name
        # Each part gets its own cut list, rebased to its own zero. Without it
        # a part is just a video: it cannot be split again, and rejoining the
        # set produces a lecture with no slide times at all. The slide NUMBERS
        # are kept as they were in the whole deck, so a part covering slides
        # 4-6 still calls them 4, 5 and 6 rather than renaming them 1, 2, 3.
        part_cuts = [c for c in cuts
                     if part.first_slide <= c["slide"] <= part.last_slide]
        if part_cuts:
            base = part.start_s
            dest.with_name(dest.stem + "_segments.json").write_text(json.dumps({
                "version": 1, "project_id": dest.stem, "master": dest.name,
                "total_s": round(part.end_s - base, 3),
                "render": render,
                "part_of": Path(video).name,
                "segments": [{**c,
                              "start_s": round(c["start_s"] - base, 3),
                              "end_s": round(c["end_s"] - base, 3),
                              "speech_start_s": round(
                                  c.get("speech_start_s", c["start_s"]) - base, 3),
                              "speech_end_s": round(
                                  c.get("speech_end_s", c["end_s"]) - base, 3)}
                             for c in part_cuts],
            }, indent=2), encoding="utf-8")
            part.cut_list = dest.stem + "_segments.json"
    kinds = {x.how for x in p.parts}
    if kinds == {"copy"}:
        log("  every part was copied without re-encoding: no quality was lost")
    elif "encode" in kinds and "copy" in kinds:
        log(f"  {sum(1 for x in p.parts if x.how == 'encode')} of {len(p.parts)} "
            f"part(s) had to be re-encoded to land on the boundary")
    (outdir / "parts.json").write_text(json.dumps(p.as_dict(), indent=2),
                                       encoding="utf-8")
    return p
