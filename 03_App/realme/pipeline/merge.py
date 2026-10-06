"""
Joining videos into one, and knowing when that is safe.

The inverse of `split`, and the one that looks trivial and is not.

`ffmpeg -f concat -c copy` is instant and lossless, and it is only CORRECT when
every input agrees on codec, profile, resolution, pixel format, frame rate,
audio codec, sample rate and channel count. Hand it two files that differ and
the good case is an error; the bad case is a file that plays the first video
and then goes black, or keeps the picture and loses the sound, or drifts
further out of sync the longer it runs. It reports success either way.

That failure is invisible until someone watches to the end, which is the worst
possible time to find it. So nothing here is assumed: every input is probed,
the parameters are compared, and the fast path is taken only when they match.
Where they do not, the inputs are normalised and re-encoded to one target, and
the report says which happened and why.

Two things travel with the video, because a merged lecture without them is a
merged lecture someone has to repair by hand:

  CAPTIONS   each input's .srt, shifted by the running offset and renumbered.
  CUT LIST   a combined `_segments.json`, so a merged video can be split again
             at slide boundaries. Only when every input has one -- a partial
             cut list would claim slide times for material it knows nothing
             about, which is worse than having none.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

from realme.core.media import probe, duration_of, require, run
from realme.core.textio import read_text
from realme.pipeline.split import _SRT_TIME, _secs, _stamp

#: How far the joined file may sit from the sum of its inputs before it is
#: called out, in seconds. Container overhead and one frame of rounding per
#: join are normal; a second is not.
DRIFT_TOLERANCE_S = 1.0

#: What must match, stream by stream, for a lossless join to be correct.
#: Anything not on this list either does not affect concatenation or is
#: implied by something that is.
#:
#: `r_frame_rate`, NOT `avg_frame_rate`. The first version compared
#: avg_frame_rate and declared three parts cut from ONE master incompatible:
#: avg_frame_rate is frames divided by duration, so it differs for every clip
#: of a different length -- 1472/59, 2072/83, 678400/27151 for three pieces of
#: the same 25 fps video. Comparing it makes a lossless join impossible for
#: exactly the files this tool exists to rejoin. r_frame_rate is the nominal
#: rate and reads 25/1 for all of them.
VIDEO_KEYS = ("codec_name", "width", "height", "pix_fmt", "profile",
              "r_frame_rate")
AUDIO_KEYS = ("codec_name", "sample_rate", "channels")

#: Re-encode settings for the mismatched case. CRF 18 is visually lossless for
#: slide content; the audio settings match what the renderer already produces,
#: so a re-encoded join sounds like the parts it was made from.
ENCODE = ("-c:v", "libx264", "-preset", "medium", "-crf", "18",
          "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k")


@dataclass
class Input:
    path: Path
    seconds: float = 0.0
    video: dict = field(default_factory=dict)
    audio: dict = field(default_factory=dict)
    has_audio: bool = True
    srt: Path | None = None
    segments: Path | None = None
    note: str = ""

    def as_dict(self) -> dict:
        d = asdict(self)
        d["path"] = str(self.path)
        d["srt"] = str(self.srt) if self.srt else ""
        d["segments"] = str(self.segments) if self.segments else ""
        return d


@dataclass
class MergePlan:
    inputs: list[Input] = field(default_factory=list)
    compatible: bool = True
    reasons: list[str] = field(default_factory=list)
    how: str = ""                     # "copy" or "encode", filled by execute
    total_s: float = 0.0
    measured_s: float | None = None
    drift_s: float | None = None
    captions: str = ""
    cut_list: str = ""
    note: str = ""

    def summary(self) -> str:
        mins = self.total_s / 60
        head = (f"{len(self.inputs)} video(s), {mins:.1f} minutes")
        if self.compatible:
            return head + " -- identical settings, so they can be joined " \
                          "without re-encoding"
        return head + f" -- settings differ ({self.reasons[0]}), so they will " \
                      f"be re-encoded to match"

    def as_dict(self) -> dict:
        d = asdict(self)
        d["inputs"] = [i.as_dict() for i in self.inputs]
        return d


def _stream(info: dict, kind: str) -> dict:
    for s in info.get("streams", []):
        if s.get("codec_type") == kind:
            return s
    return {}


def inspect(path: Path) -> Input:
    """Everything about one input that decides how it can be joined."""
    path = Path(path)
    info = probe(path)
    v, a = _stream(info, "video"), _stream(info, "audio")
    inp = Input(path=path,
                seconds=round(float(duration_of(path)), 3),
                video={k: v.get(k) for k in VIDEO_KEYS},
                audio={k: a.get(k) for k in AUDIO_KEYS},
                has_audio=bool(a))
    if not v:
        inp.note = "no video stream"
    if not a:
        # Concatenating a silent clip with a sounded one drops the audio track
        # from the join in copy mode and desynchronises it in encode mode
        # unless silence is generated. Named here; handled in `execute`.
        inp.note = (inp.note + "; " if inp.note else "") + "no audio stream"
    srt = path.with_suffix(".srt")
    if srt.is_file():
        inp.srt = srt
    seg = path.with_name(path.stem + "_segments.json")
    if seg.is_file():
        inp.segments = seg
    return inp


def plan(paths: list[Path]) -> MergePlan:
    """Probe every input and decide whether a lossless join is correct."""
    if len(paths) < 2:
        raise ValueError("Joining needs at least two videos.")
    p = MergePlan()
    for path in paths:
        path = Path(path)
        if not path.is_file():
            raise ValueError(f"No such file: {path}")
        p.inputs.append(inspect(path))
    p.total_s = round(sum(i.seconds for i in p.inputs), 3)

    first = p.inputs[0]
    for other in p.inputs[1:]:
        for key in VIDEO_KEYS:
            if first.video.get(key) != other.video.get(key):
                p.reasons.append(
                    f"{other.path.name} has {key} "
                    f"{other.video.get(key)!r} where {first.path.name} has "
                    f"{first.video.get(key)!r}")
        for key in AUDIO_KEYS:
            if first.audio.get(key) != other.audio.get(key):
                p.reasons.append(
                    f"{other.path.name} has audio {key} "
                    f"{other.audio.get(key)!r} where {first.path.name} has "
                    f"{first.audio.get(key)!r}")
    if any(not i.has_audio for i in p.inputs) and \
            any(i.has_audio for i in p.inputs):
        p.reasons.append("some inputs have an audio track and some do not")
    p.compatible = not p.reasons
    return p


# ------------------------------------------------------------------ captions

def merge_srt(inputs: list[Input], out: Path) -> Path | None:
    """One caption file for the join, each part shifted by what precedes it.

    A part with no captions still advances the clock. Skipping its duration
    would pull every later cue earlier by the length of the silent part, which
    is the same drift this project rebuilt the caption timing to avoid.
    """
    import re
    blocks, clock, n = [], 0.0, 0
    for inp in inputs:
        if inp.srt and inp.srt.is_file():
            text = inp.srt.read_text(encoding="utf-8", errors="replace")
            for block in re.split(r"\n\s*\n", text):
                m = _SRT_TIME.search(block)
                if not m:
                    continue
                a = _secs(*m.group(1, 2, 3, 4)) + clock
                b = _secs(*m.group(5, 6, 7, 8)) + clock
                n += 1
                blocks.append(f"{n}\n{_stamp(a)} --> {_stamp(b)}\n"
                              f"{block[m.end():].strip(chr(10))}")
        clock += inp.seconds
    if not blocks:
        return None
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    return out


def merge_segments(inputs: list[Input], out: Path) -> Path | None:
    """A combined cut list, so the joined video can be split again.

    Written only when EVERY input has one. A cut list covering half the file
    would give slide numbers to the first lecture and nothing to the second,
    and the splitter would happily offer to cut at a slide that is not there.

    Slide numbers are KEPT when they already increase across the join -- which
    is the case when the inputs are parts of one lecture being reassembled, and
    keeping them means the rejoined video still calls slide 21 slide 21. They
    are renumbered only when they would collide, because two slide 1s in one
    file cannot both be found.
    """
    if not all(i.segments for i in inputs):
        return None
    loaded = [json.loads(read_text(i.segments))
              for i in inputs]
    original = [s.get("slide") for d in loaded for s in d.get("segments", [])]
    keep_numbers = all(b > a for a, b in zip(original, original[1:]))
    merged, clock, slide = [], 0.0, 0
    render = {}
    for inp, data in zip(inputs, loaded):
        render = render or data.get("render", {})
        for seg in data.get("segments", []):
            slide += 1
            number = seg.get("slide", slide) if keep_numbers else slide
            merged.append({**seg, "slide": number,
                           "segment_id": slide,
                           "start_s": round(seg["start_s"] + clock, 3),
                           "end_s": round(seg["end_s"] + clock, 3),
                           "speech_start_s": round(
                               seg.get("speech_start_s", seg["start_s"]) + clock, 3),
                           "speech_end_s": round(
                               seg.get("speech_end_s", seg["end_s"]) + clock, 3),
                           "from": Path(inp.path).name})
        clock += inp.seconds
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"version": 1, "project_id": out.stem.replace("_segments", ""),
         "master": out.name.replace("_segments.json", ".mp4"),
         "total_s": round(clock, 3), "render": render,
         "merged_from": [Path(i.path).name for i in inputs],
         "slides_renumbered": not keep_numbers,
         "segments": merged}, indent=2), encoding="utf-8")
    return out


# -------------------------------------------------------------------- doing

def _listing(inputs: list[Input], where: Path) -> Path:
    where.write_text("\n".join(
        f"file '{Path(i.path).resolve().as_posix()}'" for i in inputs),
        encoding="utf-8")
    return where


def execute(p: MergePlan, out: Path, *, mode: str = "auto", log=print) -> MergePlan:
    """Join the inputs, then MEASURE the result against the sum of its parts."""
    if mode not in ("auto", "copy", "encode"):
        raise ValueError(f"mode must be auto, copy or encode, not {mode!r}")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    listing = out.with_name(f"_join_{out.stem}.txt")
    _listing(p.inputs, listing)

    use_copy = p.compatible if mode == "auto" else (mode == "copy")
    if mode == "copy" and not p.compatible:
        p.note = ("the inputs do not have matching settings, so this join may "
                  "play only the first video: " + p.reasons[0])
        log(f"  ! {p.note}")
    try:
        if use_copy:
            log("  joining without re-encoding")
            run([require("ffmpeg"), "-y", "-loglevel", "error",
                 "-f", "concat", "-safe", "0", "-i", str(listing),
                 "-c", "copy", "-movflags", "+faststart", str(out)],
                f"join {len(p.inputs)} videos")
            p.how = "copy"
        else:
            log(f"  re-encoding to one setting ({len(p.reasons)} difference(s))")
            for r in p.reasons[:4]:
                log(f"    {r}")
            # The concat DEMUXER cannot take mismatched inputs even when
            # re-encoding; the concat FILTER can, because it decodes first.
            args = [require("ffmpeg"), "-y", "-loglevel", "error"]
            for i in p.inputs:
                args += ["-i", str(i.path)]
            target = p.inputs[0]
            w = target.video.get("width") or 1920
            h = target.video.get("height") or 1080
            chain = []
            for idx, inp in enumerate(p.inputs):
                # Scale and pad rather than stretch: a 4:3 clip joined into a
                # 16:9 lecture should sit in the middle, not be distorted.
                chain.append(
                    f"[{idx}:v]scale={w}:{h}:force_original_aspect_ratio="
                    f"decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
                    f"fps=30[v{idx}]")
                if inp.has_audio:
                    chain.append(f"[{idx}:a]aresample=48000,"
                                 f"aformat=channel_layouts=stereo[a{idx}]")
                else:
                    # Silence for a silent input, so the audio track does not
                    # slide forward by that input's length for everything after.
                    chain.append(
                        f"anullsrc=channel_layout=stereo:sample_rate=48000,"
                        f"atrim=duration={inp.seconds}[a{idx}]")
            joins = "".join(f"[v{i}][a{i}]" for i in range(len(p.inputs)))
            chain.append(f"{joins}concat=n={len(p.inputs)}:v=1:a=1[v][a]")
            args += ["-filter_complex", ";".join(chain),
                     "-map", "[v]", "-map", "[a]", *ENCODE,
                     "-movflags", "+faststart", str(out)]
            run(args, f"join and re-encode {len(p.inputs)} videos")
            p.how = "encode"
    finally:
        listing.unlink(missing_ok=True)

    try:
        p.measured_s = round(float(duration_of(out)), 3)
        p.drift_s = round(p.measured_s - p.total_s, 3)
        if abs(p.drift_s) > DRIFT_TOLERANCE_S:
            extra = (f"came out {p.drift_s:+.2f}s from the {p.total_s:.2f}s "
                     f"its parts add up to")
            p.note = f"{p.note}; {extra}" if p.note else extra
            log(f"  ! {extra}")
    except Exception as e:
        p.note = f"could not measure the joined file: {e}"
        log(f"  ! {p.note}")

    srt = merge_srt(p.inputs, out.with_suffix(".srt"))
    if srt:
        p.captions = srt.name
        log(f"  captions: {srt.name}")
    cuts = merge_segments(p.inputs, out.with_name(out.stem + "_segments.json"))
    if cuts:
        p.cut_list = cuts.name
        log(f"  cut list: {cuts.name} -- the joined video can be split again")
    elif any(i.segments for i in p.inputs):
        log("  no cut list written: not every input had one, and a partial "
            "one would give slide times to material it does not cover")
    return p
