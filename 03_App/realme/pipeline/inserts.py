"""
Videos inserted between a lecture's slides, remembered by the project.

An animated explainer, a screen recording, a clip from a
colleague: something that is not a slide, but belongs between two of them.
The project keeps a list -- "after slide 12: confounding.mp4" -- in
`inserts.json`, and every render puts each one in its place, so editing the
notes and rendering again does not lose it.

What has to agree for a seamless join:

  PICTURE   a slide segment is H.264 at the lecture's size and frame rate, and
            the master is joined without re-encoding. An insert from anywhere
            else is CONFORMED once -- scaled into the frame, letterboxed if its
            shape differs, re-timed to the frame rate, given a silent track if
            it has none -- and the conformed copy is cached under its content.
  CLOCK     captions, chapters, the cut list and the final duration check all
            move by the insert's length from where it sits onward, so the
            captions stay in step with the picture.
  WORDS     an insert's own captions (`name.srt` beside it, as most
            video tools write) are carried into the lecture's, shifted to where it plays.

`after_slide` is the slide number as the editor shows it: 0 plays before
slide 1, 12 plays after slide 12.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import re
import shutil
from dataclasses import dataclass, field, asdict
from pathlib import Path

from realme.core.media import MediaError, duration_of, probe, require, run
from realme.core.textio import read_text

VIDEO_SUFFIXES = (".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi")
LIST_NAME = "inserts.json"
FOLDER = "inserts"


@dataclass
class Insert:
    id: str
    after_slide: int
    file: str                         # relative to the project folder
    title: str = ""
    captions: str | None = None       # relative to the project folder
    added: str = ""


@dataclass
class Placed:
    """An insert, conformed and located, ready to be joined."""
    insert: Insert
    after_index: int                  # -1 = before the first segment
    path: Path
    duration: float
    cues: list[tuple[float, float, str]] = field(default_factory=list)

    @property
    def title(self) -> str:
        return self.insert.title or Path(self.insert.file).stem


# ------------------------------------------------------------- the list

def load(pdir: Path) -> list[Insert]:
    f = Path(pdir) / LIST_NAME
    if not f.is_file():
        return []
    try:
        data = json.loads(read_text(f))
    except (OSError, ValueError) as e:
        raise MediaError(f"{f} could not be read ({e}). Fix or delete it.")
    out = []
    for d in data.get("inserts", []):
        out.append(Insert(**{k: d[k] for k in Insert.__dataclass_fields__
                             if k in d}))
    return sorted(out, key=lambda i: (i.after_slide, i.added))


def save(pdir: Path, items: list[Insert]) -> Path:
    f = Path(pdir) / LIST_NAME
    f.write_text(json.dumps({"version": 1,
                             "inserts": [asdict(i) for i in items]},
                            indent=2, ensure_ascii=False), encoding="utf-8")
    return f


def _safe_name(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("._") or "insert"
    return stem[:60] + Path(name).suffix.lower()


def add(pdir: Path, video: Path, after_slide: int, *, title: str = "",
        captions: Path | None = None) -> Insert:
    """Copy `video` into the project and remember where it plays.

    `captions` defaults to a `.srt` with the same name beside the video.
    """
    pdir, video = Path(pdir), Path(video)
    if not video.is_file():
        raise MediaError(f"No video at {video}")
    if video.suffix.lower() not in VIDEO_SUFFIXES:
        raise MediaError(f"{video.name} is not a video this can insert "
                         f"({', '.join(VIDEO_SUFFIXES)}).")
    if int(after_slide) < 0:
        raise MediaError("after_slide is 0 (before slide 1) or a slide number.")
    info = probe(video)
    if not any(s.get("codec_type") == "video" for s in info.get("streams", [])):
        raise MediaError(f"{video.name} has no picture in it.")
    folder = pdir / FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / _safe_name(video.name)
    n = 2
    while dest.exists() and dest.read_bytes() != video.read_bytes():
        dest = folder / f"{Path(_safe_name(video.name)).stem}_{n}{video.suffix.lower()}"
        n += 1
    if not dest.exists():
        shutil.copy2(video, dest)
    cap = captions if captions is not None else video.with_suffix(".srt")
    cap_rel = None
    if cap and Path(cap).is_file():
        cdest = dest.with_suffix(".srt")
        shutil.copy2(cap, cdest)
        cap_rel = cdest.relative_to(pdir).as_posix()
    digest = hashlib.sha256(f"{dest.name}|{after_slide}|"
                            f"{datetime.datetime.now().isoformat()}".encode())
    item = Insert(id="ins_" + digest.hexdigest()[:8], after_slide=int(after_slide),
                  file=dest.relative_to(pdir).as_posix(),
                  title=title.strip() or dest.stem.replace("_", " "),
                  captions=cap_rel,
                  added=datetime.datetime.now().isoformat(timespec="seconds"))
    items = load(pdir) + [item]
    save(pdir, items)
    return item


def remove(pdir: Path, insert_id: str) -> Insert:
    """Forget an insert. Its copied file stays in the project's folder, so
    removing one by mistake loses nothing."""
    items = load(pdir)
    keep = [i for i in items if i.id != insert_id]
    if len(keep) == len(items):
        raise MediaError(f"No insert {insert_id!r} in this lecture.")
    save(pdir, keep)
    return next(i for i in items if i.id == insert_id)


# ------------------------------------------------------- conform and place

def check_files(pdir: Path) -> list[Insert]:
    """Every insert's video is where the list says, or a MediaError now.

    Called when a render STARTS. The inserts are joined only at the very
    end, so a missing file is caught before the narration is recorded rather
    than after.
    """
    pdir = Path(pdir)
    items = load(pdir)
    gone = [i for i in items if not (pdir / i.file).is_file()]
    if gone:
        names = ", ".join(f"{i.file} ({i.title})" for i in gone[:5])
        raise MediaError(
            f"{len(gone)} inserted video(s) are missing from the project "
            f"folder: {names}{' ...' if len(gone) > 5 else ''}. Put them back "
            f"or remove those inserts, then render again. Nothing was "
            f"recorded.")
    return items


def _has_audio(path: Path) -> bool:
    return any(s.get("codec_type") == "audio"
               for s in probe(path).get("streams", []))


def conform(pdir: Path, item: Insert, *, width: int, height: int, fps: int,
            log=print) -> Path:
    """The insert as the lecture's own segments are made, cached by content.

    Same codec settings as `compose.render_segment`, so the master can still
    be joined without re-encoding a single slide.
    """
    from realme.pipeline.compose import A_ARGS, V_ARGS
    pdir = Path(pdir)
    src = pdir / item.file
    if not src.is_file():
        raise MediaError(f"Inserted video {item.file} is missing from the "
                         f"project folder. Put it back or remove the insert.")
    h = hashlib.sha256()
    with open(src, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    key = f"{h.hexdigest()[:16]}_{width}x{height}_{fps}"
    out = pdir / "_work" / "inserts" / f"{src.stem}_{key}.mp4"
    if out.is_file():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    dur = duration_of(src)
    fit = (f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
           f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,"
           f"fps={fps}")
    cmd = [require("ffmpeg"), "-y", "-i", str(src)]
    if _has_audio(src):
        amap = ["-map", "0:a:0"]
    else:
        cmd += ["-f", "lavfi", "-t", f"{dur:.3f}",
                "-i", "anullsrc=r=48000:cl=stereo"]
        amap = ["-map", "1:a:0"]
    cmd += ["-map", "0:v:0", *amap, "-vf", fit, "-r", str(fps),
            "-t", f"{dur:.3f}", *V_ARGS, *A_ARGS, str(out)]
    log(f"  preparing {src.name} to sit between the slides "
        f"({width}x{height}, {fps} fps) -- once; it is reused after this")
    run(cmd, f"conform insert {src.name}")
    return out


def _read_srt(path: Path) -> list[tuple[float, float, str]]:
    from realme.pipeline.captions import _secs
    cues, block = [], []
    for line in read_text(path).splitlines() + [""]:
        if line.strip():
            block.append(line)
            continue
        times = next((b for b in block if "-->" in b), None)
        if times:
            a, _, b = times.partition("-->")
            text = " ".join(x.strip() for x in block[block.index(times) + 1:])
            if text:
                cues.append((_secs(a), _secs(b.split()[0]), text))
        block = []
    return cues


def layout(pdir: Path, manifest, *, width: int, height: int, fps: int,
           log=print) -> list[Placed]:
    """Every insert, conformed, with the segment position it follows."""
    items = load(pdir)
    if not items:
        return []
    numbers = [seg.slide_index + 1 for seg in manifest.segments]
    placed = []
    for item in items:
        if item.after_slide <= 0:
            at = -1
        else:
            before = [i for i, n in enumerate(numbers) if n <= item.after_slide]
            at = before[-1] if before else -1
            if item.after_slide > max(numbers, default=0):
                log(f"  ! {item.title}: this lecture has {max(numbers)} slides, "
                    f"not {item.after_slide}; it plays at the end")
            elif at >= 0 and numbers[at] != item.after_slide:
                log(f"  ! {item.title}: slide {item.after_slide} is no longer "
                    f"in the deck; it plays after slide {numbers[at]}")
        path = conform(pdir, item, width=width, height=height, fps=fps, log=log)
        cues = []
        if item.captions and (Path(pdir) / item.captions).is_file():
            cues = _read_srt(Path(pdir) / item.captions)
        placed.append(Placed(insert=item, after_index=at, path=path,
                             duration=duration_of(path), cues=cues))
    return placed


def interleave(seg_videos: list[Path], placed: list[Placed]) -> list[Path]:
    """The segments with each insert after the one it follows."""
    out = [p.path for p in placed if p.after_index == -1]
    for i, v in enumerate(seg_videos):
        out.append(v)
        out += [p.path for p in placed if p.after_index == i]
    return out


def seconds(placed: list[Placed]) -> float:
    return sum(p.duration for p in placed)


def after(placed: list[Placed]) -> dict[int, float]:
    """Seconds inserted after each segment position (-1 = before the first)."""
    out: dict[int, float] = {}
    for p in placed:
        out[p.after_index] = out.get(p.after_index, 0.0) + p.duration
    return out
