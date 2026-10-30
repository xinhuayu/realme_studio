"""
A finished lecture kept as its video and text files, and edited slide by slide.

A render leaves two kinds of things behind. Text: the script as rendered,
every utterance's words and timing, the slide boundaries to the millisecond,
captions, chapters, cues. And media: one recording per utterance and one
video per slide, kept so the next render can reuse whatever did not change.
The media are most of the space -- on a server shared by a department, nearly
all of it.

This module keeps the text and the finished video, and lets the media go:

  compact     delete the per-utterance audio and the per-slide videos of
              every finished render. What remains: the .mp4, the captions,
              `<name>_segments.json` (where each slide starts and ends), the
              script as rendered with each utterance's words and timing
              (`_work/assembly/<name>.json`), the deck and the slide pictures.

  restore     cut the finished video back into one video per slide, at the
              boundaries the cut list recorded. Every slide begins on a
              keyframe -- each was encoded separately -- so this is a stream
              copy: no re-encoding, no loss.

  rerecord    re-record only the slides whose narration, picture or pacing
              changed (every utterance on such a slide is recorded again;
              there is nothing finer left to reuse), keep every other slide
              exactly as it is in the video, and join them.

  bundle      everything above in one .zip for the person to keep;
  unbundle    and back again, into a new project, so a server can keep
              nothing at all and the lecture can still be edited later.

Slides are the unit throughout. A slide is re-recorded whole or kept whole.

This sits beside the ordinary render and does not change it: a project that
was never compacted renders exactly as before, and `render` is what records
the changed slides here too.
"""
from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

from realme.core.media import MediaError, duration_of, require, run
from realme.core.textio import read_text

#: What `compact` deletes from `_work`: recordings and videos, all of which
#: are made from the text that stays. Slide pictures stay: the editor shows
#: them, and they are a small fraction of the space.
MEDIA = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg",
         ".mp4", ".mov", ".mkv", ".webm"}

#: A cut whose measured length differs from the cut list by more than this
#: is made again by re-encoding, at the exact boundaries.
CUT_TOLERANCE_S = 0.12

#: Largest a bundle may unpack to, and how many files it may hold. A bundle
#: is one lecture: a video, its captions, a deck and a few text files.
BUNDLE_MAX_BYTES = 6 * (1 << 30)
BUNDLE_MAX_FILES = 400

COMPACT_NOTE = "compacted.json"


# ------------------------------------------------------------------ state

def _work(pdir: Path) -> Path:
    return Path(pdir) / "_work"


def renders(pdir: Path) -> dict[str, dict]:
    """Finished renders that can be rebuilt from their video: name -> record.

    A render qualifies when its video, its cut list and its record (with the
    script as rendered) are all present.
    """
    from realme.pipeline.lecture import assemblies
    pdir = Path(pdir)
    out = {}
    for name, rec in assemblies(pdir).items():
        if (rec.get("manifest") and (pdir / f"{name}.mp4").is_file()
                and (pdir / f"{name}_segments.json").is_file()):
            out[name] = rec
    return out


def is_compact(pdir: Path, name: str) -> bool:
    """Has this render's per-slide media been let go?"""
    rec = renders(pdir).get(name)
    vids = (rec or {}).get("seg_videos") or []
    return bool(rec) and not (vids and all(Path(v).is_file() for v in vids))


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def media_bytes(pdir: Path) -> int:
    """What `compact` would free."""
    w = _work(pdir)
    return sum(_size(f) for f in w.rglob("*")
               if f.is_file() and f.suffix.lower() in MEDIA) if w.is_dir() else 0


# ---------------------------------------------------------------- compact

def compact(pdir: Path, log=print) -> dict:
    """Keep the finished video and the text; delete the working media.

    A render that could not be rebuilt from its video (no cut list, or
    recorded before renders kept their script) keeps its slide videos: they
    are then the only copy of that work.
    """
    pdir = Path(pdir)
    work = _work(pdir)
    if not work.is_dir():
        return {"freed_bytes": 0, "kept_renders": []}
    from realme.pipeline.lecture import assemblies
    can = renders(pdir)
    protected = set()
    for name, rec in assemblies(pdir).items():
        if name not in can:
            protected |= {Path(v).resolve() for v in rec.get("seg_videos", [])}
    freed = 0
    for f in sorted(work.rglob("*")):
        if (f.is_file() and f.suffix.lower() in MEDIA
                and f.resolve() not in protected):
            freed += _size(f)
            f.unlink()
    # Directories emptied by that, so the folder says what is left.
    for d in sorted((d for d in work.rglob("*") if d.is_dir()),
                    key=lambda d: len(d.parts), reverse=True):
        try:
            d.rmdir()
        except OSError:
            pass
    note = {"freed_bytes": freed, "kept_renders": sorted(can)}
    (work / COMPACT_NOTE).write_text(json.dumps(note, indent=2), encoding="utf-8")
    log(f"Kept the video and text files of {', '.join(sorted(can)) or 'nothing'}; "
        f"freed {freed / (1 << 20):.0f} MB of working audio and video.")
    return note


# ---------------------------------------------------------------- restore

def _cut_list(pdir: Path, name: str) -> dict:
    return json.loads(read_text(Path(pdir) / f"{name}_segments.json"))


def _copy_cut(master: Path, start: float, dur: float, out: Path) -> None:
    run([require("ffmpeg"), "-y", "-loglevel", "error",
         "-ss", f"{start:.3f}", "-i", str(master), "-t", f"{dur:.3f}",
         "-map", "0:v:0", "-map", "0:a:0", "-c", "copy",
         "-avoid_negative_ts", "make_zero", str(out)],
        f"copy slide {out.stem}")


def _exact_cut(master: Path, start: float, dur: float, out: Path,
               fps: int) -> None:
    from realme.pipeline.compose import A_ARGS, V_ARGS
    run([require("ffmpeg"), "-y", "-loglevel", "error",
         "-ss", f"{start:.3f}", "-i", str(master), "-t", f"{dur:.3f}",
         "-map", "0:v:0", "-map", "0:a:0", "-r", str(fps),
         *V_ARGS, *A_ARGS, str(out)],
        f"cut slide {out.stem}")


def restore(pdir: Path, name: str, log=print) -> list[Path]:
    """One video per slide of `name`, cut from its finished video.

    Returns the slide videos in script order, and records them as that
    render's own, so inserting a video between slides works again too.
    """
    from realme.pipeline.lecture import _record_assembly
    from realme.pipeline.split import keyframe_times
    pdir = Path(pdir)
    rec = renders(pdir).get(name)
    if not rec:
        raise MediaError(f"{name}.mp4 cannot be rebuilt here: its cut list "
                         f"({name}_segments.json) or its record is missing.")
    have = [Path(v) for v in rec.get("seg_videos", [])]
    if have and all(v.is_file() for v in have):
        return have
    cuts = _cut_list(pdir, name)
    spans = cuts.get("segments", [])
    n = len(rec["manifest"].get("segments", []))
    if len(spans) != n:
        raise MediaError(f"{name}_segments.json lists {len(spans)} slides but "
                         f"the render had {n}. Render the lecture again.")
    master = pdir / f"{name}.mp4"
    total = duration_of(master)
    fps = int(rec.get("fps") or cuts.get("render", {}).get("fps") or 30)
    keys = keyframe_times(master)

    def snap(t: float) -> float | None:
        near = [k for k in keys if abs(k - t) <= 0.25]
        return min(near, key=lambda k: abs(k - t)) if near else None

    out_dir = _work(pdir) / "video"
    out_dir.mkdir(parents=True, exist_ok=True)
    made, copied = [], 0
    log(f"Taking {n} slide(s) out of {master.name}")
    for i, sp in enumerate(spans):
        start, end = float(sp["start_s"]), min(float(sp["end_s"]), total)
        out = out_dir / f"kept_{name}_{i:03d}.mp4"
        a = snap(start)
        b = total if i == len(spans) - 1 and end >= total - 0.05 else snap(end)
        ok = False
        if a is not None and b is not None and b > a:
            _copy_cut(master, a, b - a, out)
            try:
                ok = abs(duration_of(out) - (end - start)) <= CUT_TOLERANCE_S
            except MediaError:
                ok = False
            copied += ok
        if not ok:
            _exact_cut(master, start, end - start, out, fps)
        made.append(out)
    log(f"  {copied} of {n} copied exactly as they are, "
        f"{n - copied} re-encoded at the exact boundary")
    _record_assembly(_work(pdir), {**rec, "seg_videos": [str(p) for p in made]})
    return made


# --------------------------------------------------------------- rerecord

def _slide_width(work: Path) -> int:
    """The width the slide pictures were drawn at. A picture's digest is
    how a changed slide is noticed, so a slide drawn again must be drawn at
    the same width or every slide looks changed. The editor draws them at
    1920 whatever the video's size."""
    try:
        return int(json.loads(read_text(work / "slides.json"))["width_px"])
    except (OSError, ValueError, KeyError, TypeError):
        return 1920


def _rasterize(deck: Path, work: Path, width: int) -> None:
    from realme.pipeline import slides as slides_mod
    pdf = slides_mod.deck_to_pdf(deck, work)
    slides_mod.rasterize(pdf, work / "slides", width_px=width)


def _shape(s) -> tuple:
    """What a slide's recording is made from (the same test `reassemble`
    uses to call a script edited)."""
    p = s.prosody
    return (s.slide_index, s.spoken_text.strip(), round(p.pace, 3),
            round(p.lead_in_s, 2), round(p.pause_after_s, 2),
            tuple(s.cues), s.layout)


def plan(pdir: Path, name: str, deck: Path | None = None,
         also: list[int] | None = None) -> dict:
    """Which slides of the current script are kept from the video and which
    are recorded again, and why."""
    from realme.core.schema import Manifest
    from realme.pipeline.lecture import load_script
    pdir = Path(pdir)
    rec = renders(pdir).get(name)
    if not rec:
        raise MediaError(f"No finished {name}.mp4 with its cut list here.")
    rendered = Manifest.model_validate(rec["manifest"])
    current = load_script(pdir)
    old = {s.segment_id: s for s in rendered.segments}
    digests: dict[int, str] = {}
    if deck is not None:
        from realme.pipeline import slides as slides_mod
        from realme.pipeline.revise import slide_digest
        pngs = slides_mod.slide_pages(_work(pdir) / "slides")
        digests = {i: slide_digest(p) for i, p in enumerate(pngs)}
    also = set(also or [])
    keep, redo = [], []
    for seg in current.segments:
        was = old.get(seg.segment_id)
        why = ""
        if was is None:
            why = "new"
        elif _shape(was) != _shape(seg):
            why = "narration or pacing changed"
        elif seg.slide_index + 1 in also:
            why = "asked"
        elif (digests and was.slide_digest
              and digests.get(seg.slide_index) not in (None, was.slide_digest)
              and seg.rerender is not False):
            why = "the slide's picture changed"
        (redo if why else keep).append((seg.segment_id, why))
    order_same = ([s.segment_id for s in rendered.segments]
                  == [s.segment_id for s in current.segments])
    return {"keep": [k for k, _ in keep], "redo": redo,
            "unchanged": not redo and order_same, "record": rec,
            "rendered": rendered, "current": current}


def rerecord(pdir: Path, deck: Path, tts, *, name: str, also=None,
             mode: str = "natural", pause_scale: float = 1.0,
             check_audio: bool = True, keep_lean: bool = True,
             log=print, progress=lambda *_: None) -> dict:
    """Record again only the slides that changed; keep the rest from the
    video; join. Returns what `render` returns, plus which slides were
    re-recorded.

    The render's own settings (size, frame rate, layout, highlighting) are
    used for the new slides, so they sit beside the kept ones without a seam.
    """
    from realme.pipeline import lecture as lec
    pdir, deck = Path(pdir), Path(deck)
    work = _work(pdir)
    from realme.pipeline import slides as slides_mod
    if not slides_mod.slide_pages(work / "slides"):
        _rasterize(deck, work, _slide_width(work))
    p = plan(pdir, name, deck, also)
    rec, current = p["record"], p["current"]
    voice_was = rec.get("voice", "")
    voice_now = getattr(tts, "name", tts.__class__.__name__)
    if voice_was and voice_now != voice_was:
        raise MediaError(
            f"{name}.mp4 was recorded with {voice_was}; re-recording some "
            f"slides with {voice_now} would mix two voices in one video. "
            f"Render the whole lecture again to change the voice.")
    redo_ids = {sid for sid, _ in p["redo"]}
    for sid, why in p["redo"]:
        seg = next(s for s in current.segments if s.segment_id == sid)
        log(f"slide {seg.slide_index + 1}: re-record ({why})")
    if p["unchanged"]:
        log("Nothing changed since this video was made; it is kept as it is.")
        return {"master": pdir / f"{name}.mp4", "report": json.loads(
            read_text(pdir / f"{name}_verification.json"))
            if (pdir / f"{name}_verification.json").is_file()
            else {"verdict": "PASS (unchanged)"}, "rerecorded": []}

    kept_videos = restore(pdir, name, log=log)
    by_id = {s.segment_id: (s, v) for s, v in
             zip(p["rendered"].segments, kept_videos)}

    new_by_id = {}
    if redo_ids:
        sub = current.model_copy(update={"segments": [
            s for s in current.segments if s.segment_id in redo_ids]})
        tmp = work / "rerecord"
        shutil.rmtree(tmp, ignore_errors=True)
        (tmp / "_work").mkdir(parents=True)
        # The deck's pictures are already here; the partial render reuses them.
        shutil.copytree(work / "slides", tmp / "_work" / "slides")
        out = lec.render(deck, tmp, sub, tts,
                         layout=rec.get("layout") or "slide_only",
                         width=int(rec["width"]), height=int(rec["height"]),
                         fps=int(rec["fps"]),
                         signalling=rec.get("signalling") or "highlight",
                         mode=mode, pause_scale=pause_scale,
                         check_audio=check_audio, log=log,
                         progress=lambda f: progress(0.85 * f))
        sub_rec = lec.assemblies(tmp)[Path(out["master"]).stem]
        from realme.core.schema import Manifest
        measured = Manifest.model_validate(sub_rec["manifest"]).segments
        dest = work / "video"
        dest.mkdir(parents=True, exist_ok=True)
        for seg, v in zip(measured, sub_rec["seg_videos"]):
            to = dest / f"new_{name}_{seg.segment_id:04d}_{Path(v).name}"
            shutil.move(v, to)
            new_by_id[seg.segment_id] = (seg, to)

    segs, videos = [], []
    for seg in current.segments:
        s, v = new_by_id.get(seg.segment_id) or by_id[seg.segment_id]
        segs.append(s)
        videos.append(Path(v))
    final = current.model_copy(update={"segments": segs})

    log("Joining the kept and the re-recorded slides")
    progress(0.9)
    master, srt, vtt, ch, report, placed = lec._join_master(
        pdir, final, videos, name, width=int(rec["width"]),
        height=int(rec["height"]), fps=int(rec["fps"]), log=log)
    report["voice"], report["is_draft"] = voice_was or voice_now, rec.get("is_draft")
    report["rerecorded_slides"] = sorted(s.slide_index + 1 for s in segs
                                         if s.segment_id in redo_ids)
    report["kept_slides"] = len(segs) - len(redo_ids)
    (pdir / f"{name}_verification.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    lec._write_cut_list(pdir, name, final, master, placed,
                        width=rec["width"], height=rec["height"], fps=rec["fps"],
                        layout=rec.get("layout", ""),
                        signalling=rec.get("signalling", ""),
                        voice=report["voice"])
    lec._record_assembly(work, {**rec, "seg_videos": [str(v) for v in videos],
                                "manifest": json.loads(final.model_dump_json())})
    (work / "manifest.json").write_text(final.model_dump_json(indent=2),
                                        encoding="utf-8")
    shutil.rmtree(work / "rerecord", ignore_errors=True)
    if keep_lean:
        compact(pdir, log=log)
    log(report["verdict"])
    progress(1.0)
    return {"master": master, "srt": srt, "vtt": vtt, "chapters": ch,
            "report": report, "rerecorded": report["rerecorded_slides"]}


# ----------------------------------------------------------------- bundle

_README = """This is a RealMe lecture, kept as its video and text files.

  {name}.mp4               the lecture
  {name}.srt / .vtt        captions
  {name}_chapters.txt      chapter marks (for YouTube and the like)
  {name}_segments.json     where every slide starts and ends, to the millisecond
  {name}_verification.json what the last check of the video found
  narration.txt            the narration, slide by slide
  record/                  the script as rendered, with every utterance's
                           words and timing (what RealMe needs to edit it)
  deck/                    the slides it was made from
  inserts/                 videos placed between slides, if any

To change it later: in RealMe Studio, Lecture -> Open a lecture -> Bring back
a lecture from its .zip. Edit the narration of any slide and render: only the
slides you changed are recorded again; the rest are taken from this video.
"""


def _narration(manifest: dict) -> str:
    out = []
    for s in manifest.get("segments", []):
        out.append(f"Slide {s.get('slide_index', 0) + 1}\n"
                   f"{(s.get('spoken_text') or '').strip()}\n")
    return "\n".join(out)


def bundle(pdir: Path, out_zip: Path, deck: Path | None = None) -> Path:
    """The lecture as one .zip: every finished render, its text, the deck."""
    from realme.pipeline import inserts as INS
    pdir, out_zip = Path(pdir), Path(out_zip)
    found = renders(pdir)
    if not found:
        raise MediaError("This lecture has no finished video to download yet.")
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for name, rec in found.items():
            for suffix in (".mp4", ".srt", ".vtt", "_chapters.txt",
                           "_segments.json", "_verification.json", "_cues.json"):
                f = pdir / f"{name}{suffix}"
                if f.is_file():
                    z.write(f, f.name, compress_type=zipfile.ZIP_STORED
                            if suffix == ".mp4" else zipfile.ZIP_DEFLATED)
            slim = {k: v for k, v in rec.items() if k != "seg_videos"}
            z.writestr(f"record/{name}.json", json.dumps(slim, indent=2))
        main = sorted(found, key=lambda n: (found[n].get("is_draft") is not False, n))[0]
        z.writestr("narration.txt", _narration(found[main]["manifest"]))
        script = _work(pdir) / "manifest.json"
        if script.is_file():
            z.write(script, "record/script.json")
        from realme.pipeline import slides as slides_mod
        pngs = slides_mod.slide_pages(_work(pdir) / "slides")
        if pngs:
            from PIL import Image
            with Image.open(pngs[0]) as im:
                z.writestr("record/slides.json",
                           json.dumps({"width_px": im.size[0]}))
        if deck is not None and Path(deck).is_file():
            z.write(deck, f"deck/{Path(deck).name}")
        items = INS.load(pdir)
        if items and (pdir / INS.LIST_NAME).is_file():
            z.write(pdir / INS.LIST_NAME, f"inserts/{INS.LIST_NAME}")
            for it in items:
                for rel in (it.file, it.captions):
                    if rel and (pdir / rel).is_file():
                        z.write(pdir / rel, f"inserts/{Path(rel).name}")
        z.writestr("README.txt", _README.format(name=main))
    return out_zip


def _safe_member(name: str) -> str | None:
    """The member's place in the project folder, or None to skip it."""
    p = Path(name)
    if p.is_absolute() or ".." in p.parts or "\\" in name or not p.name \
            or p.name.startswith("."):
        return None
    if len(p.parts) == 1:
        return p.name
    if len(p.parts) == 2 and p.parts[0] in ("record", "deck", "inserts"):
        return f"{p.parts[0]}/{p.name}"
    return None


def _sanitize_inserts(pdir: Path) -> None:
    """The insert list from a .zip names its files; it may name only the
    ones the .zip brought, in the project's own inserts folder.

    It comes from outside, and every path in it is opened by ffmpeg: a list
    naming `../../someone-else/voice.wav` would put another file's sound in
    this lecture. So each entry is pointed at its own name in `inserts/`,
    and one whose video did not come with it is dropped.
    """
    from realme.pipeline import inserts as INS
    f = pdir / INS.LIST_NAME
    if not f.is_file():
        return
    try:
        items = INS.load(pdir)
    except Exception:
        f.unlink()
        return
    keep = []
    for it in items:
        it.file = f"{INS.FOLDER}/{Path(it.file).name}"
        if it.captions:
            cap = f"{INS.FOLDER}/{Path(it.captions).name}"
            it.captions = cap if (pdir / cap).is_file() else None
        if (pdir / it.file).is_file() and not Path(it.file).name.startswith("."):
            keep.append(it)
    INS.save(pdir, keep)


def unbundle(zip_path: Path, pdir: Path, log=print) -> dict:
    """A bundle back into a project folder `pdir` (which must not exist)."""
    from realme.core.schema import Manifest
    from realme.pipeline.lecture import _record_assembly
    zip_path, pdir = Path(zip_path), Path(pdir)
    if pdir.exists():
        raise MediaError(f"{pdir.name} already exists.")
    try:
        z = zipfile.ZipFile(zip_path)
    except (zipfile.BadZipFile, OSError):
        raise MediaError("That is not a RealMe lecture .zip.")
    with z:
        infos = [i for i in z.infolist() if not i.is_dir()]
        if len(infos) > BUNDLE_MAX_FILES:
            raise MediaError("That .zip holds too many files to be one lecture.")
        if sum(i.file_size for i in infos) > BUNDLE_MAX_BYTES:
            raise MediaError("That .zip would unpack to more than "
                             f"{BUNDLE_MAX_BYTES >> 30} GB.")
        names = {i.filename for i in infos}
        records = sorted(n for n in names if n.startswith("record/")
                         and n.endswith(".json")
                         and n not in ("record/script.json", "record/slides.json"))
        if "record/script.json" not in names or not records:
            raise MediaError("That .zip has no RealMe record in it (record/"
                             "script.json). Was it downloaded from RealMe?")
        try:
            pdir.mkdir(parents=True)
            work = pdir / "_work"
            work.mkdir()
            for info in infos:
                where = _safe_member(info.filename)
                if where is None:
                    continue
                if where.startswith("record/"):
                    continue
                if where.startswith("deck/"):
                    dest = pdir / Path(where).name
                elif where.startswith("inserts/"):
                    from realme.pipeline import inserts as INS
                    rel = Path(where).name
                    dest = pdir / (INS.LIST_NAME if rel == INS.LIST_NAME
                                   else Path(INS.FOLDER) / rel)
                else:
                    dest = pdir / where
                dest.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out, 1 << 20)
            if "record/slides.json" in names:
                (work / "slides.json").write_bytes(z.read("record/slides.json"))
            script = Manifest.model_validate_json(z.read("record/script.json"))
            (work / "manifest.json").write_text(script.model_dump_json(indent=2),
                                                encoding="utf-8")
            kept = []
            for r in records:
                rec = json.loads(z.read(r))
                name = Path(r).stem
                if (_safe_member(f"{name}.mp4") != f"{name}.mp4"
                        or not (pdir / f"{name}.mp4").is_file()):
                    continue
                Manifest.model_validate(rec.get("manifest") or {})
                _record_assembly(work, {**rec, "name": name, "seg_videos": []})
                kept.append(name)
        except Exception:
            shutil.rmtree(pdir, ignore_errors=True)
            raise
    _sanitize_inserts(pdir)
    decks = [f for f in pdir.iterdir() if f.is_file()
             and f.suffix.lower() in {".pdf", ".pptx", ".ppt", ".odp"}]
    if decks:
        (work / "deck_path.txt").write_text(str(decks[0]), encoding="utf-8")
        # The editor shows the slides; draw them now, at the width they had.
        try:
            _rasterize(decks[0], work, _slide_width(work))
        except Exception as e:                       # e.g. .pptx, no soffice
            log(f"  the slide pictures could not be drawn yet ({e})")
    (work / COMPACT_NOTE).write_text(json.dumps(
        {"freed_bytes": 0, "kept_renders": kept, "from_bundle": zip_path.name},
        indent=2), encoding="utf-8")
    log(f"Brought back {', '.join(kept) or 'no video'} into {pdir.name}.")
    return {"project": pdir.name, "renders": kept,
            "deck": decks[0].name if decks else None}
