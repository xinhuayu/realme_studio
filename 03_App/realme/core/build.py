"""
Which release is installed, and is the zip in the folder still waiting?

**Never by timestamp.** That is the whole point of this module.

The update archive is built with every entry deliberately stamped 24 hours in
the past. ZIP stores a naive local time with no timezone; this machine builds in
UTC and Windows extracts as local, so without the backdating a user west of UTC
gets files dated in the future -- which is exactly what made Ninja fail with
"manifest still dirty after 100 tries" and cost an afternoon.

That backdating is correct, and it makes mtimes useless for answering "has this
update been applied?". The archive *file* carries the time it was copied onto
the machine, which is now; the files it extracts carry a time a day ago. So the
old check -- zip newer than cli.py -- was true immediately after a perfectly
successful update, and would have gone on nagging forever. A warning that is
always on is a warning nobody reads.

So the question is answered by CONTENT. The packager hashes everything it is
about to ship and writes that digest to `03_App/realme/BUILD_ID`, inside the
archive and into the source tree together. Comparing the archive's copy of that
one small file with the installed copy is exact, costs two reads, and does not
care what any clock says.

For an archive built before stamping existed there is a fallback that is still
timestamp-free: compare each entry's stored size against the file on disk. Sizes
live in the zip's central directory, so nothing is decompressed.
"""
from __future__ import annotations
import hashlib
import zipfile
from pathlib import Path

# The stamp travels with the package, so an install that was copied rather than
# extracted still knows what it is.
STAMP_REL = "03_App/realme/BUILD_ID"

#: The guide's video, at the root beside the slides it was made from. A render
#: leaves it inside its own project folder; `ensure_guide_video` copies it
#: here. It is optional: a release carries it when it has been rendered, and
#: the slides and narration script travel either way.
#:
#: Every name carries its version, sources and destination alike, and only a
#: source whose name states the same version is ever copied -- so rendering
#: an older guide can never overwrite a newer one.
GUIDE_VERSION = 4
_GUIDE_SOURCE_PATTERNS = ("claude_videos/guide_v{v}/guide_v{v}.mp4",
                          "guide_v{v}/guide_v{v}.mp4",
                          # where `realme lecture RealMe_Guide_v4.pdf -o guide_v4`
                          # puts it: the project is named after the deck
                          "guide_v{v}/RealMe_Guide_v{v}.mp4")

#: The guide's slides and the narration they are read from. To render the
#: video in your own voice:
#:     realme import-script RealMe_Guide_notes_v4.txt --project guide_v4
#:     realme lecture RealMe_Guide_v4.pdf --script imported -o guide_v4
GUIDE_DECK = "RealMe_Guide_v4.pdf"
GUIDE_NOTES = "RealMe_Guide_notes_v4.txt"
GUIDE_PPTX = "RealMe_Guide_v4.pptx"


def guide_video_name(version: int = GUIDE_VERSION) -> str:
    """The guide video's name at the project root, for one version."""
    return f"RealMe_Guide_narrated_v{int(version)}.mp4"


def guide_video_sources(version: int = GUIDE_VERSION) -> tuple[str, ...]:
    """Where a render of that version leaves it. Several, because the guide
    has been rendered from both a deck and a scenes module, and a release
    should pick up whichever is there rather than making somebody copy a
    file by hand and refusing when they forget."""
    return tuple(p.format(v=int(version)) for p in _GUIDE_SOURCE_PATTERNS)


def guide_version_of(path) -> int | None:
    """The guide version a path names, or None if it names none or several.

    Read from every `v<N>` token in the path (`guide_v3/guide_v3.mp4` -> 3).
    A path with no version, or with two different ones, is not trusted.
    """
    import re
    found = {int(n) for n in re.findall(r"(?:^|[/\\_.-])v(\d+)(?=$|[/\\_.-])",
                                         Path(path).as_posix())}
    return found.pop() if len(found) == 1 else None


GUIDE_VIDEO = guide_video_name()
GUIDE_VIDEO_SOURCES = guide_video_sources()


def ensure_guide_video(root: Path, log=print,
                       version: int = GUIDE_VERSION) -> bool:
    """Put the rendered guide video where a release expects it.

    Called by both packagers. Returns False, and says how to make it, when
    no rendered video exists yet; the packagers carry on without it.

    Only a source of the same `version` is ever copied, and only to that
    version's own name.
    """
    import shutil
    name = guide_video_name(version)
    sources = guide_video_sources(version)
    dst = root / name
    found = [root / s for s in sources
             if guide_version_of(s) == version and (root / s).is_file()]
    src = max(found, key=lambda f: f.stat().st_mtime) if found else None
    if dst.is_file() and (src is None
                          or dst.stat().st_mtime >= src.stat().st_mtime):
        return True
    if src is None:
        log(f"  ! {name} is missing and none of "
            f"{', '.join(sources)} is there to copy from.")
        log(f"    Optional. To include it, render the guide (README.md -> "
            f"The guide) so it lands at {sources[2] if len(sources) > 2 else sources[0]}, "
            f"or copy it to {name} directly.")
        return False
    shutil.copy2(src, dst)
    log(f"  copied {src.relative_to(root).as_posix()} -> {name} "
        f"({dst.stat().st_size / 1e6:.1f} MB)")
    return True

# What a release consists of. One definition, used by the packager and by the
# digest alike -- if these two ever disagreed, every install would report itself
# out of date forever.
SHIP_DIRS = ("00_Windows", "03_App")
SHIP_FILES = (".gitignore", "README.md", "START_HERE.md", "HANDOVER.md",
              "DESIGN.md",
              "LICENSE", "CITATION.cff",
              "realme.bat",
              # The narrated guide: the slides, the script they were read
              # from, and the video itself. The README links to all three, and
              # a link that only resolves on the machine it was written on is
              # not a link. `shipped_files` walks the numbered folders; the
              # project root is this list alone.
              #
              # It replaces the older `RealMe_Introduction.pdf` and
              # `test_slides.mp4` pair, which described a tool with one voice
              # that ran only locally. Those files are still in the working
              # tree and no longer travel.
              # v2's deck and script stay: they are small, and they are what
              # the "narrate an existing deck" walkthrough in the README is
              # built from. Its rendered video does not -- two guide videos
              # would put 45 MB in an update zip to say the same thing twice,
              # and the README gives the one command that rebuilds it.
              GUIDE_DECK, GUIDE_NOTES, GUIDE_PPTX,
              GUIDE_VIDEO)

# `_Archive` is deliberately absent. It holds the development history -- the
# superseded architecture drafts, the design reviews, the working journal that
# START_HERE.md was rewritten from. All of it is worth keeping and none of it
# is worth sending: a colleague opening the package should find a tool, not a
# record of how it came to be one. The underscore matches the convention in
# `00_Windows`, where `_`-prefixed files are the ones nobody runs directly.


def _excluded(rel: Path) -> bool:
    """
    Installed payload, caches and secrets never ship.

    `tools/` is the important one: ~3 GB of ffmpeg, model weights and a
    machine-specific compiled engine, downloaded once. An update that wrote
    there would destroy it, so it is excluded here and refused again by
    0_Update.bat -- a convention plus a check, because the convention alone is
    what conventions always are.
    """
    parts = set(rel.parts)
    if parts & {"tools", "__pycache__", ".git", "renders", "projects",
                ".ruff_cache", ".pytest_cache", "realme.egg-info"}:
        return True
    if rel.suffix in {".pyc", ".zip"}:
        return True
    # One machine's old engine path. Its own header says to delete it once
    # the migration is done; until then it must not travel.
    if rel.name == "_legacy_qwen3.bat":
        return True
    if rel.name.endswith(".env") and rel.name != "example.env":
        return True
    # A render's scratch folder, wherever one was left.
    if "_work" in parts:
        return True
    # Recordings and renders left inside the program folders (a lecture run
    # from 00_Windows writes its output there). The program needs none, and a
    # voice recording must never travel by accident. The guide's own video is
    # a root-level release file and is not affected.
    if (rel.parts and rel.parts[0] in ("00_Windows", "03_App")
            and rel.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".wav",
                                       ".m4a", ".mp3", ".flac", ".ogg", ".aac"}):
        return True
    # Anything `realme setup` retires into _Archive never ships, even from a
    # tree where setup has not run yet.
    posix = rel.as_posix()
    from realme.setup_check import RETIRED_CODE
    for rels in RETIRED_CODE.values():
        for r in rels:
            if posix == r or posix.startswith(r.rstrip("/") + "/"):
                return True
    return False


def shipped_files(root: Path) -> list[Path]:
    """Every file in a release, as paths relative to the project root, sorted.

    The BUILD_ID stamp is ALWAYS listed, whether or not it exists on disk yet.
    It used to be listed only when found, so on a fresh tree the packager
    took the file list, then wrote the stamp, and shipped a zip without it --
    the first RealMe_Update.zip from any new checkout was unstamped, and
    `update_pending` fell back to guessing by size. A release carries a stamp
    by definition; the packager writes it before reading any bytes, and
    anyone else reading the list must allow for it not existing yet.
    """
    out: list[Path] = []
    for d in SHIP_DIRS:
        base = root / d
        if not base.is_dir():
            continue
        for f in base.rglob("*"):
            if f.is_file():
                rel = f.relative_to(root)
                if not _excluded(rel):
                    out.append(rel)
    for name in SHIP_FILES:
        if (root / name).is_file():
            out.append(Path(name))
    stamp = Path(STAMP_REL)
    if (root / stamp).parent.is_dir() and stamp not in out:
        out.append(stamp)
    return sorted(out, key=lambda p: p.as_posix())


def compute_id(root: Path) -> str:
    """
    A digest of the release. Path and contents, in a fixed order.

    Paths are included because a file being *moved* is a change even when every
    byte survives, and a rename that the digest ignored would silently look like
    an update that had already been applied.
    """
    h = hashlib.sha256()
    for rel in shipped_files(root):
        if rel.as_posix() == STAMP_REL:
            continue           # the stamp cannot contain a hash of itself
        h.update(rel.as_posix().encode())
        h.update(b"\0")
        h.update(hashlib.sha256((root / rel).read_bytes()).digest())
    return h.hexdigest()[:16]


def installed_id(root: Path | None = None) -> str | None:
    """The release currently on disk, or None if this predates stamping."""
    stamp = (root / STAMP_REL) if root else Path(__file__).resolve().parents[1] / "BUILD_ID"
    try:
        from realme.core.textio import read_text
        return read_text(stamp).strip() or None
    except OSError:
        return None


def archive_id(zip_path: Path) -> str | None:
    """The release inside an update archive, without extracting it."""
    try:
        with zipfile.ZipFile(zip_path) as z:
            return z.read(STAMP_REL).decode("utf-8").strip() or None
    except (OSError, KeyError, zipfile.BadZipFile, UnicodeDecodeError):
        return None


def _differs_by_size(zip_path: Path, root: Path) -> bool:
    """
    Fallback for an unstamped archive: does any entry differ in size on disk?

    Reads only the central directory. Same-size-but-different-content slips
    through, which is why this is the fallback and not the check -- but it can
    never be fooled by a clock, and that is the failure being fixed.
    """
    try:
        with zipfile.ZipFile(zip_path) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                target = root / info.filename
                try:
                    if target.stat().st_size != info.file_size:
                        return True
                except OSError:
                    return True
    except (OSError, zipfile.BadZipFile):
        return False
    return False


def update_pending(root: Path) -> tuple[bool, str]:
    """
    Is RealMe_Update.zip in `root` a release other than the one installed?

    Returns (pending, why). `why` is short enough to print and specific enough
    to act on -- "no archive present" and "already applied" are different
    answers to "why am I not being told to update", and both are worth seeing.
    """
    zip_path = root / "RealMe_Update.zip"
    if not zip_path.is_file():
        return (False, "no RealMe_Update.zip in the folder")
    a, i = archive_id(zip_path), installed_id(root)
    if a and i:
        return (a != i, f"archive {a}, installed {i}")
    if _differs_by_size(zip_path, root):
        return (True, "unstamped archive; some files differ in size")
    return (False, "unstamped archive; every file matches in size")
