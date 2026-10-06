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

# What a release consists of. One definition, used by the packager and by the
# digest alike -- if these two ever disagreed, every install would report itself
# out of date forever.
SHIP_DIRS = ("00_Windows", "02_Research", "03_App")
SHIP_FILES = (".gitignore", "README.md", "START_HERE.md", "HANDOVER.md",
              "realme.bat",
              # The introduction deck and the demo video it points at. The
              # deck says "the video ships beside this file", which is only
              # true if it is listed here -- `shipped_files` walks the
              # numbered folders, and the project root is this list alone.
              "RealMe_Introduction.pdf", "test_slides.mp4")

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
    return False


def shipped_files(root: Path) -> list[Path]:
    """Every file in a release, as paths relative to the project root, sorted."""
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
