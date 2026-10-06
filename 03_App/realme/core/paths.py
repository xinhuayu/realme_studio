"""
Where the installed payload lives: ffmpeg, the voice engine, the draft voice.

There was a split here, and it was a bug. `_get_ffmpeg.bat` runs from the
project root, so it wrote `tools\\ffmpeg` beside `00_Windows`. `engine_home()`
resolved relative to the *package*, so it wrote `03_App\\tools\\qwen3`. Two
folders called `tools`, in different places, both correct according to whichever
piece of code you happened to read.

Nothing was broken by it -- `media.require()` searched both, and the engine
resolver found its own -- but the layout was incoherent, and one of those
folders sits inside the directory an update extracts over. Multi-gigabyte model
weights do not belong there.

So: **the project root is the one true tools directory.** Reading still checks
the old location, because an engine already installed there is 2.5 GB that
nobody should have to download again; writing only ever targets the root.
"""
from __future__ import annotations
import os
from pathlib import Path


def project_root() -> Path | None:
    """
    The folder holding 00_Windows and 03_App, if this is a checkout.

    Found by marker rather than by counting `.parent`s: a count is silently
    wrong the moment the package is installed somewhere else, and it fails as a
    path that looks plausible instead of as an error.
    """
    here = Path(__file__).resolve()
    for base in here.parents:
        if (base / "03_App").is_dir() or (base / "00_Windows").is_dir():
            return base
    return None


def package_root() -> Path:
    """…/03_App -- the directory the package was installed from."""
    return Path(__file__).resolve().parents[2]


def tools_dir() -> Path:
    """Where new installs go. One place, always."""
    env = os.environ.get("REALME_TOOLS")
    if env:
        return Path(env)
    root = project_root()
    return (root or package_root()) / "tools"


def tools_search() -> list[Path]:
    """
    Every place to LOOK, best first.

    The second entry is the legacy one. It is searched, never written to, so an
    existing install keeps working and a new one does not add to the mess.
    """
    cands = [tools_dir(), package_root() / "tools"]
    # Upward from the package, for any ancestor that has a `tools` folder.
    #
    # `tools_dir()` depends on `project_root()`, which finds the checkout by
    # looking for `03_App` -- and that only works for an editable install. A
    # package installed into site-packages has no checkout above it, so the
    # answer fell through to the CURRENT WORKING DIRECTORY, and the Studio
    # then found the voices or not depending on which folder it happened to
    # be started from. A voice is either installed or it is not; where you
    # were standing should not come into it.
    here = Path(__file__).resolve()
    for base in here.parents:
        c = base / "tools"
        if c.is_dir():
            cands.append(c)
    cands.append(Path.cwd() / "tools")
    out, seen = [], set()
    for p in cands:
        p = Path(p)
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def find_tool(name: str) -> Path | None:
    """An existing <tools>/<name> directory, wherever it already lives."""
    for base in tools_search():
        c = base / name
        if c.is_dir():
            return c
    return None
