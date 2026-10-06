"""
A revision that has been worked out but not yet accepted.

Comparing a new deck against a finished project used to apply itself. The job
deleted the old slide images, moved the new deck into the project folder and
overwrote the manifest before anyone had read a word of the result -- so a
comparison you disagreed with had already destroyed the thing you would have
gone back to. There was no back.

Everything a revision produces now lands here instead, under
`_work/pending/`, and the live project is untouched until the Save button says
so:

    _work/pending/manifest.json   the proposed script, edited in place
    _work/pending/slides/         the new deck, rasterised
    _work/pending/deck/<name>     the uploaded deck itself
    _work/pending/plan.json       what the comparison concluded

`commit` moves all four into place in one go; `discard` deletes the folder and
leaves the project exactly as it was. One module knows this layout, and every
reader asks it rather than joining the paths itself -- the editor, the slide
images, the note drafting and the renderer all go through `script_path` and
`slides_dir`.
"""
from __future__ import annotations
import json, shutil
from pathlib import Path
from realme.core.textio import read_text


def root(pdir) -> Path:
    return Path(pdir) / "_work" / "pending"


def exists(pdir) -> bool:
    return (root(pdir) / "manifest.json").is_file()


def script_path(pdir) -> Path:
    """The manifest the editor should be showing: proposed if one is waiting."""
    p = root(pdir) / "manifest.json"
    return p if p.is_file() else Path(pdir) / "_work" / "manifest.json"


def slides_dir(pdir) -> Path:
    """The slide images that go with `script_path`."""
    d = root(pdir) / "slides"
    return d if exists(pdir) and d.is_dir() else Path(pdir) / "_work" / "slides"


def plan(pdir) -> dict | None:
    p = root(pdir) / "plan.json"
    if not p.is_file():
        return None
    try:
        return json.loads(read_text(p))
    except (ValueError, OSError):
        return None


def begin(pdir) -> Path:
    """Clear any previous proposal and hand back the folder to build one in.

    Called before the upload is written, because the upload goes INTO this
    folder. An earlier version cleared the folder at the end instead, which
    deleted the deck and slides it was about to record.
    """
    pdir = Path(pdir)
    d = root(pdir)
    shutil.rmtree(d, ignore_errors=True)
    (d / "deck").mkdir(parents=True, exist_ok=True)
    return d


def finish(pdir, manifest, plan_data: dict) -> Path:
    """Record the proposed script beside the deck and slides already staged.

    Writing the manifest is what makes the proposal exist -- `exists` tests for
    it -- so it goes last, and a comparison that dies halfway leaves a folder
    the editor ignores rather than half a revision it would show.
    """
    d = root(Path(pdir))
    if not (d / "slides").is_dir():
        raise FileNotFoundError(
            "The revision has no slides; the comparison did not get far enough "
            "to be worth keeping.")
    (d / "plan.json").write_text(json.dumps(plan_data, indent=2), encoding="utf-8")
    (d / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return d


def commit(pdir) -> dict:
    """Accept the revision: the new deck, its slides and its script take over."""
    pdir = Path(pdir)
    d = root(pdir)
    if not exists(pdir):
        raise FileNotFoundError("There is no revision waiting to be applied.")
    work = pdir / "_work"
    decks = [f for f in (d / "deck").iterdir() if f.is_file()] if (d / "deck").is_dir() else []
    if not decks:
        raise FileNotFoundError(
            "The revision is missing the deck it was made from; discard it and "
            "compare again.")
    deck = decks[0]

    # Order matters. The manifest goes last, so an interruption leaves a project
    # whose script still matches its slides rather than one that points at
    # pictures that were never moved.
    if (d / "slides").is_dir():
        shutil.rmtree(work / "slides", ignore_errors=True)
        shutil.move(str(d / "slides"), str(work / "slides"))
    final = pdir / deck.name
    shutil.move(str(deck), str(final))
    (work / "deck_path.txt").write_text(str(final), encoding="utf-8")
    shutil.move(str(d / "manifest.json"), str(work / "manifest.json"))
    shutil.rmtree(d, ignore_errors=True)
    return {"deck": final.name}


def discard(pdir) -> bool:
    had = exists(pdir)
    shutil.rmtree(root(pdir), ignore_errors=True)
    return had
