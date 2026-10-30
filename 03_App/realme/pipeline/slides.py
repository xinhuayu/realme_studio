"""
Slide ingestion.

Deliberate choice: render the deck to PDF once, then rasterize at whatever DPI
the target resolution needs. The Google Slides API `getThumbnail` endpoint caps
at 1600px wide, is an 'expensive read' (60 calls/min/user), and returns URLs
that expire in 30 minutes -- it cannot feed a 1920x1080 or 4K render.
Drive `files.export` -> PDF is ONE request per deck at unlimited resolution.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from realme.core.media import require, run, MediaError
from realme.pipeline import pdfdoc


def _missing_deck(src: Path) -> str:
    """Why a deck path did not resolve, in terms of what to do about it.

    A relative path and the wrong working directory is the usual cause, so
    this names the directory searched, and looks one level up and one level
    down for a file of the same name, since a match there is most likely the
    intended file.
    """
    here = Path.cwd()
    said = [f"No deck at {src}"]
    if not src.is_absolute():
        said.append(f"  The path is relative, so it was looked for in {here}")
    found, name = [], src.name
    for root in (here.parent, here.parent.parent):
        try:
            if (root / name).is_file() and (root / name) not in found:
                found.append(root / name)
        except OSError:
            pass
    for base in (here, here.parent):
        try:
            for child in sorted(base.iterdir()):
                if child.is_dir() and (child / name).is_file():
                    if (child / name) not in found:
                        found.append(child / name)
        except OSError:
            pass
    for hit in found[:3]:
        said.append(f"  There is one here: {hit}")
    if found:
        said.append(f"  Run the command from {found[0].parent}, or give the "
                    f"path in full.")
    return "\n".join(said)


def require_deck(src) -> Path:
    """The deck, or a refusal that says where it was looked for.

    A function rather than a line inside `deck_to_pdf`, so that a caller with
    expensive work ahead of it can ask first: `write_script` runs the
    writer's preflight -- a network round trip for a hosted engine -- before
    it touches the deck, so a missing deck is caught before that.
    """
    src = Path(src)
    if not src.is_file():
        raise MediaError(_missing_deck(src))
    return src


PDF_SUFFIX = ".pdf"
OFFICE_SUFFIXES = {".pptx", ".ppt", ".odp"}
DECK_SUFFIXES = {PDF_SUFFIX} | OFFICE_SUFFIXES

#: Said whenever a slide file cannot be opened here. PDF is the format RealMe
#: expects; a PowerPoint file is a convenience that needs a converter.
SAVE_AS_PDF = ("To make the PDF: in PowerPoint, File -> Save As (or Export) "
               "-> PDF; in Google Slides, File -> Download -> PDF; in Keynote, "
               "File -> Export To -> PDF.")


def converters() -> list[str]:
    """What on this machine can turn a .pptx into a PDF. Only LibreOffice:
    PDF is the format RealMe expects, so a PowerPoint file without it is
    answered with "upload the PDF", not with automation of another program."""
    from realme.core.media import find_binary
    return ["LibreOffice"] if find_binary("soffice") else []


def deck_problem(name: str) -> str | None:
    """Why a deck with this file name cannot be read here, or None if it can.

    Asked BEFORE a project folder is made or a job is queued, so a .pptx on a
    machine with neither LibreOffice nor PowerPoint is turned away at the
    upload with what to do, not two minutes later inside a job.
    """
    suf = Path(name).suffix.lower()
    if suf == PDF_SUFFIX:
        return None
    if suf not in OFFICE_SUFFIXES:
        return (f"{Path(name).name} is not a slide deck RealMe reads. Use a "
                f"PDF. {SAVE_AS_PDF}")
    if converters():
        return None
    return (f"{Path(name).name} is a PowerPoint file, and PowerPoint files "
            f"cannot be used on this computer (LibreOffice, which converts "
            f"them, is not installed). Please upload the PDF version of the "
            f"slides instead. {SAVE_AS_PDF}")


def deck_to_pdf(src: Path, workdir: Path) -> Path:
    # Checked here as well because this is the one function every deck path
    # goes through -- the CLI, the Studio's two upload routes, `write_script`,
    # `revise` and `deck_texts` all arrive here before anything opens the
    # file.
    #
    # PDF is the format RealMe expects and needs nothing installed. A .pptx is
    # converted by LibreOffice if it is installed; without it, the refusal
    # asks for the PDF version.
    src = require_deck(src)
    suf = src.suffix.lower()
    if suf == PDF_SUFFIX:
        return src
    if suf in OFFICE_SUFFIXES:
        workdir.mkdir(parents=True, exist_ok=True)
        out = workdir / (src.stem + ".pdf")
        if out.is_file() and out.stat().st_mtime >= src.stat().st_mtime:
            return out                 # converted already, deck unchanged
        if not converters():
            raise MediaError(deck_problem(src.name))
        run([require("soffice"), "--headless", "--convert-to", "pdf",
             "--outdir", str(workdir), str(src)],
            f"LibreOffice convert {src.name}")
        if not out.exists():
            raise MediaError(f"LibreOffice produced no PDF for {src.name}. "
                             f"{SAVE_AS_PDF}")
        return out
    raise MediaError(deck_problem(src.name))


def deck_texts(deck: Path, workdir: Path) -> list[str]:
    """
    The text of every page of a deck, in page order.

    The one way to ask a deck what it says. It raises on failure rather than
    returning empty strings, which would make every slide of a revised deck
    look new.
    """
    deck, workdir = Path(deck), Path(workdir)
    if not deck.is_file():
        raise MediaError(f"The deck is not where it was recorded: {deck}")
    workdir.mkdir(parents=True, exist_ok=True)
    pdf = deck_to_pdf(deck, workdir)
    return [p.text for p in pdfdoc.read_pages(pdf)]


def rasterize(pdf: Path, outdir: Path, width_px: int = 1920) -> list[Path]:
    """Rasterize every page to PNG at the exact pixel width the video needs."""
    return pdfdoc.rasterize_pages(pdf, outdir, width_px)


@dataclass
class SlideContent:
    """What the script writer gets per slide."""
    index: int
    on_slide_text: str = ""
    speaker_notes: str = ""

    def as_prompt(self) -> str:
        parts = []
        if self.on_slide_text:
            parts.append(f"On-slide text:\n{self.on_slide_text}")
        if self.speaker_notes:
            # Notes come first: they are what the instructor meant to say,
            # whereas the on-slide text is what the audience can already read.
            parts.insert(0, f"INSTRUCTOR'S SPEAKER NOTES (follow these closely):\n"
                            f"{self.speaker_notes}")
        return "\n\n".join(parts) or "(no text on this slide)"


def extract_pdf_text(pdf: Path) -> list[str]:
    """Per-page visible text. Not speaker notes -- PDFs do not carry them."""
    try:
        return [pg.text for pg in pdfdoc.read_pages(pdf)]
    except Exception:
        return []


def _presentation(pptx: Path):
    """
    Open a .pptx, or say plainly why not.

    Raises rather than returning nothing: notes are the reason to prefer
    .pptx over .pdf, and losing them silently would make the model narrate
    the bullet points. python-pptx is a hard dependency of RealMe, so a
    missing one is a broken install, not a configuration choice.
    """
    try:
        from pptx import Presentation
    except ImportError as e:
        raise MediaError(
            "python-pptx is not installed, so speaker notes cannot be read "
            "from this deck.\n"
            "  It is a required dependency -- something is wrong with the "
            "install. Repair it:\n"
            "    00_Windows\\1_Install.bat\n"
            "  or, from 5_Command_Prompt.bat:\n"
            "    pip install python-pptx") from e
    return Presentation(str(pptx))


def _shown(pptx: Path) -> list:
    """The slides that appear in the PDF, in order.

    A hidden slide is left out of the PDF by both LibreOffice and PowerPoint,
    so reading its notes would shift every later slide's notes onto the wrong
    picture.
    """
    return [sl for sl in _presentation(pptx).slides
            if sl._element.get("show") not in ("0", "false")]


def extract_pptx_notes(pptx: Path) -> list[str]:
    """
    Real speaker notes from a .pptx.

    The most valuable grounding signal in the pipeline. The visible text is
    what the audience can already read; the notes are what the instructor
    intended to say. Narration written from the visible text alone recites
    the slide.
    """
    notes: list[str] = []
    for slide in _shown(pptx):
        text = ""
        if slide.has_notes_slide:
            frame = slide.notes_slide.notes_text_frame
            if frame is not None:
                text = (frame.text or "").strip()
        notes.append(text)
    return notes


def extract_pptx_text(pptx: Path) -> list[str]:
    """Visible text per slide, straight from the shapes."""
    out: list[str] = []
    for slide in _shown(pptx):
        chunks = [sh.text_frame.text.strip() for sh in slide.shapes
                  if getattr(sh, "has_text_frame", False) and sh.text_frame.text.strip()]
        out.append("\n".join(chunks))
    return out


def extract_content(src: Path, pdf: Path) -> list[SlideContent]:
    """
    Gather everything the script writer should see, from the best source.

    A .pptx carries real speaker notes, so read them from the original rather
    than from the PDF we rendered for rasterization. Google Slides decks should
    be pulled via `presentations.get` -> `slideProperties.notesPage` ->
    `notesProperties.speakerNotesObjectId`, which is one request per deck; the
    docs warn the shape may be absent on some slides, so handle empties.
    """
    src = Path(src)
    page_text = extract_pdf_text(pdf)
    notes: list[str] = []
    if src.suffix.lower() in {".pptx", ".ppt"}:
        notes = extract_pptx_notes(src)
        shape_text = extract_pptx_text(src)
        if shape_text and len(shape_text) >= len(page_text):
            page_text = shape_text
    # The PDF's pages are the slides the video shows, so they set the count.
    # Notes beyond them would be slides with no picture.
    n = len(page_text) if page_text else len(notes)
    return [SlideContent(index=i,
                         on_slide_text=page_text[i] if i < len(page_text) else "",
                         speaker_notes=notes[i] if i < len(notes) else "")
            for i in range(n)]


def extract_notes(pdf: Path) -> list[str]:
    """Backwards-compatible shim: per-page visible text."""
    return extract_pdf_text(pdf)


def slide_pages(outdir) -> list[Path]:
    """
    The rendered slides, in PAGE order. Every caller should use this.

    `sorted(glob("slide-*.png"))` sorts lexicographically, and for unpadded
    names `slide-1.png` … `slide-12.png` that order is

        1, 10, 11, 12, 2, 3, 4, 5, 6, 7, 8, 9

    which scrambles every deck of ten slides or more.

    Sorting on the NUMBER, parsed from the name, is correct for both padded and
    unpadded files, so older unpadded rasterisations need no redo.
    """
    import re
    out = []
    for f in Path(outdir).glob("slide-*.png"):
        m = re.search(r"slide-(\d+)", f.stem)
        if m:
            out.append((int(m.group(1)), f))
    return [f for _, f in sorted(out)]
