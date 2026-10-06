"""
Turning a join into a LECTURE, not just a file.

Merging two sections from different lectures produces a video. What it does
not produce, on its own, is anything you can edit: the narration that belongs
to those slides is still in two other projects, split between them, numbered
for decks that no longer describe what you just made. Fix a sentence in the
new lecture and there is nowhere to fix it.

So a join can also assemble a project. Each input contributes the slides it
actually covers -- its cut list records their ORIGINAL numbers, which is why
`split` keeps them -- and this gathers, in playing order:

    the narration and cue words for those slides, from each source manifest
    the slide images themselves, renumbered to their new positions
    a deck built from those images, so the result can be re-rendered

The result is an ordinary project. It opens in the editor, its notes can be
corrected, and it can be rendered again in your own voice -- or split again,
since the join already wrote a combined cut list.

WHAT IT REFUSES. A source with no cut list cannot say which slides its video
covers, and a source whose manifest does not contain those slides cannot say
what was said over them. Either way the honest answer is that the notes for
that stretch are unknown, and the assembly stops and says which input and why.
Half a manifest is worse than none: it would put one lecture's narration under
another lecture's slides, and every slide after the gap would be wrong by the
same amount -- the exact failure the revision code was rewritten to prevent.
"""
from __future__ import annotations
import json, shutil
from dataclasses import dataclass, field
from pathlib import Path
from realme.core.textio import read_text


@dataclass
class Source:
    """One input to the assembly: a video, and where its slides come from."""
    video: Path
    project: Path
    slides: list[int] = field(default_factory=list)   # ORIGINAL slide numbers
    note: str = ""


@dataclass
class Assembly:
    project_id: str
    outdir: Path
    slides: int = 0
    sources: list[str] = field(default_factory=list)
    deck: str = ""
    note: str = ""

    def summary(self) -> str:
        return (f"{self.slides} slides from {len(self.sources)} source(s), "
                f"assembled as project '{self.project_id}'")


def slide_files(project: Path) -> dict[int, Path]:
    """The rendered slide images of a project, by slide NUMBER (1-based)."""
    import re
    out = {}
    for f in (Path(project) / "_work" / "slides").glob("slide-*.png"):
        m = re.search(r"slide-(\d+)", f.stem)
        if m:
            out[int(m.group(1))] = f
    return out


def covered_slides(video: Path) -> list[int]:
    """Which original slide numbers this video contains, in order.

    From the cut list beside it. A video without one is not refused here --
    the caller decides -- but it returns nothing, and nothing is what an
    assembly cannot work from.
    """
    seg = Path(video).with_name(Path(video).stem + "_segments.json")
    if not seg.is_file():
        return []
    try:
        data = json.loads(read_text(seg))
    except (ValueError, OSError):
        return []
    return [int(s["slide"]) for s in data.get("segments", []) if "slide" in s]


def gather(sources: list[Source]) -> list[tuple[Source, int]]:
    """(source, slide number) in playing order, or raise saying what is missing."""
    problems, order = [], []
    for src in sources:
        if not src.slides:
            problems.append(
                f"{src.video.name} has no cut list, so there is no way to know "
                f"which slides it covers or what was said over them")
            continue
        manifest = Path(src.project) / "_work" / "manifest.json"
        if not manifest.is_file():
            problems.append(
                f"{src.video.name} comes from project '{Path(src.project).name}', "
                f"which no longer has its narration")
            continue
        images = slide_files(src.project)
        missing = [n for n in src.slides if n not in images]
        if missing:
            problems.append(
                f"project '{Path(src.project).name}' is missing the slide "
                f"image(s) for {', '.join(map(str, missing[:6]))}")
            continue
        order += [(src, n) for n in src.slides]
    if problems:
        raise ValueError("The notes and slides could not be carried over:\n  - "
                         + "\n  - ".join(problems))
    return order


def build_deck(images: list[Path], out_pdf: Path) -> Path:
    """One PDF page per slide image, at the image's own size.

    A merged lecture has no deck of its own, and `render` needs one: it
    rasterises from the deck whenever the slide images are missing, and the
    Studio reads `deck_path.txt` before it will render at all. Building one
    from the images that are already there costs nothing and makes the result
    an ordinary project rather than a special case.
    """
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for img in images:
        pix = fitz.Pixmap(str(img))
        page = doc.new_page(width=pix.width, height=pix.height)
        page.insert_image(fitz.Rect(0, 0, pix.width, pix.height),
                          filename=str(img))
    doc.save(str(out_pdf))
    doc.close()
    return out_pdf


def assemble(sources: list[Source], outdir: Path, project_id: str,
             *, title: str = "", log=print) -> Assembly:
    """Build a project from the slides and notes the joined videos cover."""
    from realme.core.schema import Manifest, Segment
    outdir = Path(outdir)
    work = outdir / "_work"
    (work / "slides").mkdir(parents=True, exist_ok=True)

    order = gather(sources)
    manifests: dict[Path, Manifest] = {}
    segments, images = [], []
    for position, (src, number) in enumerate(order):
        key = Path(src.project)
        if key not in manifests:
            manifests[key] = Manifest.model_validate_json(
                read_text(key / "_work" / "manifest.json"))
        by_number = {s.slide_index + 1: s for s in manifests[key].segments}
        old = by_number.get(number)
        if old is None:
            raise ValueError(
                f"Project '{key.name}' has no narration for its slide {number}, "
                f"which {src.video.name} contains. Its notes and its deck have "
                f"gone out of step; open that project and save it before "
                f"joining.")
        # A copy, renumbered to its new position. The words, the cue words and
        # the prosody travel; the measured duration does not, because it
        # describes audio that will be made again.
        segments.append(Segment(
            segment_id=position + 1, slide_index=position,
            spoken_text=old.spoken_text, cues=list(old.cues),
            prosody=old.prosody.model_copy(deep=True),
            revision="", slide_digest="", rerender=True))
        images.append(slide_files(key)[number])

    for position, img in enumerate(images):
        shutil.copy(img, work / "slides" / f"slide-{position + 1}.png")

    deck = build_deck(images, outdir / f"{project_id}_deck.pdf")
    (work / "deck_path.txt").write_text(str(deck), encoding="utf-8")

    m = Manifest(project_id=project_id,
                 title=title or project_id.replace("_", " "),
                 script_source="assembled",
                 segments=segments)
    (work / "manifest.json").write_text(m.model_dump_json(indent=2),
                                        encoding="utf-8")
    names = []
    for src in sources:
        if src.video.name not in names:
            names.append(src.video.name)
    a = Assembly(project_id=project_id, outdir=outdir, slides=len(segments),
                 sources=names, deck=deck.name)
    log(f"  {a.summary()}")
    log(f"  the notes came with the slides; open '{project_id}' in the Lecture "
        f"tab to edit or re-record it")
    return a
