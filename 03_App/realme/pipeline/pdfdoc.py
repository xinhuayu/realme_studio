"""
PDF access, with a preferred backend and a permissive fallback.

PyMuPDF does four jobs the pipeline previously spread across three libraries
and a system binary:

    word bounding boxes   was pdfplumber
    page text             was pypdf
    rasterization         was pdftoppm (poppler) -- a system install, and the
                          main friction on Windows
    image/vector detection    was nothing; needed to tell a text slide from a
                          diagram-only slide

One dependency, no system binary, and noticeably faster.

The catch, stated plainly because this project has been strict about licences:
**PyMuPDF is AGPL-3.0 or a paid Artifex commercial licence**, where pdfplumber,
pypdf and poppler are permissive. AGPL obligations attach to *network*
distribution -- running the Studio on your own machine triggers nothing, but
offering it to colleagues over a network would require offering source. Since
this project is open source that is not a practical burden, but it is a real
change and your institution may have a view.

So the backend is chosen at import: PyMuPDF if installed, the permissive stack
otherwise. Installing or not installing it is how you make the choice.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

try:
    import pymupdf as _fitz          # modern import name
    HAVE_MUPDF = True
except ImportError:                  # pragma: no cover
    try:
        import fitz as _fitz         # older wheels
        HAVE_MUPDF = True
    except ImportError:
        _fitz = None
        HAVE_MUPDF = False


@dataclass
class PageWord:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    line: int


@dataclass
class SlideImage:
    """One picture on a slide, with enough geometry to say what it is for."""
    area: float          # share of the page, 0-1
    cx: float            # centre, 0-1 across
    cy: float            # centre, 0-1 down
    furniture: bool = False   # a logo or template mark, not content

    @property
    def signature(self) -> tuple:
        """What makes this the SAME mark on another page.

        Rounded, because a logo placed by a template lands within a rounding
        error of the same spot every time but not on the exact same point.
        """
        return (round(self.area, 3), round(self.cx, 2), round(self.cy, 2))


#: A picture smaller than this share of the page, sitting outside the middle,
#: is a mark rather than a figure. Measured on real decks: the logos are 0.9%
#: and 1.0% of the page at (0.92, 0.97); the figures are 59.6% and 85.1% at
#: the centre. Nothing in between, on any slide, in any version of the deck.
FURNITURE_AREA = 0.04

#: How far from the centre a small picture must sit to be called a mark. 0.35
#: from centre in either axis is the outer third -- a corner or a margin.
FURNITURE_OFFSET = 0.35

#: How many pages a picture must recur on, identically placed, to be furniture
#: whatever its size. A logo is on every page by definition; a figure is not.
FURNITURE_SHARE = 0.6


@dataclass
class PageInfo:
    index: int
    width: float
    height: float
    text: str
    words: list[PageWord]
    image_count: int = 0
    drawing_count: int = 0
    images: list = field(default_factory=list)   # list[SlideImage]

    @property
    def content_images(self) -> list:
        """The pictures that are part of the slide's argument."""
        return [im for im in self.images if not im.furniture]

    @property
    def has_figure(self) -> bool:
        """A picture worth narrating, as opposed to a logo.

        `image_count` cannot answer this. Every page of one real deck reported
        two images and both were the same logo in the bottom-right corner; the
        re-themed version of the same deck reported none at all on those pages.
        Counting pictures counts branding.
        """
        return bool(self.content_images)

    @property
    def is_visual(self) -> bool:
        """A slide carrying a diagram rather than prose.

        Signalling needs to know: a slide with almost no text and several
        drawings has nothing to highlight from the text layer, and should fall
        back to model grounding rather than silently producing no cues.
        """
        return len(self.words) < 12 and (self.image_count + self.drawing_count) > 0

    @property
    def is_figure_slide(self) -> bool:
        """Little text AND a picture that is not a logo.

        `is_visual` is kept as it was -- the signalling code relies on it -- but
        it counts any image, so a section divider carrying only a logo looked
        like a diagram. This is the question the narration needs answered.
        """
        if len(self.words) >= 12:
            return False
        return self.has_figure or self.drawing_count > 2


def mark_furniture(pages: list) -> int:
    """Flag the logos and template marks across a whole deck.

    Two rules, and either is enough:

    REPEATED. The same picture, the same size, in the same place, on most of
    the pages. That is what a logo IS, and it needs no threshold on size -- a
    template's background band is furniture just as much as a corner mark.

    SMALL AND OUT OF THE WAY. A picture under 4% of the page sitting outside
    the middle. This catches a logo in a deck too short for repetition to mean
    anything, and it is well clear of anything measured on a real slide: the
    figures were 60% and 85% of the page and centred.

    Returns how many pictures were marked.
    """
    if not pages:
        return 0
    counts: dict = {}
    for p in pages:
        for sig in {im.signature for im in getattr(p, "images", [])}:
            counts[sig] = counts.get(sig, 0) + 1
    need = max(2, round(len(pages) * FURNITURE_SHARE))
    marked = 0
    for p in pages:
        for im in getattr(p, "images", []):
            repeated = counts.get(im.signature, 0) >= need
            peripheral = (im.area <= FURNITURE_AREA
                          and (abs(im.cx - 0.5) > FURNITURE_OFFSET
                               or abs(im.cy - 0.5) > FURNITURE_OFFSET))
            if repeated or peripheral:
                im.furniture = True
                marked += 1
    return marked


def _image_boxes(page, rect) -> list:
    """Every picture on the page, as a share of the page and a centre."""
    out = []
    try:
        infos = page.get_image_info()
    except Exception:
        return out
    area = float(rect.width) * float(rect.height) or 1.0
    for im in infos:
        try:
            x0, y0, x1, y1 = im["bbox"]
        except (KeyError, TypeError, ValueError):
            continue
        w, h = float(x1 - x0), float(y1 - y0)
        if w <= 0 or h <= 0:
            continue
        out.append(SlideImage(area=(w * h) / area,
                              cx=((x0 + x1) / 2) / float(rect.width),
                              cy=((y0 + y1) / 2) / float(rect.height)))
    return out


# ------------------------------------------------------------------ PyMuPDF

def _read_mupdf(pdf: Path) -> list[PageInfo]:
    pages: list[PageInfo] = []
    with _fitz.open(str(pdf)) as doc:
        for i, page in enumerate(doc):
            rect = page.rect
            words: list[PageWord] = []
            # get_text("words") returns (x0, y0, x1, y1, word, block, line, word_no)
            # -- MuPDF has already done the line grouping properly, so we use its
            # block/line indices instead of guessing from y-coordinates.
            for x0, y0, x1, y1, w, block, line, _ in page.get_text("words"):
                words.append(PageWord(text=w, x0=x0, y0=y0, x1=x1, y1=y1,
                                      line=block * 1000 + line))
            # renumber lines densely and in reading order
            order = {k: n for n, k in enumerate(sorted({w.line for w in words}))}
            for w in words:
                w.line = order[w.line]
            pages.append(PageInfo(
                index=i, width=float(rect.width), height=float(rect.height),
                text=page.get_text("text").strip(), words=words,
                image_count=len(page.get_images(full=True)),
                images=_image_boxes(page, rect),
                drawing_count=len(page.get_drawings())))
    return pages


def _rasterize_mupdf(pdf: Path, outdir: Path, width_px: int) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    for old in outdir.glob("slide-*.png"):
        old.unlink()
    out: list[Path] = []
    with _fitz.open(str(pdf)) as doc:
        for i, page in enumerate(doc):
            zoom = width_px / page.rect.width
            pix = page.get_pixmap(matrix=_fitz.Matrix(zoom, zoom), alpha=False)
            # Even width keeps libx264 happy without a rescale later.
            # Padded so lexicographic order matches page order for
            # anything that sorts these names without parsing them.
            p = outdir / f"slide-{i + 1:03d}.png"
            pix.save(str(p))
            out.append(p)
    return out


# --------------------------------------------------------- permissive fallback

def _read_fallback(pdf: Path) -> list[PageInfo]:
    pages: list[PageInfo] = []
    try:
        import pdfplumber
    except ImportError:
        from pypdf import PdfReader
        for i, pg in enumerate(PdfReader(str(pdf)).pages):
            pages.append(PageInfo(index=i, width=612.0, height=792.0,
                                  text=(pg.extract_text() or "").strip(), words=[]))
        return pages
    with pdfplumber.open(str(pdf)) as doc:
        for i, page in enumerate(doc.pages):
            words, tops = [], []
            for raw in page.extract_words(use_text_flow=False):
                top = float(raw["top"])
                line = next((n for n, t in enumerate(tops) if abs(t - top) < 4), None)
                if line is None:
                    tops.append(top)
                    line = len(tops) - 1
                words.append(PageWord(text=raw["text"], x0=float(raw["x0"]),
                                      y0=top, x1=float(raw["x1"]),
                                      y1=float(raw["bottom"]), line=line))
            pages.append(PageInfo(index=i, width=float(page.width),
                                  height=float(page.height),
                                  text=(page.extract_text() or "").strip(),
                                  words=words,
                                  image_count=len(page.images),
                                  drawing_count=len(page.lines) + len(page.rects)))
    return pages


def _rasterize_fallback(pdf: Path, outdir: Path, width_px: int) -> list[Path]:
    from realme.core.media import require, run, MediaError
    outdir.mkdir(parents=True, exist_ok=True)
    for old in outdir.glob("slide-*.png"):
        old.unlink()
    run([require("pdftoppm"), "-png", "-scale-to-x", str(width_px),
         "-scale-to-y", "-1", str(pdf), str(outdir / "slide")],
        f"rasterize {pdf.name}")
    # Page order, not lexicographic order: see slides.slide_pages.
    from realme.pipeline.slides import slide_pages
    pages = slide_pages(outdir)
    if not pages:
        raise MediaError(f"No pages rendered from {pdf}.")
    return pages


# ---------------------------------------------------------------- public API

def backend() -> str:
    return "pymupdf" if HAVE_MUPDF else "poppler+pdfplumber"


def read_pages(pdf: Path) -> list[PageInfo]:
    return _read_mupdf(Path(pdf)) if HAVE_MUPDF else _read_fallback(Path(pdf))


def rasterize_pages(pdf: Path, outdir: Path, width_px: int = 1920) -> list[Path]:
    pdf, outdir = Path(pdf), Path(outdir)
    return (_rasterize_mupdf(pdf, outdir, width_px) if HAVE_MUPDF
            else _rasterize_fallback(pdf, outdir, width_px))
