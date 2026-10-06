#!/usr/bin/env python3
"""
Rebuild the introduction deck after the hosted voice was added.

The deck's slides are images, not text boxes -- they were rendered once and
pasted into a .pptx -- so there is nothing in the file to edit. That leaves two
honest options: regenerate every slide and hope the new ones match the old, or
change only the sentences that stopped being true and leave every other pixel
alone. This does the second.

It can do that because the original was rendered with **DejaVu Sans**, which is
also what this renders with -- established by rendering the same words in each
candidate face and comparing against a crop of the original, not by guessing
from the look of it. Every size, position and colour below was measured off the
existing slides by scanning for text bounding boxes, and the reproduction of
slide 3 lands within a few pixels of the original on every band.

What changed, and why each one had to:

  slide 1   "It runs on your own laptop, and your voice never leaves it."
            True until the default engine became a hosted one. This is the
            first sentence of the video.
  slide 2   "Everything runs locally. The only thing that leaves the machine
            is the text of your slides, once."  Same.
  slide 3   The API key row said "free tier", which is right for drafting and
            wrong for speech: Google's free tier says human reviewers may read
            what you send.
  slide 4   New. The choice between the three voices is now the thing a new
            user most needs to understand, and nothing on the old deck said it.
  slide 12  "What it costs in time" -- half the sentence now. A hosted render
            costs money and almost no time; a local one costs time and nothing.

Everything else -- the recording advice, the editing advice, the worked
example, the consent slide -- was true before and is true now, and is not
touched.

    python 03_App/make_guide_slides.py            rebuild beside the original
    python 03_App/make_guide_slides.py --check    render and report, write nothing
"""
from __future__ import annotations
import argparse
import asyncio
import copy
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECK = ROOT / "RealMe_Guide_narrated.pptx"
W, H = 2000, 1125

# Measured from the existing slides, not chosen.
CREAM, INK, GREY, TEAL = "#FAFAF6", "#15201F", "#455659", "#0D6A62"
DARK, ON_DARK, DARK_SUB, DARK_TEAL = "#11201F", "#FFFFFF", "#A2BEB9", "#6FD3C2"
PAGE_GREY = "#7D8F95"

# Single quotes, not double: every use of this sits inside a style="..."
# attribute, and a double quote there closes the attribute at the font name --
# which renders as 16px serif and looks, at a glance, like the CSS was ignored
# rather than truncated.
FONT = "'DejaVu Sans', sans-serif"

#: A patch is a rectangle of the slide that gets cleared and redrawn. `box` is
#: (x, y, w, h) with y the top of the text's own bounding band, and the HTML is
#: rendered transparent so only the glyphs land on the slide.
PATCHES = {
    1: [dict(box=(118, 860, 1400, 180), bg=DARK, html=f"""
        <div style="font:400 44px/58px {FONT};color:{DARK_SUB}">
        Record yourself once. After that, any deck comes back as a<br>
        finished video — narration, captions, chapters. The voice is<br>
        cloned on your own laptop, or hosted, as you prefer.</div>""")],
    2: [dict(box=(118, 820, 1800, 90), bg=CREAM, html=f"""
        <div style="font:400 29px/41px {FONT};color:{GREY}">
        Three voices: a free draft voice, your own cloned here and sending nothing away, and your own
        cloned at Google —<br>
        faster, and closer to you. Whichever speaks it, you read and edit the drafted narration first.</div>""")],
    3: [dict(box=(555, 640, 1330, 60), bg=CREAM, html=f"""
        <div style="font:400 30px/38px {FONT};color:{GREY};position:relative">
        <span style="position:absolute;left:0">drafts the narration, and speaks it</span>
        <span style="position:absolute;left:804px">aistudio.google.com</span></div>"""),
       dict(box=(118, 880, 1800, 90), bg=CREAM, html=f"""
        <div style="font:400 27px/41px {FONT};color:{GREY}">
        <b style="color:{INK}">No C++ compiler. No GPU.</b> The local speech engine ships compiled and runs on the CPU; the hosted one needs<br>
        no engine at all. ffmpeg, the draft voices and the model weights are all in the package.</div>"""),
    ],
    # Slide 11's two columns are patched separately and the boxes stop at
    # x=1010, because the right-hand list starts at x=1042 and a wider erase
    # takes the top off "How not to waste it" -- which is exactly what the
    # first attempt did. Positions below are the measured bands of the left
    # column: stats at y 417, 639, 861; captions at 545, 767, 989.
    # Slide 11 is patched as two blocks, and both stop at x=1010 because the
    # right-hand list begins at 1042: a wider erase takes the top off "How not
    # to waste it", which is what the first attempt did.
    #
    # The `top:` values are box tops, but what has to line up is where the
    # GLYPHS land -- lower by the font's internal leading, measured as +4px at
    # 66px bold, +6px at 72px bold and +7px at 35px regular. So each value
    # below is (target band from the original) - (box top) - (that leading),
    # which lands first time instead of being iterated towards.
    11: [dict(box=(118, 150, 1500, 190), bg=CREAM, html=f"""
        <div style="position:absolute;left:4px;top:5px;font:700 66px/1 {FONT};
             color:{INK};letter-spacing:-0.5px">What it costs</div>
        <div style="position:absolute;left:2px;top:86px;font:400 35px/50px {FONT};
             color:{GREY}">Three voices, one bill. Only the hosted one costs money, and
             it costs<br>almost no time; the local two cost time and nothing else.</div>"""),
        dict(box=(118, 390, 892, 660), bg=CREAM, html=f"""
        <div style="position:absolute;left:10px;top:21px;font:700 72px/1 {FONT};color:{INK}">about $1</div>
        <div style="position:absolute;left:4px;top:148px;font:400 35px/1.3 {FONT};color:{GREY}">of Gemini for a 50-minute lecture</div>
        <div style="position:absolute;left:10px;top:243px;font:700 72px/1 {FONT};color:{INK}">~5 hours</div>
        <div style="position:absolute;left:4px;top:370px;font:400 35px/1.3 {FONT};color:{GREY}">of your own CPU, for the same lecture</div>
        <div style="position:absolute;left:10px;top:465px;font:700 72px/1 {FONT};color:{INK}">seconds</div>
        <div style="position:absolute;left:4px;top:592px;font:400 35px/1.3 {FONT};color:{GREY}">to hear one line before committing</div>"""),
    ],
}

#: The new slide, in the deck's own vocabulary: eyebrow, headline, subtitle,
#: rule, table, closing note. Written fresh rather than patched, so it only has
#: to match the house style and not a specific original.
NEW_SLIDE_AFTER = 3
NEW_SLIDE = f"""
<div style="position:absolute;left:122px;right:123px;top:102px">
  <div style="font:700 24px/1 {FONT};letter-spacing:2.6px;text-transform:uppercase;color:{TEAL}">The choice that matters</div>
  <div style="font:700 66px/1.06 {FONT};color:{INK};letter-spacing:-0.5px;margin-top:27px">Three voices, one recording</div>
  <div style="font:400 35px/1.32 {FONT};color:{GREY};margin-top:14px">One recording clones into either of the two that sound like you.</div>
  <div style="margin-top:37px;width:100px;height:5px;background:{TEAL}"></div>
  <table style="width:1755px;border-collapse:collapse;margin-top:92px;table-layout:fixed;font-family:{FONT}">
    <col style="width:470px"><col style="width:300px"><col style="width:330px"><col>
    <tr>
      <th style="font:700 22px/1 {FONT};letter-spacing:1.8px;text-transform:uppercase;color:{PAGE_GREY};text-align:left;padding:0 0 20px 23px">Voice</th>
      <th style="font:700 22px/1 {FONT};letter-spacing:1.8px;text-transform:uppercase;color:{PAGE_GREY};text-align:left;padding:0 0 20px 23px">Sounds like</th>
      <th style="font:700 22px/1 {FONT};letter-spacing:1.8px;text-transform:uppercase;color:{PAGE_GREY};text-align:left;padding:0 0 20px 23px">Costs</th>
      <th style="font:700 22px/1 {FONT};letter-spacing:1.8px;text-transform:uppercase;color:{PAGE_GREY};text-align:left;padding:0 0 20px 23px">What leaves the machine</th>
    </tr>
    <tr>
      <td style="font:700 30px/1.25 {FONT};color:{INK};padding:12px 0 12px 23px;border-top:2px solid #E4E6DF">gemini-tts <span style="font-weight:400;color:{TEAL}">— the default</span></td>
      <td style="font:400 30px/1.25 {FONT};color:{GREY};padding:12px 0 12px 23px;border-top:2px solid #E4E6DF">you</td>
      <td style="font:400 30px/1.25 {FONT};color:{GREY};padding:12px 0 12px 23px;border-top:2px solid #E4E6DF">about $1 an hour</td>
      <td style="font:400 30px/1.25 {FONT};color:{GREY};padding:12px 0 12px 23px;border-top:2px solid #E4E6DF">your clip, and every sentence</td>
    </tr>
    <tr>
      <td style="font:700 30px/1.25 {FONT};color:{INK};padding:12px 0 12px 23px;border-top:2px solid #E4E6DF">qwen3cpp</td>
      <td style="font:400 30px/1.25 {FONT};color:{GREY};padding:12px 0 12px 23px;border-top:2px solid #E4E6DF">you</td>
      <td style="font:400 30px/1.25 {FONT};color:{GREY};padding:12px 0 12px 23px;border-top:2px solid #E4E6DF">hours of CPU</td>
      <td style="font:400 30px/1.25 {FONT};color:{GREY};padding:12px 0 12px 23px;border-top:2px solid #E4E6DF">nothing</td>
    </tr>
    <tr>
      <td style="font:700 30px/1.25 {FONT};color:{INK};padding:12px 0 12px 23px;border-top:2px solid #E4E6DF;border-bottom:2px solid #E4E6DF">piper</td>
      <td style="font:400 30px/1.25 {FONT};color:{GREY};padding:12px 0 12px 23px;border-top:2px solid #E4E6DF;border-bottom:2px solid #E4E6DF">not you — a draft</td>
      <td style="font:400 30px/1.25 {FONT};color:{GREY};padding:12px 0 12px 23px;border-top:2px solid #E4E6DF;border-bottom:2px solid #E4E6DF">nothing</td>
      <td style="font:400 30px/1.25 {FONT};color:{GREY};padding:12px 0 12px 23px;border-top:2px solid #E4E6DF;border-bottom:2px solid #E4E6DF">nothing</td>
    </tr>
  </table>
  <div style="font:400 27px/1.38 {FONT};color:{GREY};margin-top:46px">
    <b style="color:{INK}">Draft on piper, record on one of the other two.</b>
    The hosted voice wants a paid Google project: on the free tier their terms
    say human reviewers may read what you send, and your voice is personal information.</div>
</div>
<div style="position:absolute;right:68px;bottom:50px;font:400 24px/1 {FONT};color:{PAGE_GREY}">4</div>
"""


async def render(html: str, out: Path, *, w=W, h=H, transparent=False) -> Path:
    from playwright.async_api import async_playwright
    page_css = ("background:transparent" if transparent else f"background:{CREAM}")
    doc = (f'<!doctype html><meta charset="utf-8">'
           f'<style>*{{box-sizing:border-box;margin:0;padding:0}}'
           f'body{{width:{w}px;height:{h}px;{page_css};overflow:hidden;'
           f'position:relative}}</style>{html}')
    tmp = out.with_suffix(".html")
    tmp.write_text(doc, encoding="utf-8")
    async with async_playwright() as pw:
        b = await pw.chromium.launch()
        pg = await b.new_page(viewport={"width": w, "height": h})
        await pg.goto(f"file://{tmp}")
        await pg.screenshot(path=str(out), omit_background=transparent)
        await b.close()
    tmp.unlink(missing_ok=True)
    return out


def slide_images(deck: Path) -> list[bytes]:
    from pptx import Presentation
    out = []
    for s in Presentation(str(deck)).slides:
        pics = [sh for sh in s.shapes if sh.shape_type == 13]
        if len(pics) != 1:
            raise SystemExit(f"slide with {len(pics)} pictures; this deck is "
                             f"meant to be one full-bleed image per slide")
        out.append(pics[0].image.blob)
    return out


async def build(slides_dir: Path) -> list[Path]:
    from PIL import Image
    slides_dir.mkdir(parents=True, exist_ok=True)
    blobs = slide_images(DECK)
    paths = []
    for i, blob in enumerate(blobs, 1):
        p = slides_dir / f"slide{i:02d}.png"
        p.write_bytes(blob)
        if i in PATCHES:
            im = Image.open(p).convert("RGB")
            for n, patch in enumerate(PATCHES[i]):
                x, y, w, h = patch["box"]
                im.paste(Image.new("RGB", (w, h), patch["bg"]), (x, y))
                frag = slides_dir / f"_p{i}_{n}.png"
                # The wrapper needs an explicit size. An absolutely
                # positioned box with no width is shrink-to-fit, and its
                # absolutely positioned children are then laid out inside that
                # shrunken width -- which wraps a 72px headline into a column
                # three characters wide.
                await render(f'<div style="position:absolute;left:0;top:0;'
                             f'width:{w}px;height:{h}px">{patch["html"]}</div>',
                             frag, w=w, h=h, transparent=True)
                im.paste(Image.open(frag).convert("RGBA"), (x, y),
                         Image.open(frag).convert("RGBA"))
                frag.unlink(missing_ok=True)
            im.save(p)
            print(f"  slide {i:2d}: {len(PATCHES[i])} patch(es)")
        else:
            print(f"  slide {i:2d}: unchanged")
        paths.append(p)
    new = slides_dir / "slide_new.png"
    await render(NEW_SLIDE, new)
    paths.insert(NEW_SLIDE_AFTER, new)
    print(f"  slide {NEW_SLIDE_AFTER + 1:2d}: NEW -- three voices, one recording")
    return paths


def renumber(paths: list[Path]) -> None:
    """The page number is drawn into each slide; inserting one shifts the rest."""
    from PIL import Image
    import asyncio as _a
    for i, p in enumerate(paths, 1):
        im = Image.open(p).convert("RGB")
        dark = sum(im.getpixel((20, 20))) < 200
        im.paste(Image.new("RGB", (120, 40), DARK if dark else CREAM),
                 (1880, 1040))
        frag = p.with_name(f"_n{i}.png")
        colour = "#4A5C5A" if dark else PAGE_GREY
        _a.run(render(
            # 68px from the slide's right edge, which is where the original
            # numbers sit; the patch box starts at x=1880 and is 120 wide.
            f'<div style="position:absolute;right:68px;top:10px;'
            f'font:400 24px/1 {FONT};color:{colour}">{i}</div>',
            frag, w=120, h=40, transparent=True))
        im.paste(Image.open(frag).convert("RGBA"), (1880, 1040),
                 Image.open(frag).convert("RGBA"))
        frag.unlink(missing_ok=True)
        im.save(p)


def write_deck(paths: list[Path], out: Path) -> None:
    """A fresh presentation, not a surgically edited copy of the old one.

    Deleting entries from an existing `sldIdLst` leaves the slide PARTS behind,
    and the next slide added reuses a name that is still in the package --
    `Duplicate name: ppt/slides/slide12.xml`, a file that may or may not open
    depending on which program is asked to. Every slide here is one full-bleed
    picture, so there is nothing in the original worth preserving except its
    dimensions.
    """
    from pptx import Presentation
    from pptx.util import Emu
    src = Presentation(str(DECK))
    prs = Presentation()
    prs.slide_width, prs.slide_height = src.slide_width, src.slide_height
    blank = prs.slide_layouts[6]            # the one with no placeholders
    for p in paths:
        s = prs.slides.add_slide(blank)
        s.shapes.add_picture(str(p), 0, 0, prs.slide_width, prs.slide_height)
    prs.save(str(out))


def write_pdf(paths: list[Path], out: Path) -> None:
    """The same slides as a PDF, which is the format that needs nothing.

    Reading a .pptx costs a LibreOffice install, because that is what converts
    it; RealMe says so when `soffice` is missing. A PDF deck is read directly.
    Since every slide here is already a 2000x1125 image, the PDF is those
    images at 150 dpi -- 13.33 x 7.5 inches, the same 16:9 page the .pptx has,
    with no resampling and no second rendering path to disagree with the first.
    """
    from PIL import Image
    pages = [Image.open(p).convert("RGB") for p in paths]
    pages[0].save(str(out), save_all=True, append_images=pages[1:],
                  resolution=150.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="render the slides and stop; write no deck")
    ap.add_argument("--out", type=Path, default=ROOT / "RealMe_Guide_narrated_v2.pptx")
    ap.add_argument("--slides-dir", dest="slides_dir", type=Path,
                    default=ROOT / "_guide_slides",
                    help="where the rendered slide images go")
    a = ap.parse_args()

    if not DECK.is_file():
        print(f"Cannot find {DECK}")
        return 1
    try:
        import pptx, PIL, playwright  # noqa: F401
    except ImportError as e:
        print(f"Needs python-pptx, pillow and playwright: {e}")
        return 1

    print(f"reading {DECK.name}")
    paths = asyncio.run(build(a.slides_dir))
    renumber(paths)
    print(f"\n  {len(paths)} slides in {a.slides_dir}")
    if a.check:
        print("  (--check: no deck written)")
        return 0
    write_deck(paths, a.out)
    pdf = a.out.with_suffix(".pdf")
    write_pdf(paths, pdf)
    print(f"\n  wrote {a.out}")
    print(f"  wrote {pdf}  <- use this one; a PDF deck needs no LibreOffice")
    print("\n  Next, and it has to be in this order:")
    print("    1. look at the five changed slides in that folder")
    print("    2. realme lecture RealMe_Guide_narrated_v2.pdf \\")
    print("         --script imported --tts gemini-tts")
    print("       with the notes imported first")
    return 0


if __name__ == "__main__":
    sys.exit(main())
