"""
How much a slide has to say, measured from the slide.

The length setting says what an ORDINARY slide should run to. It cannot say
what any particular slide should run to: applied uniformly it makes every
note the same length, or pads slides that lack the material.

A simple slide with one claim on it is finished in seventy words. A dense
table, a derivation, or a figure that has to be walked through is not. The
difference is visible in the slide itself, so it is measured there rather than
guessed at by the model or imposed by a global number.

The thresholds are calibrated on real decks, whose pages fall into clear
groups: figures with no prose; diagrams with ~30 words of labels; list pages
around 44 words; dense list pages around 90; and very dense pages at 130-170.

A slide with NO TEXT is not a divider. Text-less pages are usually full-page
figures -- a flow diagram, a process chart -- and a figure needs the MOST
explaining, because nothing on screen reads itself.

IMAGE COUNTS ARE NOT FIGURE COUNTS. A page's image count includes logos and
backgrounds and varies with the theme. `is_figure_slide` -- little text AND a
picture that is not a logo -- is the reliable signal.
"""
from __future__ import annotations
from dataclasses import dataclass

#: At or below this many words, with nothing drawn, the slide is a divider: a
#: title and perhaps a subtitle. Section headers are typically ~3 words;
#: the lightest real content slides ~28.
DIVIDER_WORDS = 12

#: At or above this, the slide is carrying a lot -- dense lists, a table, a
#: derivation. The gap between the 44-word list pages and the 86-96 word ones
#: is where this sits.
DENSE_WORDS = 80

#: A figure slide gets at least the ordinary band however little text it has,
#: because the words have to come from the instructor: the picture does not
#: read itself.
FIGURE_MIN_SHARE = 1.0


#: Where the ordinary band starts, as a fraction of the setting's band, for
#: the LIGHTEST ordinary slide. A slide with fifteen words of bullets is not a
#: divider and is not a dense table either, and giving it the same band as a
#: 79-word slide would pad it. 0.55 of the low end puts a light slide around
#: 55-100 words on the Standard setting, where a one-claim slide finishes.
LIGHT_LOW_SHARE, LIGHT_HIGH_SHARE = 0.55, 0.65


@dataclass
class Weight:
    kind: str            # divider | figure | ordinary | dense
    words_on_slide: int
    lines: int
    visual: bool
    why: str = ""
    content_words: int = 0      # slide text plus the author's own notes

    def band(self, lo: int, hi: int, ceiling: int) -> tuple[int, int]:
        """The words THIS slide should run to, given the setting's bands.

        The ordinary band is not one band. It covers everything between a
        divider and a dense slide, which on a real deck runs from fifteen words
        of bullets to seventy-nine, and those do not want the same narration.
        It scales with what is on the slide, so a simple slide is allowed to
        finish under a hundred words, which a single band for the whole
        middle of the range would not allow.
        """
        if self.kind == "divider":
            # No floor worth enforcing: ten words can be a finished title.
            return (0, max(40, lo // 2))
        if self.kind == "dense":
            return (hi, ceiling)
        if self.kind == "figure":
            # Nothing to scale by -- the slide has no text. A figure has to be
            # walked through, so it gets the full ordinary band.
            return (lo, hi)
        span = max(1, DENSE_WORDS - DIVIDER_WORDS)
        t = min(1.0, max(0.0, (self.content_words - DIVIDER_WORDS) / span))
        return (round(lo * (LIGHT_LOW_SHARE + (1 - LIGHT_LOW_SHARE) * t)),
                round(hi * (LIGHT_HIGH_SHARE + (1 - LIGHT_HIGH_SHARE) * t)))


def weigh(*, words_on_slide: int, lines: int, visual: bool,
          note_words: int = 0) -> Weight:
    """What kind of slide this is, from what is on it.

    `note_words` are the instructor's own speaker notes. They count toward
    density because they are the strongest statement of intent available: a
    slide with three bullets and a paragraph of notes underneath is a slide the
    author means to spend time on.
    """
    total = words_on_slide + note_words
    if visual and words_on_slide <= DIVIDER_WORDS:
        return Weight("figure", words_on_slide, lines, visual,
                      "a figure with little text: the picture does not read "
                      "itself, so the words have to", total)
    if total <= DIVIDER_WORDS and lines <= 3:
        return Weight("divider", words_on_slide, lines, visual,
                      "a title or section divider", total)
    if total >= DENSE_WORDS:
        return Weight("dense", words_on_slide, lines, visual,
                      "dense: a lot on the slide to work through", total)
    return Weight("ordinary", words_on_slide, lines, visual,
                  "one claim or one list", total)


def from_content(content, page=None) -> Weight:
    """Weigh a slide from a `SlideContent` and, if available, its `PageInfo`."""
    text = getattr(content, "on_slide_text", "") or ""
    notes = getattr(content, "speaker_notes", "") or ""
    lines = len([l for l in text.splitlines() if l.strip()])
    # `is_figure_slide`, not `is_visual`: the latter counts any image, so a
    # section divider carrying a logo would be classed as a diagram and given
    # a figure's worth of narration.
    visual = bool(getattr(page, "is_figure_slide", False)) if page is not None else False
    return weigh(words_on_slide=len(text.split()), lines=lines, visual=visual,
                 note_words=len(notes.split()))
