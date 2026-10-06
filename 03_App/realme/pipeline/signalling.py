"""
Synchronised signalling: showing the listener where to look, when.

The pedagogical case for this is stronger than the case for a talking head.
The signalling principle in multimedia learning says that cueing the relevant
region while it is being discussed improves learning; there is no comparable
evidence that a face in the corner of a slide does. And unlike an avatar this
costs no GPU, no model licence and no uncanny valley in the middle of a
derivation.

Two halves, and both are exact rather than guessed:

  WHERE   pdfplumber gives the real bounding box of every word in the deck's
          text layer. We are not asking a model to estimate coordinates -- we
          are reading them out of the PDF. For image-only slides there is a
          model-grounding fallback, but for an ordinary deck this is exact.

  WHEN    Utterance boundaries are measured with ffprobe, so they are exact.
          Inside one utterance we interpolate by character offset. That is an
          approximation, but a bounded one: utterances are capped at ~20s, and
          speech rate is near-uniform within a single breath group, so the
          error is a few hundred milliseconds on a cue that stays up for
          several seconds. Good enough, and free. `align_words()` is the hook
          for a real forced aligner if you ever want tighter timing.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path

# Words too common to be worth pointing at.
STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is",
    "are", "was", "were", "be", "been", "we", "you", "it", "that", "this",
    "these", "those", "as", "at", "by", "from", "but", "not", "can", "will",
    "our", "their", "its", "has", "have", "had", "so", "if", "then", "than",
    "when", "which", "what", "how", "why", "all", "each", "into", "over",
    "slide", "using", "used", "use", "also", "more", "most", "some", "here",
}


@dataclass
class Box:
    """Normalized to the page: 0..1 in both axes, origin top-left."""
    x0: float
    y0: float
    x1: float
    y1: float

    def union(self, other: "Box") -> "Box":
        return Box(min(self.x0, other.x0), min(self.y0, other.y0),
                   max(self.x1, other.x1), max(self.y1, other.y1))

    def pad(self, px: float = 0.002, py: float = 0.004) -> "Box":
        return Box(max(0.0, self.x0 - px), max(0.0, self.y0 - py),
                   min(1.0, self.x1 + px), min(1.0, self.y1 + py))

    def pixels(self, w: int, h: int) -> tuple[int, int, int, int]:
        x = int(self.x0 * w); y = int(self.y0 * h)
        return x, y, max(2, int((self.x1 - self.x0) * w)), max(2, int((self.y1 - self.y0) * h))


@dataclass
class Word:
    text: str
    box: Box
    line: int


@dataclass
class Cue:
    """One thing to point at, for one span of time."""
    slide_index: int
    phrase: str
    box: Box
    start_s: float
    end_s: float
    style: str = "highlight"        # highlight | underline | focus


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def extract_slide_words(pdf: Path) -> list[list[Word]]:
    """
    Real word boxes from the PDF text layer, normalized per page.

    Line numbers come from the PDF's own block/line structure rather than from
    a y-coordinate tolerance, so multi-column slides and tight leading group
    correctly instead of collapsing into one line.
    """
    from realme.pipeline import pdfdoc
    pages: list[list[Word]] = []
    for pg in pdfdoc.read_pages(pdf):
        w, h = pg.width or 1.0, pg.height or 1.0
        pages.append([Word(text=x.text, line=x.line,
                           box=Box(x.x0 / w, x.y0 / h, x.x1 / w, x.y1 / h))
                      for x in pg.words])
    return pages


def visual_slides(pdf: Path) -> set[int]:
    """
    Pages carrying a diagram rather than prose.

    These get no cues from the text layer -- there is nothing to match against.
    Reporting them is better than silently producing nothing, because "no cues
    on slide 7" is actionable and an empty result is not.
    """
    from realme.pipeline import pdfdoc
    return {p.index for p in pdfdoc.read_pages(pdf) if p.is_visual}


def find_phrase(words: list[Word], phrase: str, *, min_tokens: int = 1) -> Box | None:
    """
    Locate a phrase in a slide's text layer and return its bounding box.

    Matches on normalized tokens so punctuation, hyphenation and case do not
    defeat it, and requires the run to sit on one or two adjacent lines so a
    phrase never yields a box spanning half the slide.
    """
    target = [_norm(t) for t in phrase.split() if _norm(t)]
    if len(target) < min_tokens:
        return None
    toks = [_norm(w.text) for w in words]
    n = len(target)
    for i in range(len(toks) - n + 1):
        if toks[i:i + n] != target:
            continue
        run = words[i:i + n]
        if max(w.line for w in run) - min(w.line for w in run) > 1:
            continue
        box = run[0].box
        for w in run[1:]:
            box = box.union(w.box)
        return box
    return None


def candidate_phrases(words: list[Word], *, max_len: int = 5,
                      skip_title: bool = True) -> list[str]:
    """
    Phrases on the slide worth pointing at: contiguous, on one line, and
    carrying at least one content word. Longer phrases first, so a specific
    match wins over a generic one.

    Title lines are excluded. Highlighting the slide's own title while the
    narrator reads the slide's own title is signal with no information in it --
    the listener already knows where the title is, and a cue that fires on
    something obvious trains them to ignore the next one.
    """
    out: list[tuple[int, str]] = []
    by_line: dict[int, list[Word]] = {}
    for w in words:
        by_line.setdefault(w.line, []).append(w)
    title_lines: set[int] = set()
    if skip_title and words:
        top = min(w.box.y0 for w in words)
        title_lines = {w.line for w in words if w.box.y0 < top + 0.06}
    for line_no, line_words in by_line.items():
        if line_no in title_lines:
            continue
        toks = [w.text for w in line_words]
        for size in range(min(max_len, len(toks)), 0, -1):
            for i in range(len(toks) - size + 1):
                span = toks[i:i + size]
                if not any(_norm(t) and _norm(t) not in STOP and len(_norm(t)) > 2
                           for t in span):
                    continue
                out.append((size, " ".join(span)))
    seen: set[str] = set()
    ranked = []
    for size, phrase in sorted(out, key=lambda p: -p[0]):
        k = _norm(phrase)
        if k in seen:
            continue
        seen.add(k)
        ranked.append(phrase)
    return ranked


def _haystack(narration: str) -> str:
    return " " + re.sub(r"\s+", " ", narration.lower()) + " "


def _find_in_narration(phrase: str, hay: str):
    """Where this phrase is said, or None. One matcher, two callers.

    Auto cues and authored cues must agree about what counts as "said on this
    slide", or a phrase the author typed would be rejected by rules the
    automatic path never applied.
    """
    toks = phrase.split()
    pattern = r"\W+".join(re.escape(re.sub(r"[^a-z0-9]+", "", t.lower()))
                          for t in toks if re.sub(r"[^a-z0-9]+", "", t))
    if not pattern:
        return None
    return re.search(r"\b" + pattern + r"\b", hay)


def manual_cues(phrases, narration: str, words: list[Word]
                ) -> tuple[list[tuple[str, Box, int]], list[str]]:
    """
    Cues the author typed, rather than ones inferred.

    A cue needs two things: a place on the slide to point at, and a moment in
    the narration to point at it. An authored phrase can fail either, and the
    two failures need different fixes -- so they are reported separately
    instead of the cue silently not appearing.

    Authored cues are not capped and not filtered by length: a one-word cue is
    a deliberate choice here, where in the automatic path it would be noise.
    """
    hay = _haystack(narration)
    hits, problems = [], []
    for phrase in phrases:
        phrase = phrase.strip()
        if not phrase:
            continue
        box = find_phrase(words, phrase, min_tokens=1)
        m = _find_in_narration(phrase, hay)
        if box is None and m is None:
            problems.append(f"{phrase!r}: not on the slide and not in the "
                            f"narration")
        elif box is None:
            problems.append(f"{phrase!r}: said, but not found on the slide - "
                            f"nothing to point at")
        elif m is None:
            problems.append(f"{phrase!r}: on the slide, but not said in this "
                            f"slide's narration - nothing to point at it with")
        else:
            hits.append((phrase, box, m.start() - 1))
    return sorted(hits, key=lambda h: h[2]), problems


def auto_cues(narration: str, words: list[Word], *, max_cues: int = 6,
              min_tokens: int = 2) -> list[tuple[str, Box, int]]:
    """
    Derive cues with no model involved: find phrases that appear both on the
    slide and in the narration, and point at them when they are said.

    Deterministic, free, and it degrades to "no cues" rather than to "wrong
    cues" -- which is the right failure direction for something overlaid on
    a lecture. An LLM can supply richer cues; this is the floor, not the ceiling.
    """
    hay = _haystack(narration)
    hits: list[tuple[str, Box, int]] = []
    claimed: list[tuple[int, int]] = []
    for phrase in candidate_phrases(words):
        if len(phrase.split()) < min_tokens:
            continue
        m = _find_in_narration(phrase, hay)
        if not m:
            continue
        if any(a <= m.start() <= b or a <= m.end() <= b for a, b in claimed):
            continue
        box = find_phrase(words, phrase, min_tokens=min_tokens)
        if box is None:
            continue
        claimed.append((m.start(), m.end()))
        hits.append((phrase, box, m.start() - 1))
        if len(hits) >= max_cues:
            break
    return sorted(hits, key=lambda h: h[2])


def time_cues(hits: list[tuple[str, Box, int]], narration: str, utterances,
              slide_index: int, *, hold_s: float = 2.6,
              style: str = "highlight", offset_s: float = 0.0) -> list[Cue]:
    """
    Turn character offsets into times using the MEASURED utterance boundaries.

    Utterance starts and ends are exact (ffprobe). Only the position inside an
    utterance is interpolated, which is why utterances are kept short.
    """
    spans: list[tuple[int, int, float, float]] = []   # char0, char1, t0, t1
    # `offset_s` is the slide's silent lead-in: speech starts that much after
    # the slide appears, so every cue shifts with it or they all fire early.
    cursor_char, cursor_t = 0, offset_s
    for u in utterances:
        idx = narration.find(u.text, cursor_char)
        if idx < 0:
            idx = cursor_char
        spans.append((idx, idx + len(u.text), cursor_t, cursor_t + (u.duration_s or 0.0)))
        cursor_char = idx + len(u.text)
        cursor_t += (u.duration_s or 0.0)

    cues: list[Cue] = []
    total = cursor_t
    for phrase, box, offset in hits:
        t = None
        for c0, c1, t0, t1 in spans:
            if c0 <= offset <= c1 and c1 > c0:
                t = t0 + (t1 - t0) * ((offset - c0) / (c1 - c0))
                break
        if t is None:
            t = total * (offset / max(len(narration), 1))
        start = max(0.0, t - 0.15)
        cues.append(Cue(slide_index=slide_index, phrase=phrase, box=box,
                        start_s=start, end_s=min(total, start + hold_s), style=style))

    # Never show two cues at once -- simultaneous highlights signal nothing.
    cues.sort(key=lambda c: c.start_s)
    for a, b in zip(cues, cues[1:]):
        a.end_s = min(a.end_s, max(a.start_s + 0.8, b.start_s - 0.05))
    return cues


HIGHLIGHT = "yellow@0.30"
UNDERLINE = "#2B44B8@0.85"
DIM = "black@0.45"


def filtergraph(cues: list[Cue], width: int, height: int,
                *, slide_w: int | None = None, slide_h: int | None = None,
                x_off: int = 0, y_off: int = 0) -> str:
    """
    ffmpeg filter chain for one segment's cues.

    `slide_w/h` and the offsets let the caller map normalized page coordinates
    onto wherever the slide actually sits in the frame -- which matters as soon
    as the slide is letterboxed or shares the frame with an avatar.
    """
    sw = slide_w or width
    sh = slide_h or height
    parts: list[str] = []
    for c in cues:
        x, y, w, h = c.box.pad().pixels(sw, sh)
        x += x_off
        y += y_off
        en = f"enable='between(t,{c.start_s:.3f},{c.end_s:.3f})'"
        if c.style == "underline":
            bar = max(3, int(sh * 0.006))
            parts.append(f"drawbox=x={x}:y={y + h}:w={w}:h={bar}:"
                         f"color={UNDERLINE}:t=fill:{en}")
        elif c.style == "focus":
            # Dim everything except the region: four rects around it, which is
            # cheaper and more portable than an inverse mask.
            for rx, ry, rw, rh in ((0, 0, width, y),
                                   (0, y + h, width, max(0, height - y - h)),
                                   (0, y, x, h),
                                   (x + w, y, max(0, width - x - w), h)):
                if rw > 0 and rh > 0:
                    parts.append(f"drawbox=x={rx}:y={ry}:w={rw}:h={rh}:"
                                 f"color={DIM}:t=fill:{en}")
        else:
            parts.append(f"drawbox=x={x}:y={y}:w={w}:h={h}:"
                         f"color={HIGHLIGHT}:t=fill:{en}")
    return ",".join(parts)


def cue_report(cues: list[Cue]) -> list[dict]:
    return [{"slide": c.slide_index + 1, "phrase": c.phrase, "style": c.style,
             "start_s": round(c.start_s, 2), "end_s": round(c.end_s, 2),
             "box": [round(v, 4) for v in (c.box.x0, c.box.y0, c.box.x1, c.box.y1)]}
            for c in cues]
