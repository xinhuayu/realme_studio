"""
Re-recording only what changed.

A figure is replaced, a definition is corrected, two slides are inserted in
the middle -- and only the affected slides should be rendered again.

The render ledger caches each segment's audio on its text and voice, and each
segment's video on top of that, including the slide PICTURE (see
lecture.render), so an untouched slide costs nothing to "re-render". This
module compares two versions of a deck to say which slides changed.

Two separate questions:

  WHICH OLD SLIDE IS THIS?  decides where the narration and cue words go.
  DOES IT NEED RE-RENDERING? decides what the render costs.

Re-exporting a deck with a different theme changes every pixel while changing
no content: the narration still belongs to the same slide, and the video
still has to be remade. Comparing pictures answers only the second question
and would move notes onto the wrong slides.

Identity comes from TEXT. A slide is what it says, and text survives a change
of font, a removed logo, a different page number and a re-export. Across a
re-theme, text overlap separates the same slide (about 0.93 and up), an
edited slide (about 0.85-0.91) and a different slide (0.31 or less); pixel
distance does not.

Word overlap answers WHICH SLIDE, and only that. Whether the wording changed is
answered exactly, by `wording_change` -- an overlap score cannot tell six
changed words on a long slide from none.

Pixels decide the one case text cannot: a slide with no text at all. Two
image-only slides look identical to a text comparison -- both empty -- so
pixels are consulted there, and only there.
"""
from __future__ import annotations
import hashlib
from dataclasses import dataclass
from pathlib import Path


def slide_digest(png: Path) -> str:
    """A short content hash of a rendered slide.

    Content, not modification time: re-rasterising an unchanged deck produces
    new files with new timestamps and identical pixels, and mtimes would call
    every slide changed.
    """
    return hashlib.sha256(Path(png).read_bytes()).hexdigest()[:16]


#: Text overlap above which two slides are the same slide. Well below an
#: edited slide's ~0.85 and well above an unrelated one's ~0.31.
SAME_SLIDE_TEXT = 0.62

#: There is no threshold for "did the wording change". Word-set overlap
#: measures how MUCH changed, and the question is WHETHER anything did: six
#: changed words on a 91-word slide score 0.97, the same as an untouched
#: re-themed slide. So matched slides are instead required to be
#: word-for-word identical after normalisation.
#:
#: Exactness errs on the safe side. A false "edited" (only a ligature moved)
#: costs one unticked checkbox; a false "unchanged" costs a wrong video.
#: `wording_change` forgives the page number explicitly and reports everything
#: else with the words that changed, so a false alarm can be dismissed on
#: sight.

#: A slide with fewer than this many words cannot be identified by text.
#: Three, not four: a three-word title slide still matches by text across a
#: re-theme, whereas the pixel path would see the re-theme as a new slide.
MIN_WORDS_FOR_TEXT = 3

#: Pixel distance below which two TEXT-LESS slides are the same slide.
#: Loose, because it is comparing across a possible re-theme: the same
#: image-only slide typically scores under 3 and different ones over 14.
IDENTITY_PIXELS = 10.0

#: Pixel distance above which a matched slide needs a NEW FRAME. Tight,
#: because any visible difference means the rendered video is out of date --
#: including a re-theme that changed no words.
REDRAW_PIXELS = 2.0


def _page_numbers(new_i: int, old_j: int) -> set:
    """
    The page numbers these two slides might be printing.

    Both the position and, for a deck that is an excerpt, nothing reliable at
    all -- an excerpt may start at printed page 4 while sitting at position 1.
    So this removes the obvious candidates and the threshold absorbs the rest,
    since the number cannot be predicted.
    """
    return {str(new_i + 1), str(old_j + 1)}


def _tokens(text: str) -> set:
    import re
    return set(re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower()).split())


def text_similarity(a: str, b: str, ignore: set | None = None) -> float:
    """
    Jaccard overlap of word sets. Order-insensitive on purpose: moving a
    bullet up a slide does not make it a different slide.

    `ignore` drops tokens that are page furniture rather than content, such
    as the page number printed in the corner, so two otherwise identical
    slides score 1.0.
    """
    A, B = _tokens(a), _tokens(b)
    if ignore:
        A, B = A - ignore, B - ignore
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def _seq(text: str) -> list[str]:
    """Normalised word tokens, in order.

    Normalisation removes what a re-export perturbs without changing what is
    said: ligatures and curly punctuation (NFKC and a few explicit pairs),
    soft hyphens, words broken across a line by a hyphen, letter case, and all
    whitespace. What survives is the words themselves, in the order spoken.
    """
    import re, unicodedata
    t = unicodedata.normalize("NFKC", text or "")
    for a, b in (("\u2018", "'"), ("\u2019", "'"), ("\u201c", '"'),
                 ("\u201d", '"'), ("\u2013", "-"), ("\u2014", "-"),
                 ("\u00ad", "")):
        t = t.replace(a, b)
    t = re.sub(r"-\s*\n\s*", "", t)          # hyphenated across a line break
    return re.findall(r"[a-z0-9]+(?:['.\-][a-z0-9]+)*", t.lower())


def wording_change(before: str, after: str,
                   ignore: set | None = None) -> tuple[bool, str]:
    """
    Did the words change, and which ones?

    Exact after normalisation -- see the note above `MIN_WORDS_FOR_TEXT`. A
    difference made up entirely of tokens in `ignore` is forgiven, which is how
    the page number printed in the corner stops two identical slides from
    reading as edited. The forgiveness is applied to the DIFFERENCE rather than
    by deleting those tokens up front, so a body word that happens to be "2"
    still counts.

    Returns (changed, description). The description names the changed words
    so the reader can check the flag.
    """
    import difflib
    A, B = _seq(before), _seq(after)
    sm = difflib.SequenceMatcher(None, A, B, autojunk=False)
    added, removed, first = 0, 0, ""
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        gone, came = A[i1:i2], B[j1:j2]
        if ignore and all(t in ignore for t in gone + came):
            continue                          # the page number, and nothing else
        added += len(came)
        removed += len(gone)
        if not first:
            if came and gone:
                first = f"\u201c{' '.join(gone)}\u201d \u2192 \u201c{' '.join(came)}\u201d"
            elif came:
                first = f"added \u201c{' '.join(came)}\u201d"
            else:
                first = f"dropped \u201c{' '.join(gone)}\u201d"
            if len(first) > 60:
                first = first[:57] + "\u2026"
    if not (added or removed):
        return False, ""
    counts = ", ".join(p for p in (f"{added} word{'s' if added != 1 else ''} added"
                                   if added else "",
                                   f"{removed} removed" if removed else "") if p)
    return True, f"{counts}: {first}" if first else counts


@dataclass
class SlideChange:
    slide_number: int            # 1-based, in the NEW deck
    status: str                  # same | edited | moved | added
    old_slide_number: int | None = None
    note: str = ""
    #: Whether the FRAME must be made again. Independent of `status`: a slide
    #: whose wording is untouched still needs a new frame if the deck was
    #: re-themed, and its narration still carries across.
    needs_render: bool = False


@dataclass
class RevisionPlan:
    changes: list[SlideChange]
    removed: list[int]

    @property
    def to_render(self) -> list[int]:
        return [c.slide_number for c in self.changes if c.needs_render]

    @property
    def to_write(self) -> list[int]:
        """Slides whose narration must be written or revised by a person."""
        return [c.slide_number for c in self.changes
                if c.status in ("added", "edited")]

    def summary(self) -> str:
        n = {s: sum(1 for c in self.changes if c.status == s)
             for s in ("same", "moved", "edited", "added")}
        return (f"{len(self.changes)} slides: {n['same']} unchanged, "
                f"{n['moved']} moved, {n['edited']} edited, {n['added']} new, "
                f"{len(self.removed)} removed; "
                f"{len(self.to_render)} need re-rendering")


def compare(old_digests: list[str], new_digests: list[str]) -> RevisionPlan:
    """Exact-digest comparison, for two rasterisations of the SAME file.

    Kept for the in-project case where the deck has not been re-exported.
    `match_slides` is what a revised deck should use.
    """
    pool: dict[str, list[int]] = {}
    for i, d in enumerate(old_digests):
        pool.setdefault(d, []).append(i + 1)
    matched_old: set[int] = set()
    match_for: dict[int, int] = {}
    for i, d in enumerate(new_digests):
        where = pool.get(d) or []
        if where:
            old_n = where.pop(0)
            matched_old.add(old_n)
            match_for[i + 1] = old_n
    changes: list[SlideChange] = []
    for i in range(len(new_digests)):
        n = i + 1
        if n in match_for:
            old_n = match_for[n]
            changes.append(SlideChange(n, "same" if old_n == n else "moved",
                                       old_n,
                                       "" if old_n == n else f"was slide {old_n}",
                                       needs_render=False))
        elif n <= len(old_digests) and n not in matched_old:
            changes.append(SlideChange(n, "edited", n, "the picture is different",
                                       needs_render=True))
        else:
            changes.append(SlideChange(n, "added", None, "new slide",
                                       needs_render=True))
    removed = [i + 1 for i in range(len(old_digests)) if (i + 1) not in matched_old]
    return RevisionPlan(changes, removed)


def match_slides(old_texts: list[str], new_texts: list[str], *,
                 pixel_distance=None) -> RevisionPlan:
    """
    Pair new slides with old ones by what they SAY, and decide separately
    whether each needs a new frame.

    `pixel_distance(old_i, new_i) -> float` returns the perceptual distance
    between two slide images, or None if unavailable. It is passed in rather
    than computed here so this module needs no image tooling.

    One distance, two thresholds, because the two questions want opposite
    leniency. Identity must survive a re-theme, so it is generous. Redrawing
    must catch a re-theme, so it is strict. A single threshold would either
    misplace notes or keep stale frames.
    """
    n_old, n_new = len(old_texts), len(new_texts)
    wordy = [len(_tokens(t)) >= MIN_WORDS_FOR_TEXT for t in new_texts]

    # Best available partner for each new slide, greedily, best score first --
    # so a strong match claims its partner before a weak one can take it.
    scores = []
    for i in range(n_new):
        for j in range(n_old):
            if wordy[i] and len(_tokens(old_texts[j])) >= MIN_WORDS_FOR_TEXT:
                sc = text_similarity(new_texts[i], old_texts[j],
                                     _page_numbers(i, j))
                if sc >= SAME_SLIDE_TEXT:
                    scores.append((sc, i, j))
            elif not wordy[i] and len(_tokens(old_texts[j])) < MIN_WORDS_FOR_TEXT:
                # Neither has usable text: only the picture can say. Scored by
                # closeness so the best candidate wins, not merely a passing
                # one -- an added graph and an unchanged photograph can both
                # sit under the threshold, and only one of them is the match.
                if pixel_distance is not None:
                    d = pixel_distance(j, i)
                    if d is not None and d <= IDENTITY_PIXELS:
                        scores.append((1.0 - d / (IDENTITY_PIXELS * 100), i, j))
    scores.sort(reverse=True)

    match: dict[int, int] = {}
    taken_old: set[int] = set()
    for sc, i, j in scores:
        if i in match or j in taken_old:
            continue
        match[i] = j
        taken_old.add(j)

    changes: list[SlideChange] = []
    for i in range(n_new):
        n = i + 1
        if i not in match:
            changes.append(SlideChange(n, "added", None, "new slide",
                                       needs_render=True))
            continue
        j = match[i]
        # OLD first, NEW second: the description says what happened to the
        # slide, and the reverse order would report an insertion as a deletion.
        changed, what = wording_change(old_texts[j], new_texts[i],
                                       _page_numbers(i, j))
        moved = (j != i)
        # The frame is remade whenever the picture is not the same picture --
        # a re-theme counts, even though the words did not change.
        redraw = True
        if pixel_distance is not None:
            d = pixel_distance(j, i)
            if d is not None:
                redraw = d > REDRAW_PIXELS
        if not changed:
            status = "moved" if moved else "same"
            note = f"was slide {j + 1}" if moved else ""
        else:
            status = "edited"
            note = what
            if moved:
                note += f", was slide {j + 1}"
        changes.append(SlideChange(n, status, j + 1, note, needs_render=redraw))
    removed = [j + 1 for j in range(n_old) if j not in taken_old]
    return RevisionPlan(changes, removed)
