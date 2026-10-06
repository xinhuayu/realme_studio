"""
Narrate a text file. No deck, no slides, no script writer.

The lecture pipeline assumes a deck: it rasterises slides, asks a model to
write narration from them, composites video. None of that is wanted when the
text already exists and the only question is what it sounds like read aloud --
a test transcript, a paper abstract, a set of notes, a passage to check a
lexicon entry against.

What this does NOT do is re-implement the render. Chunking, control markers,
the lexicon, the speaker's measured pause rhythm and the utterance cache are
the same functions the lecture uses, called the same way. That matters more
here than anywhere else: the whole point of narrating a test transcript is to
hear what a lecture will sound like, and a bench that renders differently from
the renderer answers a question nobody asked. The bench had exactly that bug,
twice -- its own splitter, and its own joining without pauses.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

from realme.core.ledger import RenderLedger
from realme.core.media import duration_of
from realme.pipeline.speak import speak_segment, join_utterances
from realme.text.lexicon import Lexicon

# A blank line is a paragraph break, and a paragraph break is a longer breath
# than a full stop. Measured from read-aloud speech this sits near 0.7s; the
# sentence-level gaps inside a paragraph come from the speaker's own rhythm.
PARAGRAPH_GAP_S = 0.7


@dataclass
class NarrationResult:
    wav: Path
    duration_s: float
    paragraphs: int
    utterances: int
    words: int
    warnings: list[str] = field(default_factory=list)
    srt: Path | None = None
    transcript: Path | None = None


def read_text(source: Path) -> str:
    """
    Plain text, Markdown, or Word.

    Word is read by unzipping the document and pulling the text runs, because
    requiring python-docx for the one job of reading paragraphs out of a file
    the user already has is a dependency for nothing. A .docx is a zip with XML
    inside; `w:p` is a paragraph and `w:t` is a text run.
    """
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"No such file: {source}")
    if source.suffix.lower() == ".docx":
        import re
        import zipfile
        with zipfile.ZipFile(source) as z:
            xml = z.read("word/document.xml").decode("utf-8", "replace")
        paras = []
        for block in re.split(r"</w:p>", xml):
            runs = re.findall(r"<w:t[^>]*>(.*?)</w:t>", block, flags=re.S)
            line = "".join(runs).strip()
            if line:
                # Unescape the five XML entities by hand; nothing else appears
                # in a text run.
                for a, b in (("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                             ("&quot;", '"'), ("&apos;", "'")):
                    line = line.replace(a, b)
                paras.append(line)
        return "\n\n".join(paras)
    return source.read_text(encoding="utf-8", errors="replace")


def to_paragraphs(text: str) -> list[str]:
    """
    Blank-line separated blocks, with Markdown headings dropped.

    A heading is a visual signal, not something a narrator says. "## Methods"
    read aloud as "hash hash Methods" is the kind of thing that survives all the
    way into a finished recording because nobody listens to the first ten
    seconds twice.
    """
    out: list[str] = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        lines = []
        for line in block.split("\n"):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if set(stripped) <= {"-", "=", "*", "_"} and len(stripped) >= 3:
                continue                      # a setext rule or a divider
            if stripped:
                lines.append(stripped)
        joined = " ".join(lines).strip()
        if joined:
            out.append(joined)
    return out


def _write_srt(cues: list[tuple[float, float, str]], out: Path) -> Path:
    from realme.pipeline.captions import _ts
    lines = []
    for n, (start, end, text) in enumerate(cues, 1):
        lines += [str(n), f"{_ts(start)} --> {_ts(end)}", text, ""]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def narrate(source: Path, out_wav: Path, tts, *, mode: str = "natural",
            pace: float = 1.0, paragraph_gap_s: float = PARAGRAPH_GAP_S,
            captions: bool = True, acronym_mode: str | None = None,
            log=print) -> NarrationResult:
    source, out_wav = Path(source), Path(out_wav)
    paragraphs = to_paragraphs(read_text(source))
    if not paragraphs:
        raise ValueError(f"{source} has no text to read.")

    work = out_wav.parent / f"{out_wav.stem}_work"
    # The audio directory, not just the work root: `speak_segment` hands the
    # adapter a path and expects it to be writable. Most adapters happen to
    # create the parent themselves, which is exactly why a caller that does not
    # is a bug waiting for the one adapter that does not bother.
    (work / "audio").mkdir(parents=True, exist_ok=True)
    # Keyed on text AND voice fingerprint, so editing one sentence re-renders
    # one sentence and changing the reference re-renders everything. Reading
    # the same transcript again after a one-word fix is then seconds, not
    # minutes -- which is the difference between iterating and not bothering.
    ledger = RenderLedger(work / "ledger.json")
    lexicon = Lexicon()

    tts.preflight()

    rhythm: dict = {}
    ref = getattr(tts, "reference_wav", None)
    if ref and Path(ref).exists():
        from realme.pipeline.prosody import measure_reference_pauses
        try:
            rhythm = measure_reference_pauses(ref)
            if rhythm.get("median_pause_s"):
                log(f"  speaker rhythm: median pause "
                    f"{rhythm['median_pause_s']}s, {rhythm['breaths']} "
                    f"breath-length gaps")
        except Exception:
            rhythm = {}

    words = sum(len(p.split()) for p in paragraphs)
    log(f"  {len(paragraphs)} paragraph(s), {words} words, split '{mode}'")

    all_utts, warnings, cues = [], [], []
    clock = 0.0
    # A mistyped marker is silent in the audio -- `[[emphasise]]` is simply not
    # a control, so it survives normalization and gets read aloud, or dropped.
    # Check before synthesising anything: the whole point of the markers is
    # that you cannot hear the difference between one that worked and one that
    # was ignored until you listen to the whole file.
    from realme.text.controls import parse_controls
    unknown: list[str] = []
    for para in paragraphs:
        for mk in parse_controls(para).unknown_markers:
            if mk not in unknown:
                unknown.append(mk)
    if unknown:
        warnings.append("unrecognised marker(s): " + ", ".join(unknown)
                        + " - these are not controls and will be read or "
                          "dropped, not obeyed")
        log("  " + warnings[-1])

    for i, para in enumerate(paragraphs):
        utts = speak_segment(para, tts, work / "audio", i, ledger,
                             lexicon=lexicon, pace=pace, rhythm=rhythm,
                             mode=mode, acronym_mode=acronym_mode, log=log)
        for u in utts:
            warnings.extend(u.warnings or [])
            if u.text.strip():
                cues.append((clock, clock + u.duration_s, u.text.strip()))
            clock += u.duration_s
            # The last utterance of a paragraph carries the paragraph gap; the
            # rest keep the gap their own punctuation earned.
            if u is utts[-1] and i < len(paragraphs) - 1:
                u.gap_ms = max(u.gap_ms, int(paragraph_gap_s * 1000))
            clock += u.gap_ms / 1000.0
        all_utts.extend(utts)
        log(f"  [{i + 1}/{len(paragraphs)}] {len(utts)} utterance(s)")

    join_utterances(all_utts, out_wav)
    total = duration_of(out_wav)

    transcript = work / "spoken.txt"
    transcript.write_text(
        "\n".join(u.text for u in all_utts if u.text.strip()) + "\n",
        encoding="utf-8")
    srt = _write_srt(cues, out_wav.with_suffix(".srt")) if captions else None

    return NarrationResult(wav=out_wav, duration_s=total,
                           paragraphs=len(paragraphs), utterances=len(all_utts),
                           words=words, warnings=warnings, srt=srt,
                           transcript=transcript)
