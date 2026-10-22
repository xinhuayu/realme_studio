"""
Delivery units and punctuation-aware pauses.

Adopted from the real_voice_qwen3 project, which solved this better than the
flat inter-utterance gap RealMe had. Three ideas worth keeping:

1. **Split at punctuation, not at a character count.** A clause boundary is a
   real prosodic boundary; 260 characters is not. Hard sentence stops close a
   group and are never merged, so a long passage may exceed the soft target
   rather than run two sentences into one model call.

2. **Vary the pace slightly, alternating.** A constant rate over thirty minutes
   reads as a metronome. ±2.5% per unit, nudged by question and exclamation
   marks, is audible as life without being audible as an effect.

3. **Derive the pauses from the speaker's own recording.** If your reference
   shows you leave 600 ms between sentences, the synthesis leans that way —
   bounded, so one unusually long pause in the recording does not propagate
   everywhere. This is the part that makes long-form narration sound like a
   person rather than a reader.

Chinese and English punctuation are both handled, since decks and quotations
are not reliably one language.
"""
from __future__ import annotations
import re
from typing import Optional

HARD_STOP = ".!?。！？"
CLAUSE_STOP = ",:;、，：；—–"


#: Periods that end a word without ending a sentence. Short and specific: this
#: is an exception list, and a long one would mean the rule is wrong.
ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "prof", "st", "jr", "sr", "vs", "etc", "eg",
    "ie", "cf", "al", "fig", "figs", "eq", "no", "vol", "ch", "approx",
    "u.s", "u.k", "e.g", "i.e", "p", "pp", "ca",
}


def _ends_abbreviation(text: str, i: int) -> bool:
    """Is the period at `i` the end of an abbreviation rather than a sentence?"""
    j = i - 1
    while j >= 0 and (text[j].isalnum() or text[j] == "."):
        j -= 1
    return text[j + 1:i].lower().rstrip(".") in ABBREVIATIONS


def _punctuation_units(text: str) -> list[tuple[str, bool]]:
    """
    Split at sentence/clause punctuation, keeping the boundary character.

    A period is not always a full stop. This split on the one inside "15.3
    percent", which produced the utterance "Only 15." followed by "3 percent
    of that sample…" -- found by reading a real lecture script aloud, and it
    would have hit every decimal, p-value and effect size in a statistics
    course. Three cases are excluded now: a decimal point between digits, a
    period with no whitespace after it (inside a token), and a known
    abbreviation.
    """
    units: list[tuple[str, bool]] = []
    start = 0
    for i, ch in enumerate(text):
        if ch not in HARD_STOP + CLAUSE_STOP:
            continue
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if ch == ".":
            prev = text[i - 1] if i else ""
            if prev.isdigit() and nxt.isdigit():
                continue                      # 15.3 -- a number, not two
            if nxt and not nxt.isspace():
                continue                      # inside a token: a URL, a file
            if _ends_abbreviation(text, i):
                continue                      # Dr. Smith, e.g., et al.
        if (nxt and not nxt.isspace() and ch in CLAUSE_STOP
                and not ("　" <= nxt <= "鿿")):
            # CJK punctuation is a boundary with no following space; an English
            # comma normally needs whitespace before it counts as one.
            continue
        boundary = i + 1
        if nxt and nxt.isspace():
            boundary += len(nxt)
        candidate = text[start:boundary].strip()
        if candidate:
            units.append((candidate, ch in HARD_STOP))
        start = boundary
    rest = text[start:].strip()
    if rest:
        units.append((rest, False))
    return units


def split_delivery_units(text: str, *, mode: str = "natural") -> list[str]:
    """
    Sentence/clause units, merging only within a sentence when there are too many.

    `stable` keeps the whole passage as one unit; `sentence` splits at full
    stops only; `natural` and `expressive` split at clauses too. Sentences are
    never merged together merely to hit the soft target.
    """
    cleaned = re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n")).strip()
    if not cleaned:
        return []
    if mode == "stable":
        return [cleaned]
    if mode == "sentence":
        # Whole sentences, never split at a comma.
        #
        # Clause splitting exists for the autoregressive engine, where short
        # calls bound the damage when one goes wrong. It costs something a
        # non-autoregressive engine does not have to pay: piper renders a whole
        # sentence with one intonation arc, and cutting "If the outcome is
        # rare," off from its own conclusion makes it fall like a full stop.
        # For piper and anything built on it, this is the honest unit.
        out, current = [], []
        for unit, hard in _punctuation_units(cleaned):
            current.append(unit)
            if hard:
                out.append(" ".join(current).strip())
                current = []
        if current:
            out.append(" ".join(current).strip())
        return [u for u in out if u]
    maximum = 16 if mode == "expressive" else 12
    raw = _punctuation_units(cleaned)
    if len(raw) <= maximum:
        return [u for u, _ in raw]

    grouped: list[list[tuple[str, bool]]] = []
    current: list[tuple[str, bool]] = []
    for unit in raw:
        current.append(unit)
        if unit[1]:
            grouped.append(current)
            current = []
    if current:
        grouped.append(current)

    while sum(len(g) for g in grouped) > maximum:
        candidates = [g for g in grouped if len(g) > 1]
        if not candidates:
            break
        target = max(candidates, key=len)
        first, second = target[0], target[1]
        target[:2] = [(f"{first[0]} {second[0]}".strip(), second[1])]
    return [u for g in grouped for u, _ in g]


def speed_multiplier(unit: str, index: int, total: int,
                     *, mode: str = "natural") -> float:
    """A deliberately small, alternating pace change for one unit."""
    if mode == "stable" or total <= 1:
        return 1.0
    m = 0.975 if index % 2 else 1.025
    if "?" in unit or "？" in unit:
        m -= 0.015
    elif "!" in unit or "！" in unit:
        m += 0.015
    if len(unit) > 180:
        m -= 0.01
    if mode == "expressive":
        m = 1.0 + (m - 1.0) * 1.35
    return max(0.94, min(1.06, round(m, 3)))


def gap_ms(unit: str, *, mode: str = "natural", pause_scale: float = 1.0,
           reference_pause_seconds: Optional[float] = None,
           reference_longest_pause_seconds: Optional[float] = None,
           breath_pause_candidates: int = 0) -> int:
    """
    A bounded pause after one unit, shaped by the speaker's own rhythm.

    The observed-pause lift is capped (700 ms after a sentence, 420 ms after a
    clause) and scaled down, so a single long hesitation in the reference does
    not turn into a long hesitation everywhere.
    """
    pause_scale = max(0.85, min(1.25, float(pause_scale)))
    if mode == "stable":
        return 0
    cleaned = unit.strip()
    hard = cleaned.endswith(tuple(HARD_STOP))
    clause = cleaned.endswith(tuple(CLAUSE_STOP))
    if "?" in cleaned or "？" in cleaned:
        base = 220 if mode in ("natural", "sentence") else 255
    elif "!" in cleaned or "！" in cleaned:
        base = 195 if mode in ("natural", "sentence") else 225
    elif hard:
        base = 185 if mode in ("natural", "sentence") else 215
    elif clause:
        base = 125 if mode in ("natural", "sentence") else 150
    else:
        base = 110 if mode in ("natural", "sentence") else 125

    # The lift from the speaker's own rhythm, reduced in October 2026.
    #
    # It was double-counting. The median pause measured by
    # `measure_reference_pauses` is the median across EVERY gap in the
    # reference, most of them mid-sentence breaths -- and it was being applied
    # at every sentence boundary, on top of the trailing breath the engine
    # already renders because each clause is synthesised as its own utterance.
    # One instructor's clip (0.336 s median, 1.192 s longest, 18 breath-length
    # gaps) produced 442 ms after every sentence against a 185 ms floor: a
    # measured 2.4x, audible as a lecture that drags.
    #
    # The factors are halved and the ceilings lowered, so the rhythm still
    # shows -- 238 ms rather than 185 ms for that clip -- without being the
    # dominant term. `pause_scale` tunes it per project; `--prosody-mode
    # stable` removes inserted gaps altogether.
    if reference_pause_seconds and reference_pause_seconds > 0:
        limit = 480 if hard else 300
        factor = 0.6 if hard else 0.45
        base = max(base, min(limit, round(reference_pause_seconds * 1000 * factor)))
    if hard and reference_longest_pause_seconds and reference_longest_pause_seconds > 0:
        base = max(base, min(420, round(reference_longest_pause_seconds * 1000 * 0.20)))
    # No separate breath bump. It added 25 ms on top of a lift that already
    # encodes breathing, which is the same quantity counted twice.
    return round(base * pause_scale)


def measure_reference_pauses(wav) -> dict:
    """
    Measure the speaker's natural pause rhythm from their reference recording.

    Uses ffmpeg's silencedetect, so it needs nothing beyond what is already
    installed. Returns the median and longest gap plus a count of breath-length
    candidates, which is exactly what `gap_ms` consumes.
    """
    import subprocess
    from statistics import median
    from realme.core.media import require
    proc = subprocess.run(
        [require("ffmpeg"), "-hide_banner", "-i", str(wav),
         "-af", "silencedetect=noise=-32dB:d=0.18", "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    gaps = [float(m) for m in re.findall(r"silence_duration:\s*([\d.]+)", proc.stderr)]
    # Trailing silence is not a speech pause; ignore anything implausibly long.
    gaps = [g for g in gaps if 0.18 <= g <= 2.0]
    if not gaps:
        return {"median_pause_s": None, "longest_pause_s": None, "breaths": 0}
    return {"median_pause_s": round(median(gaps), 3),
            "longest_pause_s": round(max(gaps), 3),
            "breaths": sum(1 for g in gaps if 0.25 <= g <= 0.8)}
