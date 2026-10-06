"""
Spelling out initialisms, by default.

The rule this implements: **an unfamiliar all-caps term is spelled out unless it
is a word people actually say.** WIOA is "double-you eye oh ay". NASA is "nassa".
Getting it backwards is not a small blemish -- "wai-oh-ah" in the middle of a
lecture on the Workforce Innovation and Opportunity Act sounds like the speaker
does not know the field.

The default is spelling because the failure modes are not symmetric. An
initialism read as a word is a mispronunciation an expert audience notices
immediately. A genuine acronym spelled out is merely stilted, and is caught the
first time you listen. So the safe direction is the default, and the exceptions
are enumerated.

Three ways a term escapes being spelled:

  1. It is in WORDLIKE below -- pronounced as a word by everyone.
  2. It has a lexicon entry, which always wins. That is the override: you
     decided, once, and the decision is recorded.
  3. It is not shaped like an initialism at all (has lowercase, is one letter,
     is longer than six).
"""
from __future__ import annotations
import re

# All-caps terms that are ordinary spoken words. Kept deliberately short: this
# is a list of exceptions, and a long exception list means the rule is wrong.
# Add to it when you meet one, rather than guessing at what might come up.
WORDLIKE = {
    # General
    "NASA", "NATO", "UNESCO", "UNICEF", "OPEC", "ASCII", "SCUBA", "LASER",
    "RADAR", "AIDS", "SARS", "COVID", "PIN", "RAM", "ROM", "GIF", "JPEG",
    "PNG", "PDF", "SQL", "JSON", "YAML", "HTML", "CSS", "API", "USB",
    # Health, epidemiology, statistics -- your field, so worth being right about
    "ANOVA", "MANOVA", "LASSO", "MICE", "SMART", "CONSORT", "PRISMA",
    "STROBE", "MOOSE", "GRADE", "NHANES", "BRFSS", "SEER", "MESA",
    "REGARDS", "CHARGE", "PROMIS", "SLEEP",
}

# Shaped like an initialism: 2-6 letters, all capitals. Longer than that and it
# is usually a name in caps; shorter and it is usually a variable.
CANDIDATE = re.compile(r"\b[A-Z][A-Z0-9]{1,5}\b")

# A trailing plural or possessive should not stop a term being recognised:
# "IRBs" and "IRB's" are the same initialism.
SUFFIXED = re.compile(r"\b([A-Z][A-Z0-9]{1,5})('?s)\b")


def spell(token: str) -> str:
    """
    'WIOA' -> 'W I O A'.

    Single spaces, not hyphens or full stops: every engine here reads spaced
    capitals letter by letter, while 'W.I.O.A.' invites a pause after each one
    and 'W-I-O-A' can be read as a hyphenated word.
    """
    return " ".join(token)


def is_wordlike(token: str) -> bool:
    return token.upper() in WORDLIKE


# "Pre-ETS" is one term, not the word "Pre" next to the initialism "ETS". The
# speller matched only the capitals and produced "Pre-E T S", ignoring the
# lexicon entry for the whole compound -- because it looked up "ETS", which
# nobody had written an entry for. Compounds are now looked up whole first.
COMPOUND_PREFIX = re.compile(r"([A-Za-z][A-Za-z0-9]*-)$")


def compound_at(text: str, start: int, token: str) -> str:
    """The hyphenated term this match belongs to, or just the match."""
    m = COMPOUND_PREFIX.search(text[:start])
    return (m.group(1) + token) if m else token


def find(text: str, lexicon=None) -> list[str]:
    """Which terms in this text would be spelled out. Order preserved, unique."""
    out, seen = [], set()
    for m in CANDIDATE.finditer(text):
        tok = m.group(0)
        if tok in seen or is_wordlike(tok):
            continue
        whole = compound_at(text, m.start(), tok)
        if lexicon is not None and (lexicon.get(tok) or lexicon.get(whole)):
            continue          # you already decided; the lexicon wins
        if not any(c.isalpha() for c in tok):
            continue          # pure digits are a number, not an initialism
        seen.add(tok)
        out.append(tok)
    return out


#: How much to spell.
#:
#:   "all"    every initialism-shaped token that the lexicon does not claim.
#:            The default, and the instruction is "do not guess": the only
#:            thing that stops a term being spelled is a decision someone
#:            recorded in the lexicon.
#:   "known"  also skip the WORDLIKE list below.
#:   "off"    spell nothing.
#:
#: The default moved from "known" to "all" after WORDLIKE was found to contain
#: a term this project gets wrong in its own field: BRFSS is said "B R F S S"
#: by the people who use it, and the list asserted it was a word. A curated
#: list of exceptions is still a list of guesses about how other people speak,
#: and it is wrong silently. The lexicon is the override that is not a guess --
#: an entry means somebody decided, once, and it is written down.
DEFAULT_MODE = "all"


def _skip_wordlike(mode: str) -> bool:
    return mode == "known"


# "Type II error", "Stage IV", "Phase III trial", "Class I evidence": a Roman
# numeral after one of these is a number, and the speller was reading it as
# "I I". It becomes the Arabic number, which every engine says correctly. A
# bare "IV" with no such word before it is left to the speller -- in this
# field that is usually intravenous, and "I V" is right.
ROMAN_CUES = ("type", "types", "stage", "stages", "phase", "phases", "class",
              "classes", "grade", "grades", "level", "levels", "group", "part",
              "chapter", "section", "table", "figure", "model", "wave",
              "trimester", "period", "category", "tier", "cohort", "study",
              "act", "schedule", "world war", "mark", "generation", "version")
_ROMAN = {"II": 2, "III": 3, "IV": 4, "VI": 6, "VII": 7, "VIII": 8, "IX": 9,
          "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, "XV": 15, "XVI": 16,
          "XVII": 17, "XVIII": 18, "XIX": 19, "XX": 20}
_ROMAN_RE = re.compile(
    r"\b(" + "|".join(re.escape(c) for c in ROMAN_CUES) + r")\s+("
    + "|".join(_ROMAN) + r")\b(?!['\-]?[A-Za-z0-9])", re.IGNORECASE)


def roman_after_cue(text: str) -> str:
    return _ROMAN_RE.sub(lambda m: f"{m.group(1)} {_ROMAN[m.group(2).upper()]}",
                         text)


def expand(text: str, lexicon=None, enabled: bool = True,
           mode: str | None = None) -> tuple[str, list[str]]:
    """
    Spell out unfamiliar initialisms. Returns (text, what_was_spelled).

    Applied to the NORMALISED text, before the lexicon: a lexicon entry for a
    term takes precedence, and terms it does not mention fall through to here.
    """
    import os
    mode = (mode or os.environ.get("REALME_SPELL_ACRONYMS")
            or DEFAULT_MODE).strip().lower()
    if not enabled or mode == "off":
        return text, []
    wordlike_wins = _skip_wordlike(mode)
    spelled: list[str] = []

    def sub_suffixed(m: re.Match) -> str:
        tok, suffix = m.group(1), m.group(2)
        whole = compound_at(m.string, m.start(), tok)
        if (wordlike_wins and is_wordlike(tok)) or (
                lexicon is not None
                and (lexicon.get(tok) or lexicon.get(whole))):
            return m.group(0)
        if not any(c.isalpha() for c in tok):
            return m.group(0)
        if tok not in spelled:
            spelled.append(tok)
        # "IRBs" -> "I R Bs" would be read as "bee-ess". Say the plural.
        return spell(tok) + ("'s" if suffix == "'s" else "s")

    def sub_plain(m: re.Match) -> str:
        tok = m.group(0)
        whole = compound_at(m.string, m.start(), tok)
        if (wordlike_wins and is_wordlike(tok)) or (
                lexicon is not None
                and (lexicon.get(tok) or lexicon.get(whole))):
            return tok
        if not any(c.isalpha() for c in tok):
            return tok
        if tok not in spelled:
            spelled.append(tok)
        return spell(tok)

    text = roman_after_cue(text)
    text = SUFFIXED.sub(sub_suffixed, text)
    text = CANDIDATE.sub(sub_plain, text)
    return text, spelled
