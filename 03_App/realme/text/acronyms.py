"""
Spelling out initialisms, by default.

The rule this implements: **an unfamiliar all-caps term is spelled out unless it
is a word people actually say.** WIOA is "double-you eye oh ay". NASA is "nassa".

The default is spelling because the failure modes are not symmetric. An
initialism read as a word is a mispronunciation an expert audience notices
immediately. A genuine acronym spelled out is merely stilted, and is caught on
the first listen. So the safe direction is the default, and the exceptions
are enumerated.

Three ways a term escapes being spelled:

  1. It is in WORDLIKE below -- pronounced as a word by everyone.
  2. It has a lexicon entry, which always wins. That is the override: a
     recorded decision.
  3. It is not shaped like an initialism at all (has lowercase, is one letter,
     is longer than six).
"""
from __future__ import annotations
import re

# All-caps terms that are ordinary spoken words. Kept deliberately short: a
# list of exceptions. Add to it when one is met, rather than guessing ahead.
WORDLIKE = {
    # General
    "NASA", "NATO", "UNESCO", "UNICEF", "OPEC", "ASCII", "SCUBA", "LASER",
    "RADAR", "AIDS", "SARS", "COVID", "PIN", "RAM", "ROM", "GIF", "JPEG",
    "PNG", "PDF", "SQL", "JSON", "YAML", "HTML", "CSS", "API", "USB",
    # Health, epidemiology, statistics
    "ANOVA", "MANOVA", "LASSO", "MICE", "SMART", "CONSORT", "PRISMA",
    "STROBE", "MOOSE", "GRADE", "NHANES", "BRFSS", "SEER", "MESA",
    "REGARDS", "CHARGE", "PROMIS", "SLEEP",
}

# Ordinary English words that happen to be written in capitals. Read as words
# in EVERY mode, because no mode of "spell unfamiliar initialisms" should turn
# THE into "tee aitch ee". The case this is for is a logical or emphatic line
# in ordinary prose -- "A AND B, but NOT C" -- where the connectives are
# capitalised and nothing else is.
#
# It is a list, not a rule about capitals, and that is deliberate. Treating a
# RUN of capitalised words as a sentence would read NHANES BRFSS SEER as three
# words. So a whole heading in capitals is still spelled unless its words are
# listed here or the lexicon claims them.
#
# What is deliberately NOT here: OR, IT, US, NO, IN, IS, AS, AT, BY. Each is
# both an ordinary word and something people genuinely spell out, so they stay
# spelled and the lexicon settles any particular case. ALL is absent because
# in health sciences it is acute lymphoblastic leukaemia.
READ_AS_WORDS = {
    "AND", "NOT", "NOR", "NAND", "IF", "THEN", "ELSE", "THE", "FOR", "WITH",
    "FROM", "INTO", "OVER", "UNDER", "ANY", "NONE", "BOTH", "EACH", "BUT",
    "ONLY", "ALSO", "SAME", "NEXT", "LAST", "FIRST", "TRUE", "FALSE", "YES",
    "NEW", "OLD", "HIGH", "LOW", "MORE", "LESS", "END", "WHEN", "WHERE",
    "WHAT", "HOW", "WHY", "WHO",
}

# Said a particular way, neither spelled nor read as written.
SAY_AS = {"XOR": "ex or", "XNOR": "ex nor"}

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


# "Pre-ETS" is one term, not the word "Pre" next to the initialism "ETS".
# Compounds are looked up whole first, so a lexicon entry for "Pre-ETS" wins
# over spelling "ETS".
COMPOUND_PREFIX = re.compile(r"([A-Za-z][A-Za-z0-9]*-)$")


def compound_at(text: str, start: int, token: str) -> str:
    """The hyphenated term this match belongs to, or just the match."""
    m = COMPOUND_PREFIX.search(text[:start])
    return (m.group(1) + token) if m else token


def spoken_as(tok: str, whole: str, lexicon=None,
              wordlike_wins: bool = False) -> str | None:
    """How this token should be said, or None to spell it out.

    One place, so that `find` (what the page lists) and `expand` (what the
    engine is handed) can never disagree about a term. The order is the order
    of authority: a decision somebody recorded beats a table, a table beats
    the default, and the default is to spell.
    """
    if lexicon is not None and (lexicon.get(tok) or lexicon.get(whole)):
        return tok                     # a recorded decision; the lexicon wins
    if tok in SAY_AS:
        return SAY_AS[tok]
    if tok in READ_AS_WORDS:
        return tok                     # an English word in capitals
    if wordlike_wins and is_wordlike(tok):
        return tok
    if not any(c.isalpha() for c in tok):
        return tok                     # pure digits are a number
    return None


def find(text: str, lexicon=None) -> list[str]:
    """Which terms in this text would be spelled out. Order preserved, unique."""
    out, seen = [], set()
    for m in CANDIDATE.finditer(text):
        tok = m.group(0)
        if tok in seen:
            continue
        whole = compound_at(text, m.start(), tok)
        if spoken_as(tok, whole, lexicon, wordlike_wins=True) is not None:
            continue
        seen.add(tok)
        out.append(tok)
    return out


#: How much to spell.
#:
#:   "all"    every initialism-shaped token that the lexicon does not claim.
#:            The default: the only thing that stops a term being spelled is
#:            a decision recorded in the lexicon.
#:   "known"  also skip the WORDLIKE list below.
#:   "off"    spell nothing.
#:
#: The default is "all" because WORDLIKE is a curated list of guesses about
#: how people speak, and can be silently wrong (BRFSS, for instance, is said
#: "B R F S S" by those who use it). A lexicon entry is a recorded decision.
DEFAULT_MODE = "all"


def _skip_wordlike(mode: str) -> bool:
    return mode == "known"


# "Type II error", "Stage IV", "Phase III trial", "Class I evidence": a Roman
# numeral after one of these is a number, not an initialism to spell as
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
        said = spoken_as(tok, whole, lexicon, wordlike_wins)
        if said is not None:
            return m.group(0) if said == tok else said + suffix
        if tok not in spelled:
            spelled.append(tok)
        # "IRBs" -> "I R Bs" would be read as "bee-ess". Say the plural.
        return spell(tok) + ("'s" if suffix == "'s" else "s")

    def sub_plain(m: re.Match) -> str:
        tok = m.group(0)
        whole = compound_at(m.string, m.start(), tok)
        said = spoken_as(tok, whole, lexicon, wordlike_wins)
        if said is not None:
            return said
        if tok not in spelled:
            spelled.append(tok)
        return spell(tok)

    text = roman_after_cue(text)
    text = SUFFIXED.sub(sub_suffixed, text)
    text = CANDIDATE.sub(sub_plain, text)
    return text, spelled
