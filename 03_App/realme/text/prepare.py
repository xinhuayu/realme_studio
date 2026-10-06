"""
The gate every line of narration passes through before synthesis.

    raw script -> normalize -> lint -> lexicon injection -> engine

Nothing reaches a TTS engine without going through `prepare_for_speech`. That
is the whole point: the engines' own front-ends fail silently and differently,
so the pipeline never relies on them.
"""
from __future__ import annotations
from dataclasses import dataclass, field

from realme.text.normalize import normalize, UnspokenSymbol
from realme.text.lexicon import Lexicon
from realme.text.lexicon import espeak_phonemes
from realme.text.acronyms import expand as expand_acronyms
from realme.text.mathspeech import MathSpeech, expand_math


@dataclass
class PreparedText:
    raw: str
    normalized: str
    engine_text: str
    engine: str
    lexicon_terms: list[str] = field(default_factory=list)
    math: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# Terms the phonemizer is known to mangle in ways a reader would not predict.
# The linter suggests, it does not rewrite - silent rewriting is the disease.
def lint(text: str, lexicon: Lexicon) -> list[str]:
    """
    Flag risky tokens BEFORE any audio is made.

    The cheap, high-yield check: any all-caps token of 2-6 letters that is not
    in the lexicon and that the phonemizer turns into something word-shaped
    rather than letter-shaped. That is the 'OR -> or', 'CI -> sigh',
    'AUC -> awk' failure class, and it is invisible in the script.
    """
    import re
    warnings: list[str] = []
    seen: set[str] = set()
    for tok in re.findall(r"\b[A-Z][A-Za-z]{1,5}\b", text):
        if tok in seen or tok.isalpha() and tok.islower():
            continue
        seen.add(tok)
        if not tok.isupper() or lexicon.get(tok):
            continue
        spelled = espeak_phonemes(" ".join(tok))
        asis = espeak_phonemes(tok)
        if not asis:
            warnings.append(f"{tok!r}: phonemizer produces nothing - it will be silent")
        elif asis != spelled and len(asis) < len(spelled) * 0.7:
            warnings.append(
                f"{tok!r}: read as a word ({asis!r}), not as letters ({spelled!r}). "
                f"Add a lexicon entry if it should be spelled out.")
    return warnings


def prepare_for_speech(text: str, engine: str,
                       lexicon: Lexicon | None = None,
                       *, strict: bool = False,
                       math: MathSpeech | None = None,
                       spell_acronyms: bool = True,
                       acronym_mode: str | None = None) -> PreparedText:
    lx = lexicon or Lexicon()
    # Math first: it emits ordinary words, which then face the same symbol
    # checks as everything else rather than sneaking past them.
    if "$" in text or "\\(" in text:
        if math is None:
            # The shared cache, so `realme math --approve` reaches a render.
            from realme.text.mathspeech import MathSpeech, default_cache_path
            math = MathSpeech(cache_path=default_cache_path())
        text_with_math, found = expand_math(text, math)
    else:
        text_with_math, found = text, []
    report = normalize(text_with_math, strict=strict)
    warnings = list(report.dropped)
    for m in found:
        if not m["spoken"]:
            warnings.append(f"could not convert LaTeX {m['latex']!r} to speech - "
                            f"it will be read literally")
        elif not m["approved"]:
            warnings.append(f"math not yet approved: {m['latex']!r} -> "
                            f"{m['spoken']!r}")
    # Spell out unfamiliar initialisms BEFORE the lexicon runs, so a lexicon
    # entry for a term still overrides this (expand() skips anything the
    # lexicon knows). Default on: see realme/text/acronyms.py for why the safe
    # direction is spelling rather than guessing.
    spoken_text, spelled = expand_acronyms(report.normalized, lx,
                                           enabled=spell_acronyms,
                                           mode=acronym_mode)
    if spelled:
        warnings.append("spelled out as letters: " + ", ".join(spelled)
                        + " (add a lexicon entry to say one differently)")
    engine_text, used = lx.apply(spoken_text, engine)
    # Lint the text the ENGINE receives, not the text the author wrote.
    #
    # It used to run on the normalised text, before the speller and the
    # lexicon had touched it, and so warned "'ETS': read as a word, add a
    # lexicon entry" about a term the very next line spelled correctly, and
    # about NHANES, which is meant to be said as a word. Both complaints were
    # about work already done. A lint that reports non-problems is worse than
    # no lint, because it trains you to skip the ones that are real.
    warnings += lint(engine_text, lx)
    return PreparedText(raw=text, normalized=report.normalized,
                        engine_text=engine_text, engine=engine,
                        lexicon_terms=used, math=found, warnings=warnings)
