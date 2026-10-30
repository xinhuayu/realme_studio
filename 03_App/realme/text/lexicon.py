"""
One lexicon, many engines.

Each engine wraps phonetic content in its own syntax, but the content is the
same. So we keep a single YAML source of truth and emit whatever the selected
engine wants:

    Piper / espeak     [[ ˌaɪpˌiːtˌiːdˈʌbəljuː ]]      inline IPA
    IndexTTS 2.5       <IPTW|AY1 P IY1 T IY1 ...>      inline CMU
    VoxCPM2            {AY1 P IY1 T IY1 ...}           inline ARPAbet
    MOSS-TTS           /ˌaɪpˌiːtˌiːdˈʌbəljuː/          inline IPA
    Azure/ElevenLabs   PLS <lexeme><phoneme>           uploaded XML
    engines with none  "I P T W"                        respelling fallback

Each entry also records the WRONG pronunciations observed from the engines.
They become competing variants in a forced-alignment dictionary, which is how
the verifier decides, from the audio itself, which one the model said. The
lexicon and the verifier share one data source.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from xml.sax.saxutils import escape
from realme.core.textio import read_text
import shutil
import subprocess


def espeak_phonemes(text: str, voice: str = "en-us") -> str:
    """
    Ask espeak-ng for its OWN phoneme mnemonics for a piece of text.

    espeak's `[[ ]]` injection does NOT accept IPA, despite IPA being what
    `--ipa` prints. Given IPA it emits nothing at all for that word -- the
    term is silently deleted from the audio:

        espeak-ng -q --ipa=3 "the [[<IPA for IPTW>]] method"  ->  "the method"
        espeak-ng -q --ipa=3 "the [[a'Ipi:ti:d'VbLju:]] method" -> works

    So mnemonics are never hand-authored. A human writes a respelling they
    can read and check ("I P T W", "HEN-sel"); this function converts it. The
    machine-generated string is the one that reaches the engine.
    """
    if not shutil.which("espeak-ng"):
        return ""
    out = subprocess.run(["espeak-ng", "-v", voice, "-q", "-x", text],
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    return out.stdout.strip().replace("\n", " ")


@dataclass
class Entry:
    term: str
    ipa: str = ""            # espeak-flavour IPA, for Piper / MOSS
    arpabet: str = ""        # CMU, for IndexTTS / VoxCPM / ElevenLabs
    respelling: str = ""     # last resort for engines with no phoneme input
    wrong: list[str] = field(default_factory=list)   # observed failures (IPA)
    note: str = ""

    case_sensitive: bool = True

    def suppresses_spelling(self) -> bool:
        """
        Does this entry exist to stop the acronym speller rather than to change
        a pronunciation?

        `Pre-ETS` is the case. espeak already says it correctly, so as a
        phoneme override it is a no-op -- but without the entry the speller
        turns it into "Pre-E T S". Removing it would break the term it
        protects, so `check` does not suggest that.
        """
        return bool(re.search(r"[A-Z]{2,}", self.term))

    def spoken_fallback(self) -> str:
        return self.respelling or self.term

    def espeak_block(self) -> str:
        """Inline `[[ ]]` block in espeak's own mnemonics, derived not authored."""
        ph = espeak_phonemes(self.spoken_fallback())
        return f"[[{ph}]]" if ph else self.spoken_fallback()


# Each of these is phonemized wrongly by `espeak-ng -v en-us --ipa=3 -q`.
# Terms espeak already gets right (DAG, RCT, HR, ANOVA, Bonferroni,
# heteroskedasticity...) are deliberately absent: an unnecessary override
# only adds noise.
SEED: list[Entry] = [
    # Vocational rehabilitation. Practitioners say WIOA as "why-OH-uh", not
    # "double-you eye oh ay", and Pre-ETS is one compound term. Other terms in
    # the field -- PSE, IPE, IDD, RSA -- are correctly spelled letter by
    # letter and need no entry.
    Entry("WIOA", ipa="waɪˈoʊə", respelling="why-oh-uh",
          wrong=["ˌdʌbəljuːˌaɪˌoʊˈeɪ"],
          note="Workforce Innovation and Opportunity Act; said as a word"),
    # No IPA: espeak already says this correctly, and an override that changes
    # nothing is noise in `realme lexicon check`. The entry exists to stop the
    # acronym speller and to give engines without a phoneme interface a
    # respelling.
    Entry("Pre-ETS", respelling="Pre E T S",
          note="Pre-Employment Transition Services. Said as letters, "
               "not 'pree-ets'."),
    Entry("CIE", ipa="ˌsiːˌaɪˈiː", respelling="C I E",
          note="competitive integrated employment; spelled out"),

    Entry("IPTW", ipa="ˌaɪpˌiːtˌiːdˈʌbəljuː", arpabet="AY1 P IY1 T IY1 D AH1 B AH0 L Y UW0",
          respelling="I P T W", wrong=["ˈɪptwə"],
          note="espeak says 'IP-twuh'"),
    Entry("AIPW", ipa="ˌeɪˌaɪpˌiːdˈʌbəljuː", respelling="A I P W", wrong=["ˈeɪpwə"],
          note="espeak says 'aypwuh'"),
    Entry("OR", ipa="ˌoʊˈɑːɹ", respelling="odds ratio", wrong=["ˈɔːɹ"],
          note="CRITICAL: espeak reads the odds ratio as the conjunction 'or'. "
               "Respelling to 'odds ratio' is usually the better fix."),
    Entry("CI", ipa="ˌsiːˈaɪ", respelling="C I", wrong=["sˈaɪ"],
          note="espeak says 'sigh'"),
    Entry("DAGs", ipa="dˈæɡz", respelling="dags", wrong=["dˈæɡ"],
          note="espeak DELETES the plural s"),
    Entry("LOCF", ipa="ˌɛlˌoʊsˌiːˈɛf", respelling="L O C F", wrong=["lˈɑːkf"]),
    Entry("MCAR", ipa="ˌɛmˌsiːˌeɪˈɑːɹ", respelling="M C A R", wrong=["məkˈɑːɹ"]),
    Entry("MAR", ipa="ˌɛmˌeɪˈɑːɹ", respelling="M A R", wrong=["mˈɑːɹ"]),
    Entry("SES", ipa="ˌɛsˌiːˈɛs", respelling="S E S", wrong=["sˈɛᵻz"],
          note="espeak says 'sez'"),
    Entry("NHIS", ipa="ˌɛnˌeɪtʃˌaɪˈɛs", respelling="N H I S", wrong=["ˈɛnhˈaɪz"]),
    Entry("UKB", ipa="ˌjuːˌkeɪˈbiː", respelling="U K B", wrong=["ˈʌkb"]),
    Entry("ICD", ipa="ˌaɪˌsiːˈdiː", respelling="I C D", wrong=["ˈaɪkd"]),
    Entry("NIH", ipa="ˌɛnˌaɪˈeɪtʃ", respelling="N I H", wrong=["nˈɪ"]),
    Entry("IRB", ipa="ˌaɪˌɑːɹˈbiː", respelling="I R B", wrong=["ˈɜːb"]),
    Entry("ITT", ipa="ˌaɪˌtiːˈtiː", respelling="I T T", wrong=["ˈɪt"]),
    Entry("AUC", ipa="ˌeɪˌjuːˈsiː", respelling="A U C", wrong=["ˈɔːk"]),
    Entry("ROC", ipa="ˌɑːɹˌoʊˈsiː", respelling="R O C", wrong=["ɹˈɑːk"],
          note="espeak says 'rock'; some fields do say 'rock curve' - your call"),
    Entry("GEE", ipa="ˌdʒiːˌiːˈiː", respelling="G E E", wrong=["dʒˈiː"]),
    Entry("Haenszel", ipa="hˈɛnsəl", respelling="HEN-sel", wrong=["hˈiːnszəl"]),
    Entry("PICO", ipa="pˈiːkoʊ", respelling="PEE-koh", note="verify against your usage"),
    Entry("a priori", ipa="ˌeɪpɹaɪˈɔːɹaɪ", respelling="ay pry-OR-eye",
          wrong=["ɐ pɹaɪˈɔːɹi"], case_sensitive=False,
          note="espeak gives the anglicised 'uh pry-OR-ee'"),
]


def phoneme_syntax_for(engine: str) -> str | None:
    """
    Which inline pronunciation syntax this engine understands, from the adapter
    that has to live with the answer. Falls back to a fixed name map for an
    engine that is not in the registry.
    """
    try:
        from realme.adapters.tts import REGISTRY
        cls = REGISTRY.get(engine)
        if cls is not None:
            return getattr(cls, "phoneme_syntax", None)
    except Exception:
        pass
    return {"espeak": "espeak", "moss": "moss",
            "indextts": "indextts", "voxcpm": "voxcpm"}.get(engine)


def load_user_entries() -> list["Entry"]:
    p = user_lexicon_path()
    if not p.is_file():
        return []
    try:
        data = json.loads(read_text(p))
    except (OSError, json.JSONDecodeError) as e:
        import sys as _sys
        print(f"  [lexicon] {p} could not be read ({e}); using the built-in "
              f"entries only", file=_sys.stderr, flush=True)
        return []
    out = []
    for d in data.get("entries", []):
        try:
            out.append(Entry(**d))
        except TypeError:
            continue
    return out


def save_user_entries(entries: list["Entry"]) -> Path:
    p = user_lexicon_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(
        {"version": 1, "entries": [asdict(e) for e in entries]},
        indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def user_lexicon_path() -> Path:
    """Where a person's own pronunciations live, outside the source tree."""
    from realme.core.env import data_home
    return data_home() / "lexicon.json"


class Lexicon:
    def __init__(self, entries: list[Entry] | None = None,
                 *, user: bool = True):
        """
        SEED, then the user's own file on top.

        Warnings elsewhere end "add a lexicon entry"; this is where such
        entries go. A user's pronunciations live in the data directory next to
        the profile, not in a source tree that an update overwrites.
        """
        self.entries: dict[str, Entry] = {}
        self.user_terms: set[str] = set()
        for e in (entries if entries is not None else SEED):
            self.add(e)
        if entries is None and user:
            for e in load_user_entries():
                self.add(e)
                self.user_terms.add(e.term)

    def add(self, e: Entry) -> None:
        self.entries[e.term if e.case_sensitive else e.term.lower()] = e

    def get(self, term: str) -> Entry | None:
        return self.entries.get(term) or self.entries.get(term.lower())

    # ------------------------------------------------------------- storage
    @classmethod
    def load(cls, path: Path) -> "Lexicon":
        data = json.loads(read_text(path))
        return cls([Entry(**d) for d in data["entries"]])

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"version": 1, "entries": [asdict(e) for e in self.entries.values()]},
            indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    # ------------------------------------------------------------- emitters

    def _pattern(self) -> re.Pattern:
        terms = sorted(
            {e.term for e in self.entries.values()}, key=len, reverse=True)
        return re.compile(r"(?<![\w-])(" + "|".join(map(re.escape, terms)) + r")(?![\w-])")

    def apply(self, text: str, engine: str) -> tuple[str, list[str]]:
        """
        Inject pronunciations for `engine`. Returns (text, terms_applied).
        Engines with no phoneme interface get the respelling instead, and the
        caller can see from the returned list exactly what was substituted.

        Which syntax an engine takes is asked of the ADAPTER, not decided here,
        so the fact is held in one place (e.g. `PiperTTS.phoneme_syntax` is
        None, and piper gets no espeak blocks).
        """
        used: list[str] = []
        syntax = phoneme_syntax_for(engine)

        def repl(m):
            e = self.get(m.group(1))
            if e is None:
                return m.group(0)
            used.append(e.term)
            if syntax == "espeak":
                return e.espeak_block()
            if syntax == "moss" and e.ipa:
                return f"/{e.ipa}/"
            if syntax == "indextts" and e.arpabet:
                return f"<{e.term}|{e.arpabet}>"
            if syntax == "voxcpm" and e.arpabet:
                return "{" + e.arpabet + "}"
            return e.spoken_fallback()

        return self._pattern().sub(repl, text), used

    def to_pls(self, lang: str = "en-US") -> str:
        """W3C PLS 1.0 — Azure custom lexicon and ElevenLabs dictionaries."""
        lines = ['<?xml version="1.0" encoding="UTF-8"?>',
                 '<lexicon version="1.0" xmlns="http://www.w3.org/2005/01/pronunciation-lexicon"',
                 f'  alphabet="ipa" xml:lang="{lang}">']
        for e in self.entries.values():
            lines.append("  <lexeme>")
            lines.append(f"    <grapheme>{escape(e.term)}</grapheme>")
            if e.ipa:
                lines.append(f"    <phoneme>{escape(e.ipa)}</phoneme>")
            elif e.respelling:
                lines.append(f"    <alias>{escape(e.respelling)}</alias>")
            lines.append("  </lexeme>")
        lines.append("</lexicon>")
        return "\n".join(lines)

    def to_espeak_dict(self) -> str:
        """espeak-ng `<lang>_extra` source, compiled with `espeak-ng --compile`."""
        rows = []
        for e in self.entries.values():
            ph = espeak_phonemes(e.spoken_fallback())
            if ph:
                rows.append(f"{e.term}\t{ph}")
        return "\n".join(rows)

    # ------------------------------------------------------------- self-check
    def check(self) -> list[str]:
        """
        Validate every entry against the actual phonemizer.

        Catches the two ways a lexicon rots: an entry whose respelling produces
        no phonemes (the term would vanish from the audio), and an entry that
        is not actually needed because the engine already says it correctly.
        Run this in CI; it costs milliseconds.
        """
        problems = []
        for e in self.entries.values():
            fixed = espeak_phonemes(e.spoken_fallback())
            if not fixed:
                problems.append(f"{e.term}: respelling {e.spoken_fallback()!r} "
                                f"produces NO phonemes - the word would be deleted")
                continue
            raw = espeak_phonemes(e.term)
            if raw and raw == fixed and not e.suppresses_spelling():
                problems.append(f"{e.term}: override is a no-op "
                                f"(engine already says {raw!r}) - consider removing")
        return problems

    def to_mfa_dict(self) -> str:
        """
        Montreal Forced Aligner dictionary with COMPETING VARIANTS.

        Each term appears twice: once with the intended pronunciation, once
        with the observed failure. After synthesis the audio is aligned against
        this dictionary and the phone tier read: whichever variant the aligner
        chose is what the model actually said. That turns open-set 'did it sound
        wrong?' into closed-set 'which of these two did it say?', which needs no
        confidence threshold.
        """
        rows = []
        for e in self.entries.values():
            if e.ipa:
                rows.append(f"{e.term}\t1.0\t{' '.join(_ipa_phones(e.ipa))}")
            for w in e.wrong:
                rows.append(f"{e.term}\t1.0\t{' '.join(_ipa_phones(w))}")
        return "\n".join(rows)


# IPA modifiers that attach to the preceding symbol rather than standing alone.
_ATTACH = "\u02d0\u0361\u035c\u0303\u031d\u031e\u0329\u02b0\u02b2\u02b7"
_STRESS = "\u02c8\u02cc"


def _ipa_phones(ipa: str) -> list[str]:
    """
    Split an IPA string into phone-sized units for a forced-alignment
    dictionary. Naive per-character splitting turns the length mark and the
    stress marks into their own 'phones', which no aligner has models for.
    """
    phones: list[str] = []
    for ch in ipa:
        if ch in _STRESS or ch.isspace():
            continue
        if ch in _ATTACH and phones:
            phones[-1] += ch
        else:
            phones.append(ch)
    return phones
