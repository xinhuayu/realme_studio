"""
Text normalization for spoken academic content.

Why this module exists, in one example:

    $ espeak-ng -v en-us --ipa=3 -q "beta-hat"      ->  bˈeɪɾə hˈæt
    $ espeak-ng -v en-us --ipa=3 -q "β̂"             ->  bˈeɪɾə

The combining circumflex is silently dropped, so the ESTIMATOR becomes the
PARAMETER: a false statement, with no error anywhere in the pipeline. The same
silent deletion happens to "<", ">" and "|":

    "p < 0.05"        -> "p zero point zero five"       (the claim is gone)
    "Pr(T>t|X=x)"     -> "P R T T X equals X"           (word salad)

Every TTS front-end normalizes differently and they all fail quietly. So this
module does it, and FAILS LOUDLY: after normalization the text must contain
nothing but letters, digits and a small set of safe punctuation. Anything else
is an unhandled symbol and raises, rather than being silently eaten downstream.
"""
from __future__ import annotations
import re
import unicodedata
from dataclasses import dataclass, field

# After normalization, only these may remain. Everything else must have been
# turned into words, or we refuse to synthesize.
_ASCII_SAFE = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?'\"()-\n"


def _is_safe(ch: str) -> bool:
    """
    ASCII, any precomposed letter, and any letter of a script without case.

    Šidák and Poisson pass on the second rule. The third is for the scripts
    this engine speaks: Chinese, Japanese and Korean letters are Unicode
    category Lo -- "letter, other", the category for writing systems with no
    upper and lower case -- so a rule accepting only Ll and Lu would delete
    every character of a [[zh-CN]] block.

    A letter in a script the selected voice cannot speak is the engine's
    problem to report, not something to silently remove here.
    """
    return ch in _ASCII_SAFE or (
        ch.isalpha() and unicodedata.category(ch) in ("Ll", "Lu", "Lo"))


SAFE = set(_ASCII_SAFE)

GREEK = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon",
    "ζ": "zeta", "η": "eta", "θ": "theta", "ι": "iota", "κ": "kappa",
    "λ": "lambda", "μ": "mu", "ν": "nu", "ξ": "xi", "π": "pi", "ρ": "rho",
    "σ": "sigma", "τ": "tau", "υ": "upsilon", "φ": "phi", "χ": "chi",
    "ψ": "psi", "ω": "omega",
    "Α": "capital alpha", "Β": "capital beta", "Γ": "capital gamma",
    "Δ": "delta", "Θ": "capital theta", "Λ": "capital lambda", "Ξ": "capital xi",
    "Π": "capital pi", "Σ": "capital sigma", "Φ": "capital phi",
    "Ψ": "capital psi", "Ω": "capital omega",
}

# Combining marks that carry statistical meaning. Order matters: these run
# before any NFC/NFD juggling so we never lose one.
COMBINING = {
    "̂": "hat",      # β̂  estimator
    "̄": "bar",      # x̄  sample mean
    "̃": "tilde",    # x̃  median / transformed
    "̇": "dot",      # ẋ
    "⃗": "vector",   # x⃗
    "̆": "breve",
}

SUPERSCRIPT = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5",
               "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "ⁿ": "n",
               "⁻": "-"}    # 10⁻³ -- keep the minus, not "10 cubed"
SUBSCRIPT = {"₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4", "₅": "5",
             "₆": "6", "₇": "7", "₈": "8", "₉": "9"}

# Operators. Spoken forms are chosen to sound like a lecturer, not a calculator.
OPERATORS = [
    ("≤", " is less than or equal to "), ("≥", " is greater than or equal to "),
    ("≠", " is not equal to "), ("≈", " is approximately "),
    ("≡", " is identical to "), ("∝", " is proportional to "),
    ("→", " goes to "), ("←", " comes from "), ("⇒", " implies "),
    ("↔", " is equivalent to "), ("∈", " in "), ("∉", " not in "),
    ("∑", " the sum of "), ("∏", " the product of "), ("∫", " the integral of "),
    ("√", " the square root of "), ("∞", " infinity "), ("∂", " partial "),
    ("±", " plus or minus "), ("×", " times "), ("÷", " divided by "),
    # U+2212 is the true minus sign: never a hyphen or a range dash, so it is
    # always "minus" (as "-" is inside math). Deleting it would read "−3"
    # as "3" and "a − b" as "a b".
    ("−", " minus "),
    ("·", " dot "), ("⊥", " independent of "), ("∼", " distributed as "),
    ("∀", " for all "), ("∃", " there exists "), ("∩", " intersect "),
    ("∪", " union "), ("⊂", " is a subset of "), ("°", " degrees "),
    ("–", "-"), ("—", "-"), ("‑", "-"),          # dashes -> plain hyphen
    ("“", '"'), ("”", '"'), ("‘", "'"), ("’", "'"), ("…", "..."),
    (" ", " "), (" ", " "), ("​", ""),
]

UNIT_WORDS = {
    "mg": "milligrams", "kg": "kilograms", "g": "grams", "dL": "deciliter",
    "mL": "milliliters", "L": "liters", "mmHg": "millimeters of mercury",
    "yr": "year", "y": "year", "d": "day", "h": "hour", "wk": "week",
    "person-years": "person years", "py": "person years",
}


class UnspokenSymbol(ValueError):
    """Raised when text still contains a symbol we would not say aloud."""


@dataclass
class NormalizationReport:
    original: str
    normalized: str
    changes: list[tuple[str, str]] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.changes:
            return "no changes"
        return "; ".join(f"{a!r}->{b!r}" for a, b in self.changes[:12])


def _expand_combining(text: str) -> tuple[str, list]:
    """β̂ -> 'beta hat'. Must run before anything strips marks."""
    changes = []
    out = []
    decomposed = unicodedata.normalize("NFD", text)
    i = 0
    while i < len(decomposed):
        ch = decomposed[i]
        marks = []
        j = i + 1
        while j < len(decomposed) and unicodedata.combining(decomposed[j]):
            marks.append(decomposed[j])
            j += 1
        base = GREEK.get(ch, ch)
        if marks and not ch.isalpha():
            # A mark on no letter at all -- at the start of the text, on a
            # space, a digit or punctuation -- is stray, not notation. It is
            # left for the final check, which reports it (and in pipeline mode
            # drops it); it is not spoken as "hat", and it never crashes.
            out.append(decomposed[i:j])
        elif marks:
            # Distinguish spelling from notation, without a hand-maintained list.
            # If base+mark composes to a single real Unicode character, it is a
            # letter of some language's alphabet -- "e" + acute is é,
            # "n" + tilde is enye -- and must pass through untouched. If it does
            # NOT compose, no alphabet uses it, so it is mathematical notation:
            # beta+circumflex is an estimator, x+macron is a sample mean. Those
            # get spoken. So "é" survives and "beta hat" is heard.
            #
            # Except a marked Latin letter standing ALONE: ŷ and ȳ compose (to
            # Welsh and Livonian letters; ᾱ to a Greek one), so "ŷ" would pass
            # through and be read as a plain "y" -- the fitted value said as
            # the outcome. A single letter with a hat or bar between
            # non-letters is notation; inside
            # a word ("tŷ", "Māori") it is still spelling.
            composed = unicodedata.normalize("NFC", ch + "".join(marks))
            alone = ((ch.isascii() or ch in GREEK)
                     and not (i and (decomposed[i - 1].isalpha()
                                     or unicodedata.combining(decomposed[i - 1])))
                     and not (j < len(decomposed) and decomposed[j].isalpha()))
            notation = alone and all(m in COMBINING for m in marks)
            if len(composed) == 1 and not notation:
                out.append(composed)
                i = j
                continue
            # Marks with no spoken name stay on the letter (composed where an
            # alphabet has the result), so a stack such as acute + circumflex
            # is said "á hat" and an unknown mark on its own is reported by
            # the final check rather than raising here.
            unknown = "".join(m for m in marks if m not in COMBINING)
            spoken = [COMBINING[m] for m in marks if m in COMBINING]
            head = (unicodedata.normalize("NFC", ch + unknown) if unknown
                    else base)
            piece = " ".join([head] + spoken)
            if spoken:
                changes.append((unicodedata.normalize("NFC", decomposed[i:j]),
                                piece))
            out.append(" " + piece + " ")
        else:
            if ch in GREEK:
                changes.append((ch, GREEK[ch]))
                out.append(" " + GREEK[ch] + " ")
            else:
                out.append(ch)
        i = j
    return unicodedata.normalize("NFC", "".join(out)), changes


#: Spoken forms for the punctuation of code and markup, longest first so
#: `%>%` is matched before `>` and `<-` before `<`. Each is something a
#: lecturer actually says; the comparison step after this would otherwise
#: read `fit <- glm(...)` as "fit is less than minus glm".
CODE_TOKENS = [
    ("%>%", " pipe "), ("|>", " pipe "), ("<-", " gets "), ("->", " goes to "),
    ("=>", " goes to "), ("==", " equals "), ("::", " colon colon "),
]
_CURRENCY_WORDS = {"€": "euros", "£": "pounds", "¥": "yen"}
_SAID_AS = {"24/7": "twenty-four seven", "50/50": "fifty-fifty",
            "9/11": "nine eleven"}


def _markup(text: str) -> tuple[str, list]:
    """Markdown, links, HTML tags, code operators and the like, as spoken.

    Narration is pasted from slides, notes and chat, and carries their
    formatting. Each rule prevents a specific mis-reading: "## Methods" as
    "number number Methods", a link read out as its URL with the slashes
    deleted, `<br>` as "is less than br is greater than", `fit <- glm()` as
    "fit is less than minus glm", "#MeToo" as "number MeToo", "€50" as "euros
    50", and "24/7" as "24 out of 7".
    """
    changes = []

    def sub(pat, rep, label, flags=0):
        nonlocal text
        text, n = re.subn(pat, rep, text, flags=flags)
        if n:
            changes.append((label, rep if isinstance(rep, str) else label))

    # Marks meant for the reader's eye, never the ear: "<!-- check -->",
    # "~~old~~ new" and "teh [sic]" would otherwise be read out symbol by
    # symbol. A comment and struck-through text are not narration.
    sub(r"<!--.*?-->", " ", "html comment", flags=re.S)
    sub(r"~~(?=\S)[^~\n]*?(?<=\S)~~", " ", "struck-through text")
    sub(r"<(del|s|strike)>[^<]*</\1>", " ", "struck-through text",
        flags=re.I)
    sub(r"\s*[\[(]\s*[*_]?sic[*_]?!?\s*[\])]", "", "[sic]", flags=re.I)
    # Markdown structure at the start of a line: headings, quotes, rules.
    sub(r"(?m)^[ \t]{0,3}#{1,6}[ \t]+", "", "markdown heading")
    sub(r"(?m)^[ \t]{0,3}>[ \t]?", "", "markdown quote")
    sub(r"(?m)^[ \t]*(?:-{3,}|\*{3,}|_{3,})[ \t]*$", "", "markdown rule")
    # [text](url) -> text; ![alt](img) -> alt.
    sub(r"!?\[([^\]]+)\]\((?:[^)\s]+)\)", r"\1", "markdown link")
    # A bare web address is said by its site: "https://www.cdc.gov/flu/x.html"
    # -> "cdc.gov". Nobody reads a path aloud.
    # The host never ends in a dot and the path never does either, so the
    # full stop after "Visit https://www.cdc.gov." stays a full stop.
    path = r"(?:/(?:[^\s)\]>,;]*[^\s)\]>,;.])?)?"
    sub(r"\bhttps?://(?:www\.)?([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)" + path,
        r"\1", "web address")
    sub(r"\bwww\.([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)" + path, r"\1",
        "web address")
    # <tag> and </tag>: the tag's name, not two comparisons around it.
    sub(r"</?([A-Za-z][A-Za-z0-9-]*)\s*/?>", r" \1 ", "html tag")
    # Code operators, before the comparison step sees their < and >.
    for tok, word in CODE_TOKENS:
        if tok in text:
            text = text.replace(tok, word)
            changes.append((tok, word.strip()))
    # A hashtag is a hashtag; "#12" stays "number 12" (handled later).
    sub(r"(?<![\w&])#(?=[A-Za-z])", "hashtag ", "hashtag")
    # A hash sign on its own, as in "In R, # starts a comment".
    sub(r"(?<!\S)#(?=\s)", "hash", "hash sign")
    # Escapes in code: "\n" is "backslash n". A longer \word is LaTeX and is
    # read as the word.
    sub(r"\\([A-Za-z])\b", r" backslash \1 ", "escape")
    # Money in other currencies, number first, as with dollars.
    sub(r"([€£¥])\s?(\d[\d,]*(?:\.\d+)?)(\s?(?:thousand|million|billion)\b)?",
        lambda m: f"{m.group(2)}{m.group(3) or ''} {_CURRENCY_WORDS[m.group(1)]}",
        "currency")
    # Fixed expressions written with a slash.
    for written, said in _SAID_AS.items():
        if re.search(rf"(?<![\d/]){re.escape(written)}(?![\d/])", text):
            text = re.sub(rf"(?<![\d/]){re.escape(written)}(?![\d/])", said, text)
            changes.append((written, said))
    # A ratio "3:1" or "1:2:1" is said with "to"; a clock time "9:30" is not.
    sub(r"\b(\d+)((?::\d+)+)\b",
        lambda m: m.group(0) if re.fullmatch(r"\d{1,2}:\d{2}", m.group(0))
        else " to ".join([m.group(1)] + m.group(2)[1:].split(":")), "ratio")
    # A tilde before a number is "approximately"; elsewhere (y ~ x in a model
    # formula) it is "tilde", which is what is said in a stats class.
    sub(r"~\s*(?=\d)", " approximately ", "~number")
    sub(r"~", " tilde ", "tilde")
    return text, changes


def _comparisons(text: str) -> tuple[str, list]:
    """The single most dangerous class: ASCII comparison operators."""
    changes = []
    subs = [
        (r"<=", " is less than or equal to "),
        (r">=", " is greater than or equal to "),
        (r"!=", " is not equal to "),
        (r"<", " is less than "),
        (r">", " is greater than "),
    ]
    for pat, rep in subs:
        if re.search(pat, text):
            changes.append((pat, rep.strip()))
            text = re.sub(pat, rep, text)
    return text, changes


def _pipes(text: str) -> tuple[str, list]:
    """'Pr(T>t|X=x)' — the bar is conditioning, and it is always deleted."""
    if "|" not in text:
        return text, []
    return re.sub(r"\s*\|\s*", " given ", text), [("|", "given")]


def _superscripts(text: str) -> tuple[str, list]:
    changes = []

    def sup(m):
        digits = "".join(SUPERSCRIPT[c] for c in m.group(0))
        word = {"2": "squared", "3": "cubed"}.get(digits, f"to the power {digits}")
        changes.append((m.group(0), word))
        return " " + word + " "

    text = re.sub("[" + "".join(SUPERSCRIPT) + "]+", sup, text)

    def sub_(m):
        digits = "".join(SUBSCRIPT[c] for c in m.group(0))
        changes.append((m.group(0), f"sub {digits}"))
        return f" sub {digits} "

    text = re.sub("[" + "".join(SUBSCRIPT) + "]+", sub_, text)
    # caret form: x^2, beta^T
    text = re.sub(r"\^2\b", " squared ", text)
    text = re.sub(r"\^3\b", " cubed ", text)
    text = re.sub(r"\^T\b", " transpose ", text)
    text = re.sub(r"\^\{?T\}?", " transpose ", text)
    text = re.sub(r"\^\{?([-−]?\d+)\}?", r" to the power \1 ", text)
    return text, changes


def _slashes(text: str) -> tuple[str, list]:
    """
    '/' is genuinely ambiguous and every reading is wrong somewhere:
      12/34      -> "twelve out of thirty-four"
      mg/dL      -> "milligrams per deciliter"
      and/or     -> "and or"
      24/7       -> "twenty-four seven"
    The two unambiguous cases are handled and the rest REFUSED, rather than
    guessed silently.
    """
    changes = []
    text = re.sub(r"\band\s*/\s*or\b", "and or", text, flags=re.I)
    unit_pat = "|".join(sorted(map(re.escape, UNIT_WORDS), key=len, reverse=True))
    text, n = re.subn(rf"\b({unit_pat})\s*/\s*({unit_pat})\b",
                      lambda m: f"{UNIT_WORDS[m.group(1)]} per {UNIT_WORDS[m.group(2)]}",
                      text)
    if n:
        changes.append(("unit/unit", "per"))
    text, n = re.subn(r"(?<=\d)\s*/\s*(?=\d)", " out of ", text)
    if n:
        changes.append(("n/n", "out of"))
    # word/word -- "retrospective/prospective", "hazard ratio / rate ratio".
    # In academic prose this is almost always "or". That reading is taken and
    # RECORDED, so the decision shows up in the change log and in the script
    # editor, rather than being a silent guess or a hard stop on ordinary
    # writing.
    text, n = re.subn(r"(?<=[A-Za-z])\s*/\s*(?=[A-Za-z])", " or ", text)
    if n:
        changes.append(("word/word", "or"))
    return text, changes


# Everyday prose symbols. Separate from OPERATORS because these are
# unambiguous; the operators above are the ones that change meaning when
# dropped.
PROSE = [
    ("&", " and "), ("@", " at "), ("+", " plus "), ("~", " approximately "),
    ("[", " ("), ("]", ") "), ("{", " ("), ("}", ") "), ("|", " given "),
    ("_", " "), ("\\", " "), ("*", " "), ("#", " number "), ("$", " dollars "),
    ("`", ""), ("^", " "),
    ("€", " euros "), ("£", " pounds "), ("§", " section "), ("†", " "), ("‡", " "),
    ("•", ". "), ("·", " "), ("→", " goes to "), ("™", " "), ("®", " "), ("©", " "),
]


def _prose(text: str) -> tuple[str, list]:
    changes = []
    for sym, word in PROSE:
        if sym in text:
            text = text.replace(sym, word)
            if word.strip():
                changes.append((sym, word.strip()))
    return text, changes


def _misc(text: str) -> tuple[str, list]:
    changes = []
    for sym, word in OPERATORS:
        if sym in text:
            text = text.replace(sym, word)
            if word.strip() not in {"-", '"', "'", "...", ""}:
                changes.append((sym, word.strip()))
    # percent
    text = re.sub(r"\s*%", " percent", text)
    # equals (leave '=' inside no context; always spoken)
    text = re.sub(r"\s*=\s*", " equals ", text)
    # numeric ranges: 0.61-0.85 -> 0.61 to 0.85
    text = re.sub(r"(?<=\d)\s*-\s*(?=\d)", " to ", text)
    # thousands separators: 1,247 -> 1247 so the engine reads it as a number
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    # p-values: keep "p" as a word
    text = re.sub(r"\bp\s+equals\b", "p equals", text)
    return text, changes


def normalize(text: str, *, strict: bool = True) -> NormalizationReport:
    """
    Turn written academic prose into text that is safe to speak.

    strict=True (default) raises UnspokenSymbol if anything unhandled remains,
    rather than letting it be silently deleted downstream.
    """
    original = text
    changes: list[tuple[str, str]] = []
    for step in (_expand_combining, _markup, _comparisons, _pipes,
                 _superscripts, _slashes, _misc, _prose):
        text, c = step(text)
        changes.extend(c)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text).strip()

    bad = sorted({c for c in text if not _is_safe(c)})
    dropped: list[str] = []
    if bad:
        detail = ", ".join(f"{c!r} (U+{ord(c):04X} {unicodedata.name(c, '?')})"
                           for c in bad)
        if strict:
            # Library and CI behaviour: refuse.
            raise UnspokenSymbol(
                f"Text still contains symbols with no spoken form: {detail}\n"
                f"  in: {text[:160]!r}\n"
                f"Add a rule to realme/text/normalize.py, or rewrite the sentence. "
                f"Do NOT ignore this: the TTS front-end will delete these silently."
            )
        # Pipeline behaviour: drop them, but REPORT every one. The instructor
        # sees "I removed ` from slide 2" in the script editor rather than a
        # stack trace. The invariant is "never delete without saying so".
        for c in bad:
            text = text.replace(c, " ")
        dropped = [f"removed {c!r} (U+{ord(c):04X} {unicodedata.name(c, '?')}) "
                   f"- no spoken form defined" for c in bad]
        text = re.sub(r"[ \t]+", " ", text).strip()
    return NormalizationReport(original=original, normalized=text,
                               changes=changes, dropped=dropped)
