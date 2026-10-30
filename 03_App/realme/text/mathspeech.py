"""
Speaking mathematics.

Pipeline:  LaTeX --latex2mathml--> MathML --Speech Rule Engine--> spoken draft
                                                  |
                                    optional LLM rewrite to lecture register
                                                  |
                                    hash-keyed cache of APPROVED strings

Measured here: latex2mathml takes 0.25 ms per expression, SRE about 9 ms after
a 135 ms one-time startup. Both are free at lecture scale.

SRE's `clearspeak` domain is genuinely good at structure and genuinely wrong at
domain semantics, and it is worth seeing both:

    \\sum_{i=1}^{n}\\frac{Y_iA_i}{\\pi(X_i)}
      -> "the sum from i equals 1 to n of the fraction with numerator
          Y sub i A sub i and denominator pi of open paren X sub i close paren"
      exactly how a lecturer says it.

    \\beta^{\\top}          -> "beta raised to the down tack power"   (transpose)
    \\widehat{HR}          -> "H of R hat"                           (hazard ratio hat)

No rule engine can fix the second pair, because "HR means hazard ratio" is your
knowledge, not a property of the MathML. So SRE drafts and a human (or an LLM
you then approve) corrects -- and the approved string is cached against a hash
of the LaTeX, so a deck's fifty recurring expressions are corrected once and
then reused deterministically forever.
"""
from __future__ import annotations
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

# Domain fixes applied to SRE output before anyone sees it. These are the
# failures that recur in every statistics deck.
POST: list[tuple[str, str]] = [
    (r"\braised to the down tack power\b", "transpose"),
    (r"\bdown tack\b", "transpose"),
    (r"\bdivides\b", "given"),                    # conditioning bar
    (r"\bopen paren\b", ""), (r"\bclose paren\b", ""),
    (r"\bnormal ([A-Z])", r"\1"),
    (r"\bmodifying above (\w+) with caret\b", r"\1 hat"),
    (r"\bmodifying above (\w+) with bar\b", r"\1 bar"),
    (r"\bStartFraction\b", "the fraction"), (r"\bEndFraction\b", ""),
    (r"\bsigma summation\b", "the sum"),
    (r"\s{2,}", " "),
]

# Rules that are only correct given what was in the LaTeX. `\sim` between two
# operands is "distributed as" in statistics, but SRE says "tilde" -- and
# "tilde" is also the right word for an accent, so the fix cannot be applied
# blindly. Keying on the source resolves it exactly.
CONDITIONAL: list[tuple[str, str, str]] = [
    (r"\\sim", r"\btilde\b", "distributed as"),
    (r"\\perp|\\independent", r"\b(?:up tack|perpendicular)\b", "independent of"),
    (r"\\to|\\rightarrow", r"\bright arrow\b", "goes to"),
    (r"\\approx", r"\balmost equals\b", "is approximately"),
    (r"\\propto", r"\bproportional to\b", "is proportional to"),
]


def default_cache_path() -> Path:
    """The one math cache: the CLI approves into it and every render reads it.

    `realme math` used to default to `./math_cache.json` and the renderer
    built a MathSpeech with no cache at all, so an approval never reached a
    render and every lecture warned "math not yet approved" forever.
    """
    from realme.core.env import data_home
    return data_home() / "math_cache.json"


# ---------------------------------------------------------------------------
# A plain reader, for when the Speech Rule Engine is not installed.
#
# SRE is a Node program and most machines here do not have it, so every
# `$\\hat\\beta$` used to be left as written and then read symbol by symbol:
# "dollars hat beta dollars". This reads the common statistical notation the
# way a lecturer would. It is a draft like any other: never cached (so
# installing SRE later still takes over), and reported as not yet approved.

_GREEK = ("alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta "
          "iota kappa lambda mu nu xi pi rho sigma tau upsilon phi varphi chi "
          "psi omega Gamma Delta Theta Lambda Xi Pi Sigma Phi Psi Omega").split()
_WORDS = {
    "times": " times ", "cdot": " times ", "pm": " plus or minus ",
    "mp": " minus or plus ", "leq": " <= ", "le": " <= ", "geq": " >= ",
    "ge": " >= ", "neq": " != ", "ne": " != ", "approx": " is approximately ",
    "sim": " is distributed as ", "to": " goes to ", "rightarrow": " goes to ",
    "infty": " infinity ", "sum": " the sum of ", "prod": " the product of ",
    "int": " the integral of ", "log": " log ", "ln": " natural log ",
    "exp": " exp ", "Pr": " the probability ", "mid": " given ",
    "perp": " independent of ", "in": " in ", "partial": " partial ",
    "ldots": " dot dot dot ", "cdots": " dot dot dot ", "dots": " dot dot dot ",
    "propto": " is proportional to ", "top": " transpose ",
}
_DROP = ("left", "right", "big", "Big", "bigg", "Bigg", "displaystyle",
         "quad", "qquad", "limits", "nolimits")
_BRACE = r"\{([^{}]*)\}"


def plain_latex(latex: str) -> str:
    """Common LaTeX as words, with no converter. Comparisons are left as
    symbols for the normaliser, which already says them."""
    t = latex
    # Limits first, while they are still attached to the operator.
    lim = r"(?:\{([^{}]*)\}|(\w))"
    for op, word in (("sum", "the sum"), ("prod", "the product"),
                     ("int", "the integral")):
        t = re.sub(r"\\" + op + r"\s*_" + lim + r"\s*\^" + lim,
                   lambda m: f" {word} from {m.group(1) or m.group(2)} to "
                             f"{m.group(3) or m.group(4)} of ", t)
        t = re.sub(r"\\" + op + r"\s*_" + lim,
                   lambda m: f" {word} over {m.group(1) or m.group(2)} of ", t)
    for _ in range(4):                      # nested braces, innermost first
        t = re.sub(r"\\[dt]?frac" + _BRACE + _BRACE, r" \1 over \2 ", t)
        t = re.sub(r"\\sqrt" + _BRACE, r" the square root of \1 ", t)
        t = re.sub(r"\\(?:wide)?hat" + _BRACE, r" \1 hat ", t)
        t = re.sub(r"\\(?:bar|overline)" + _BRACE, r" \1 bar ", t)
        t = re.sub(r"\\(?:wide)?tilde" + _BRACE, r" \1 tilde ", t)
        t = re.sub(r"\\(?:text|mathrm|mathbf|mathit|mathcal|operatorname|"
                   r"textrm|textbf|mbox)" + _BRACE, r" \1 ", t)
        t = re.sub(r"\^\{\s*\\top\s*\}|\^\\top", " transpose ", t)
        t = re.sub(r"\^\{([^{}]*)\}", r" to the power \1 ", t)
        t = re.sub(r"_\{([^{}]*)\}", r" sub \1 ", t)
    # Unbraced accents: \hat\beta, \hat x.
    t = re.sub(r"\\(?:hat|widehat)\s*(\\?[A-Za-z]+)", r" \1 hat ", t)
    t = re.sub(r"\\(?:bar|overline)\s*(\\?[A-Za-z]+)", r" \1 bar ", t)
    t = re.sub(r"\^2\b", " squared ", t)
    t = re.sub(r"\^3\b", " cubed ", t)
    t = re.sub(r"\^(\\?\w)", r" to the power \1 ", t)
    t = re.sub(r"_(\\?\w)", r" sub \1 ", t)
    t = re.sub(r"\\[,;:! ]", " ", t)
    for name in _DROP:
        t = re.sub(r"\\" + name + r"\b", " ", t)

    def command(m):
        name = m.group(1)
        if name in _GREEK:
            return f" {name.lower() if name.startswith('var') else name} " \
                .replace("var", "")
        return _WORDS.get(name, f" {name} ")
    t = re.sub(r"\\([A-Za-z]+)", command, t)
    t = t.replace("{", " ").replace("}", " ")
    # In mathematics a hyphen is always a minus sign.
    t = re.sub(r"(?<!\w)-(?=\s*[\w(])|(?<=[\w)\s])-(?=\s*[\w(])", " minus ", t)
    # And a slash is always division. Left for the normaliser, `$a/b$` met
    # the prose rule for "and/or" and was read "a or b"; `$x/2$` lost it.
    t = re.sub(r"\s*/\s*", " over ", t)
    return re.sub(r"\s+", " ", t).strip()


class MathSpeech:
    def __init__(self, cache_path: Path | None = None, domain: str = "clearspeak"):
        self.domain = domain
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, dict] = {}
        if self.cache_path and self.cache_path.exists():
            from realme.core.textio import read_text
            self.cache = json.loads(read_text(self.cache_path))

    # ------------------------------------------------------------------ tools
    @staticmethod
    def available() -> dict:
        try:
            import latex2mathml.converter  # noqa: F401
            has_l2m = True
        except ImportError:
            has_l2m = False
        return {"latex2mathml": has_l2m, "sre": bool(shutil.which("sre"))}

    @staticmethod
    def key(latex: str) -> str:
        return hashlib.sha256(latex.strip().encode()).hexdigest()[:16]

    # ------------------------------------------------------------- conversion
    def _sre(self, mathml: str) -> str:
        exe = shutil.which("sre")
        if not exe:
            return ""
        proc = subprocess.run([exe, "--domain", self.domain, "--modality", "speech"],
                              input=mathml, capture_output=True, text=True, encoding="utf-8", errors="replace")
        return proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""

    def draft(self, latex: str) -> str:  # noqa: C901
        try:
            import latex2mathml.converter as conv
            mathml = conv.convert(latex)
        except Exception:
            return ""
        spoken = self._sre(mathml)
        for pat, rep in POST:
            spoken = re.sub(pat, rep, spoken)
        for src_pat, pat, rep in CONDITIONAL:
            if re.search(src_pat, latex):
                spoken = re.sub(pat, rep, spoken)
        return spoken.strip()

    # ------------------------------------------------------------------ cache
    def speak(self, latex: str) -> tuple[str, bool]:
        """Return (spoken, approved). Approved strings never regenerate."""
        k = self.key(latex)
        hit = self.cache.get(k)
        if hit:
            return hit["spoken"], hit.get("approved", False)
        spoken = self.draft(latex)
        # An empty draft means a converter was missing. Caching it would make
        # installing the converter later change nothing.
        if spoken:
            self.cache[k] = {"latex": latex, "spoken": spoken, "approved": False}
            self._flush()
            return spoken, False
        # No converter: the plain reader, uncached for the same reason.
        return plain_latex(latex), False

    def approve(self, latex: str, spoken: str) -> None:
        self.cache[self.key(latex)] = {"latex": latex, "spoken": spoken,
                                       "approved": True}
        self._flush()

    def pending(self) -> list[dict]:
        return [v for v in self.cache.values() if not v.get("approved")]

    def _flush(self) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache, indent=2,
                                                  ensure_ascii=False), encoding="utf-8")


MATH_SPAN = re.compile(r"\$([^$]+)\$|\\\(([^)]+)\\\)")
#: `\( ... \)` only. Dollar spans are found by `dollar_math_spans`, because
#: a dollar sign is far more often money than mathematics in a lecture.
PAREN_SPAN = re.compile(r"\\\(([^)]+)\\\)")


def looks_like_math(inner: str, after: str = "") -> bool:
    """Is the text between two dollar signs inline math, or two amounts?

    `$150,000 for a pilot. Coordination $15,000` has two dollar signs, and
    the old pattern read everything between them as LaTeX -- then crashed
    looking for the math cache, on slide 84 of a 90-slide lecture. TeX's own
    convention separates the cases: inline math hugs its dollar signs
    (`$x$`, never `$ x $`), whereas the second of two amounts follows a
    space and is followed by a digit. Prose between them gives it away too.
    """
    s = inner
    if not s.strip() or s[0].isspace() or s[-1].isspace():
        return False
    if after[:1].isdigit():            # "... $15,000": that $ opens an amount
        return False
    if re.search(r"[\\^_{}]", s):
        return True
    # Ordinary words of four letters or more are prose, not symbols.
    return not re.search(r"[A-Za-z]{4,}", s)


def dollar_math_spans(text: str) -> list[tuple[int, int, str]]:
    """(start, end, latex) for each `$...$` that is math, left to right.

    A dollar sign that does not open math is left where it is, for the
    currency pass to read as money; it is never consumed as a closing sign,
    so a real `$\\beta$` after an amount is still found.
    """
    out, i = [], 0
    while True:
        i = text.find("$", i)
        if i < 0:
            return out
        if i and text[i - 1] == "\\":          # \$ is a literal dollar
            i += 1
            continue
        j = text.find("$", i + 1)
        while j > 0 and text[j - 1] == "\\":
            j = text.find("$", j + 1)
        if j < 0:
            return out
        inner = text[i + 1:j]
        if looks_like_math(inner, text[j + 1:j + 2]):
            out.append((i, j + 1, inner))
            i = j + 1
        else:
            i += 1


def expand_math(text: str, ms: MathSpeech | None = None) -> tuple[str, list[dict]]:
    """
    Replace `$...$` and `\\(...\\)` spans with their spoken form.

    Runs BEFORE normalization, so the words it produces then pass through the
    same symbol checks as everything else. Anything it cannot convert is left
    in place and reported -- so it surfaces as a warning rather than being read
    aloud as backslashes.
    """
    ms = ms or MathSpeech()
    found: list[dict] = []

    def say(latex, original):
        spoken, approved = ms.speak(latex)
        found.append({"latex": latex, "spoken": spoken, "approved": approved})
        return f" {spoken} " if spoken else original

    parts, last = [], 0
    for a, b, latex in dollar_math_spans(text):
        parts += [text[last:a], say(latex, text[a:b])]
        last = b
    text = "".join(parts) + text[last:]
    return PAREN_SPAN.sub(lambda m: say(m.group(1), m.group(0)), text), found
