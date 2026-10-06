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
    from realme.core.paths import data_home
    return data_home() / "math_cache.json"


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

    def repl(m):
        latex = m.group(1) or m.group(2) or ""
        spoken, approved = ms.speak(latex)
        found.append({"latex": latex, "spoken": spoken, "approved": approved})
        return f" {spoken} " if spoken else m.group(0)

    return MATH_SPAN.sub(repl, text), found
