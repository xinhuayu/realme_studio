#!/usr/bin/env python3
"""
Dollar signs, markup, code and LaTeX: what each is said as.

Narration such as "$150,000 ... $15,000" holds two dollar signs that could be
taken for a LaTeX span, which would read "150,000 for a one-year pilot.
Coordination" as an equation. So:

  * the math cache path resolves;
  * an amount is money, said as people say it: "150,000 dollars";
  * real inline math beside an amount is still found as math;
  * math the converter cannot read is never half-read as a price.
"""
from __future__ import annotations
import os, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

_ISO = tempfile.mkdtemp(prefix="t_text_")
os.environ["REALME_HOME"] = str(Path(_ISO) / "home")
os.environ["HOME"] = _ISO
os.environ["USERPROFILE"] = _ISO

from t_util import check, summary          # noqa: E402

SLIDE_84 = ("Available: $150,000 for a one-year pilot. Coordination $15,000 "
            "plus evaluation and advisory input $15,000. Remaining $120,000: "
            "choose a coherent intervention. For example, 120 times ($600 "
            "exercise + $400 transport) = $120,000.")


def main() -> int:
    from realme.text.mathspeech import (default_cache_path, dollar_math_spans,
                                        looks_like_math)
    from realme.text.prepare import prepare_for_speech, speak_currency

    print("\nthe math cache is found (the crash)")
    try:
        path = default_cache_path()
        ok = path.name == "math_cache.json"
    except Exception as e:
        ok, path = False, e
    check(ok, "default_cache_path() resolves", str(path))

    print("\nthe slide that failed now prepares, and says money as money")
    try:
        p = prepare_for_speech(SLIDE_84, "piper")
        text, warn = p.engine_text, p.warnings
    except Exception as e:
        text, warn = f"{type(e).__name__}: {e}", ["crashed"]
    check("150000 dollars for a one-year pilot" in text.replace(",", ""),
          "$150,000 is read as 150,000 dollars", text[:90])
    check("dollars 1" not in text and not any("LaTeX" in w for w in warn),
          "and no amount is mistaken for an equation", str(warn)[:120])
    check("600 dollars exercise plus 400 dollars transport" in text,
          "two amounts in one bracket are two amounts", text[-90:])

    print("\nhow an amount is said")
    for given, said in (("$5", "5 dollars"), ("$1", "1 dollar"),
                        ("$5.50", "5 dollars and 50 cents"),
                        ("$3.00", "3 dollars"), ("$0.01", "0 dollars and 1 cent"),
                        ("$1.2 million", "1.2 million dollars"),
                        ("$2M", "2 million dollars"), ("$5k", "5 thousand dollars"),
                        ("$150,000", "150,000 dollars")):
        got = speak_currency(given)
        check(got == said, f"{given} -> {said}", got)
    check(speak_currency("costs $ more") == "costs $ more",
          "a dollar sign with no number is left alone")

    print("\ninline math is still math")
    for src, want in (("the estimator $\\hat\\beta$ here", ["\\hat\\beta"]),
                      ("when $p < 0.05$ and $n = 120$", ["p < 0.05", "n = 120"]),
                      ("coverage $1-\\alpha$", ["1-\\alpha"]),
                      ("costs $1.2 million and slope $\\beta_1$", ["\\beta_1"]),
                      ("$5 and $10", []),
                      ("$150,000 for a pilot. Coordination $15,000", []),
                      ("a literal \\$5 and $x$", ["x"])):
        got = [t for _, _, t in dollar_math_spans(src)]
        check(got == want, f"{src!r} -> math {want}", str(got))
    check(looks_like_math("x^2") and not looks_like_math(" x ")
          and not looks_like_math("600 exercise + "),
          "math hugs its dollar signs; prose and padding do not")

    print("\nmath that cannot be read is not half-read as money")
    p = prepare_for_speech("with $1-\\alpha$ coverage", "piper")
    check("1 dollar" not in p.engine_text,
          "$1-\\alpha$ is never read as 'one dollar'", p.engine_text)
    print("\nmarkup, code and web text, as a lecturer would say it")
    # (written, must contain, must not contain) -- the third column is the
    # wrong, literal reading of each.
    said = [
        ("## Methods\nWe used Cox models.", "Methods", "number"),
        ("> A quoted line.", "A quoted line", "greater than"),
        ("[CDC guidance](https://www.cdc.gov/flu/index.html) is clear.",
         "guidance is clear", "https"),
        ("Visit https://www.cdc.gov.", "Visit cdc.gov.", "https"),
        ("Visit www.who.int/news for more.", "Visit who.int for more", "news"),
        ("Use <code> tags and <br> breaks.", "code tags and br breaks",
         "less than"),
        ("In R: fit <- glm(y ~ x + age)", "fit gets glm", "less than"),
        ("Arrow: A -> B, and B => C.", "A goes to B, and B goes to C",
         "greater than"),
        ("x == y", "x equals y", "equals equals"),
        ("df %>% filter(x)", "df pipe filter", "percent"),
        ("#MeToo trended.", "hashtag MeToo", "number"),
        ("In R, # starts a comment.", "hash starts", "number"),
        ("See slide #12.", "number 12", "hashtag"),
        ("Use \\n for a new line.", "backslash n", "Use n "),
        ("€50 and £1.2 million", "50 euros and 1.2 million pounds",
         "euros 50"),
        ("24/7 access and a 50/50 split", "twenty-four seven",
         "out of 7"),
        ("The ratio was 3:1.", "3 to 1", "3:1"),
        ("Class starts at 9:30.", "9:30", "9 to 30"),
        ("~50 participants", "approximately 50", "tilde"),
        ("y ~ x1 + x2", "y tilde x1", "approximately"),
        ("p < 0.05 and BMI >= 30", "is greater than or equal to 30", "<"),
    ]
    for written, want, never in said:
        got = prepare_for_speech(written, "piper").engine_text
        check(want in got and never not in got,
              f"{written!r} -> ...{want}...", got)

    print("\nLaTeX is read even without the Speech Rule Engine")
    from realme.text.mathspeech import plain_latex
    for latex, want in ((r"\hat{\beta}", "beta hat"),
                        (r"\hat\beta_1", "beta hat sub 1"),
                        (r"\bar{x} \pm 1.96 \frac{s}{\sqrt{n}}",
                         "x bar plus or minus 1.96 s over the square root of n"),
                        (r"1-\alpha", "1 minus alpha"),
                        (r"\sum_{i=1}^{n} x_i", "the sum from i=1 to n of x sub i"),
                        (r"Y \sim N(\mu, \sigma^2)", "is distributed as"),
                        (r"\beta^{\top} X", "beta transpose X"),
                        (r"\widehat{HR}", "HR hat")):
        got = plain_latex(latex)
        check(want in got, f"{latex} -> {want}", got)
    p = prepare_for_speech("The estimator $\\hat{\\beta}$ is unbiased.", "piper")
    check("beta hat" in p.engine_text and "dollars" not in p.engine_text,
          "in narration: $\\hat{\\beta}$ is 'beta hat', not 'dollars hat beta "
          "dollars'", p.engine_text)
    check(any("not yet approved" in w for w in p.warnings),
          "and it is reported as a draft, for `realme math` to approve")

    print("\nthe minus sign U+2212 is 'minus', never deleted")
    for written, want in (("\u22123 degrees", "minus 3 degrees"),
                          ("a \u2212 b", "a minus b"),
                          ("x = \u22122", "x equals minus 2"),
                          ("$a \u2212 b$", "a minus b"),
                          ("10\u207b\u00b3", "10 to the power -3")):
        p = prepare_for_speech(written, "piper", strict=True)
        check(want in p.engine_text and "\u2212" not in p.engine_text,
              f"{written!r} -> ...{want}...", p.engine_text)

    print("\na hat or bar on a letter that happens to compose is still spoken")
    for written, want in (("\u0233 is the mean", "y bar is the mean"),
                          ("y\u0304 is the mean", "y bar is the mean"),
                          ("\u0177 is fitted", "y hat is fitted"),
                          ("y\u0302 is fitted", "y hat is fitted"),
                          ("\u0232 and \u0176", "Y bar and Y hat"),
                          ("p\u0302 and \u03b2\u0302", "p hat and beta hat"),
                          ("\u03b8\u0304 and x\u0304", "theta bar and x bar")):
        got = prepare_for_speech(written, "piper").engine_text
        check(want in got, f"{written!r} -> ...{want}...", got)
    for word in ("Hern\u00e1n", "M\u0101ori", "t\u0177 bach", "se\u00f1or",
                 "na\u00efve"):
        got = prepare_for_speech(word, "piper").engine_text
        check(got == word, f"but {word!r} inside a word is spelling", got)

    print("\ncombining marks in odd places never crash the normaliser")
    from realme.text.normalize import normalize, UnspokenSymbol
    for odd in ("\u0302abc", "x\u0302\u0304", "a\u0301\u0302", " \u0302 ",
                "3\u0302", "x.\u0302", ",\u0304", "\u20d7", "q\u0327",
                "\u0302"):
        try:
            p = prepare_for_speech(odd, "piper")
            ok, detail = True, p.engine_text
        except Exception as e:
            ok, detail = False, f"{type(e).__name__}: {e}"
        try:
            normalize(odd, strict=True)
        except UnspokenSymbol:
            pass                        # strict mode's documented refusal
        except Exception as e:
            ok, detail = False, f"strict: {type(e).__name__}: {e}"
        check(ok, f"{odd!r} degrades gracefully", detail)
    check(prepare_for_speech("x\u0302\u0304", "piper").engine_text
          == "x hat bar", "two marks on one letter are both said")
    check("hat" not in prepare_for_speech(" \u0302 3\u0302", "piper").engine_text,
          "a stray mark on a space or digit is not said as 'hat'")

    print("\ninside math, a slash is division")
    for written, want in (("$a/b$", "a over b"), ("$x/2$", "x over 2"),
                          ("\\(a/b\\)", "a over b")):
        got = prepare_for_speech(written, "piper").engine_text
        check(want in got and " or " not in got,
              f"{written!r} -> {want}", got)
    check(prepare_for_speech("and/or", "piper").engine_text == "and or",
          "while prose and/or is still 'and or'")

    print("\nmarks for the eye are not read aloud")
    for written, want, never in (
            ("They wrote teh [sic] result.", "They wrote teh result.", "sic"),
            ("teh (sic) result", "teh result", "sic"),
            ("~~old~~ new value", "new value", "tilde"),
            ("<del>old</del> new value", "new value", "del"),
            ("a <!-- check this --> b", "a b", "less than"),
            ("~50 and y ~ x", "approximately 50 and y tilde x", "@"),
            ("the basic idea", "the basic idea", "@")):
        got = prepare_for_speech(written, "piper").engine_text
        check(want in got and never not in got, f"{written!r} -> {want}", got)
    return summary()


if __name__ == "__main__":
    raise SystemExit(main())
