"""
What a turn is DOING, and what that should sound like.

The script writer labels every turn with a stance in the manifest. The
bookend filter also reads it, checking for "opening" and "closing".

Two requirements for using it to drive delivery:

1. A FIXED vocabulary. The response schema declares stance as a bare string,
   so the model returns whatever it likes: "statement", "rebuttal",
   "counterpoint", "agreeing", "Question". `normalize` folds free text onto
   this list and says so when it cannot.

2. A RELATIVE effect. The dialogue pace is measured -- `realme voice pace`
   sets it and `casting.DIALOGUE_SPEED` is the fallback. A stance that SET
   the pace would undo that measurement. These are multipliers on top of the
   measured pace, and the adapters take such a multiplier per utterance.

The numbers below are deliberately small placeholders. Piper's control is
length_scale, which is coarse: push it and an eager turn sounds fast-forwarded
rather than eager. They are starting points for measurement, not findings.
"""
from __future__ import annotations

#: The stances a turn may have. Anything outside this list is an error the
#: caller is told about, never a silent fallback.
STANCES = ("neutral", "open", "press", "concede", "question", "qualify",
           "agree", "summarise", "close")

#: Plain-language gloss, for the prompt and for the report.
MEANING = {
    "neutral":   "just saying something; no particular move",
    "open":      "starting the conversation",
    "press":     "pushing on what was just said",
    "concede":   "granting part of the other speaker's point",
    "question":  "asking for something specific",
    "qualify":   "narrowing or hedging a claim, including one's own",
    "agree":     "agreeing and adding to it",
    "summarise": "drawing the thread together",
    "close":     "ending the conversation",
}

#: How the model's free text maps onto the vocabulary, collected from what the
#: writer returns.
SYNONYMS = {
    # "statement" is the writer's default and means nothing in particular. It
    # maps to neutral, NOT to qualify: a turn the model did not characterise
    # does not get the prosody of a hedge.
    "statement": "neutral", "claim": "neutral", "assertion": "neutral",
    "point": "neutral", "remark": "neutral",
    "hedge": "qualify", "caveat": "qualify", "clarification": "qualify",
    "limitation": "qualify", "nuance": "qualify",
    "rebuttal": "press", "counterpoint": "press", "challenge": "press",
    "objection": "press", "pushback": "press", "critique": "press",
    "disagree": "press", "counter": "press",
    "concession": "concede", "acknowledgement": "concede",
    "acknowledgment": "concede", "conceding": "concede",
    "ask": "question", "query": "question", "probe": "question",
    "agreement": "agree", "agreeing": "agree", "support": "agree",
    "endorsement": "agree", "elaboration": "agree", "build": "agree",
    "summary": "summarise", "synthesis": "summarise",
    "conclusion": "summarise", "wrapup": "summarise", "recap": "summarise",
    "opening": "open", "intro": "open", "introduction": "open", "greeting": "open",
    "closing": "close", "signoff": "close", "outro": "close", "farewell": "close",
}

#: Pace multiplier per stance, applied ON TOP of the measured dialogue pace.
#:
#: PLACEHOLDERS. The spread is deliberately narrow -- +-6% at the extremes --
#: because the mechanism is length_scale and the failure mode of a coarse pace
#: knob is a turn that sounds sped up rather than animated.
PACE = {
    "neutral":   1.00,   # no stance, no change: never a guess
    "open":      1.00,
    "press":     1.06,   # leaning in
    "concede":   0.95,   # giving ground takes a moment
    "question":  1.02,
    "qualify":   0.97,   # a hedge is careful
    "agree":     1.03,
    "summarise": 0.96,   # drawing together, unhurried
    "close":     0.97,
}

#: What a turn gets when the writer said nothing we recognise.
#:
#: Its pace is exactly 1.0, so a turn that cannot be classified is not
#: retimed.
NEUTRAL = "neutral"


def normalize(raw: str | None) -> tuple[str, bool]:
    """
    Fold a stance the model wrote onto the fixed vocabulary.

    Returns (stance, recognised). `recognised` is False when the text meant
    nothing to us and the neutral stance was substituted. It is reported,
    because a writer that has drifted to a vocabulary of its own shows up only
    as a run of unrecognised stances.
    """
    key = (raw or "").strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    if not key:
        return NEUTRAL, False
    if key in STANCES:
        return key, True
    if key in SYNONYMS:
        return SYNONYMS[key], True
    # British and American spellings of the one stance that has both.
    if key in ("summarize", "summarizing", "summarising"):
        return "summarise", True
    return NEUTRAL, False


def pace_for(stance: str) -> float:
    """The pace multiplier for a stance. Unknown stances get 1.0, not a guess."""
    return PACE.get(stance, 1.0)


def prompt_vocabulary() -> str:
    """
    The stance list as the script writer should be told it.

    Generated from the vocabulary rather than typed into the prompt, so the
    prompt cannot list a stance the code does not handle.
    """
    lines = [f"  {s}: {MEANING[s]}" for s in STANCES]
    return ("Label every turn with exactly one stance from this list, using "
            "the word as written:\n" + "\n".join(lines))


def _check_tables() -> None:
    """The three tables must describe the same vocabulary.

    Run at import, because an edit that adds a stance to one table and forgets
    another is silent until a particular turn happens to use it -- and then it
    is a KeyError in the middle of a render, or a stance with no pace that
    quietly does nothing.
    """
    missing_pace = set(STANCES) - set(PACE)
    missing_meaning = set(STANCES) - set(MEANING)
    extra = (set(PACE) | set(MEANING)) - set(STANCES)
    bad_syn = {k: v for k, v in SYNONYMS.items() if v not in STANCES}
    problems = []
    if missing_pace:
        problems.append(f"no pace for {sorted(missing_pace)}")
    if missing_meaning:
        problems.append(f"no meaning for {sorted(missing_meaning)}")
    if extra:
        problems.append(f"{sorted(extra)} is not in STANCES")
    if bad_syn:
        problems.append(f"synonyms point outside the vocabulary: {bad_syn}")
    if PACE.get(NEUTRAL) != 1.0:
        problems.append(f"the neutral stance must not change the pace "
                        f"(it is {PACE.get(NEUTRAL)})")
    if problems:
        raise RuntimeError("realme.expressive.stance tables disagree: "
                           + "; ".join(problems))


_check_tables()
