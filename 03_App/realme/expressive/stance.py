"""
What a turn is DOING, and what that should sound like.

The script writer already labels every turn with a stance and has since
dialogue was built. Until now exactly one place read it -- the bookend filter,
checking for "opening" and "closing" -- so a signal the model produces for
every turn has been sitting unused in the manifest.

Two things had to be true before it could drive anything:

1. A FIXED vocabulary. The response schema declares stance as a bare string,
   so the model returns whatever it likes: "statement", "rebuttal",
   "counterpoint", "agreeing", "Question". Six spellings of one intent cannot
   be mapped to anything, which is the same lesson the control markers taught.
   `normalize` folds free text onto this list and says so when it cannot.

2. A RELATIVE effect. The dialogue pace is measured, not guessed -- `realme
   voice pace` sets it and `casting.DIALOGUE_SPEED` is the fallback. A stance
   that SET the pace would silently undo that measurement. These are
   multipliers on top of whatever pace was measured, and the adapters already
   take exactly such a multiplier per utterance.

The numbers below are deliberate placeholders, and small. Piper's control is
length_scale, which is coarse: push it and an eager turn sounds fast-forwarded
rather than eager. They are starting points for measurement, not findings, and
nothing should cite them until the listening test has been run.
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

#: How the model's free text maps onto the vocabulary. Collected from what the
#: writer actually returned, not invented: "statement" is its overwhelming
#: default and means nothing in particular, so it maps to the neutral stance.
SYNONYMS = {
    # "statement" is the writer's default and means nothing in particular. It
    # maps to neutral, NOT to qualify: a turn the model did not characterise
    # must not quietly acquire the prosody of a hedge.
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
#: knob is a turn that sounds sped up rather than animated. The listening test
#: decides whether these move at all, and in which direction.
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
#: Its pace is exactly 1.0, and that is the point. This was "qualify" until a
#: test asked what an unrecognised stance does, and the answer was that it
#: slowed the turn by 3% -- a prosodic decision about a turn we had just
#: admitted we could not classify. Not knowing has to mean doing nothing.
NEUTRAL = "neutral"


def normalize(raw: str | None) -> tuple[str, bool]:
    """
    Fold a stance the model wrote onto the fixed vocabulary.

    Returns (stance, recognised). `recognised` is False when the text meant
    nothing to us and the neutral stance was substituted -- reported rather
    than hidden, because a writer that has drifted to a vocabulary of its own
    shows up as a run of unrecognised stances and nothing else.
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
    prompt cannot list a stance the code does not handle -- which is how a
    model ends up confidently emitting a label nothing reads.
    """
    lines = [f"  {s}: {MEANING[s]}" for s in STANCES]
    return ("Label every turn with exactly one stance from this list, using "
            "the word as written:\n" + "\n".join(lines))


def _check_tables() -> None:
    """The three tables must describe the same vocabulary.

    Run at import, because an edit that adds a stance to one table and forgets
    another is silent until a particular turn happens to use it -- and then it
    is a KeyError in the middle of a render, or worse, a stance with no pace
    that quietly does nothing. An earlier edit to this file dropped "open" from
    the pace table exactly this way.
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
