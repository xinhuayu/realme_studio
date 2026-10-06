"""
A scripted exchange with hand-labelled stances, so the experiment can run
without a model call.

Written by hand rather than generated, because the question under test is
whether a KNOWN stance produces an audible difference. A generated script
would confound the prosody with whatever the writer happened to say that run,
and no amount of repetition would separate them afterwards.

The stances are deliberately adjacent and contrasting -- a press answered by a
concession, a question answered by a qualification -- because the claim being
tested is about contrast BETWEEN neighbouring turns, and a script that never
changes stance cannot show it either way.

The content is epidemiological and real enough to argue about; a conversation
about nothing is delivered like a conversation about nothing.
"""
from __future__ import annotations

#: (speaker, stance, text). `speaker` is "instructor" or "guest", matching the
#: dialogue pipeline's own two voices.
EXCHANGE = [
    ("instructor", "open",
     "Let's take on the claim that a well-adjusted observational study can "
     "stand in for a trial."),
    ("guest", "press",
     "It can't, and the reason isn't statistical. You adjust for what you "
     "measured. The trial handles what nobody thought to measure."),
    ("instructor", "concede",
     "That's fair. Randomisation buys you the unmeasured confounders, and no "
     "amount of regression does."),
    ("guest", "question",
     "So what would change your mind? What would a study have to show?"),
    ("instructor", "qualify",
     "A negative control outcome that comes out null. Not proof, but it "
     "tells you the residual confounding isn't running the result."),
    ("guest", "agree",
     "That I'll take. And it's cheap, which is why it's strange how rarely "
     "anyone reports one."),
    ("instructor", "press",
     "Rarely reported, or rarely run? Those are different failures, and only "
     "one of them is fixable by a journal."),
    ("guest", "summarise",
     "Either way the honest version is the same: say what you adjusted for, "
     "say what you couldn't, and stop calling it causal."),
    ("instructor", "close",
     "That's where we'll leave it. Measure what you adjust for, and be loud "
     "about what you can't."),
]


def words(text: str) -> int:
    return len([w for w in text.split() if any(c.isalnum() for c in w)])


def flattened() -> list[tuple[str, str, str]]:
    """The same exchange with every stance neutral: the control arm.

    The words are identical, so anything that differs between the two arms is
    the pacing and not the script.
    """
    return [(spk, "neutral", text) for spk, _, text in EXCHANGE]
