"""
The greeting a podcast opens with and the sign-off it closes on.

The opening says who is speaking, what the question is and who is answering
it; the close tells the listener the episode has ended.

These are TEMPLATES, not model output: they say the same thing every episode,
and they contain facts that must be exact -- the host's name and the topic as
typed. A language model asked to write a greeting may alter them.

They are ordinary turns, so they get the instructor's voice, the lexicon, the
control markers and the pause rhythm like everything else. `[[pause]]` after
the topic is deliberate: an announcement wants a beat before the conversation
starts.
"""
from __future__ import annotations

from realme.core.schema import Turn

MODE_FRAMING = {
    "debate": "we are going to argue about",
    "socratic": "we are going to think through",
    "interview": "we are going to look closely at",
}

OPENING = ("Hello, and welcome. I am {host}, and today {framing} {topic} "
           "[[pause:700]] Joining me is {guest}, who is going to push back. "
           "Let's begin.")

CLOSING = ("That is where we will leave it. Thanks to {guest}, and thanks to "
           "you for listening. [[pause:500]] I am {host}. Until next time.")


def _clean_topic(topic: str) -> str:
    """
    A topic typed as a question reads badly dropped into a sentence, and it
    already carries its own terminal punctuation -- so the template must not
    add a second one (e.g. "...immortal time bias?.").
    """
    t = " ".join(topic.split()).rstrip(" .")
    if t.endswith("?"):
        return "the question: " + t          # already ends in "?"
    return t + "."


def opening_turn(topic: str, mode: str, host: str, guest: str) -> Turn:
    return Turn(
        turn_id=-1, speaker_id="instructor", speaker_name=host,
        voice="instructor", stance="opening",
        spoken_text=OPENING.format(
            host=host, guest=guest, topic=_clean_topic(topic),
            framing=MODE_FRAMING.get(mode, MODE_FRAMING["debate"])))


def closing_turn(host: str, guest: str, last_id: int) -> Turn:
    return Turn(
        turn_id=last_id + 1, speaker_id="instructor", speaker_name=host,
        voice="instructor", stance="closing",
        spoken_text=CLOSING.format(host=host, guest=guest))


def wrap(script, host: str, guest: str) -> None:
    """
    Add the bookends to a script, in place, at most once.

    Idempotent because a dialogue resumes from `dialogue.json` on a re-run,
    which must not add a second greeting.
    """
    stances = {t.stance for t in script.turns}
    if "opening" in stances or "closing" in stances:
        return
    if not script.turns:
        return
    last = max(t.turn_id for t in script.turns)
    script.turns.insert(0, opening_turn(script.topic, script.mode, host, guest))
    script.turns.append(closing_turn(host, guest, last))
    # Renumber from zero. The opening was built with turn_id -1 to sort ahead
    # of turn 0, and the id is not just a label: it names the wav on disk
    # (`seg_-01_u00.wav`) and seeds the ledger key as `turn_id * 1000 + i`,
    # where a negative turn overlaps the turn below it. Contiguous ids, one
    # file each, no collisions.
    for i, t in enumerate(script.turns):
        t.turn_id = i
