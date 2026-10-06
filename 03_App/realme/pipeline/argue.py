"""
Live argument mode: you against an interlocutor, in real time.

This is a different thing from the scripted debate in `dialogue.py`, and the
difference matters. A scripted debate is a model arguing with itself in two
voices -- pleasant to listen to, but you are not in it, so it cannot surprise
you. Live mode puts you in the exchange, which is where the ideas actually come
from. Afterwards the transcript can be rendered to audio, so you get both.

The interlocutor is deliberately not agreeable. Most of the value in arguing
with a model is lost the moment it starts saying "that's a great point" -- so
the system prompt forbids exactly that, and requires it to name the specific
step in your reasoning it is attacking.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from realme.core.textio import read_text

STANCES = {
    "adversary": """You are a sharp, well-read methodologist arguing AGAINST the \
user's position. You are not hostile, but you are not accommodating either.

Rules you must follow:
- Never open with praise. No "great point", "excellent question", "you're right that".
- Attack the WEAKEST SPECIFIC STEP in what they just said. Name it explicitly:
  quote or paraphrase the exact claim you are contesting before you contest it.
- Ground every objection in something concrete -- a named method, a study design,
  a failure mode, a counterexample, an assumption that does not hold.
- One objection at a time, developed properly. Do not spray five weak ones.
- If they answer you well, say so plainly and move to your next-strongest
  objection. Conceding a point is not losing; it is how the argument advances.
- If they are simply right and you have nothing left, say that, and say what
  would have to be true for them to be wrong.
- 100-200 words per turn. This is speech, not an essay.""",

    "socratic": """You are a patient interlocutor drawing out the user's reasoning \
by questioning it. Ask one real question per turn -- the question you actually \
want answered, not a rhetorical one. Follow the thread they open rather than \
your own agenda. Never lecture. Never answer your own question. When their \
answer exposes an inconsistency, point at it and ask about it rather than \
declaring it. 80-150 words.""",

    "steelman": """You are constructing the STRONGEST possible version of the \
position the user is arguing against. Not a caricature -- the case its best \
defenders would actually make, with their real evidence and their real reasons. \
State it in the first person as if you held it. Where the position has a genuine \
weakness, name that too: a steelman that hides its own soft spots is useless. \
150-250 words.""",

    "referee": """You are a methodological referee reviewing the exchange so far. \
Identify: what has actually been established, what remains contested, where each \
side is talking past the other, and the single question that would most advance \
the argument if answered next. Be concrete and brief.""",
}


@dataclass
class Move:
    speaker: str          # "instructor" | "interlocutor"
    text: str
    stance: str = ""


@dataclass
class Argument:
    topic: str
    stance: str = "adversary"
    context: str = ""
    moves: list[Move] = field(default_factory=list)

    def history(self) -> list[dict]:
        return [{"role": "user" if m.speaker == "instructor" else "model",
                 "text": m.text} for m in self.moves]

    def system_prompt(self) -> str:
        base = STANCES.get(self.stance, STANCES["adversary"])
        head = f"The topic under argument is: {self.topic}\n"
        if self.context:
            head += f"\nBackground material the user has supplied:\n{self.context[:120_000]}\n"
        return head + "\n" + base

    def save(self, path: Path) -> Path:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "Argument":
        d = json.loads(read_text(path))
        return cls(topic=d["topic"], stance=d.get("stance", "adversary"),
                   context=d.get("context", ""),
                   moves=[Move(**m) for m in d["moves"]])

    def to_markdown(self) -> str:
        lines = [f"# {self.topic}", "", f"_Live argument, stance: {self.stance}_", ""]
        for m in self.moves:
            who = "**You**" if m.speaker == "instructor" else "**Interlocutor**"
            lines += [f"{who} — {m.text}", ""]
        return "\n".join(lines)

    def to_dialogue_script(self):
        """Hand the finished argument to the audio pipeline unchanged."""
        from realme.core.schema import DialogueScript, Turn
        slug = "".join(c if c.isalnum() else "_" for c in self.topic).strip("_")[:60]
        return DialogueScript(
            session_id=slug or "argument", topic=self.topic, mode="debate",
            script_source="live-argument",
            turns=[Turn(turn_id=i + 1, speaker_id=m.speaker,
                        speaker_name="Instructor" if m.speaker == "instructor"
                                     else "Interlocutor",
                        voice="instructor" if m.speaker == "instructor" else "guest",
                        stance=m.stance or "statement", spoken_text=m.text)
                   for i, m in enumerate(self.moves)])


def reply(arg: Argument, writer, user_text: str) -> str:
    """One exchange: record the user's move, get the interlocutor's answer."""
    arg.moves.append(Move(speaker="instructor", text=user_text))
    answer = writer.chat(arg.system_prompt(), arg.history()[:-1], user_text)
    arg.moves.append(Move(speaker="interlocutor", text=answer, stance=arg.stance))
    return answer
