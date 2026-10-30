"""
Several recordings of one voice, kept comparable.

A register is the same person delivering the same words differently:
explaining, pressing, conceding, wondering. Enrol them and the dialogue can
pick the matching one per turn -- and the difference between two of them is a
direction in speaker-embedding space, which is what makes continuous
expressive control possible without any training.

Both uses depend on the SAME thing: that the only difference between the files
is the delivery. That is much harder to get than it sounds, and two ways of
losing it are built into the tools.

LOUDNESS NORMALISATION ERASES THE SIGNAL. The enrolment chain ends in
`loudnorm=I=-16`, which brings every file to the same perceived level. Run it
over four registers separately and the pressing take and the conceding take
come out equally loud -- and loudness is one of the two or three things that
MAKE them different. The set is levelled ONCE, by a single correction measured
across all of it, so the registers keep their loudness relative to each other
while the set as a whole lands where it should. This is the same reasoning
`_measured_loudnorm` already applies within one file, one level up.

THE CHANNEL TRAVELS WITH THE VOICE. A speaker embedding encodes the room, the
microphone, the distance and the gain along with the speaker. If the neutral
take comes from an old enrolment and the others from a new session, their
difference contains the equipment, and the "emotion vector" is partly a change
of microphone. Registers are therefore recorded as a SET, in one sitting, and
this records which sitting each came from so a mismatch can be reported rather
than silently averaged in.
"""
from __future__ import annotations
import json, time
from dataclasses import dataclass, asdict
from pathlib import Path

from realme.core.media import require, run, duration_of

#: The registers a dialogue can ask for, and what each one is.
#:
#: Deliberately the same words as the stance vocabulary, because the whole
#: point is that a turn labelled `press` can be spoken in the pressing voice.
#: `explaining` is the neutral one and is required: every difference is
#: measured against it.
#: Each register, and the situation that produces it.
#:
#: Described as a listener rather than as a technique on purpose. "Raise your
#: pitch and speed up by ten percent" produces an impression of a person doing
#: that; picturing the colleague who just waved your point away produces the
#: delivery itself. The imagined listener does more work than any instruction
#: about pitch.
REGISTERS = {
    "explaining": "your ordinary teaching voice — as if to a student in office "
                  "hours who is following you perfectly well. The baseline "
                  "everything else is measured against.",
    "pressing":   "as if to a colleague who just waved this away, and you want "
                  "it to land. Not louder for its own sake — you mean it.",
    "conceding":  "as if the objection was fair and you are granting it. "
                  "Slower, softer, no hurry to get to the 'but'.",
    "wondering":  "as if at a whiteboard, genuinely unsure. Thinking aloud, "
                  "not quizzing anyone.",
}
NEUTRAL = "explaining"

#: Target loudness for the SET, in LUFS. The same figure the single-reference
#: enrolment uses, so a register and the old reference sit at the same level.
TARGET_LUFS = -16.0

#: How far apart two takes must be, on each axis, before the difference is
#: worth believing as delivery rather than as noise in the measurement.
#:
#: Three axes, because loudness alone is a weak test. Somebody asked to read a
#: passage four times can easily produce four takes at four levels that are
#: otherwise identical -- and, much more likely, four takes that sound the same
#: in every way. A register differs from neutral if ANY of these moves: the
#: question "are these four deliveries?" is answered yes by any real
#: difference, and no only when none of them shows one.
MEANINGFUL_LU = 0.7          # loudness, LU
MEANINGFUL_ST = 0.4          # median pitch, semitones
MEANINGFUL_RATE = 0.06       # speaking rate, as a fraction

#: Shortest take worth enrolling. Below this the speaker encoder is
#: characterising a phrase rather than a delivery, and the difference between
#: two such takes is mostly which words happened to be in them.
MIN_SECONDS = 20.0


#: What to read, four times.
#:
#: THE SAME WORDS EVERY TIME. This is not a preference. The difference between
#: two takes is what becomes the emotion direction, so if the words differ the
#: difference contains the words -- you would be measuring "sentences about
#: confounding minus sentences about bias" and calling it warmth.
#:
#: Which makes the passage hard to write, because it has to be deliverable
#: FOUR ways. A text that is inherently a question cannot be conceded; a text
#: that is inherently a concession cannot be pressed. So this one is built from
#: four moves that each register can lean on in turn: a claim to press, a limit
#: to concede, an open question to wonder at, and a plain definition to explain.
#: FOUR DIFFERENT DELIVERIES, not four readings. Reading it the way the words
#: suggest, four times, produces four identical takes and every direction is
#: zero -- there is nothing to extract from a difference that does not exist.
#: What stays the same is the WORDS. What must change is everything else.
#:
#: "One register throughout" means within a take: on the conceding take, read
#: even the insisting paragraph in a conceding way. That is what makes one take
#: differ from another as a whole, rather than all four converging on whatever
#: each paragraph invites.
#:
#: About 150 words, which is roughly 75 seconds at a normal teaching pace --
#: comfortably past the 20-second floor, and long enough for the speaker
#: encoder to hear a delivery rather than a phrase. Phonetically it is ordinary
#: academic English on purpose: no tongue-twisters, because a stumble is a
#: worse artefact than a dull sentence.
PASSAGE = """\
Confounding is not a statistical problem. It is a question about how people
came to be in the groups you are comparing, and no amount of adjustment will
answer it if the thing that sorted them was never measured.

That is the part I want to insist on. You can control for age, for sex, for
stage at diagnosis, and still be looking at a difference that was decided
before the study began.

Now, I will grant that this can be overstated. Plenty of well-adjusted
observational studies have held up perfectly well against the trials that came
later, and pretending otherwise is its own kind of dishonesty.

So what would actually change our minds here? What would a study have to show,
for us to say the residual confounding is small enough to live with? I do not
think we have a good answer yet.
"""


@dataclass
class Register:
    name: str
    path: str = ""
    seconds: float = 0.0
    lufs_in: float = 0.0        # as recorded, before the shared correction
    gain_db: float = 0.0        # the one correction applied to the whole set
    session: str = ""           # which sitting this came from
    note: str = ""
    #: Measured from the treated file, so a set can be told whether it really
    #: contains four deliveries or one delivery recorded four times.
    pitch_hz: float = 0.0
    pitch_range_st: float = 0.0
    rate_wps: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def _loudness(path: Path) -> float:
    """Integrated loudness in LUFS, measured not guessed."""
    import re, subprocess
    proc = subprocess.run(
        [str(require("ffmpeg")), "-hide_banner", "-nostats", "-i", str(path),
         "-filter:a", "loudnorm=print_format=json", "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    blob = proc.stderr
    m = re.search(r'"input_i"\s*:\s*"(-?\d+(?:\.\d+)?)"', blob)
    if m:
        return float(m.group(1))
    raise ValueError(f"could not measure the loudness of {Path(path).name}")


def prepare(takes: dict, outdir: Path, *, session: str = "",
            log=print) -> list[Register]:
    """
    Treat a whole set of register recordings together.

    `takes` maps register name -> the raw recording. Every file gets the SAME
    gain correction, chosen so the set as a whole sits at the target: the
    neutral take is the anchor, because it is the one every difference is
    measured against and the one the ordinary lecture voice comes from.

    Returns the registers, in the order given, with what was measured.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    session = session or time.strftime("%Y%m%d-%H%M%S")

    unknown = [k for k in takes if k not in REGISTERS]
    if unknown:
        raise ValueError(
            f"Not a register: {', '.join(unknown)}. "
            f"The registers are: {', '.join(REGISTERS)}.")
    if NEUTRAL not in takes:
        raise ValueError(
            f"The '{NEUTRAL}' take is required: it is the baseline every other "
            f"register is measured against, and it has to come from the same "
            f"sitting as the rest.")

    # CONVERT FIRST, then measure.
    #
    # `to_reference_wav` carries the guard that matters here: it compares the
    # source duration with the converted one and refuses a silent truncation.
    # iPhone Voice Memos can write ALAC, which a minimal ffmpeg build decodes
    # part of and then gives up on, exiting 0 -- this project has already been
    # caught by that once, on a 60-second take that arrived as a few seconds.
    # Measuring the raw file first would have taken the loudness of a fragment
    # and put it into the gain for the whole SET.
    from realme.core.media import to_reference_wav
    staging = outdir / "_raw"
    staging.mkdir(parents=True, exist_ok=True)
    converted, measured = {}, {}
    for name, src in takes.items():
        src = Path(src)
        if not src.is_file():
            raise ValueError(f"No such recording: {src}")
        wav = to_reference_wav(src, staging / f"{name}.wav")
        converted[name] = wav
        measured[name] = _loudness(wav)
        log(f"  {name:<12} {duration_of(wav):5.1f}s  "
            f"{measured[name]:6.1f} LUFS as recorded")

    # ONE correction for the set, anchored on the neutral take. Levelling each
    # file to the target separately would make the pressing and conceding
    # takes equally loud, which is most of what tells them apart.
    gain = TARGET_LUFS - measured[NEUTRAL]
    log(f"  one correction of {gain:+.1f} dB for the whole set, from the "
        f"'{NEUTRAL}' take")

    out = []
    for name, src in takes.items():
        dst = outdir / f"{name}.wav"
        run([require("ffmpeg"), "-y", "-loglevel", "error",
             "-i", str(converted[name]),
             "-filter:a", f"volume={gain:.2f}dB,alimiter=limit=0.97",
             "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", str(dst)],
            f"level {name}")
        try:
            from realme.expressive.prosody import measure as _measure
            pr = _measure(dst, words=len(PASSAGE.split()))
        except Exception:
            pr = None
        secs = float(duration_of(dst))
        if secs < MIN_SECONDS:
            raise ValueError(
                f"The '{name}' take is only {secs:.1f}s. A register needs at "
                f"least {MIN_SECONDS:.0f} seconds: the speaker encoder is "
                f"describing a delivery, and a few words do not contain one.")
        r = Register(name=name, path=str(dst),
                     seconds=round(float(duration_of(dst)), 2),
                     lufs_in=round(measured[name], 1),
                     gain_db=round(gain, 2), session=session)
        if pr is not None and pr.ok:
            r.pitch_hz = round(pr.f0_median_hz, 1)
            r.pitch_range_st = round(pr.f0_range_st, 2)
            r.rate_wps = round(pr.rate_wps, 2)
        out.append(r)

    # Distinctness is judged AFTER the whole set exists, because every axis is
    # measured against the neutral take and that take is one of the set.
    base = next((r for r in out if r.name == NEUTRAL), None)
    for r in out:
        if base is None or r.name == NEUTRAL:
            continue
        moved = _differences(r, base)
        if not moved:
            r.note = (f"indistinguishable from '{NEUTRAL}': same loudness, "
                      f"same pitch, same pace. Read again, differently.")
            log(f"  ! {r.name}: {r.note}")
        else:
            log(f"  {r.name:<12} differs by {', '.join(moved)}")
    (outdir / "registers.json").write_text(
        json.dumps([r.as_dict() for r in out], indent=2), encoding="utf-8")
    return out


def _differences(r, base) -> list:
    """Which axes actually moved, relative to the neutral take."""
    import math
    moved = []
    lu = r.lufs_in - base.lufs_in
    if abs(lu) >= MEANINGFUL_LU:
        moved.append(f"{lu:+.1f} LU")
    if r.pitch_hz > 0 and base.pitch_hz > 0:
        st = 12 * math.log2(r.pitch_hz / base.pitch_hz)
        if abs(st) >= MEANINGFUL_ST:
            moved.append(f"{st:+.1f} semitones")
    if r.rate_wps > 0 and base.rate_wps > 0:
        frac = (r.rate_wps - base.rate_wps) / base.rate_wps
        if abs(frac) >= MEANINGFUL_RATE:
            moved.append(f"{frac * 100:+.0f}% pace")
    return moved


def report(regs: list) -> str:
    """What was recorded, and whether the takes are actually different."""
    if not regs:
        return "no registers enrolled"
    base = next((r for r in regs if r.name == NEUTRAL), regs[0])
    lines = [f"{'register':<12} {'length':>7} {'loud':>8} {'pitch':>8} "
             f"{'range':>7} {'pace':>7}   vs " + NEUTRAL,
             "-" * 78]
    for r in regs:
        rel = r.lufs_in - base.lufs_in
        moved = "baseline" if r.name == NEUTRAL else (
            ", ".join(_differences(r, base)) or "NO DIFFERENCE")
        lines.append(f"{r.name:<12} {r.seconds:6.1f}s {rel:+7.1f}LU "
                     f"{r.pitch_hz:7.0f}Hz {r.pitch_range_st:6.1f}st "
                     f"{r.rate_wps:6.2f}w/s   {moved}")
    flat = [r.name for r in regs if r.name != NEUTRAL and not _differences(r, base)]
    if flat:
        lines += ["",
                  f"! {', '.join(flat)} came out the same as '{NEUTRAL}' on "
                  f"every axis measured.",
                  "  Four takes that sound alike carry no direction: the whole "
                  "method is the",
                  "  DIFFERENCE between them. Record those again, further apart."]
    sessions = {r.session for r in regs}
    if len(sessions) > 1:
        lines += ["",
                  "! These came from more than one sitting. The difference "
                  "between two registers",
                  "  then contains the room and the microphone as well as the "
                  "delivery. Re-record",
                  "  them together before using them as an emotion direction."]
    return "\n".join(lines)
