"""
Run the same exchange under different conditions and report what changed.

The design is paired: identical words, identical voices, identical seed
conditions where the engine allows it -- only the condition differs. The
engine samples, so two renders of one sentence already differ, and a small
effect can only be attributed in a paired design.

Repeats are required. A codec language model at temperature 0.9 produces a
different draw every time. `repeat` renders the whole exchange N times per
condition and the report carries the spread, so an effect smaller than the
noise shows as such.

The output is two things:

  a table of measurements
  a pair of audio files with a hidden key, for a blind listening comparison

The listening comparison is the primary result; every number here is a proxy
for whether a listener hears two people talking.
"""
from __future__ import annotations
import json, random
from dataclasses import dataclass
from pathlib import Path

from realme.expressive import sample as SAMPLE
from realme.expressive.contrast import dynamics
from realme.expressive.prosody import measure
from realme.expressive.voices import StancePaced

#: Silence inserted between turns when the exchange is joined, in seconds.
#: A real dialogue render has its own gap logic; this is only so the listening
#: copy does not run the turns together.
TURN_GAP_S = 0.35


@dataclass
class Condition:
    name: str
    strength: float          # 0.0 is the control: the unwrapped adapter
    note: str = ""
    #: How the voices are wrapped for this condition, as wrap(adapter,
    #: strength) -> adapter. None means stance pacing.
    #:
    #: A callable rather than a flag, so conditions can compare DIFFERENT
    #: KINDS of control (pacing, register switching, embedding shift)
    #: without a flag per kind.
    wrap: object = None


def preflight(adapters: dict, log=print) -> "bool | str":
    """True when every voice can run, or a message saying which cannot and why.

    Returns rather than raises so the caller decides; the runner turns it
    into a clean exit with an instruction rather than a traceback.
    """
    from realme.adapters.base import AdapterUnavailable
    broken = []
    for role, tts in adapters.items():
        check = getattr(tts, "preflight", None)
        if check is None:
            continue
        try:
            check()
            log(f"  {role}: {getattr(tts, 'name', tts)} ready")
        except AdapterUnavailable as e:
            broken.append(f"{role} voice ({getattr(tts, 'name', tts)}): {e}")
        except Exception as e:                  # a broken engine, not a missing one
            broken.append(f"{role} voice ({getattr(tts, 'name', tts)}): "
                          f"{type(e).__name__}: {e}")
    if broken:
        return ("Not every voice can run, so nothing was rendered:\n\n  "
                + "\n\n  ".join(broken))
    return True


def _render_turns(adapters, turns, outdir: Path, strength: float,
                  log=print, wrap=None) -> list[tuple[str, Path, int]]:
    """Render one pass of the exchange. Returns [(speaker, wav, words)]."""
    outdir.mkdir(parents=True, exist_ok=True)
    make = wrap or (lambda a, st: StancePaced(a, enabled=st > 0, strength=st))
    wrapped = {k: make(a, strength) for k, a in adapters.items()}
    out = []
    for i, (speaker, stance, text) in enumerate(turns):
        tts = wrapped.get(speaker) or next(iter(wrapped.values()))
        setter = getattr(tts, "set_stance", None)
        if setter is None:
            # The unwrapped control arm. Duck-typed rather than assumed,
            # because a condition may hand back the bare adapter.
            st, known = (stance or "neutral"), True
        else:
            st, known = setter(stance)
        if not known:
            log(f"    turn {i + 1}: stance {stance!r} is not in the vocabulary; "
                f"treated as neutral and left at the measured pace")
        wav = outdir / f"turn_{i + 1:02d}_{speaker}_{st}.wav"
        tts.synthesize(text, wav)
        out.append((speaker, wav, SAMPLE.words(text)))
    return out


def _join(wavs: list[Path], out: Path) -> Path:
    """Concatenate renders into one listenable file, resampling if needed.

    The two dialogue voices need not agree on sample rate -- piper is 22.05 kHz
    and the cloned engine 24 kHz -- and concatenating those without conversion
    plays one speaker at the wrong speed, which would be heard as a prosody
    difference.
    """
    import numpy as np
    from realme.verify.acoustic import read_wav
    from realme.expressive.wavio import write_wav, resample
    pieces, rate = [], None
    for w in wavs:
        try:
            y, sr = read_wav(Path(w))
        except Exception:
            continue
        if rate is None:
            rate = sr
        elif sr != rate:
            y = resample(y, sr, rate)
        pieces.append(np.asarray(y, dtype=np.float64))
        pieces.append(np.zeros(int((rate or 22050) * TURN_GAP_S)))
    return write_wav(out, np.concatenate(pieces) if pieces else np.zeros(1),
                     rate or 22050)


def run(adapters: dict, outdir: Path, *, conditions=None, repeat: int = 3,
        turns=None, log=print, seed: int | None = None) -> dict:
    """
    Render the exchange under each condition, `repeat` times, and measure.

    `adapters` is {"instructor": tts, "guest": tts} -- the same shape
    `casting.cast()` returns, so the real dialogue voices can be handed
    straight in without this module knowing how they were built.
    """
    outdir = Path(outdir)
    turns = turns or SAMPLE.EXCHANGE
    conditions = conditions or [
        Condition("flat", 0.0, "the control: adapters exactly as they are"),
        Condition("stance", 1.0, "pace multipliers from the turn's stance"),
    ]
    # Check every voice BEFORE rendering anything. Loading the cloned engine
    # and rendering one speaker's turns takes minutes, so a voice that cannot
    # start should fail the run at once.
    ready = preflight(adapters, log=log)
    if ready is not True:
        raise RuntimeError(ready)

    # Check the arms are actually different arms. `casting.cast` may already
    # have wrapped a voice (e.g. in a register switcher), which can make two
    # conditions identical.
    #
    # The render ledger's key settles it: `voice_fingerprint()` is defined as
    # "everything that changes the audio", so two conditions that fingerprint
    # alike ARE alike. Asked at a stance that exercises the difference, since
    # every wrapper is transparent at neutral.
    distinct = distinctness(adapters, conditions)
    if distinct is not True:
        raise RuntimeError(distinct)

    results = {}
    for cond in conditions:
        runs = []
        for r in range(repeat):
            log(f"  {cond.name}, pass {r + 1} of {repeat}")
            rendered = _render_turns(adapters, turns,
                                     outdir / cond.name / f"pass{r + 1}",
                                     cond.strength, log=log, wrap=cond.wrap)
            measured = [(spk, measure(w, words=n)) for spk, w, n in rendered]
            d = dynamics(measured)
            runs.append(d)
            log(f"    {d.summary()}")
            if r == 0:
                _join([w for _, w, _ in rendered],
                      outdir / f"{cond.name}_exchange.wav")
        results[cond.name] = {"condition": cond.name,
                              "strength": cond.strength,
                              "note": cond.note,
                              "runs": [d.as_dict() for d in runs]}
    key = _blind(outdir, [c.name for c in conditions], seed=seed, log=log)
    (outdir / "measurements.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return {"conditions": results, "listening": key,
            "report": report(results)}


def distinctness(adapters: dict, conditions, *, stance: str = "press"):
    """True when every condition wraps the voice differently, else a message.

    Compares `voice_fingerprint()` across the conditions, which is the string
    the render ledger uses to decide whether two renders are the same render.
    Two arms with the same fingerprint produce the same audio.
    """
    seen, who = {}, {}
    for cond in conditions:
        make = cond.wrap or (lambda a, st: StancePaced(a, enabled=st > 0,
                                                       strength=st))
        try:
            w = make(next(iter(adapters.values())), cond.strength)
        except Exception as e:
            return (f"the {cond.name!r} arm could not be built: {e}")
        setter = getattr(w, "set_stance", None)
        if setter:
            setter(stance)
        try:
            fp = str(w.voice_fingerprint())
        except Exception as e:
            return f"the {cond.name!r} arm has no usable fingerprint: {e}"
        if fp in seen:
            return (f"the {cond.name!r} and {seen[fp]!r} arms are the same "
                    f"condition: both fingerprint as\n\n    {fp}\n\n"
                    f"Nothing was rendered. A comparison between two copies "
                    f"of one thing has an\nanswer before it starts.")
        seen[fp], who[cond.name] = cond.name, fp
    return True


def _blind(outdir: Path, names: list[str], *, seed=None, log=print) -> dict:
    """Copy the joined exchanges to neutral names, with the key written aside.

    Knowing which file is the treatment biases the listener. The key goes in
    a separate file so it can be read after listening.
    """
    import shutil
    rng = random.Random(seed)
    labels = [chr(ord("A") + i) for i in range(len(names))]
    shuffled = names[:]
    rng.shuffle(shuffled)
    mapping = {}
    for label, name in zip(labels, shuffled):
        src = outdir / f"{name}_exchange.wav"
        if src.is_file():
            shutil.copy(src, outdir / f"listen_{label}.wav")
            mapping[label] = name
    (outdir / "listening_key.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    log(f"  listening copies: {', '.join('listen_' + l + '.wav' for l in mapping)} "
        f"(the key is in listening_key.json -- listen first)")
    return mapping


def report(results: dict) -> str:
    """A table with the spread across repeats, not a single number."""
    lines = []
    head = f"{'condition':10} {'speaker':11} {'within st':>10} {'between st':>11} {'sep':>6} {'rate %':>8}"
    lines.append(head)
    lines.append("-" * len(head))
    for name, data in results.items():
        per_speaker: dict[str, list] = {}
        for run_d in data["runs"]:
            for s in run_d["speakers"]:
                per_speaker.setdefault(s["speaker"], []).append(s)
        for speaker, rows in sorted(per_speaker.items()):
            def band(key):
                vals = [r[key] for r in rows]
                lo, hi = min(vals), max(vals)
                mean = sum(vals) / len(vals)
                return f"{mean:.2f}", f"{hi - lo:.2f}"
            w, wr = band("within_pitch_st")
            b, br = band("between_pitch_st")
            sep, _ = band("separation")
            rt, _ = band("between_rate_pct")
            lines.append(f"{name:10} {speaker:11} {w:>7}±{wr:<2} {b:>8}±{br:<2} "
                         f"{sep:>6} {rt:>8}")
    lines.append("")
    lines.append("within  = pitch movement INSIDE a turn. Instability; lower is better.")
    lines.append("between = pitch movement ACROSS that speaker's turns. Intent; higher is")
    lines.append("          what expressive control is trying to buy.")
    lines.append("sep     = between / within. A screening ratio, not a verdict.")
    lines.append("±       = the spread across repeats. An effect smaller than this is")
    lines.append("          not an effect yet.")
    return "\n".join(lines)
