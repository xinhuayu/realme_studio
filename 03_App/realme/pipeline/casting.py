"""
Who speaks, in which voice, at what pace. **One place. Only this one.**

The CLI grew `--speed`, `--guest-voice` and a female-guest default; the Studio
went on calling `build(name)` twice with nothing, so a dialogue generated from
the web page ran at the lecture pace with two identical voices. The settings
existed and the web page could not reach them.

This is the same failure the adapter factory was written to end -- two callers,
two ideas of what a dialogue needs -- one level up. Casting a dialogue is a
decision, not a pair of constructor calls, so it lives in a function both
callers use.

A voice is named `engine` or `engine:voice`, e.g. `piper:en_US-lessac-medium`.
The suffix is a piper voice file and applies only to piper, because `model`
means the checkpoint to Qwen3 and handing it a piper voice name would send it
looking for a TTS model called "en_US-lessac-medium".
"""
from __future__ import annotations

#: Conversation reads faster than narration at the same rate, so a dialogue
#: runs a notch below the draft pace. Judgment, not measurement -- `--speed`
#: and the Studio's control both override it, and if a measured number ever
#: replaces this it should replace it here.
DIALOGUE_SPEED = 0.72

PIPER_ENGINES = ("piper",)


def split_spec(spec: str) -> tuple[str, str | None]:
    """`piper:en_US-lessac-medium` -> ('piper', 'en_US-lessac-medium')."""
    engine, _, voice = (spec or "").partition(":")
    return engine.strip() or "piper", (voice.strip() or None)


def other_draft_voice(voice: str | None = None) -> str:
    """
    The stock draft voice that is NOT the one given.

    Derived from the voice list rather than hardcoded to a female name, so it
    still gives two different speakers if the default ever changes.
    """
    from realme.adapters.tts import DRAFT_VOICES, PiperTTS, draft_voice
    current = voice or draft_voice(None)
    others = [v for v in DRAFT_VOICES if v != current]
    return others[0] if others else PiperTTS.DEFAULT_VOICE


def voice_overrides(engine: str, voice: str | None, speed: float | None) -> dict:
    out: dict = {}
    if engine in PIPER_ENGINES:
        if voice:
            out["model"] = voice
        if speed:
            out["speed"] = float(speed)
    return out


def cast(instructor: str, guest: str | None = None, *, speed: float | None = None,
         profile=None, build=None, log=None, registers: bool = True) -> dict:
    """
    Build the two adapters a dialogue needs.

    The guest defaults to the DRAFT voice, never to the instructor's engine:
    with a cloning instructor, `guest or instructor` produced a debate between
    two copies of the same person.

    `registers=False` returns the instructor UNWRAPPED. An experiment that
    compares ways of controlling the voice has to start from a voice that is
    not already controlled -- the first `--compare` run asked for a "plain"
    arm, got a register-switching one, and then crashed trying to shift it,
    which was the honest outcome of a silently non-plain control.
    """
    if build is None:
        from realme.adapters.factory import build as build
    host_engine, host_voice = split_spec(instructor)
    guest_engine, guest_voice = split_spec(guest or "piper")
    if guest_voice is None and guest_engine in PIPER_ENGINES:
        # Contrast with whatever the instructor is using, so the two speakers
        # are never the same voice.
        guest_voice = other_draft_voice(
            host_voice if host_engine in PIPER_ENGINES else None)

    if speed is None:
        # This is where a measured pace belongs: two voices alternating, where
        # one rushing the other is audible. Asked for THIS guest voice, because
        # the ratio is yours divided by that voice's, and voices differ.
        speed = guest_speed(guest_voice, profile, log=log)

    kw = {"profile": profile} if profile is not None else {}
    cast_out = {
        "instructor": build(host_engine, **kw,
                            **voice_overrides(host_engine, host_voice, speed)),
        "guest": build(guest_engine, **kw,
                       **voice_overrides(guest_engine, guest_voice, speed)),
    }
    if registers:
        cast_out["instructor"] = with_registers(cast_out["instructor"], profile,
                                                log=log)
    return cast_out


def guest_speed(voice: str | None, profile=None, *, log=None,
                measure: bool = True) -> float:
    """
    How fast the guest speaks, measured for THIS voice, measuring it if needed.

    The constant below was the fallback for a voice nobody had measured, and it
    is the worst of the three possible answers. `measured_speed` is careful:
    handed a voice it has no number for, it returns None rather than a ratio
    belonging to a different speaker. Casting then applied 0.72 anyway -- a
    figure never measured against anything -- and 0.72 happened to be SLOWER
    than the ratio measured against Ryan, applied to Amy, who is slower than
    Ryan at speed 1.0 to begin with. Two penalties, neither of them measured.
    That is the guest who drags.

    So when the number is missing, get the number. Piper runs at about RTF
    0.08 and the calibration is sixty words, so measuring a voice costs a few
    seconds, once ever -- it is written back to the profile and the next cast
    reads it. The constant survives only for the case where there is nothing
    to measure from: no enrolment recording, no transcript, no piper.
    """
    from realme.adapters.tts import measured_speed, speed_from
    # The profile in hand first, the one on disk only when there is none.
    # Asking the disk while holding an updated profile is how the same voice
    # got measured twice.
    got = (speed_from(profile.data, voice) if profile is not None else None)
    if got is None:
        got = measured_speed(voice)
    if got is not None:
        return got
    if not (measure and voice and profile is not None):
        return DIALOGUE_SPEED
    try:
        from pathlib import Path as _P
        from realme.core.env import data_home
        from realme.enrollment.pace import calibrate
        ref = profile.data.get("voice_reference")
        text = getattr(profile, "reference_text", None)
        if not ref or not _P(ref).is_file() or not text:
            raise ValueError("no enrolment recording and transcript to measure "
                             "your own rate from")
        if log:
            log(f"  {voice} has never been paced against you; measuring "
                f"(a few seconds, once)")
        r = calibrate(ref, text, voice=voice, workdir=data_home() / "voice",
                      log=(lambda *_: None))
        # Stored in the durable shape -- your rate once, each voice's beside
        # it -- so a later voice change is arithmetic rather than another
        # render, and so this never runs twice for the same voice.
        from realme.adapters.tts import rates as _rates
        mine, table = _rates(profile.data)
        table[r["voice"]] = r["voice_wpm"]
        profile.update({"your_wpm": mine or r["your_wpm"],
                        "draft_voice_wpm": table})
        if log:
            log(f"    you {r['your_wpm']:.1f} w/min, {r['voice']} "
                f"{r['voice_wpm']:.1f} w/min  ->  speed {r['speed']:.3f}")
        return float(r["speed"])
    except Exception as e:
        # Never fail a render over a pace measurement, but never do it
        # silently either: a guest at the fallback is a guest that will be
        # complained about, and the complaint should arrive with its cause.
        if log:
            log(f"  could not measure {voice} against your rate ({e}); "
                f"falling back to {DIALOGUE_SPEED:g} — run "
                f"`realme voice pace --all` to fix this properly")
        return DIALOGUE_SPEED


def with_registers(adapter, profile, *, log=None):
    """Wrap a cloned voice so each turn is spoken in its matching register.

    Only when registers have actually been enrolled, and only for a voice that
    clones -- a stock piper voice has no reference to swap. Without them this
    returns the adapter untouched, so a dialogue behaves exactly as it did
    before anyone recorded anything.
    """
    from pathlib import Path as _P
    if profile is None or not getattr(adapter, "is_voice_clone", False):
        return adapter
    try:
        regs = dict(profile.data.get("voice_registers") or {})
    except Exception:
        return adapter
    regs = {k: v for k, v in regs.items() if v and _P(v).is_file()}
    if len(regs) < 2:
        return adapter          # one register is just the reference again
    if not getattr(adapter, "accepts_reference_per_call", False):
        # The wrapper works by passing a different reference to each call.
        # An engine that ignores `voice=` would take the wrapper, carry a
        # fingerprint saying "reg=pressing", and render every turn the same.
        if log:
            log(f"  registers enrolled, but the {getattr(adapter, 'name', '?')} "
                f"engine cannot switch its reference per turn; speaking "
                f"everything in the neutral voice")
        return adapter
    try:
        from realme.expressive.voices import StanceVoiced
    except ImportError:
        return adapter          # the exploratory package is optional
    if log:
        log(f"  registers: {', '.join(sorted(regs))}")
    return StanceVoiced(adapter, regs, log=log)


def with_expression(adapter, profile, *, alpha=None, alphas=None,
                    renorm: bool = True, log=None):
    """Wrap a cloned voice so each turn is shifted along its emotion direction.

    The continuous alternative to `with_registers`, and deliberately NOT
    combined with it: alpha = 1 along a direction is the register itself, so
    the two are the same control at different resolutions, and stacking them
    would encode a register and then discard it.

    Returns the adapter untouched whenever anything is missing -- no
    directions extracted, a voice that does not clone, an engine that takes a
    reference file rather than a vector -- so a dialogue behaves exactly as it
    did before anyone ran `realme voice tau`.
    """
    from pathlib import Path as _P
    if profile is None or not getattr(adapter, "is_voice_clone", False):
        return adapter
    if not hasattr(adapter, "set_expression"):
        if log:
            log("  emotion directions need the in-process qwen3 engine; "
                "rendering in the plain voice")
        return adapter
    try:
        from realme.core.env import data_home
        from realme.expressive import tau as TAU
        from realme.expressive.voices import ExpressionShifted
    except ImportError:
        return adapter          # the exploratory package is optional
    store = data_home() / "voice" / "registers" / "tau.json"
    if not _P(store).is_file():
        return adapter
    try:
        ts = TAU.load(store)
        TAU.check_engine(ts, adapter)
    except Exception as e:
        # Loudly. A direction set that cannot be used is worth one line; a
        # dialogue that quietly renders flat when the user asked for
        # expression is the kind of null result that gets believed.
        if log:
            log(f"  emotion directions not used: {e}")
        return adapter
    if not ts.directions:
        return adapter
    w = ExpressionShifted(adapter, ts, alpha=alpha, alphas=alphas,
                          renorm=renorm, log=log)
    if log:
        log("  expression: " + ", ".join(
            f"{d} at {w.alpha_for(d):+.2f}" for d in sorted(ts.directions)))
    return w


def speaker_name(spec: str, adapter=None, *, fallback: str = "my guest") -> str:
    """
    What the host calls this voice out loud.

    A podcast introduces its co-host by name, and the name a synthetic
    interlocutor has is its voice: Ryan, Lessac. The script writer labels the
    second speaker "AI Interlocutor", which is a slot rather than a name and
    reads as a stage direction when spoken.

    Taken from the curated voice labels rather than parsed out of the file
    name, so "Ryan - male, US" yields "Ryan" and a voice added later with a
    proper label needs no code change. The file name is the fallback for a
    hand-installed voice that is not in the list.
    """
    import re
    engine, voice = split_spec(spec)
    if engine in PIPER_ENGINES:
        from realme.adapters.tts import DRAFT_VOICES, draft_voice
        v = voice or getattr(adapter, "model", None) or draft_voice(None)
        label = DRAFT_VOICES.get(v)
        if label:
            return re.split(r"\s*[—-]\s+", label)[0].strip()
        parts = str(v).split("-")
        if len(parts) >= 2 and parts[1]:
            return parts[1].capitalize()
    return fallback


def describe(spec: str, adapter) -> str:
    """One line naming the voice, for the log and the web page alike."""
    engine, _ = split_spec(spec)
    model = getattr(adapter, "model", None)
    clip = getattr(adapter, "reference_wav", None)
    if clip:
        from pathlib import Path
        return f"{engine} cloning {Path(clip).name}"
    return f"{engine}{f' ({model})' if model else ''}"
