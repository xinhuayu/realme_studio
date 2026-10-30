"""
Building a voice adapter. **One place. Only this one.**

Some engines need several inputs to be usable: Qwen3's preflight requires both
the reference clip and its transcript. A caller that builds an adapter itself
and omits one gets "qwen3 unavailable" even when the engine is installed and
working.

So nothing constructs a TTS adapter directly. Every caller (the Studio's
render path, `/api/doctor`, `realme bench`, ...) comes through here, and
inherits every input the engines need whether or not that caller knew to ask.
"""
from __future__ import annotations
from pathlib import Path

from realme.adapters.base import AdapterUnavailable


def voice_inputs(profile=None) -> dict:
    """
    Everything known about the user's voice, from the profile.

    Falls back to the environment for the transcript, since an older setup may
    have `REALME_QWEN3_REF_TEXT` set rather than a transcript in the profile.
    """
    import os
    if profile is None:
        from realme.core.env import data_home
        from realme.app.profile import Profile
        profile = Profile(data_home() / "profile")
    ref = profile.data.get("voice_reference")
    text = ""
    try:
        text = profile.reference_text
    except AttributeError:
        text = (profile.data.get("voice_reference_text") or "").strip()
    if not text:
        text = os.environ.get("REALME_QWEN3_REF_TEXT", "").strip()
    return {"reference_wav": Path(ref) if ref else None,
            "reference_text": text, "profile": profile}


def build(name: str, *, profile=None, reference_wav: Path | None = None,
          reference_text: str | None = None, quiet: bool = False, **overrides):
    """
    Construct the named adapter with everything it can use.

    Explicit arguments win over the profile, so a caller that genuinely means
    "this clip, not the enrolled one" still can -- `realme bench --reference`,
    for instance.
    """
    from realme.adapters.tts import REGISTRY
    cls = REGISTRY.get(name)
    if cls is None:
        raise AdapterUnavailable(
            f"Unknown engine '{name}'. Known: {', '.join(sorted(REGISTRY))}")

    inputs = voice_inputs(profile)
    ref = reference_wav if reference_wav is not None else inputs["reference_wav"]
    text = reference_text if reference_text is not None else inputs["reference_text"]

    # Convert here, once, for every engine. Unconverted, each engine fails in
    # its own way -- "Format not recognised", "Not a RIFF file" -- and neither
    # message names the real problem. Phones record .m4a, so this is the
    # normal input.
    if ref is not None and Path(ref).is_file():
        try:
            from realme.core.media import ensure_reference_wav
            ref = ensure_reference_wav(Path(ref))
        except Exception:
            pass          # a conversion failure must not hide the real error

    # Adapters differ in what they accept, and that is fine -- a draft voice has
    # no reference to clone from. Ask the constructor what it takes and pass
    # exactly that.
    #
    # Falling back on TypeError alone is unsafe: an unrelated keyword the
    # adapter does not take (e.g. device="cuda" for the C++ adapter) would
    # make every attempt fail until one with NO reference succeeded, and the
    # engine would speak in an unconditioned default voice. Introspection
    # drops the reference only when THAT adapter does not take it -- never as
    # collateral from some other argument.
    import inspect
    # An engine that says it does not clone is not offered the clip at all,
    # so no warning is printed about an engine that never uses it. The
    # warning still fires for an engine that CLAIMS to clone and then refuses
    # the reference.
    if not getattr(cls, "is_voice_clone", False):
        ref, text = None, None
    offered = {"reference_wav": ref, "reference_text": text, **overrides}

    # Half the registry entries are lazy wrappers -- a class whose __new__
    # imports the real adapter and returns it, so the engine is only imported
    # if it is used. Their own signature is `(*args, **kwargs)`, which tells us
    # nothing and, read naively, says "accepts everything" (Qwen3DllTTS, for
    # one, does not take reference_text).
    #
    # So introspection is used where it is informative, and where it is not
    # there is a ladder -- with the rule that REFERENCE_WAV IS IN EVERY RUNG.
    # Narrowing happens only among the optional extras.
    try:
        params = inspect.signature(cls.__init__).parameters
        informative = not (len(params) <= 3 and
                           any(q.kind is q.VAR_KEYWORD for q in params.values()))
        accepted = set(params)
    except (TypeError, ValueError):
        informative, accepted = False, set()

    if informative:
        attempts = [{k: v for k, v in offered.items() if k in accepted}]
    else:
        extras = {k: v for k, v in offered.items()
                  if k not in ("reference_wav", "reference_text")}
        attempts = [
            offered,                                        # everything
            {"reference_wav": ref, **extras},               # no transcript
            {"reference_wav": ref, "reference_text": text},  # no overrides
            {"reference_wav": ref},                         # the clip alone
        ]

    built, last = None, None
    for kwargs in attempts:
        try:
            built = cls(**kwargs)
            break
        except TypeError as e:
            last = e
    if built is None:
        # An engine that will not take the reference at all. Build it without
        # the clip rather than fail the render, and always say so.
        import sys as _sys
        if ref is not None and not quiet:
            print(f"  [{name}] could not pass the reference clip to this "
                  f"engine ({last}) -- trying without it", file=_sys.stderr,
                  flush=True)
        bare = {k: v for k, v in offered.items()
                if k not in ("reference_wav", "reference_text")}
        for kwargs in (bare, {}):
            try:
                built = cls(**kwargs)
                break
            except TypeError as e:
                last = e
        if built is None:
            raise AdapterUnavailable(
                f"Could not construct '{name}': {last}") from last

    # Say whose voice this is. Here, once, for every caller -- the lecture,
    # the podcast, the bench, `narrate`, the Studio's render queue -- so they
    # all report it the same way.
    #
    # `is_voice_clone` rather than "did it take the argument": espeak and the
    # other placeholders inherit a constructor that ACCEPTS reference_wav and
    # ignores it, so asking about the signature would have called that a clone.
    if not quiet:
        announce(name, built, ref)
    return built


def announce(name: str, adapter, ref) -> None:
    """One line on stderr saying which voice is about to be used."""
    import sys as _sys
    # An engine whose clone does not live in a local file has to answer this
    # itself. The reasoning below reads `reference_wav` and would otherwise
    # say "no reference enrolled" about a hosted voice that is enrolled, or
    # "cloning from voice_reference.wav" about a hosted voice that is not.
    own = getattr(adapter, "voice_note", None)
    if callable(own):
        try:
            print(f"  [{name}] {own()}", file=_sys.stderr, flush=True)
            return
        except Exception:
            pass          # a broken note must not stop a render
    clones = getattr(adapter, "is_voice_clone", False)
    got = getattr(adapter, "reference_wav", None)
    if clones and got:
        msg = f"cloning from {Path(got).name}"
    elif clones and ref is not None:
        msg = ("WARNING: this engine clones, but the reference never reached "
               "it -- the voice will not be yours")
    elif clones:
        msg = ("WARNING: no reference enrolled -- a default speaker, drifting "
               "between utterances. Fix with: realme voice enroll <file>")
    elif ref is not None:
        msg = "its own voice (this engine does not clone); your clip is unused"
    else:
        msg = "its own voice (this engine does not clone)"
    print(f"  [{name}] {msg}", file=_sys.stderr, flush=True)
