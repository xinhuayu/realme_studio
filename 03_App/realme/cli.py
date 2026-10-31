"""RealMe CLI. Every command either produces a verified artifact or exits nonzero."""
from __future__ import annotations
import argparse
import os, sys, json
from pathlib import Path
from realme.core.textio import read_text as _TXT


def _tts(name: str | None, args, **extra):
    """
    Every CLI command that speaks gets its adapter from here, through the one
    factory the Studio and the benchmark also use (realme/adapters/factory.py),
    so each engine gets its reference clip and transcript.
    """
    name = name or _profile_tts()
    from realme.adapters.factory import build
    from realme.adapters.base import AdapterUnavailable
    overrides = {}
    ref = getattr(args, "reference_wav", None)
    device = getattr(args, "device", None)
    if device:
        overrides["device"] = device
    overrides.update({k: v for k, v in extra.items() if v is not None})
    try:
        tts = build(name, reference_wav=ref, **overrides)
    except AdapterUnavailable as e:
        sys.exit(str(e))
    # The speed saved for this voice in the Studio, so the command line and
    # the page sound the same.
    try:
        from realme.core.env import data_home
        from realme.app.profile import Profile
        from realme.pipeline.casting import with_voice_speed
        tts = with_voice_speed(tts, name,
                               Profile(data_home() / "profile").data)
    except Exception:
        pass
    # The spending cap: --max-spend on the command, else REALME_MAX_SPEND_USD.
    # Applied here because every command that speaks gets its engine here.
    from realme.pipeline.speak import spend_capped
    try:
        return spend_capped(tts, getattr(args, "max_spend", None))
    except ValueError as e:
        sys.exit(str(e))


def _profile_tts() -> str:
    """The engine the profile selects, for commands that did not name one.

    Falls back to piper when there is no profile or it names no engine: a
    real voice, and audibly not the instructor's.
    """
    try:
        from realme.core.env import data_home
        from realme.app.profile import Profile
        return (Profile(data_home() / "profile").data["adapters"].get("tts")
                or "piper")
    except Exception:
        return "piper"


def _pause_scale() -> float:
    """The profile's pause scale, or 1.0. Read here so `--pause-scale` and
    the Studio's render both end up asking the same stored number."""
    try:
        from realme.core.env import data_home
        from realme.app.profile import Profile
        return float(Profile(data_home() / "profile").data.get("pause_scale", 1.0))
    except Exception:
        return 1.0


def _display_name() -> str:
    try:
        from realme.core.env import data_home
        from realme.app.profile import Profile
        return (Profile(data_home() / "profile").data.get("display_name")
                or "your host")
    except Exception:
        return "your host"


def _writer(name: str, args):
    from realme.adapters.script_writer import GeminiScriptWriter, PlaceholderScriptWriter
    if name == "imported":
        # No writer: lecture.write_script reuses the script already saved in
        # the project.
        return None
    from realme.adapters.script_writer_more import WRITERS, model_for
    if name == "gemini":
        from realme.adapters.script_writer import DEFAULT_MODEL
        return GeminiScriptWriter(model=model_for("gemini", args.model) or DEFAULT_MODEL)
    if name in WRITERS:
        return WRITERS[name](model=model_for(name, getattr(args, "model", None)))
    if name == "placeholder":
        print("!! Using PLACEHOLDER narration. Output is a pipeline test, not a lecture.",
              file=sys.stderr)
        return PlaceholderScriptWriter()
    sys.exit(f"Unknown script writer '{name}'.")


def _warn_if_update_pending() -> None:
    """
    Say so when the RealMe_Update.zip sitting in the folder is a different
    release from the one running.

    Answered by content, never by timestamp. Archive entries are deliberately
    backdated 24 hours (see realme.core.build), so the archive file is always
    "newer" than the files it extracted, and an mtime check would always fire.

    An unapplied update otherwise shows up only as "unrecognized arguments"
    for a flag the documentation describes.
    """
    try:
        from realme.core.paths import project_root
        from realme.core.build import update_pending
        root = project_root()
        if root is None:
            return
        pending, why = update_pending(root)
        if pending:
            print("  ! RealMe_Update.zip is a different version from the code "
                  f"running ({why}).\n"
                  "    Run 00_Windows\\0_Update.bat to apply it.\n",
                  file=sys.stderr)
    except Exception:
        pass          # a warning must never be what breaks a command



# Voice tuning flags, accepted ANYWHERE on the command line.
#
# These are declared on the top-level parser, which in argparse means they must
# appear before the subcommand: `realme --vc-tau 0.15 bench …`. After it --
# `realme bench --engines … --vc-tau 0.15` -- argparse would report
# `unrecognized arguments: --vc-tau 0.15`.
#
# argparse has no clean way to accept an option in both positions -- `parents=`
# on the subparsers makes the subparser's own default overwrite the value given
# before the subcommand. So these are taken out of argv before argparse sees it,
# and passed on through the environment, which is where every consumer already
# reads them from.
TUNING_FLAGS = {
    "--piper-voice": ("REALME_PIPER_VOICE", None),
    "--piper-speed": ("REALME_PIPER_SPEED", "float"),
    "--vc-topk": ("REALME_VC_TOPK", "int"),
    "--vc-tau": ("REALME_VC_TAU", "float"),
    "--vc-polish": ("REALME_VC_POLISH", ("off", "light", "steady", "tight")),
    "--tts-temperature": ("REALME_TTS_TEMPERATURE", "float"),
    "--tts-top-k": ("REALME_TTS_TOP_K", "int"),
    "--tts-top-p": ("REALME_TTS_TOP_P", "float"),
    "--tts-repetition-penalty": ("REALME_TTS_REPETITION_PENALTY", "float"),
}


def _voice_gemini(a) -> int:
    """
    `realme voice gemini` -- enrol, inspect and forget a cloud-cloned voice.

    Kept apart from `voice enroll` deliberately. Enrolling with Qwen3 copies a
    file into the local profile directory; enrolling here uploads the voice to
    Google, where it is kept for a year, so it is a separate command rather
    than a flag.
    """
    import datetime, json
    from pathlib import Path as _P
    from realme.core.env import data_home
    from realme.app.profile import Profile
    from realme.adapters.base import AdapterUnavailable
    from realme.adapters import tts_gemini as G

    prof = Profile(data_home() / "profile")
    g = dict(prof.data.get("gemini_voice") or {})

    if a.ack_paid:
        print(G.TIER_NOTICE)
        print()
        g["paid_tier_ack"] = True
        prof.update({"gemini_voice": g})
        print("Recorded: this key bills to a paid project.")
        return 0

    if a.list_voices:
        try:
            rows = G.list_voices()
        except AdapterUnavailable as e:
            print(f"Could not ask Google: {e}")
            return 1
        if not rows:
            print("No replicated voices in this project.")
            return 0
        here = g.get("voice")
        print(f"{len(rows)} replicated voice(s) in your Google project:\n")
        for r in rows:
            mark = " <- the one this profile uses" if r["voice"] == here else ""
            print(f"  {r['voice']}  {r.get('display_name') or '(no name)'}")
            print(f"      model {r['model'] or '?'}   expires {r['ttl']}{mark}")
        if not here:
            print("\nThis profile is not using any of them. Adopt one with:")
            print(f"  realme voice gemini --use {rows[0]['voice']}")
        return 0

    if a.use_voice:
        v = a.use_voice.strip()
        if not v.startswith(("voice_", "voicekey_")):
            print("That does not look like a Gemini voice id. They start with "
                  "voice_ or voicekey_. `realme voice gemini --list` shows "
                  "yours.")
            return 1
        try:
            G.get_voice(v)
        except AdapterUnavailable as e:
            print(f"Google does not recognise {v}: {e}")
            return 1
        g = G.with_voice(g, v, stored=v.startswith("voice_"))
        if not g.get("created_at"):
            import datetime as _dt
            g["created_at"] = _dt.datetime.now().isoformat(timespec="seconds")
        prof.update({"gemini_voice": g})
        print(f"This profile now speaks with {v}.")
        print("Nothing was uploaded; the voice already existed.")
        return 0

    if a.gem_pace:
        from realme.enrollment.pace import calibrate_engine
        from realme.core.env import data_home as _dh
        ref = prof.data.get("voice_reference")
        text = (prof.data.get("voice_reference_text") or "").strip()
        try:
            text = prof.reference_text or text
        except AttributeError:
            pass
        if not ref or not text:
            print("Measuring a pace needs the enrolment recording AND its "
                  "transcript:\n"
                  '  realme voice enroll <file> --transcript-file <txt>')
            return 1
        if not g.get("voice"):
            print("No Gemini voice enrolled yet; there is nothing to measure.")
            return 1
        if not a.target_wpm and not prof.transcript_matches_recording:
            print(prof.transcript_note or
                  "The transcript and the recording are not known to match.")
            return 1
        print("Rendering a short sample through the cloned voice "
              "(about one cent)...\n")
        try:
            r = calibrate_engine("gemini-tts", _P(ref), text,
                                 target_wpm=a.target_wpm,
                                 workdir=_dh() / "_work")
        except Exception as e:
            print(f"Could not measure it: {e}")
            return 1
        g["rate_match"] = r["pace"]
        g["rate_match_for"] = g.get("voice")
        prof.update({"gemini_voice": g})
        if not a.target_wpm:
            prof.note_transcript_pairing()
        if r["pace"] == 1.0:
            print(f"\n  stored: no correction. Your "
                  f"{r['target_wpm']:.0f} words a minute of speech against "
                  f"the clone's {r['engine_wpm']:.0f} is inside take-to-take "
                  f"variation, so nothing is retimed.")
        else:
            print(f"\n  stored: every Gemini render is now retimed by "
                  f"{r['pace']:.3f}, measured for {g.get('voice')}")
            print(f"  (you {r['target_wpm']:.0f} words/min of speech, "
                  f"the clone {r['engine_wpm']:.0f})")
        print(f"  listen to the sample: {r['sample_wav']}")
        print("\n  Audio already rendered keeps the pace it was made with; "
              "the key\n  carries this number, so the next render is a fresh "
              "take rather than a\n  cache hit at the old speed.")
        return 0

    if a.max_chars is not None:
        if a.max_chars < 0:
            print("--max-chars cannot be negative.")
            return 1
        # 0 means "use the pipeline's chunking" and is distinct from None,
        # which means "use the measured default".
        g["max_chars"] = int(a.max_chars)
        prof.update({"gemini_voice": g})
        if a.max_chars:
            print(f"Gemini TTS will now be sent up to {a.max_chars} characters "
                  f"per call (~{a.max_chars // 6} words).")
            print("The pipeline still splits at clause boundaries; it just "
                  "merges more of them into one call.")
        else:
            print("Back to the pipeline's chunking (260 characters a call) -- "
                  "smaller than\nthis engine needs, but never hurried.")
        return 0

    if a.forget:
        voice = g.get("voice") if a.forget == "__profile__" else a.forget
        if not voice:
            print("No Gemini voice is enrolled. Name one to delete it: "
                  "realme voice gemini --forget voice_...")
            return 0
        try:
            G.delete_voice(voice)
            print(f"Deleted {voice} from your Google project.")
        except AdapterUnavailable as e:
            print(f"Could not delete it at Google ({e}).")
            print("Check the console at aistudio.google.com so it does not "
                  "sit there.")
        # Only clear the profile if the voice deleted was the profile's. The
        # consent clip is kept either way: it is a local recording, not a
        # Google artifact, and can be reused for a later enrolment.
        if voice == g.get("voice"):
            prof.update({"gemini_voice": G.with_voice(g, "", created_at=None)})
        return 0

    if a.enroll:
        ref = prof.data.get("voice_reference")
        if not ref or not _P(ref).is_file():
            print("No voice reference enrolled yet. Record 10-30 seconds of "
                  "clean speech and run:")
            print("  realme voice enroll my_take.wav --transcript \"...\"")
            return 1
        consent = a.consent_wav or g.get("consent_recording")
        if not consent or not _P(consent).is_file():
            print("Google needs its OWN consent clip, in its exact words. "
                  "RealMe's consent recording says something broader and will "
                  "be rejected.\n")
            print("Record yourself saying exactly this, in the same room and "
                  "at the same distance as your reference clip:\n")
            print(f"    {G.CONSENT_SENTENCE}\n")
            print("Then:  realme voice gemini --enroll --consent-wav consent.wav")
            return 1
        if not (os.environ.get("REALME_GEMINI_PAID") == "1"
                or g.get("paid_tier_ack")):
            print(G.TIER_NOTICE)
            return 1

        # Converted the same way the reference was, because `_inline_audio`
        # refuses anything that is not already a RIFF WAV, and phone
        # recordings are usually .m4a.
        from realme.core.media import ensure_reference_wav
        consent_wav = ensure_reference_wav(_P(consent))

        # Measured before it is sent. Google runs a speaker check on the pair
        # and refuses with an HTTP 500 minutes later; the two causes it
        # refuses for can both be checked locally in a second.
        rep = G.consent_check(_P(ref), _P(consent_wav))
        if rep["score"] is not None:
            print(f"  acoustic resemblance between the two clips: "
                  f"{rep['score']:.3f}")
        for note in rep["notes"]:
            print(f"  ! {note}")
        if rep["notes"] and not a.anyway:
            print("\n  Not uploading. Google's own guidance is to record both "
                  "clips on the\n  same microphone in the same room, one "
                  "after the other.\n"
                  "  Send it anyway with --anyway.")
            return 1

        print(f"Uploading {_P(ref).name} and {_P(consent_wav).name} to Google...")
        try:
            made = G.create_voice(_P(ref), _P(consent_wav),
                                  store=not a.stateless,
                                  log=lambda m: print(m),
                                  display_name=(prof.data.get("display_name")
                                                or "RealMe instructor"))
        except AdapterUnavailable as e:
            print(f"\nGoogle refused.\n\n{e}")
            return 1
        g = G.with_voice(
            g, made["voice"], model=made["model"], stored=made["stored"],
            consent_recording=str(consent_wav),
            created_at=datetime.datetime.now().isoformat(timespec="seconds"))
        prof.update({"gemini_voice": g})
        print(f"\nEnrolled: {made['voice']}")
        print(f"  expires: {made['ttl']}")
        print(f"  (if anything below goes wrong, that id is safe: "
              f"`realme voice gemini --use {made['voice']}`)")
        print("\nNext, measure how much text it takes in one call:")
        print("  python 03_App\\probe_gemini_tts.py")
        print("\nThen select it for a render with  --tts gemini-tts, or in "
              "the Studio's Recording menu.")
        return 0

    # No flags: say where things stand, and what the next step is.
    print("Gemini TTS -- a cloned voice that lives in Google's project, not "
          "on this machine.\n")
    have_key = bool(os.environ.get("GEMINI_API_KEY"))
    print(f"  API key              : {'set' if have_key else 'MISSING (GEMINI_API_KEY)'}")
    print(f"  paid tier confirmed  : {'yes' if g.get('paid_tier_ack') else 'no'}")
    print(f"  replicated voice     : {g.get('voice') or 'none enrolled'}")
    if g.get("voice"):
        print(f"    model              : {g.get('model')}")
        print(f"    created            : {g.get('created_at')}")
        print(f"    expires            : "
              f"{'1 year from last use' if g.get('stored') else '7 days'}")
    mc = g.get("max_chars")
    print(f"  characters per call  : "
          + (f"{mc}" if mc else
             (f"{G.MEASURED_MAX_CHARS} (the measured default)" if mc is None
              else "260 (the pipeline's own)")))
    made, out = G.calls_today(), G.exhausted_today()
    print(f"  requests today       : {made}, counted here "
          f"(Google reports no balance)")
    if out:
        print(f"    Google said the daily allowance was gone at {out}. It "
              f"resets on the calendar\n    date; a render that stops resumes "
              f"from where it stopped.")
    print(f"\n  Google's consent sentence, to record in your own voice:\n")
    print(f"    {G.CONSENT_SENTENCE}\n")
    if not g.get("paid_tier_ack"):
        print(G.TIER_NOTICE)
    elif not g.get("voice"):
        print("  Next:  realme voice gemini --enroll --consent-wav consent.wav")
    else:
        print("  Measure the chunk size:  python 03_App\\probe_gemini_tts.py")
    return 0


def _take_tuning_flags(argv: list[str]) -> list[str]:
    """Strip the tuning flags out of argv into the environment. Exit with an
    error on a value that is not usable, e.g. `--vc-tau banana`."""
    out, i = [], 0
    while i < len(argv):
        token = argv[i]
        name, eq, inline = token.partition("=")
        if name not in TUNING_FLAGS:
            out.append(token)
            i += 1
            continue
        env, kind = TUNING_FLAGS[name]
        if eq:
            value = inline
        elif i + 1 < len(argv):
            value = argv[i + 1]
            i += 1
        else:
            sys.exit(f"{name} needs a value")
        if kind == "float":
            try:
                float(value)
            except ValueError:
                sys.exit(f"{name} takes a number, not {value!r}")
        elif kind == "int":
            try:
                int(value)
            except ValueError:
                sys.exit(f"{name} takes a whole number, not {value!r}")
        elif isinstance(kind, tuple) and value not in kind:
            sys.exit(f"{name} must be one of: {', '.join(kind)}")
        os.environ[env] = value
        i += 1
    return out


def main(argv=None):
    """The entry point. `_main` does the work.

    This exists to report the two failures that are the reader's to fix as
    sentences instead of tracebacks. `MediaError` and `AdapterUnavailable` are
    raised with a message written for a person -- "No deck at ...", "ffmpeg is
    not on PATH" -- and a traceback would bury it under frames from pymupdf
    or httpx.

    Only those two. Anything else is a fault in this program and keeps its
    traceback.
    """
    from realme.core.media import MediaError
    from realme.adapters.base import AdapterUnavailable
    argv = list(argv) if argv is not None else sys.argv[1:]
    if "--json" in argv:
        return _main_json([t for t in argv if t != "--json"])
    try:
        return _main(argv)
    except (MediaError, AdapterUnavailable) as e:
        print(f"\n{e}", file=sys.stderr)
        return 1


#: What a command reports when run with --json. Filled by `_result`; read by
#: `_main_json`. A module global because the commands are one long function
#: with many return points.
_RESULT: dict = {}


def _result(*, report=None, files=None, warnings=None, **extra) -> None:
    """Record what this command made, for --json. Harmless without it.

    `files` values are made absolute, so an agent can open them without
    knowing which directory the command ran in.
    """
    if report is not None:
        _RESULT["report"] = report
    if files is not None:
        if isinstance(files, dict):
            _RESULT["files"] = {k: str(Path(v).resolve()) for k, v in files.items()}
        else:
            _RESULT["files"] = [str(Path(v).resolve()) for v in files]
    if warnings is not None:
        _RESULT["warnings"] = list(warnings)
    _RESULT.update(extra)


def _main_json(argv) -> int:
    """`realme <command> ... --json`: one JSON object on stdout, prose on stderr.

    For a program driving RealMe (a script, an AI agent). The human-readable
    lines still appear -- on stderr, where a terminal shows them and a parser
    does not read them. The object is always printed, success or not:

        {"ok": true, "command": "lecture", "exit_code": 0,
         "report": {...}, "files": {...}, "warnings": [...]}
        {"ok": false, "command": "lecture", "exit_code": 1, "error": "..."}

    Commands that have not been taught to fill `report` still give ok,
    exit_code and error, which is enough to branch on.
    """
    import contextlib
    from realme.core.media import MediaError
    from realme.adapters.base import AdapterUnavailable, QuotaExceeded
    _RESULT.clear()
    out = sys.stdout
    payload: dict = {}
    try:
        with contextlib.redirect_stdout(sys.stderr):
            code = _main(argv)
        code = int(code or 0)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr)
            code, payload["error"] = 1, e.code.strip()
        else:
            code = int(e.code or 0)
    except QuotaExceeded as e:
        print(f"\n{e}", file=sys.stderr)
        code, payload["error"] = 1, str(e)
        payload["stopped"] = ("spend_cap" if type(e).__name__ == "SpendCapReached"
                              else "quota")
    except (MediaError, AdapterUnavailable, ValueError) as e:
        print(f"\n{e}", file=sys.stderr)
        code, payload["error"] = 1, str(e)
    result = {"ok": code == 0, "command": _RESULT.pop("command", None),
              "exit_code": code}
    result.update(_RESULT)
    result.update(payload)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str), file=out)
    return code


def _main(argv=None):
    argv = list(argv) if argv is not None else sys.argv[1:]
    argv = _take_tuning_flags(argv)
    _warn_if_update_pending()
    from realme.core.env import load as _load_env
    _load_env()
    # Set before anything builds text, so every path -- narrate, lecture,
    # dialogue, bench, the Studio in-process -- sees the same rule.
    # Read AND remove, in any position. These are declared on the top-level
    # parser so --help documents them, which means argparse would reject them
    # after a subcommand, e.g. `realme narrate x.txt --spell-acronyms known`.
    # Same treatment as the tuning flags.
    for flag, env in (("--spell-acronyms", "REALME_SPELL_ACRONYMS"),
                      ("--seconds-per-slide", "REALME_SECONDS_PER_SLIDE")):
        while flag in argv:
            i = argv.index(flag)
            if i + 1 >= len(argv):
                sys.exit(f"{flag} needs a value")
            os.environ[env] = argv[i + 1]
            del argv[i:i + 2]
    # allow_abbrev=False is required.
    #
    # argparse classifies every token in argv as an option or a value BEFORE
    # the subcommand positional is consumed, and it resolves unique prefixes
    # while doing so. With abbreviation on, the top-level parser would match
    # `--tts` in `realme narrate x.txt --tts qwen3cpp` against its own
    # --tts-temperature / --tts-top-k / --tts-top-p / --tts-repetition-penalty
    # and fail with "ambiguous option" before the subparser saw it.
    #
    # The tuning flags above are stripped from argv before parsing anyway and
    # are declared here only so --help documents them.
    p = argparse.ArgumentParser("realme", allow_abbrev=False,
                                description="Pedagogical digital twin pipeline")
    # Pinned in one place, overridable without editing code: Google retires
    # preview models on its own schedule, so REALME_GEMINI_MODEL can follow
    # without a source change.
    from realme.adapters.script_writer import DEFAULT_MODEL as _GEMINI_DEFAULT
    p.add_argument("--model",
                   default=os.environ.get("REALME_GEMINI_MODEL")
                   or _GEMINI_DEFAULT)
    # No default for --device: only the torch-based adapters have a device at
    # all; left unset, each picks its own (see adapters/factory.py).
    p.add_argument("--seconds-per-slide", dest="secs_per_slide", type=int,
                   default=None,
                   help="Upper limit on narration per slide, in seconds. 110 "
                        "by default (~256 words). A limit, not a target: a "
                        "section divider should still get a sentence or two.")
    p.add_argument("--spell-acronyms", dest="spell_acronyms", default=None,
                   choices=["all", "known", "off"],
                   help="all (default): spell every initialism the lexicon "
                        "does not claim. known: also skip a built-in list of "
                        "acronyms said as words. off: spell none.")
    p.add_argument("--json", action="store_true",
                   help="Print one JSON object on stdout (ok, exit_code, "
                        "report, files, warnings, error) and send the "
                        "readable lines to stderr. Works after any command: "
                        "`realme lecture deck.pdf --json`. For scripts "
                        "and AI agents.")
    p.add_argument("--device", default=None,
                   help="Torch device for the GPU engines (chatterbox, "
                        "indextts). Ignored by piper and qwen3cpp.")
    p.add_argument("--reference-wav", type=Path)
    p.add_argument("--tts-temperature", type=float, default=None,
                   help="qwen3cpp sampling temperature. 0.9 is the default; "
                        "lower is flatter and more repeatable, higher is more "
                        "expressive and more likely to drift.")
    p.add_argument("--tts-top-k", type=int, default=None,
                   help="qwen3cpp top-k. 50 is the default; lower is safer.")
    p.add_argument("--tts-top-p", type=float, default=None,
                   help="qwen3cpp nucleus sampling. 1.0 is the default.")
    p.add_argument("--tts-repetition-penalty", dest="tts_rep", type=float,
                   default=None,
                   help="qwen3cpp repetition penalty. 1.05 is the default; it "
                        "guards the stutter-and-loop failure.")
    p.add_argument("--vc-tau", type=float, default=None,
                   help="OpenVoice flow temperature. 0.3 is the default; lower "
                        "is cleaner and more deterministic, higher moves "
                        "further toward your timbre and warbles more.")
    p.add_argument("--vc-polish", default=None,
                   choices=["off", "light", "steady", "tight"],
                   help="Compression and headroom on converted audio "
                        "(default: steady).")
    p.add_argument("--vc-topk", type=int, default=None,
                   help="Voice conversion smoothing: how many of your own "
                        "frames are averaged into each output frame. 4 is the "
                        "default; 1 keeps the most detail.")
    p.add_argument("--piper-speed", type=float, default=None,
                   help="Pace of the draft voice: 1.0 is its own, 0.9 is a "
                        "tenth slower. Conversion keeps the pace it is given.")
    p.add_argument("--piper-voice", default=None,
                   help="Piper voice name, e.g. a voice fine-tuned on your own "
                        "corpus.")
    sub = p.add_subparsers(dest="cmd", required=True)

    co = sub.add_parser("corpus", help="Build the recording corpus for a "
                                       "fine-tuned voice of your own")
    co.add_argument("action", choices=["script", "add", "split", "status", "export"])
    co.add_argument("--from", dest="source", type=Path,
                    help="For `script`: your own writing to draw sentences from")
    co.add_argument("--minutes", type=float, default=35.0,
                    help="For `script`: how much speech to ask for")
    co.add_argument("--script", dest="script_file", type=Path,
                    help="For `split`: the script that was read")
    co.add_argument("--text", default="", help="For `add`: what was said")
    co.add_argument("--silence-db", dest="silence_db", type=int, default=-35)
    co.add_argument("--min-silence", dest="min_silence", type=float, default=0.6)
    co.add_argument("-o", "--out", type=Path)
    co.add_argument("audio", nargs="?", type=Path)

    d = sub.add_parser("doctor", help="Check which adapters can actually run here")

    md = sub.add_parser("models", help="Ask Google which Gemini models this "
                                       "API key can call")
    md.add_argument("--all", action="store_true",
                    help="Every model, not only the ones that can write a "
                         "script (generateContent)")

    vo = sub.add_parser("voice", help="Check, polish and bake your voice reference")
    # `analyze` is an alias for `check`.
    vo.add_argument("action", choices=["check", "analyze", "polish", "compare",
                                       "bake", "consent", "resemblance",
                                       "enroll", "show", "transcript", "pace", "registers",
                                       "tau", "gemini", "takes", "speed",
                                       "choose"])
    vo.add_argument("--against", type=Path, help="Second WAV, for resemblance")
    vo.add_argument("--also", type=Path, action="append", default=[],
                    metavar="WAV",
                    help="For `takes`: another candidate recording. Repeat it. "
                         "Compares them on the axes that separate one clone "
                         "from another, before you spend a clone on one.")
    vo.add_argument("wav", nargs="?", type=Path)
    # `off` is the default: process only what a recording needs, and say what
    # that is rather than treating every take alike. A full chain removes
    # detail from a clean recording -- dynamics, pace, the breathing pauses
    # between sentences.
    vo.add_argument("--take", action="append", default=[], metavar="NAME=FILE",
                    help="With `registers`: one recording per register, e.g. "
                         "--take pressing=press.wav. Repeat for each. All of "
                         "them are levelled together so the difference between "
                         "them stays the delivery.")
    vo.add_argument("--all", action="store_true",
                    help="With `pace`: measure every offered draft voice, not "
                         "just one. The match is then computed per voice, "
                         "which is what stops a guest being slowed by a ratio "
                         "measured against a different speaker.")
    vo.add_argument("--voice", default=None,
                    help="For `pace`: which draft voice to calibrate "
                         "(default: the one in use). For `speed`: the voice "
                         "to set, as the Studio names it -- gemini-tts, "
                         "qwen3cpp, piper:en_US-amy-medium")
    vo.add_argument("--percent", type=float, default=None,
                    help="For `speed`: how much faster (+) or slower (-) this "
                         "voice speaks everywhere, -30 to 30; 0 clears it. "
                         "Applied to the finished take, so nothing is "
                         "recorded again. Without --percent, lists them.")
    vo.add_argument("--choice", choices=["gemini", "local", "none"],
                    help="For `choose`: whether RealMe speaks in a cloned voice "
                         "-- gemini (cloned at Google), local (the free "
                         "qwen3cpp clone, from the RealMe developer) or none "
                         "(the free draft voices). Without it, asks.")
    vo.add_argument("--if-new", dest="if_new", action="store_true",
                    help="For `choose`: ask only on a new installation; an "
                         "existing choice or an enrolled voice is kept.")
    vo.add_argument("--preset", default="off",
                    choices=["off", "clean", "natural", "warm"])
    vo.add_argument("--name", default="", help="For the consent script")
    vo.add_argument("--transcript", default="",
                    help="For `enroll`: exactly what you said, word for word. "
                         "Either the text itself or the path to a .txt file - "
                         "a long transcript belongs in a file.")
    vo.add_argument("--transcript-file", dest="transcript_file", type=Path,
                    help="For `enroll`: same thing, said explicitly.")
    vo.add_argument("--consent-take", action="store_true",
                    help="For `enroll`: store as the consent recording rather "
                         "than the voice reference")
    vo.add_argument("-o", "--out", type=Path)
    # ---- `voice gemini`: enrol a replicated voice with Google.
    vo.add_argument("--enroll", action="store_true",
                    help="For `gemini`: create the replicated voice from your "
                         "enrolled reference clip plus the Google consent clip")
    vo.add_argument("--consent-wav", dest="consent_wav", type=Path,
                    help="For `gemini --enroll`: the recording of Google's "
                         "exact consent sentence (`realme voice gemini` prints "
                         "it). Defaults to the one already stored.")
    vo.add_argument("--stateless", action="store_true",
                    help="For `gemini --enroll`: a 7-day voicekey_ instead of "
                         "a stored voice_ that lives a year from last use")
    vo.add_argument("--acknowledge-paid", dest="ack_paid", action="store_true",
                    help="For `gemini`: confirm this API key bills to a PAID "
                         "project, having read what that changes")
    vo.add_argument("--max-chars", dest="max_chars", type=int,
                    help="For `gemini`: characters per API call, as measured "
                         "by probe_gemini_tts.py. Unset = the pipeline default.")
    vo.add_argument("--list", dest="list_voices", action="store_true",
                    help="For `gemini`: every replicated voice in your Google "
                         "project, including ones RealMe has lost track of")
    vo.add_argument("--use", dest="use_voice", default="",
                    help="For `gemini`: adopt a voice that already exists at "
                         "Google (voice_... or voicekey_...) without uploading "
                         "anything again")
    vo.add_argument("--pace", dest="gem_pace", action="store_true",
                    help="For `gemini`: measure how fast the cloned voice "
                         "actually speaks and store the correction that makes "
                         "it match you")
    vo.add_argument("--target-wpm", dest="target_wpm", type=float,
                    help="With `gemini --pace`: the rate to match, in words "
                         "per minute of speech. Default: the enrolment "
                         "recording's own. Give it explicitly when the "
                         "enrolment was read more carefully than you lecture.")
    vo.add_argument("--anyway", action="store_true",
                    help="For `gemini --enroll`: upload even though the two "
                         "recordings do not look like they were made on the "
                         "same equipment")
    vo.add_argument("--forget", nargs="?", const="__profile__", default=None,
                    metavar="VOICE",
                    help="For `gemini`: delete a replicated voice from your "
                         "Google project. With no value, the one this profile "
                         "uses; with a voice_... id, that one -- which is how "
                         "an orphan from a failed enrolment is cleaned up.")

    mg = sub.add_parser("migrate",
                        help="Package this working install for another machine")
    mg.add_argument("--out", type=Path,
                    help="Where to write the .zip (default: beside the project)")
    # The sentinel is a name, not an empty path: `Path("")` normalises to
    # `Path(".")`, which is indistinguishable from someone typing `.`.
    mg.add_argument("--restore", type=Path, metavar="FOLDER", nargs="?",
                    const=Path("__auto__"),
                    help="On the NEW machine: copy the packaged voice and "
                         "profile into your data directory. FOLDER is where you "
                         "unzipped the package (the folder holding "
                         "`profile_data`); omit it and the usual places are "
                         "searched.")
    mg.add_argument("--include-key", dest="include_key", action="store_true",
                    help="Also carry your .env. Off by default -- an API key in "
                         "a zip travels further than you intend.")
    mg.add_argument("--no-voice", dest="voice", action="store_false",
                    help="Leave your enrolled voice and profile out, for a "
                         "package going to ANOTHER PERSON rather than to "
                         "another machine of yours. They record their own; the "
                         "packaged instructions tell them how.")
    mg.add_argument("--full", action="store_true",
                    help="Include the retired engines' source too")
    # The local Qwen3 engine is 2.2 GB of a 2.8 GB package. It is a standard
    # part of RealMe, and without it the recipient needs a compiler, so it
    # travels by default whenever it is installed here.
    mg.add_argument("--with-qwen", dest="qwen", action="store_const", const=True,
                    default=None,
                    help="Carry the local Qwen3 engine (~2.2 GB). The default "
                         "when it is installed here")
    mg.add_argument("--no-qwen", dest="qwen", action="store_const", const=False,
                    help="Leave the local Qwen3 engine out; the recipient uses "
                         "Gemini and can add it with 3_Get_Voice_Engine.bat")
    mg.add_argument("--dry-run", dest="dry_run", action="store_true",
                    help="Show what would be packaged, and how big, without "
                         "writing it")

    ky = sub.add_parser("key", help="Store an API key in a .env file")
    ky.add_argument("name", nargs="?", help="e.g. GEMINI_API_KEY")
    ky.add_argument("value", nargs="?")
    ky.add_argument("--where", action="store_true",
                    help="Show which .env files are being read")

    en = sub.add_parser("engine", help="Install or check the local voice engine")
    # No `vc` action. Voice conversion (piper + kNN-VC / OpenVoice) is not
    # offered: the C++ engine clones directly, and the converters need a torch
    # stack at startup. Its installer module remains in the tree.
    en.add_argument("action", choices=["status", "install", "retire", "cpp"])
    en.add_argument("--portable", action="store_true",
                    help="For `cpp build`: shorthand for --target-cpu avx2. "
                         "Build a binary that runs on an older machine.")
    en.add_argument("--target-cpu", dest="target_cpu", default="native",
                    choices=["native", "avx2", "baseline"],
                    help="For `cpp build`: which instruction set to compile "
                         "for. `native` is this machine and may not run "
                         "elsewhere; `avx2` runs on any Core chip since 2013; "
                         "`baseline` assumes nothing (Atom/Celeron).")
    en.add_argument("--gpu", default="auto",
                    choices=["auto", "cuda", "vulkan", "metal", "off"],
                    help="For `cpp build`: which GPU backend to compile in. "
                         "auto detects; cuda is NVIDIA-only and fastest; "
                         "vulkan works on any vendor; metal is macOS.")
    en.add_argument("--quant", default="f16", choices=["f16", "q8_0"],
                    help="For `cpp convert`: weight precision. q8_0 is ~28%% "
                         "smaller and often faster on CPU.")
    en.add_argument("--cpp-action", dest="cpp_action",
                    choices=["status", "build", "convert", "all", "selftest",
                             "rebuild", "rebuild-all", "quantize",
                             "export-source", "export-release",
                             "export-server", "integrate"],
                    default="all",
                    help="For `cpp`: which step (default: all)")
    #: Engines by name, so the command line shows which engines were asked
    #: for.
    en.add_argument("targets", nargs="*", metavar="NAME",
                    help="For `install`: which engines -- draft (the piper "
                         "voices used for drafting), clone (qwen3cpp: your own "
                         "voice; weights + GGUF + C++ build). With no name: both defaults, draft and "
                         "clone. For `retire`: the old engine folder to "
                         "reclaim.")
    en.add_argument("--python", dest="drop_python", action="store_true",
                    help="For `retire`: reclaim the Python engine's own payload "
                         "(weights + venv) now that the C++ model is built. "
                         "Refuses unless the converted GGUF is complete.")
    en.add_argument("--delete", action="store_true",
                    help="For `retire`: actually delete. Without it you get a "
                         "report and nothing is touched.")
    en.add_argument("--archive-to", type=Path,
                    help="For `retire --delete`: where to put the data RealMe "
                         "did not adopt (default: alongside the old folder)")
    en.add_argument("--from", dest="source", type=Path,
                    help="Copy from an existing real_voice_qwen3 folder "
                         "instead of downloading")
    en.add_argument("--model", default="Qwen3-TTS-12Hz-0.6B-Base")
    en.add_argument("--with-my-voice", dest="with_my_voice", action="store_true",
                    help="For `cpp --cpp-action export-server`: put your RealMe "
                         "voice recording in the bundle as the voice 'realme'")
    en.add_argument("--source", dest="engine_source", default="realme",
                    choices=["realme", "original"],
                    help="For `cpp --cpp-action build`: realme (the default) "
                         "builds RealMe's engine source; original builds the "
                         "original upstream source, unchanged, into a folder "
                         "of its own for comparison (RealMe never uses it)")
    en.add_argument("--export-to", dest="export_to", type=Path, default=None,
                    help="For `cpp --cpp-action export-source`: a new folder "
                         "for RealMe's revised qwen3-tts.cpp source")
    en.add_argument("--force", action="store_true",
                    help="Rebuild even if it looks installed")
    en.add_argument("--full", action="store_true",
                    help="Install PyTorch with CUDA and qwen-tts with all its "
                         "dependencies. Only if the lean install misbehaves.")

    bn = sub.add_parser("bench", help="Time the same script through two "
                                      "engines and compare the audio")
    bn.add_argument("--engines", default="qwen3,qwen3cpp,piper",
                    help="Comma-separated (default: qwen3,qwen3cpp,piper). "
                         "qwen3cpp is in-process; qwen3cpp-cli is the "
                         "subprocess version.")
    bn.add_argument("--script", type=Path,
                    help="A text file, written as prose. Split into utterances "
                         "the same way `realme lecture` splits it.")
    bn.add_argument("--repeat", type=int, default=1,
                    help="Run the whole thing N times with nothing changed, "
                         "and report the spread. The engine samples without a "
                         "seed, so one run is one draw; this is how you find "
                         "out whether a difference you are about to act on is "
                         "bigger than the noise.")
    bn.add_argument("--threads", type=int, default=0,
                    help="Override the thread count, to isolate its effect. "
                         "0 = automatic (up to 8).")
    bn.add_argument("--mode", default="natural",
                    choices=["natural", "sentence", "stable", "expressive"],
                    help="How the script is split. `stable` keeps paragraphs "
                         "whole -- fewer, longer calls.")
    bn.add_argument("--minutes", type=float, default=0.0,
                    help="Use only the first N minutes' worth of narration. "
                         "Three engines over five minutes is hours of compute.")
    bn.add_argument("--yes", action="store_true",
                    help="Skip the time estimate confirmation")
    bn.add_argument("--reference", default=None,
                    help="`tuned` (default: the take you adjusted in the "
                         "Studio), `raw` (the recording it came from), or a "
                         "path to any other clip.")
    bn.add_argument("--out", type=Path, help="Where to write the audio")
    bn.add_argument("--ref-seconds", dest="ref_seconds", type=float, default=0.0,
                    help="Use only the first N seconds of the reference. The "
                         "clip is prefilled on every call, so its length costs "
                         "time on every utterance -- this is how you find out "
                         "what length is actually worth paying for.")
    bn.add_argument("--rescore", action="store_true",
                    help="Recompute like_you from a previous run's audio in "
                         "--out, without synthesizing anything again.")
    bn.add_argument("--lines", type=int, default=0,
                    help="Use only the first N utterances")

    pb = sub.add_parser("bench-parallel",
                        help="Would several local voice engines at once record "
                             "faster than one? Measured on this machine")
    pb.add_argument("--engine", default="qwen3cpp",
                    help="The engine to measure (default: qwen3cpp, the local "
                         "clone). `selftest` checks the benchmark itself.")
    pb.add_argument("--layouts", default="",
                    help="Workers x threads each, comma separated, one worker "
                         "first: 1x6,2x4,3x3. Default: chosen from the number "
                         "of cores.")
    pb.add_argument("--repeat", type=int, default=2,
                    help="Runs per layout (default 2). The engine samples "
                         "without a seed, so one run is one draw.")
    pb.add_argument("--lines", type=int, default=12,
                    help="Sentences per run (default 12; with --batch, at "
                         "least two passes for every engine).")
    pb.add_argument("--batch", type=int, default=1,
                    help="Sentences per pass in each engine (default 1, one "
                         "at a time). E.g. --layouts 1x6,2x3 --batch 8: one "
                         "engine at 6 threads against two at 3, all in passes.")
    pb.add_argument("--script", type=Path, default=None,
                    help="A text file to read instead of the built-in "
                         "sentences.")
    pb.add_argument("--reference", default=None,
                    help="As for `realme bench`: tuned (default), raw, or a "
                         "path.")
    pb.add_argument("--out", type=Path, default=None,
                    help="Where the audio and results go (default: "
                         "<data folder>/bench/parallel).")
    pb.add_argument("--yes", action="store_true",
                    help="Skip the time estimate confirmation")

    gp = sub.add_parser("gemini-parallel",
                        help="How many Gemini requests at once your Google "
                             "project takes; saves the answer for lectures")
    gp.add_argument("--levels", default="1,2,3,4",
                    help="How many at once to try, in order (default 1,2,3,4)")
    gp.add_argument("--per-level", dest="per_level", type=int, default=0,
                    help="Requests per level (default: twice the level, at "
                         "least 4)")
    gp.add_argument("--yes", action="store_true",
                    help="Skip the cost confirmation")
    gp.add_argument("--use", type=int, default=0,
                    help="Record N at a time from now on (default 2; at most "
                         "what was measured). Sends nothing.")
    gp.add_argument("--out", type=Path, default=None,
                    help="Where results and audio go (default: <data "
                         "folder>/bench/gemini_parallel)")

    et = sub.add_parser("engine-timing",
                        help="Where the local clone's time goes: builds a "
                             "stopwatch copy of the engine and measures it")
    et.add_argument("--threads", default="4,6,8,12",
                    help="Thread counts to compare (default 4,6,8,12; 4 is "
                         "what the engine uses today)")
    et.add_argument("--sentences", type=int, default=3,
                    help="Sentences per thread count (default 3), after one "
                         "to warm up")
    et.add_argument("--rebuild", action="store_true",
                    help="Build the stopwatch copy again even if it exists")
    et.add_argument("--build-only", dest="build_only", action="store_true",
                    help="Build the stopwatch copy and stop -- on a computer "
                         "with a compiler, to copy to one without")
    et.add_argument("--reference", default=None,
                    help="As for `realme bench`: tuned (default), raw, or a "
                         "path")
    et.add_argument("--out", type=Path, default=None,
                    help="Where results go (default: <data folder>/bench/timing)")
    et.add_argument("--variants", default="",
                    help="With --compare: which runs, e.g. fast,q8 (default "
                         "before,after,snake,fast; q8 is the 8-bit model, voc "
                         "the 32-bit vocoder)")
    et.add_argument("--compare", action="store_true",
                    help="Measure the engine with RealMe's speed changes off "
                         "and on, at this computer's thread count, and check "
                         "the sound is identical")

    bb = sub.add_parser("bench-batch",
                        help="Does recording several sentences in one pass "
                             "help on this computer? Saves the answer")
    bb.add_argument("--sizes", default="1,2,4,8,12,16",
                    help="Sentences per pass to compare, starting at 1, up to 16 "
                         "(default 1,2,4,8,12,16; sizes that do not fit in free memory are skipped)")
    bb.add_argument("--sentences", type=int, default=16,
                    help="Sentences recorded at each size (default 16; a size "
                         "records whole passes, so 12 per pass records 12)")
    bb.add_argument("--reference", default=None,
                    help="As for `realme bench`: tuned (default), raw, or a path")
    bb.add_argument("--out", type=Path, default=None,
                    help="Where results go (default: <data folder>/bench/batch)")
    bb.add_argument("--quant", default=None, choices=["f16", "q8_0"],
                    help="Which model to measure (default: the one lectures "
                         "use, REALME_QWEN3CPP_QUANT)")
    bb.add_argument("--no-save", dest="no_save", action="store_true",
                    help="Measure and keep the results and audio, but leave "
                         "the pass size lectures use as it was (for "
                         "comparisons)")

    na = sub.add_parser("narrate", help="Read a text file aloud in your voice "
                                        "- no deck, no script writer")
    na.add_argument("source", type=Path,
                    help="A .txt, .md or .docx file. Blank lines separate "
                         "paragraphs; [[pause]] and [[slow]] markers work "
                         "exactly as they do in a lecture script.")
    na.add_argument("-o", "--out", type=Path,
                    help="Output .wav (default: alongside the source)")
    na.add_argument("--tts", default=None,
                    help="Voice. Default: your profile's (gemini-tts unless "
                         "you chose otherwise). piper for a free draft, "
                         "qwen3cpp for the optional local engine")
    na.add_argument("--mode", default="natural",
                    choices=["natural", "sentence"],
                    help="How to split: 'natural' breaks at clauses, "
                         "'sentence' only at full stops")
    na.add_argument("--pace", type=float, default=1.0,
                    help="Pace for this reading, as a multiplier. "
                         "Deterministic - the right lever for speed, not "
                         "temperature. Multiplies with the voice's own saved "
                         "speed (`realme voice speed`).")
    na.add_argument("--paragraph-gap", dest="paragraph_gap", type=float,
                    default=0.7, help="Seconds of silence between paragraphs")
    na.add_argument("--no-captions", dest="captions", action="store_false",
                    help="Skip the .srt")
    na.add_argument("--max-spend", dest="max_spend", type=float, default=None,
                    help="Stop before spending more than this many dollars on "
                         "this render (paid engines only; default: "
                         "REALME_MAX_SPEND_USD, else no cap). What was "
                         "recorded is kept, so re-running with a higher cap "
                         "continues.")

    # A finished animated explainer joins a lecture through `realme insert`.
    ins = sub.add_parser("insert",
                         help="Play a video between a lecture's slides -- an "
                              "animated explainer, a screen recording -- "
                              "in this and every later render")
    ins.add_argument("project", type=Path,
                     help="The lecture's folder: the -o of `realme lecture`, "
                          "or its folder under RealMeStudio\\projects")
    ins.add_argument("video", type=Path, nargs="?",
                     help="The video to add. Its .srt beside it, if any, is "
                          "carried into the lecture's captions.")
    ins.add_argument("--after", type=int, default=None,
                     help="The slide it plays after; 0 = before slide 1")
    ins.add_argument("--title", default="",
                     help="Its chapter title (default: the file name)")
    ins.add_argument("--captions", type=Path, default=None,
                     help="Its captions, if not NAME.srt beside the video")
    ins.add_argument("--remove", metavar="ID", default=None,
                     help="Take an insert out (the IDs are listed with no "
                          "other arguments)")
    ins.add_argument("--apply", action="store_true",
                     help="Re-join the last render with the inserts as they "
                          "are now: seconds, nothing recorded")

    sp = sub.add_parser("splice", help="Reuse an old lecture video for the "
                                      "slides that did not change")
    sp.add_argument("video", type=Path, help="The old rendered .mp4")
    sp.add_argument("segments", type=Path,
                    help="Its <deck>_segments.json cut list")
    sp.add_argument("deck", type=Path, help="The NEW slide deck")
    sp.add_argument("--rerecord", default="",
                    help="Slide numbers in the NEW deck to re-record whatever "
                         "the comparison says, comma separated. Use this for "
                         "small text edits: those are inside the noise of "
                         "video compression and are not detected.")
    sp.add_argument("-o", "--out", type=Path, default=Path("./splice"))

    sl = sub.add_parser("split", help="Cut a rendered lecture into parts at "
                                      "slide boundaries")
    sl.add_argument("video", type=Path, help="The rendered .mp4")
    sl.add_argument("--segments", type=Path, default=None,
                    help="Its <name>_segments.json. Found beside the video if "
                         "not given.")
    sl.add_argument("--at", default="",
                    help="Slide numbers where a NEW part begins, comma "
                         "separated: --at 9,21,34. Slide 1 always begins the "
                         "first part. Omit to just list the slides and stop.")
    sl.add_argument("--titles", default="",
                    help="A title per part, separated by |. Used for the file "
                         "names; a blank one falls back to the slide range.")
    sl.add_argument("--mode", choices=("auto", "copy", "encode"), default="auto",
                    help="auto copies where the cut point is a keyframe and "
                         "re-encodes only where it is not (default). copy is "
                         "instant but wrong off a keyframe; encode is exact "
                         "everywhere and slow.")
    sl.add_argument("--srt", type=Path, default=None,
                    help="Captions to slice per part. Found beside the video "
                         "if not given.")
    sl.add_argument("-o", "--out", type=Path, default=None,
                    help="Where the parts go. Default: <video>_parts/")
    sl.add_argument("--projects", action="store_true",
                    help="Make each part a lecture of its own -- its slides, "
                         "notes, captions and a deck -- so it can be edited "
                         "and re-recorded, not just played.")

    mg = sub.add_parser("merge", help="Join videos into one, losslessly where "
                                      "their settings match")
    mg.add_argument("videos", type=Path, nargs="+",
                    help="The videos to join, in the order they should play")
    mg.add_argument("-o", "--out", type=Path, required=True,
                    help="The joined .mp4")
    mg.add_argument("--mode", choices=("auto", "copy", "encode"), default="auto",
                    help="auto joins without re-encoding when every input has "
                         "the same settings and re-encodes when they differ "
                         "(default). copy never re-encodes, which produces a "
                         "broken file from mismatched inputs. encode always "
                         "re-encodes.")
    mg.add_argument("--force", action="store_true",
                    help="Append whatever these are. No notes, no slides and "
                         "no cut list are required or produced -- for an old "
                         "recording, or something made outside RealMe.")
    mg.add_argument("--check", action="store_true",
                    help="Report whether they can be joined losslessly, and "
                         "write nothing")

    sub.add_parser("blocked", help="What Windows has recently refused to run "
                                   "(Defender vs Application Control)")

    su = sub.add_parser("setup", help="Check the environment and print exactly "
                                      "what to install, for this machine")
    su.add_argument("--voice", default="en_US-lessac-medium")
    su.add_argument("--no-install", dest="install", action="store_false",
                    help="Only report. By default, any required Python package "
                         "that is missing is installed into this environment "
                         "before the summary is printed; everything that is "
                         "not pip's to install is printed as a command either "
                         "way.")
    # An alias for the default behaviour, accepted but hidden from --help so
    # existing instructions that use it keep working.
    su.add_argument("--fix", action="store_true", help=argparse.SUPPRESS)

    lec = sub.add_parser("lecture", help="Slides -> narrated video")
    lec.add_argument("deck", type=Path,
                     help="The slides as a PDF (a .pptx needs LibreOffice)")
    lec.add_argument("-o", "--outdir", type=Path, default=Path("./out"))
    lec.add_argument("--tts", default=None,
                     help="Voice engine (default: the one in your profile)")
    lec.add_argument("--max-spend", dest="max_spend", type=float, default=None,
                     help="Stop before spending more than this many dollars on "
                         "this render (paid engines only; default: "
                         "REALME_MAX_SPEND_USD, else no cap). What was "
                         "recorded is kept, so re-running with a higher cap "
                         "continues.")
    lec.add_argument("--script", default="gemini",
                     choices=["gemini", "claude", "openai", "placeholder",
                              "imported"],
                     help="'imported' reuses the script already in the project "
                          "(see: realme import-script)")
    lec.add_argument("--prosody-mode", dest="prosody_mode", default="natural",
                     choices=["natural", "sentence", "stable", "expressive"],
                     help="`stable` renders each slide as ONE synthesis call "
                          "instead of splitting it into clauses. Cheaper per "
                          "second of audio, but a one-word fix then re-renders "
                          "the whole slide.")
    lec.add_argument("--pause-scale", dest="pause_scale", type=float,
                     default=None,
                     help="Scale the silence between clauses and sentences "
                          "(0.85-1.25; default from your profile, 1.0). "
                          "Lower tightens a lecture that drags; "
                          "--prosody-mode stable removes the gaps entirely.")
    lec.add_argument("--layout", default="slide_only",
                     choices=["slide_only", "pip", "side_by_side"])
    lec.add_argument("--width", type=int, default=1920)
    lec.add_argument("--height", type=int, default=1080)
    lec.add_argument("--no-audio-check", action="store_true",
                     help="Skip the acoustic pronunciation check")
    lec.add_argument("--signalling", default="highlight",
                     choices=["highlight", "underline", "focus", "off"],
                     help="Cue the slide region being discussed (default: highlight)")
    lec.add_argument("--context", default="", help="Course context for the script writer")
    lec.add_argument("--style", default="", help="Your teaching-voice notes")

    pod = sub.add_parser("dialogue", help="Paper/topic -> two-voice podcast or debate")
    pod.add_argument("topic")
    pod.add_argument("-s", "--source", type=Path)
    pod.add_argument("-o", "--outdir", type=Path, default=Path("./out_dialogue"))
    pod.add_argument("--tts", default=None,
                     help="Instructor engine (default: your profile's)")
    pod.add_argument("--guest-tts", dest="guest_tts", default=None,
                     help="Engine for the guest. Defaults to the draft voice "
                          "(piper), NOT to whatever the instructor uses -- two "
                          "speakers with one voice is not a dialogue, and with "
                          "a cloning instructor it would make the guest you.")
    pod.add_argument("--script", default="gemini",
                     choices=["gemini", "claude", "openai", "placeholder"])
    pod.add_argument("--mode", default="debate", choices=["socratic", "debate", "interview"])
    pod.add_argument("--turns", type=int, default=16,
                     help="Turns to script. Was 8, when a turn was up to 150 "
                          "words; turns are three or four sentences now, so "
                          "the same ground needs about twice as many.")
    pod.add_argument("--guest-voice", dest="guest_voice", default=None,
                     help="Draft voice for the guest when --guest-tts is "
                          "piper. Defaults to the stock voice the instructor "
                          "is NOT using, so the two speakers differ.")
    pod.add_argument("--speed", type=float, default=0.72,
                     help="Pace of any piper voice in this dialogue. The "
                          "lecture default is 0.8; conversation reads faster "
                          "than narration at the same rate, so this is a notch "
                          "slower. 1.0 is piper's own pace.")
    pod.add_argument("--turn-gap", dest="turn_gap", type=float, default=0.6,
                     help="Seconds of silence between turns")
    pod.add_argument("--no-bookends", dest="bookends", action="store_false",
                     help="Skip the opening greeting and the sign-off")
    pod.add_argument("--host-name", dest="host_name", default=None,
                     help="Name spoken in the greeting (default: your profile "
                          "display name)")

    arg = sub.add_parser("argue", help="Argue with an AI interlocutor, live")
    arg.add_argument("topic", nargs="?", default="")
    arg.add_argument("--stance", default="adversary",
                     choices=["adversary", "socratic", "steelman"])
    arg.add_argument("-s", "--source", type=Path, help="Paper or notes to ground it")
    arg.add_argument("-o", "--outdir", type=Path, default=Path("./arguments"))
    arg.add_argument("--resume", type=Path, help="Continue a saved argument")

    si = sub.add_parser("import-script",
                        help="Turn one narration file (.txt or .md, or a RealMe "
                             "narration .json) into slide sections")
    si.add_argument("path", type=Path)
    si.add_argument("--project", default="", help="Project id to write into")

    se = sub.add_parser("export-script",
                        help="Write a lecture's narration as a notes file "
                             "([[slide N: Title]] sections), as import-script reads")
    se.add_argument("source", type=Path,
                    help="The lecture's folder, its _work/manifest.json, or a "
                         "lecture name under RealMeStudio\\projects")
    se.add_argument("-o", "--out", type=Path, default=None,
                    help="The .md file to write (default: <lecture>_notes.md "
                         "in the current folder)")

    pv = sub.add_parser("preview", help="Hear one utterance before rendering anything")
    pv.add_argument("text", help="A sentence or two of narration")
    pv.add_argument("--tts", default=None)
    pv.add_argument("-i", "--index", type=int, default=0)
    pv.add_argument("-o", "--out", type=Path, default=Path("./preview.wav"))

    mth = sub.add_parser("math", help="Inspect the spoken form of LaTeX")
    mth.add_argument("latex", nargs="?", help="A LaTeX expression")
    mth.add_argument("--pending", action="store_true", help="List unapproved entries")
    mth.add_argument("--cache", type=Path, default=Path("./math_cache.json"))

    lex = sub.add_parser("lexicon", help="Inspect, add to, or check the "
                                        "pronunciation lexicon")
    lex.add_argument("action",
                     choices=["check", "list", "emit", "add", "remove", "where"])
    lex.add_argument("term", nargs="?", help="For add/remove")
    lex.add_argument("--engine", default="piper")
    lex.add_argument("--respelling", default="",
                     help="How to say it, in ordinary letters: "
                          "\"en-haynes\". Works with every engine.")
    lex.add_argument("--ipa", default="",
                     help="IPA, for engines that take phonemes")
    lex.add_argument("--arpabet", default="", help="CMU ARPAbet")
    lex.add_argument("--note", default="", help="Why, for your future self")
    lex.add_argument("--no-spell", dest="no_spell", action="store_true",
                     help="Only stop the acronym speller touching this term, "
                          "without changing how it is said")

    ck = sub.add_parser("check-audio",
                        help="Ask a rendered WAV which pronunciation it contains")
    ck.add_argument("wav", type=Path)
    ck.add_argument("terms", nargs="+", help="Lexicon terms to listen for")
    ck.add_argument("--tts", default=None)

    st = sub.add_parser("studio", help="Launch the RealMe Studio web app")
    st.add_argument("--host", default="127.0.0.1",
                    help="Address to listen on. Anything but loopback "
                         "(127.0.0.1, localhost, ::1) needs --allow-network")
    st.add_argument("--port", type=int, default=8000)
    st.add_argument("--allow-network", action="store_true",
                    help="Allow a --host other machines can reach. The "
                         "Studio has no login: only on a network you trust")
    st.add_argument("--colab", action="store_true",
                    help="Open through Colab's built-in port proxy instead of a tunnel")

    ra = sub.add_parser("render-argument", help="Turn a saved argument into audio")
    ra.add_argument("argument", type=Path)
    ra.add_argument("-o", "--outdir", type=Path, default=Path("./arguments"))
    ra.add_argument("--tts", default=None)
    ra.add_argument("--guest-tts", default=None)

    ver = sub.add_parser("verify", help="Independently check a rendered file")
    ver.add_argument("media", type=Path)

    a = p.parse_args(argv)
    _RESULT["command"] = a.cmd + (f" {a.action}" if isinstance(
        getattr(a, "action", None), str) else "")

    # One setting, honoured by every path that ends up constructing a piper
    # voice. Set in the environment rather than passed through each call
    # site, so every consumer reads it from one place.
    if getattr(a, "piper_voice", None):
        os.environ["REALME_PIPER_VOICE"] = a.piper_voice
    if getattr(a, "piper_speed", None):
        os.environ["REALME_PIPER_SPEED"] = str(a.piper_speed)
    if getattr(a, "vc_topk", None):
        os.environ["REALME_VC_TOPK"] = str(a.vc_topk)
    if getattr(a, "vc_tau", None):
        os.environ["REALME_VC_TAU"] = str(a.vc_tau)
    if getattr(a, "vc_polish", None):
        os.environ["REALME_VC_POLISH"] = a.vc_polish

    if a.cmd == "insert":
        from realme.pipeline import inserts as INS
        from realme.pipeline import lecture as L
        from realme.core.media import MediaError
        pdir = a.project
        if not (pdir / "_work" / "manifest.json").is_file():
            sys.exit(f"{pdir} is not a lecture folder (no _work/manifest.json).")
        try:
            if a.video is not None:
                if a.after is None:
                    sys.exit("After which slide? `--after 12` (0 = before "
                             "slide 1)")
                item = INS.add(pdir, a.video, a.after, title=a.title,
                               captions=a.captions)
                print(f"Added {item.title} after slide {item.after_slide} "
                      f"({item.id})" + ("  + its captions" if item.captions
                                         else ""))
            if a.remove:
                gone = INS.remove(pdir, a.remove)
                print(f"Removed {gone.title} (the file stays in "
                      f"{pdir / INS.FOLDER})")
            items = INS.load(pdir)
            if not items:
                print("No inserted videos in this lecture.")
            else:
                print("Inserted videos (each render puts them in place):")
                for i in items:
                    where = ("before slide 1" if i.after_slide == 0
                             else f"after slide {i.after_slide}")
                    print(f"  {i.id}  {where:<18} {i.title}"
                          + ("  [captions]" if i.captions else ""))
            result = {"inserts": [i.__dict__ for i in items]}
            if a.apply:
                out = L.reassemble(pdir, log=print)
                result["report"] = out["report"]
                _result(files={k: v for k, v in out.items() if k != "report"})
                print(f"\n  {out['master']}")
            _result(report=result)
        except MediaError as e:
            sys.exit(str(e))
        if a.apply and not str(result["report"]["verdict"]).startswith("PASS"):
            return 1
        if not a.apply and items and a.video is not None:
            print("\n  Render the lecture, or run this again with --apply to "
                  "put it into the last render now.")
        return 0

    if a.cmd == "narrate":
        from realme.pipeline import narrate as N
        if not a.source.is_file():
            sys.exit(f"No such file: {a.source}")
        out = a.out or a.source.with_suffix(".wav")
        if out.resolve() == a.source.resolve():
            sys.exit("The output would overwrite the source. Pass -o.")
        tts = _tts(a.tts, a)
        print(f"Narrating {a.source}")
        print(f"  voice    : {a.tts or _profile_tts()}")
        # The reference is announced by the factory, for every command at once.
        print(f"  output   : {out}")
        try:
            r = N.narrate(a.source, out, tts, mode=a.mode, pace=a.pace,
                          paragraph_gap_s=a.paragraph_gap,
                          captions=a.captions, log=print)
        except (ValueError, FileNotFoundError) as e:
            sys.exit(str(e))
        mins, secs = divmod(r.duration_s, 60)
        print(f"\n  {int(mins)}m {secs:04.1f}s of audio from {r.words} words "
              f"({r.utterances} utterances)")
        if r.srt:
            print(f"  captions : {r.srt}")
        print(f"  spoken   : {r.transcript}")
        if r.warnings:
            # Shown prominently: the lint flags likely mispronunciations, the
            # most common fault in a finished recording.
            seen = []
            for w in r.warnings:
                if w not in seen:
                    seen.append(w)
            print(f"\n  {len(seen)} lint warning(s):")
            for w in seen[:8]:
                print(f"    {w}")
            print("  Fix with a lexicon entry, then run again - only the "
                  "changed lines re-render.")
        print(f"\n  {out}")
        _result(report={"duration_s": round(r.duration_s, 2), "words": r.words,
                        "utterances": r.utterances},
                warnings=list(dict.fromkeys(r.warnings or [])),
                files={"audio": out, "transcript": r.transcript,
                       **({"captions": r.srt} if r.srt else {})})
        return 0

    if a.cmd == "models":
        from realme.adapters.script_writer import list_models, DEFAULT_MODEL
        try:
            models = list_models()
        except RuntimeError as e:
            sys.exit(str(e))
        wanted = models if a.all else [
            m for m in models if "generateContent" in m["methods"]]
        current = getattr(a, "model", None) or DEFAULT_MODEL
        print(f"  {len(wanted)} model(s) this key can call"
              + ("" if a.all else " for writing a script") + ":\n")
        for m in wanted:
            mark = " <-- configured" if m["id"] == current else ""
            ctx = (f"{m['input_tokens']:>9,} in" if m["input_tokens"] else
                   "          ")
            print(f"  {m['id']:<44} {ctx}  {m['label'][:28]}{mark}")
        if not any(m["id"] == current for m in models):
            print(f"\n  WARNING: the configured model '{current}' is NOT in "
                  f"this list.")
            print("  A retired preview fails as a bare HTTP 404, which reads "
                  "like a broken")
            print("  endpoint rather than a model that no longer exists. Pick "
                  "one above:")
            print("    realme lecture deck.pdf --model <id>")
            print("  or set it once in your .env:  REALME_GEMINI_MODEL=<id>")
        return 0

    if a.cmd == "voice":
        from realme.core.media import MediaError
        from realme.enrollment import voice as V
        from realme.app.profile import Profile
        import datetime
        if a.action == "choose":
            from realme.core.env import data_home
            from realme.enrollment.choice import choose_voice
            prof = Profile(data_home() / "profile")
            got = choose_voice(prof, choice=a.choice, if_new=a.if_new)
            _result(report=got)
            return 0
        if a.action == "speed":
            # A voice's own speed, saved in the profile -- the same setting as
            # the Lecture tab's "Speed of this voice". Not pace: pace belongs
            # to a script or a render and is recorded into the take; this
            # belongs to the voice and stretches the take afterwards.
            from realme.core.env import data_home
            from realme.pipeline.casting import voice_key, VOICE_SPEED_RANGE
            prof = Profile(data_home() / "profile")
            speeds = dict(prof.data.get("voice_speeds") or {})
            if a.percent is not None:
                if not a.voice:
                    sys.exit("Which voice? `realme voice speed --voice "
                             "piper:en_US-amy-medium --percent 15`")
                lo, hi = ((x - 1) * 100 for x in VOICE_SPEED_RANGE)
                if not lo <= a.percent <= hi:
                    sys.exit(f"--percent must be between {lo:.0f} and {hi:.0f}")
                key = voice_key(a.voice)
                if abs(a.percent) < 0.05:
                    speeds.pop(key, None)
                else:
                    speeds[key] = round(1 + a.percent / 100, 4)
                prof.data["voice_speeds"] = speeds
                prof.save()
                print(f"Saved: {key} "
                      + (f"{a.percent:+g}%" if abs(a.percent) >= 0.05
                         else "at its own speed"))
            pcs = {k: round((float(v) - 1) * 100, 1) for k, v in speeds.items()}
            _result(report={"speeds_percent": pcs})
            if not pcs:
                print("No voice speeds saved; every voice speaks at its own.")
            else:
                print("Saved voice speeds (applied wherever the voice speaks):")
                for k, v in sorted(pcs.items()):
                    print(f"  {k:<32} {v:+g}%")
            return 0
        if a.action == "show":
            from realme.core.env import data_home
            prof = Profile(data_home() / "profile")
            txt = prof.reference_text
            f = prof.reference_text_path
            print(f"reference : {prof.data.get('voice_reference') or 'not set'}")
            print(f"transcript: {(txt[:66] + '...') if len(txt) > 66 else (txt or 'not set')}")
            print(f"            {f}  [{'present' if f.is_file() else 'no file - using profile.json'}]")
            print(f"consent   : {prof.data.get('consent_recording') or 'not set'}")
            print(f"preset    : {prof.data.get('voice_preset') or 'none (unprocessed)'}")
            print(f"ready     : {prof.ready_for_cloning}")
            # Two folders hold voice files and only one is used. Dates alone
            # cannot show whether they are the same recording -- a copy has
            # a new timestamp and identical audio -- so compare the content.
            import hashlib
            from realme.core.media import duration_of

            def facts(f: Path) -> str:
                if not f.is_file():
                    return "not present"
                try:
                    secs = f"{duration_of(f):.1f}s"
                except Exception:
                    secs = "unreadable"
                h = hashlib.sha256(f.read_bytes()).hexdigest()[:10]
                import datetime as _dt
                when = _dt.datetime.fromtimestamp(f.stat().st_mtime)
                return f"{secs:>7}  {f.stat().st_size:>9,} bytes  {h}  {when:%Y-%m-%d %H:%M}"

            inuse = prof.root / "baked_assets" / "voice_reference.wav"
            rawdir = prof.root.parent / "voice"
            raw = rawdir / "reference_raw.wav"
            print()
            print("               duration        size  content     modified")
            print(f"  IN USE   {facts(inuse)}")
            print(f"           {inuse}")
            print(f"  scratch  {facts(raw)}")
            print(f"           {raw}")
            print()
            if inuse.is_file() and raw.is_file():
                same = inuse.read_bytes() == raw.read_bytes()
                if same:
                    print("  Same recording. Nothing is pending.")
                else:
                    newer = "scratch" if raw.stat().st_mtime > inuse.stat().st_mtime else "in-use"
                    print("  DIFFERENT recordings.")
                    if newer == "scratch":
                        print("  The scratch copy is newer, which means you uploaded "
                              "something in the")
                        print("  Studio and never pressed \"Save this voice\". Your "
                              "renders still use")
                        print("  the in-use file above -- that is the safe direction, "
                              "not a problem.")
                        print("  Save it in the Studio only if that newer upload is "
                              "the one you want.")
                    else:
                        print("  The in-use file is newer, so it was set outside the "
                              "Studio (enrol).")
                        print("  Baking in the Studio would replace it with the older "
                              "scratch take;")
                        print("  RealMe refuses that, but re-upload before tuning to "
                              "be certain.")
            if rawdir.is_dir():
                n = len(list(rawdir.glob("preset_*.wav")))
                if n:
                    print(f"\n  Also {n} preset preview(s) in {rawdir.name}\\ -- "
                          f"scratch, safe to delete.")
            return 0

        if a.action == "transcript":
            # Attach or replace the transcript without re-enrolling; the audio
            # is already stored and converted.
            from realme.core.env import data_home
            prof = Profile(data_home() / "profile")
            src = a.transcript_file or a.transcript or (str(a.wav) if a.wav else "")
            if not src:
                sys.exit('realme voice transcript "what you said"   '
                         "(or a path to a .txt file)")
            text, tsource = V.read_transcript(src)
            if not text:
                sys.exit(f"The transcript from {tsource} is empty.")
            dst = prof.set_reference_text(text)
            print(f"Transcript saved: {len(text.split())} words from {tsource}")
            print(f"  -> {dst}")
            ref = prof.data.get("voice_reference")
            if ref and Path(ref).is_file():
                from realme.core.media import duration_of
                for note in V.transcript_fit(text, duration_of(Path(ref))):
                    print(f"  - {note}")
            else:
                print("  No reference audio enrolled yet: "
                      "realme voice enroll my_take.m4a")
            return 0

        if a.action == "registers":
            from realme.core.env import data_home as _dh
            from realme.enrollment import registers as REG
            prof = Profile(_dh() / "profile")
            if not a.take:
                have = prof.data.get("voice_registers") or {}
                # Already enrolled: show what was MEASURED, not the
                # instructions for doing it again. The distinctness table
                # shows whether the takes really are distinct deliveries; it
                # is stored at enrolment and re-printed here.
                store = _dh() / "voice" / "registers" / "registers.json"
                if have and store.is_file():
                    try:
                        import json as _json
                        regs = [REG.Register(**d) for d in
                                _json.loads(_TXT(store))]
                    except Exception as e:
                        regs = []
                        print(f"  (could not read {store}: {e})")
                    if regs:
                        print(REG.report(regs))
                        print("\n  Re-record any of them with --take "
                              "<name>=<file>; the whole set is levelled\n"
                              "  together, so pass all four when you do.")
                        print("\n  Emotion directions from these takes:   "
                              "realme voice tau")
                        return 0
                print("Registers are several takes of YOUR voice, the same "
                      "words delivered differently.\n")
                for name, what in REG.REGISTERS.items():
                    mark = "recorded" if name in have else "   —    "
                    print(f"  {mark}  {name:<12} {what}")
                print("\n  Record them in ONE sitting, same room, same "
                      "microphone, same distance:")
                print("  the difference between two takes is what drives this, "
                      "and a change of")
                print("  equipment between them becomes part of it.\n")
                print("  FOUR DIFFERENT DELIVERIES of the same words. Not "
                      "four readings — reading it")
                print("  the way the words suggest, four times, gives four "
                      "identical takes and there")
                print("  is no difference to extract. The words stay the same; "
                      "everything else changes.\n")
                print("  Read THE SAME passage every time — the difference "
                      "between two takes is")
                print("  what this measures, so different words become part "
                      "of it:\n")
                for line in REG.PASSAGE.strip().splitlines():
                    print(f"    {line}")
                print("\n  How to deliver each one:")
                for name, what in REG.REGISTERS.items():
                    print(f"    {name:<12} {what}")
                print()
                print("    realme voice registers --take explaining=neutral.wav "
                      "\\\n"
                      "                           --take pressing=press.wav "
                      "\\\n"
                      "                           --take conceding=concede.wav "
                      "\\\n"
                      "                           --take wondering=wonder.wav")
                return 0
            takes = {}
            for spec in a.take:
                name, _, path = spec.partition("=")
                if not path:
                    sys.exit(f"--take wants name=file, got {spec!r}")
                takes[name.strip()] = Path(path.strip())
            try:
                regs = REG.prepare(takes, _dh() / "voice" / "registers",
                                   log=print)
            except (ValueError, OSError) as e:
                sys.exit(str(e))
            print()
            print(REG.report(regs))
            prof.update({"voice_registers": {r.name: r.path for r in regs}})
            print(f"\n  Saved. A dialogue now speaks each turn in the register "
                  f"matching its stance.")
            return 0

        if a.action == "tau":
            # Emotion directions, from the registers already enrolled.
            #
            # Extraction needs the engine, not a file format: a direction is
            # only meaningful in the embedding space of the encoder that made
            # it, so this builds the adapter the way a render builds it rather
            # than constructing one here.
            from realme.core.env import data_home as _dh
            from realme.expressive import tau as TAU
            prof = Profile(_dh() / "profile")
            have = prof.data.get("voice_registers") or {}
            if len(have) < 2:
                sys.exit(
                    "Emotion directions are DIFFERENCES between registers, so "
                    "at least two\nare needed and they must come from one "
                    "sitting:\n\n    realme voice registers")
            missing = [n for n, pth in have.items() if not Path(pth).is_file()]
            if missing:
                sys.exit(f"These registers are recorded in your profile but "
                         f"the audio is gone: {', '.join(missing)}. "
                         f"Re-run `realme voice registers`.")
            # Built the way a render builds it, from the profile's own
            # choice -- not from a name hardcoded here -- so the directions
            # are in the embedding space of the engine renders use.
            from realme.adapters.factory import build as _build
            spec = (prof.data.get("adapters", {}) or {}).get("tts", "gemini-tts")
            eng = _build(spec, quiet=True)
            eng.preflight()
            if not hasattr(eng, "set_expression"):
                sys.exit(
                    f"Your voice engine is {getattr(eng, 'name', spec)}, which "
                    f"speaks from a reference FILE.\nAn emotion direction is a "
                    f"vector, and only the in-process qwen3 engine takes one:\n"
                    f"\n    realme setup adapters --tts qwen3cpp")
            out = _dh() / "voice" / "registers" / "tau.json"
            print(f"Extracting speaker embeddings ({len(have)} takes) — this "
                  f"runs the speaker\nencoder once per register and is the "
                  f"whole cost of the method.\n")
            try:
                ts = TAU.extract(eng, have, log=print)
            except (ValueError, TypeError, KeyError) as e:
                sys.exit(str(e))
            TAU.save(ts, out)
            print()
            print(TAU.report(ts))
            print(f"\n  Saved to {out}")
            print("\n  Nothing renders differently yet: directions are "
                  "opt-in, and the lecture path\n  does not know they exist. "
                  "To hear them, and to check the engine actually\n  responds "
                  "to a shifted embedding:\n\n"
                  "    python 03_App\\explore_expressive.py --tau "
                  "--out runs\\tau1")
            return 0

        if a.action == "pace":
            from realme.core.env import data_home as _dh
            if getattr(a, "all", False):
                from realme.enrollment import pace as P
                prof = Profile(_dh() / "profile")
                ref = prof.data.get("voice_reference")
                if not ref or not Path(ref).is_file():
                    sys.exit("No reference enrolled yet: realme voice enroll <file>")
                print("Measuring your rate, and every offered draft voice")
                try:
                    r = P.calibrate_all(Path(ref), prof.reference_text,
                                        workdir=_dh() / "voice", log=print)
                except ValueError as e:
                    sys.exit(str(e))
                prof.update({"your_wpm": r["your_wpm"],
                             "draft_voice_wpm": r["voice_wpm"]})
                print("\n  Saved. Each voice is now matched to your pace "
                      "individually, so a\n  guest is not slowed by a "
                      "ratio measured against a different voice.")
                return 0
            from realme.core.env import data_home
            from realme.enrollment import pace as P
            prof = Profile(data_home() / "profile")
            ref = prof.data.get("voice_reference")
            if not ref or not Path(ref).is_file():
                sys.exit("No reference enrolled yet: realme voice enroll <file>")
            print("Matching the draft voice to your own pace")
            try:
                r = P.calibrate(Path(ref), prof.reference_text,
                                voice=a.voice or None,
                                workdir=data_home() / "voice", log=print)
            except ValueError as e:
                sys.exit(str(e))
            prof.update({"draft_speed": r["speed"], "draft_pace_measured": r})
            print(f"\n  Saved. Dialogues now run the draft voice at speed "
                  f"{r['speed']:.3f}, matching your own rate.")
            print("  Lecture and narration drafts keep the voice's own pace: "
                  "a draft is\n  replaced by your cloned voice before anything "
                  "is published, so slowing\n  it only lengthens the "
                  "iteration loop.")
            if r["clamped"]:
                print("  Note: the raw ratio was outside the sane range and "
                      "was clamped.\n  That usually means the enrolment "
                      "transcript does not match the audio.")
            print("  Override for one run with --piper-speed.")
            return 0

        if a.action == "enroll":
            from realme.core.env import data_home
            if not a.wav or not a.wav.exists():
                sys.exit("Point me at a recording: "
                         "realme voice enroll my_take.m4a --transcript \"...\"")
            prof = Profile(data_home() / "profile")
            kind = "consent_recording" if a.consent_take else "voice_reference"
            dst = prof.store_audio(a.wav, kind)
            print(f"Stored as {dst}")
            print(f"  converted from {a.wav.suffix or 'raw'} to 24 kHz mono WAV")
            transcript, tsource = "", ""
            src = a.transcript_file or a.transcript
            if src and not a.consent_take:
                try:
                    transcript, tsource = V.read_transcript(src)
                except OSError as e:
                    sys.exit(f"Could not read the transcript: {e}")
                if not transcript:
                    sys.exit(f"The transcript from {tsource} is empty.")
                dst = prof.set_reference_text(transcript)
                print(f"  transcript saved ({len(transcript.split())} words "
                      f"from {tsource})")
                print(f"    -> {dst}")
            if kind == "voice_reference" and prof.transcript_note:
                print(f"\n  {prof.transcript_note}")
            # Analyse it now, so a take that is too quiet or noisy is
            # reported at enrolment.
            r = V.analyze(dst)
            print(f"\n  {r.duration_s}s, {r.sample_rate} Hz, {r.channels}ch")
            print(f"  peak {r.peak_db} dBFS | SNR ~{r.snr_db} dB")
            print(f"  {r.verdict}")
            for pr in r.problems:
                print(f"    problem: {pr}")
            for ad in r.advice:
                print(f"    - {ad}")
            if not a.consent_take:
                for note in V.transcript_fit(transcript, r.duration_s):
                    print(f"    - {note}")
                if not transcript:
                    print("\n  Add it with either of these:")
                    print('    realme voice enroll <file> --transcript "what you said"')
                    print("    realme voice enroll <file> --transcript my_script.txt")
            return 0

        if a.action == "takes":
            cands = [c for c in ([a.wav] if a.wav else []) + list(a.also) if c]
            if len(cands) < 1:
                print("Point me at the takes:\n"
                      "  realme voice takes take1.wav --also take2.wav "
                      "--also take3.wav")
                return 1
            missing = [c for c in cands if not Path(c).is_file()]
            if missing:
                print("Not there: " + ", ".join(str(m) for m in missing))
                return 1
            V.compare_takes([Path(c) for c in cands])
            return 0

        if a.action == "gemini":
            return _voice_gemini(a)

        if a.action == "consent":
            print(V.consent_text(a.name, datetime.date.today().isoformat()))
            print("\nRead that aloud, in the same room and at the same distance "
                  "as your reference recording, and keep the file.")
            return 0
        if not a.wav or not a.wav.exists():
            sys.exit("Point me at a WAV: realme voice check my_reference.wav")

        if a.action in ("check", "analyze"):
            r = V.analyze(a.wav)
            print(f"{a.wav.name}: {r.duration_s}s, {r.sample_rate} Hz, "
                  f"{r.channels}ch")
            print(f"  peak {r.peak_db} dBFS | rms {r.rms_db} dB | "
                  f"noise floor {r.noise_floor_db} dB | SNR ~{r.snr_db} dB")
            # The same features `bench` scores on, so "check this clip" and
            # "score this clip" describe a file in the same terms, including
            # dynamic range.
            try:
                from realme.enrollment.voice import acoustic_signature
                sig = acoustic_signature(a.wav)
                if sig:
                    print(f"  pitch {sig.get('pitch_median_hz') or 0:.0f} Hz | "
                          f"variation {sig.get('pitch_variation_semitones') or 0:.1f} st | "
                          f"centroid {sig.get('spectral_centroid_hz') or 0:.0f} Hz | "
                          f"dynamic range {sig.get('dynamic_range_db') or 0:.1f} dB")
            except Exception:
                pass
            print(f"\n  {r.verdict}\n")
            for pr in r.problems:
                print(f"  problem: {pr}")
            for ad in r.advice:
                print(f"  - {ad}")
            return 0 if r.ok() else 1

        if a.action == "resemblance":
            from realme.enrollment.voice import acoustic_signature, compare_signatures
            if not a.against or not a.against.exists():
                sys.exit("Give both files: realme voice resemblance rendered.wav "
                         "--against reference.wav")
            r = compare_signatures(acoustic_signature(a.against),
                                   acoustic_signature(a.wav))
            print(f"resemblance: {r['score_percent']}%  "
                  f"({r['feature_count']} features)\n")
            for k, v in r["features"].items():
                print(f"  {k:26} {v['reference']:>10} -> {v['candidate']:>10}"
                      f"   {v['similarity']}")
            print(f"\n  {r['interpretation']}")
            return 0

        if a.action == "compare":
            outdir = a.out or Path("./voice_compare")
            made = V.compare(a.wav, outdir)
            print("Listen to these and pick one:\n")
            for name, path in made.items():
                print(f"  {name:9} {V.PRESETS[name]['label']}\n            {path}")
            print("\nThen:  realme voice bake %s --preset <name>" % a.wav)
            return 0

        if a.action == "polish":
            out = a.out or a.wav.with_name(a.wav.stem + f"_{a.preset}.wav")
            V.beautify(a.wav, out, a.preset)
            r = V.analyze(out)
            print(f"Wrote {out}")
            print(f"  peak {r.peak_db} dBFS | SNR ~{r.snr_db} dB | {r.verdict}")
            return 0

        # bake: polish and install as the profile's reference
        prof = Profile(Path(os.environ.get("REALME_HOME",
                                           Path.home() / "RealMeStudio")) / "profile")
        try:
            raw = prof.store_audio(a.wav, "voice_reference_raw")
        except MediaError as e:
            sys.exit(f"Could not read that recording:\n  {e}")
        baked = prof.root / "baked_assets" / "voice_reference.wav"
        V.beautify(raw, baked, a.preset)
        for what in getattr(V.treat, "last_applied", []):
            print(f"  - {what}")
        if a.preset == "off":
            # Untouched is the default, but list any repairs the take would
            # benefit from.
            wants = V.repairs_for(raw)
            if len(wants) > 1 or any("rumble" not in w for _, w in wants):
                print("  This recording could benefit from:")
                for _, why in wants:
                    print(f"    - {why}")
                print("  Try:  realme voice bake %s --preset clean" % raw)
        prof.update({"voice_reference": str(baked), "voice_preset": a.preset})
        r = V.analyze(baked)
        print(f"Baked with preset '{a.preset}' -> {baked}")
        print(f"  peak {r.peak_db} dBFS | SNR ~{r.snr_db} dB | {r.verdict}")
        print("\nThis is what the cloning backend will learn from. "
              "Raw original kept alongside it.")
        return 0

    if a.cmd == "key":
        from realme.core import env as _env
        if a.where or not a.name:
            print(f"New keys are written to:  {_env.data_home() / '.env'}\n")
            print("Looked for .env in (first match wins per key):")
            found = []
            for p in _env.candidates():
                mark = "[found]" if p.exists() else "       "
                print(f"  {mark} {p}")
                if p.exists():
                    found.append(p)
            if len(found) > 1:
                # Two files holding the same key is the confusing case: one is
                # being ignored and nothing on screen would say which.
                print("\n  Note: more than one .env exists. Earlier entries "
                      "win.\n  If a key looks stale, it is coming from the "
                      "first one listed.")
            print("\nKeys currently visible to RealMe:")
            for k in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                      "ELEVENLABS_API_KEY", "REALME_ELEVEN_VOICE_ID",
                      "REALME_GOOGLE_VOICE_KEY"):
                v = os.environ.get(k)
                print(f"  {k:26} {'set (' + v[:6] + '...)' if v else 'not set'}")
            # Not secrets, so shown in full, alongside the keys, to confirm
            # the .env settings were read.
            print("\nSettings from the same files:")
            from realme.adapters.script_writer import DEFAULT_MODEL as _dm
            for k, dflt in (("REALME_GEMINI_MODEL", _dm),
                            ("REALME_HOME", str(_env.data_home()))):
                v = os.environ.get(k)
                print(f"  {k:26} {v if v else f'not set (using {dflt})'}")
            if not a.name:
                print("\nTo store one:  realme key GEMINI_API_KEY your-key-here")
            return 0
        if not a.value:
            sys.exit("Give the value too:  realme key GEMINI_API_KEY your-key-here")
        path = _env.write_key(a.name, a.value)
        print(f"Stored {a.name} in {path}")
        print("It takes effect immediately - no need to reopen the terminal.")
        return 0

    if a.cmd == "migrate":
        from realme import migrate as M
        from realme.core.env import data_home
        from realme.core.paths import project_root
        if a.restore is not None:
            try:
                M.restore(None if a.restore.name == "__auto__"
                          else a.restore, data_home())
            except ValueError as e:
                sys.exit(str(e))
            print("\n  Now run: realme setup")
            return 0
        root = project_root()
        if root is None:
            sys.exit("Run this from inside the RealMe folder.")
        # The voice and the API key travel together or not at all.
        #
        # --no-voice means this package is for another person, who must not
        # receive the API key. Refuse rather than warn.
        if not a.voice and a.include_key:
            sys.exit(
                "--no-voice packages this for another person, and --include-key "
                "would put your\nAPI key in their hands. Pick one. They can set "
                "their own key in one command:\n\n    realme key "
                "GEMINI_API_KEY <their key>")
        groups = M.collect(root, data_home(), include_key=a.include_key,
                           lean=not a.full, include_voice=a.voice,
                           include_qwen=a.qwen)
        carries_qwen = bool(groups.get("tools/qwen3cpp"))
        pl = M.plan(groups)
        print(f"Packaging {root}")
        for r in pl["rows"]:
            print(f"  {r['group']:<18} {r['files']:>5} files  "
                  f"{r['bytes'] / 1e6:>8.1f} MB")
        print(f"  {'TOTAL':<18} {pl['files']:>5} files  "
              f"{pl['total_bytes'] / 1e9:>8.2f} GB uncompressed")
        print("")
        problems = 0
        for ok, msg in M.preflight(root, data_home(), groups,
                                   include_voice=a.voice):
            print(f"  [{'ok ' if ok else '!! '}] {msg}")
            problems += 0 if ok else 1
        if problems:
            print(f"\n  {problems} thing(s) to fix before this package is worth "
                  f"carrying.")
        # Empty on purpose is not "nothing found": a colleague's package has
        # no voice or scripts, and leaves the local clone out unless asked.
        expected_empty = set()
        if not a.voice:
            expected_empty |= {"voice", "scripts"}
        if not carries_qwen:
            expected_empty.add("tools/qwen3cpp")
        missing = [r["group"] for r in pl["rows"]
                   if r["files"] == 0 and r["group"] not in expected_empty]
        if missing:
            print(f"\n  ! nothing found for: {', '.join(missing)}")
        if not a.include_key:
            print("  (your .env is NOT included -- run `realme key` on the new "
                  "machine, or pass --include-key)")
        if not a.voice:
            print("  (your VOICE is NOT included -- this package is for another "
                  "person, and the\n   instructions inside tell them how to "
                  "record their own)")
        # Check for an illegal-instruction crash on another CPU:
        # build_info.json records what the binary was compiled for, so this
        # can say whether THIS package will run elsewhere.
        try:
            from realme.engines.cpp import status as _cpp_status
            built = _cpp_status().get("built_cpu", "unknown")
        except Exception:
            built = "unknown"
        print("")
        if not carries_qwen:
            pass                      # no engine binary in this package
        elif built in ("avx2", "baseline"):
            print(f"  C++ engine CPU target: {built} - this binary will run on "
                  f"another x86 machine.")
        else:
            print(f"  ! C++ engine CPU target: {built}. It was compiled for THIS "
                  f"processor and may\n    die with an illegal instruction on an "
                  f"older machine. Build for a portable\n    target first, then "
                  f"package again:\n"
                  f"      realme engine cpp --cpp-action build --force "
                  f"--target-cpu avx2")
        _result(report={"rows": pl["rows"], "files": pl["files"],
                        "total_bytes": pl["total_bytes"],
                        "includes_voice": bool(a.voice),
                        "includes_qwen": carries_qwen,
                        "includes_key": bool(a.include_key),
                        "engine_cpu_target": built},
                warnings=[m for ok, m in M.preflight(root, data_home(), groups,
                                                     include_voice=a.voice)
                          if not ok])
        # A folder given as --out gets the default name inside it; resolved
        # before the dry run returns, so the dry run says where it would go.
        try:
            out = M.resolve_out(a.out, root.parent,
                                M.PACKAGE_NAME[bool(a.voice)])
        except ValueError as e:
            sys.exit(str(e))
        print(f"\n  {'Would write' if a.dry_run else 'Writing'}: {out}")
        if a.dry_run:
            return 0
        print("")
        M.write_package(out, groups,
                        readme=(M.README if a.voice
                                else M.readme_fresh(carries_qwen)))
        # Read back from the finished archive, not from the plan that built
        # it, so the report states what the zip actually contains.
        leaked = M.personal_entries(out)
        if a.voice:
            _result(files={"package": out}, personal_files=len(leaked))
            print(f"\n  Contains your enrolled voice ({len(leaked)} personal "
                  f"files) -- this package is for\n  another machine of YOURS.")
            print("\n  Copy that to the new machine, unzip it, and read "
                  "README_MIGRATION.md inside.")
        elif leaked:
            sys.exit(f"\n  STOP. --no-voice was set, but the finished archive "
                     f"still contains:\n    "
                     + "\n    ".join(leaked[:8])
                     + f"\n\n  {out} has NOT been cleared for sending. This is "
                       f"a bug; report it\n  rather than deleting the entries "
                       f"by hand.")
        else:
            _result(files={"package": out}, checked_no_personal_files=True)
            print("\n  Checked the finished archive: no voice, no profile, no "
                  "key. Safe to send.")
            print(f"\n  Send {out.name}. They unzip it and read "
                  "README_MIGRATION.md inside.")
        return 0

    if a.cmd == "corpus":
        from realme.enrollment import corpus as C
        if a.action == "script":
            if not a.source or not Path(a.source).is_file():
                sys.exit("Point --from at a file of your own writing: lecture "
                         "notes, a paper, anything in your own words.")
            text = Path(a.source).read_text(encoding="utf-8", errors="ignore")
            lines = C.make_script(text, a.minutes)
            out = a.out or (C.corpus_home() / "recording_script.txt")
            C.write_script(lines, out)
            words = sum(len(s.split()) for s in lines)
            print(f"{len(lines)} lines, {words} words, about "
                  f"{words / C.WORDS_PER_MINUTE:.0f} minutes of speech")
            print(f"  {out}")
            print("\nRead it in one sitting if you can - one voice, one room, "
                  "one mic distance.")
            return 0
        if a.action == "add":
            if not a.audio or not a.text:
                sys.exit("Need both: realme corpus add TAKE.wav --text \"what you said\"")
            t = C.add_take(a.audio, a.text)
            print(f"[{t.id}] {t.duration_s:.1f}s  {t.verdict}")
            for p_ in t.problems:
                print(f"   ! {p_}")
            for w_ in t.warnings:
                print(f"   - {w_}")
            return 0
        if a.action == "split":
            if not a.audio or not a.script_file:
                sys.exit("Need both: realme corpus split SESSION.wav --script script.txt")
            script = C.read_script(a.script_file)
            takes = C.split_session(a.audio, script, silence_db=a.silence_db,
                                    min_silence=a.min_silence)
            if not takes:
                return 1
            bad = [t for t in takes if t.problems]
            warned = [t for t in takes if t.warnings and not t.problems]
            print(f"imported {len(takes)} takes: {len(takes) - len(bad)} usable, "
                  f"{len(bad)} to re-record")
            for t in bad[:10]:
                print(f"   ! [{t.id}] {t.problems[0]}")
            for t in warned[:5]:
                print(f"   - [{t.id}] {t.warnings[0]}")
            return 0
        if a.action == "status":
            st = C.status()
            print(f"corpus      : {st['home']}")
            print(f"takes       : {st['takes']}  ({st['usable']} usable)")
            print(f"speech      : {st['minutes']} minutes of usable audio")
            print(f"to re-record: {st['flagged']}")
            for tid, why in st["problems"]:
                print(f"   ! [{tid}] {why}")
            for tid, why in st["warnings"]:
                print(f"   - [{tid}] {why}")
            print(f"\n{st['verdict']}")
            return 0
        if a.action == "export":
            out = a.out or (C.corpus_home() / "export")
            try:
                r = C.export(out)
            except ValueError as e:
                sys.exit(str(e))
            print(f"{r['takes']} takes, {r['minutes']} minutes -> {r['dest']}")
            print("  metadata.csv, wavs/, README_TRAINING.md")
            return 0

    if a.cmd == "engine":
        from realme.engines import install as eng
        if a.action == "status":
            st = eng.status(model=a.model)
            from realme.core.paths import tools_dir, package_root
            legacy = package_root() / "tools"
            if Path(st["home"]).parent == legacy and legacy != tools_dir():
                print(f"tools dir   : {legacy}  (legacy location, still used)")
                print(f"              new installs would go to {tools_dir()}")
            else:
                print(f"tools dir   : {tools_dir()}")
            print(f"engine home : {st['home']}")
            print(f"runtime     : {'ok' if st['runtime_works'] else 'missing'}"
                  f"  {st['runtime']}")
            print(f"model       : {'ok' if st['model_present'] else 'missing'}"
                  f"  {st['model']}")
            d = st["draft"]
            print(f"draft voice : {'ok' if d['ready'] else 'missing'}"
                  f"  {d['voice']}"
                  + ("" if d["ready"] else
                     f"  (piper-tts {'ok' if d['code'] else 'missing'},"
                     f" voice {'ok' if d['voice_present'] else 'missing'})"))
            if st["ready"]:
                print("\nReady.")
                return 0
            # Show why, not just that. The install runs in a window that
            # closes, so print the tail of its log here.
            from realme.core.env import data_home
            logfile = data_home() / "engine-install.log"
            print("\nNot installed.")
            if logfile.is_file():
                lines = logfile.read_text(encoding="utf-8",
                                          errors="ignore").splitlines()
                start = 0
                for i, ln in enumerate(lines):
                    if ln.startswith("=== engine install"):
                        start = i
                tail = [ln for ln in lines[start:] if ln.strip()][-18:]
                if tail:
                    print(f"\nLast attempt ({logfile}):")
                    for ln in tail:
                        print(f"  {ln}")
            else:
                print("  No install has been attempted yet.")
            print("\nRun:  realme engine install --from \"<your old folder>\"")
            return 1
        if a.action == "cpp":
            from realme.engines import cpp as C
            if a.cpp_action == "status":
                st = C.status()
                print(f"home       : {st['home']}")
                print(f"built from : {'ok' if st['built_from_present'] else 'MISSING'}"
                      f"  {st['built_from']}  (RealMe's engine source)")
                print(f"original   : {'ok' if st['source_present'] else 'MISSING'}"
                      f"  {st['source']}  (kept as the record; built only "
                      f"with --source original)")
                print(f"ggml       : {'ok (vendored, with the original)' if st['ggml_present'] else 'MISSING'}")
                print(f"compiler   : {st['compiler']}  {st['compiler_detail']}")
                print(f"cmake      : {st['cmake'] or 'not found (installed on demand)'}")
                print(f"ninja      : {st['ninja'] or 'not found (installed on demand)'}")
                glslc = st.get("glslc") or "not found"
                nvcc = st.get("nvcc") or "not found"
                print(f"vulkan sdk : {glslc}  (glslc; only for --gpu vulkan)")
                print(f"cuda       : {nvcc}  (nvcc; only for --gpu cuda)")
                label = {"ok": "ok", "missing": "not built",
                         "blocked": "BLOCKED by Defender",
                         "appcontrol": "BLOCKED by Application Control",
                         "wrongcpu": "CRASHES (built for a CPU feature this "
                                     "machine lacks; rebuild with "
                                     "--target-cpu)",
                         "broken": "will not start"
                         }.get(st["binary_state"], st["binary_state"])
                print(f"binary     : {label}  {st['cli']}")
                if st["binary_state"] in ("blocked", "appcontrol"):
                    from realme.engines import cpp as _C
                    print()
                    print(f"  {st['binary_detail']}")
                    print()
                    print(_C.DEFENDER_ADVICE.format(build=Path(st["cli"]).parent)
                          if st["binary_state"] == "blocked"
                          else _C.APPCONTROL_ADVICE)
                print(f"weights    : {'ok' if st['weights_ready'] else 'not converted'}"
                      f"  {st['weights']}")
                # The distinction that decides whether the model is loaded once
                # or once per utterance.
                print(f"GPU present: {st['gpu_present']['name'] if st['gpu_present'] else 'none detected'}")
                from realme.core.cpu import describe as _cpu
                c = _cpu()
                print(f"this CPU   : {c['name']}  [{c['arch']}]")
                print(f"built for  : {st['built_gpu']} (CPU target: "
                      f"{st.get('built_cpu', '?')})")
                if st.get("built_cpu") in ("native", "native (assumed)") \
                        and c["x86"]:
                    print("             native = this processor only; "
                          "--target-cpu avx2 travels")
                if st["built_gpu"] in ("off", "unknown (built before this was recorded)") \
                        and st["gpu_recommended"] != "off":
                    print(f"             ! a GPU backend is available and NOT "
                          f"compiled in.")
                    print(f"             Rebuild:  realme engine cpp "
                          f"--cpp-action build --force --gpu {st['gpu_recommended']}")
                elif st["gpu_present"] and st["built_gpu"] == "off":
                    for ln in st["gpu_advice"].splitlines():
                        print(f"             {ln}")
                print(f"in-process : {'ok' if st['in_process'] else 'NO'}  "
                      f"{st['in_process_detail']}")
                if st["in_process"]:
                    print("             `qwen3cpp` loads the model once and "
                          "keeps it open.")
                else:
                    print("             `qwen3cpp` cannot run in-process; only "
                          "`qwen3cpp-cli` will work,")
                    print("             and that reloads 2.2 GB per utterance.")
                if st["ready"]:
                    print("\nReady.")
                elif st["can_build"]:
                    print("\nEverything needed to build is present. Run:  "
                          "realme engine cpp")
                else:
                    print(f"\nCannot build yet:\n  {st['compiler_detail']}")
                return 0 if st["ready"] else 1
            if a.cpp_action == "selftest":
                return 0 if C.selftest() else 1
            if a.cpp_action == "export-server":
                if not a.export_to:
                    sys.exit("Say where:  --export-to C:\\path\\to\\new-folder")
                voice = None
                if a.with_my_voice:
                    from realme.app.profile import Profile
                    from realme.core.env import data_home
                    ref = Profile(data_home() / "profile").data.get("voice_reference")
                    if not ref or not Path(ref).is_file():
                        sys.exit("No voice recording in your profile yet "
                                 "(Twin Setup -> Your voice).")
                    voice = Path(ref)
                try:
                    C.export_server_bundle(a.export_to, voice_wav=voice)
                except Exception as e:
                    sys.exit(f"\n{e}")
                return 0
            if a.cpp_action == "integrate":
                try:
                    C.integrate()
                except Exception as e:
                    sys.exit(f"\n{e}")
                print("Rebuild the voice engine to use it (_Rebuild_Voice_Engine.bat).")
                return 0
            if a.cpp_action == "build" and a.engine_source == "original":
                try:
                    C.build(force=True, gpu="off", portable=a.portable,
                            target_cpu=a.target_cpu, source="original")
                except Exception as e:
                    sys.exit(f"\n{e}")
                return 0
            if a.cpp_action == "export-release":
                if not a.export_to:
                    sys.exit("Say where:  --export-to C:\\path\\to\\new-folder")
                try:
                    C.export_release(a.export_to)
                except Exception as e:
                    sys.exit(f"\n{e}")
                return 0
            if a.cpp_action == "export-source":
                if not a.export_to:
                    sys.exit("Say where:  --export-to C:\\path\\to\\new-folder")
                try:
                    C.export_source(a.export_to)
                except Exception as e:
                    sys.exit(f"\n{e}")
                return 0
            if a.cpp_action == "quantize":
                try:
                    out = C.make_q8(force=a.force)
                except Exception as e:
                    sys.exit(f"\n{e}")
                print(f"\nThe 8-bit model is in {out}. Lectures use it unless "
                      f"told otherwise; the 16-bit one again after:\n"
                      f"    realme key REALME_QWEN3CPP_QUANT f16   (or Twin "
                      f"Setup)")
                return 0
            if a.cpp_action in ("rebuild", "rebuild-all"):
                try:
                    if a.cpp_action == "rebuild-all":
                        C.rebuild_all()
                    else:
                        C.rebuild()
                except Exception as e:
                    sys.exit(f"\n{e}\n\nThe working engine was kept.")
                from realme.adapters.tts_qwen3_dll import default_threads
                print(f"\nRebuilt. The engine now uses {default_threads()} "
                      f"threads on this computer.")
                C.ensure_q8()
                return 0 if C.status()["ready"] else 1
            try:
                if a.cpp_action in ("build", "all"):
                    C.build(force=a.force, gpu=a.gpu, portable=a.portable,
                            target_cpu=a.target_cpu)
                if a.cpp_action in ("convert", "all"):
                    C.convert(force=a.force, quant=a.quant)
            except Exception as e:
                sys.exit(str(e))
            st = C.status()
            if st["ready"]:
                C.ensure_q8()
            print("Ready." if st["ready"] else "Something is still missing.")
            return 0 if st["ready"] else 1

        if a.action == "retire":
            if a.drop_python:
                return 0 if eng.drop_python(delete=a.delete)["deleted"] or \
                    not a.delete else 1
            from realme.engines import retire as ret
            folder = (Path(a.targets[0]) if a.targets else None) or a.source
            if not folder:
                sys.exit("Which folder? realme engine retire <old folder>")
            pl = ret.plan(folder, model=a.model)
            if not a.delete:
                ret.report(pl)
                return 0 if pl.ready else 1
            try:
                ret.delete(pl, a.archive_to)
            except Exception as e:
                sys.exit(str(e))
            return 0
        # Which engines were asked for. Aliases because the same thing has a
        # user-facing name ("the draft voice"), an engine name ("piper") and a
        # role ("clone"), and each maps to the same install.
        ALIASES = {
            "draft": "draft", "piper": "draft", "drafts": "draft",
            "clone": "clone", "qwen3cpp": "clone", "cpp": "clone",
            "qwen3": "clone", "voice": "clone",
            "all": "all",
        }
        names = [n.strip().lower() for n in (a.targets or [])]
        unknown = [n for n in names if n not in ALIASES]
        if unknown:
            sys.exit(f"Unknown engine name(s): {', '.join(unknown)}\n"
                     f"    draft      the piper voices (Ryan, Amy and Cori, "
                     f"~120 MB, no build)\n"
                     f"    clone      your own voice: weights, GGUF "
                     f"conversion, C++ build\n"
                     f"    all        the same as no name\n"
                     f"  With no name: both defaults -- piper for drafting, "
                     f"qwen3cpp for your own voice.")
        want = {ALIASES[n] for n in names}
        if "all" in want or not want:
            # The two engines this system actually runs on: piper for drafting
            # and qwen3cpp for the instructor's own voice. Bare `install`
            # includes the C++ build, so the main engine is usable afterwards.
            want = {"draft", "clone"}

        if want == {"draft"}:
            # Just the piper voices: ~120 MB and a pip install, without the
            # full installer's 2.5 GB of model weights.
            st = eng.install_draft_voices(eng.engine_home(), log=print)
            ok = all(v.get("ready") for v in st.values())
            print("\nDraft voices ready." if ok else
                  "\nSomething is missing -- realme doctor says what.")
            return 0 if ok else 1

        build_cpp = "clone" in want
        # Tee to a log. This install is long and its console window closes,
        # so the log keeps any failure readable afterwards.
        from realme.core.env import data_home
        logfile = data_home() / "engine-install.log"
        logfile.parent.mkdir(parents=True, exist_ok=True)
        with open(logfile, "a", encoding="utf-8") as fh:
            import datetime
            fh.write(f"\n=== engine install {datetime.datetime.now():%Y-%m-%d %H:%M} ===\n")

            def log(msg=""):
                print(msg)
                fh.write(str(msg) + "\n")
                fh.flush()

            try:
                st = eng.install(source=a.source, model=a.model, force=a.force,
                                 lean=not a.full, log=log)
            except Exception as e:
                log(f"Install failed: {e}")
                print(f"\nFull log: {logfile}")
                return 1
        if not st["ready"]:
            print(f"\nFull log: {logfile}")
            return 1
        if build_cpp:
            # The remaining half. `install` lays down the runtime, the weights
            # and the draft voices; the C++ engine still has to be converted
            # and compiled.
            from realme.engines import cpp as C
            print("\n[4/5] Converting the weights to GGUF")
            try:
                C.convert(force=a.force, quant=a.quant)
            except Exception as e:
                print(f"  convert failed: {e}")
                return 1
            print("\n[5/5] Building the C++ engine for this machine")
            try:
                C.build(force=a.force, gpu=a.gpu, portable=a.portable,
                        target_cpu=a.target_cpu)
            except Exception as e:
                print(f"  build failed: {e}")
                print("  The draft voice and the weights are installed; only "
                      "the compile failed.")
                return 1
            C.ensure_q8()                 # the model lectures use by default
            print()
            ok = C.selftest()
            print("\nThe cloning engine is ready." if ok else
                  "\nBuilt, but the self-test did not pass.")
            print("  realme doctor            every engine's state")
            return 0 if ok else 1
        return 0

    if a.cmd == "gemini-parallel":
        from realme import gemparallel as GP
        from realme.adapters.base import AdapterUnavailable
        from realme.core.env import data_home
        if a.use:
            return GP.choose(a.use)
        try:
            levels = [int(x) for x in a.levels.replace(" ", "").split(",") if x]
            if not levels or levels[0] != 1 or any(l < 1 or l > 8 for l in levels):
                raise ValueError
        except ValueError:
            sys.exit("--levels starts at 1 and goes up to 8, like 1,2,3,4")
        try:
            return GP.main_cli(levels, a.per_level or None, a.yes,
                               a.out or (data_home() / "bench" / "gemini_parallel"))
        except AdapterUnavailable as e:
            sys.exit(f"\n{e}")

    if a.cmd == "engine-timing":
        from realme import bench as B
        from realme.core.env import data_home
        from realme.engines import cpp_timing as T
        try:
            threads = [int(x) for x in a.threads.replace(" ", "").split(",") if x]
            if not threads or any(t < 1 or t > 64 for t in threads):
                raise ValueError
        except ValueError:
            sys.exit("--threads takes numbers between 1 and 64, like 4,6,8,12")
        ref = None
        if not a.build_only:
            ref, label = B.resolve_reference(a.reference)
            if ref is not None and Path(ref).is_file():
                from realme.core.media import ensure_reference_wav
                ref = ensure_reference_wav(Path(ref))
            print(f"  reference: {ref}  [{label}]")
        try:
            if a.compare and not a.build_only:
                from realme.adapters.tts_qwen3_dll import default_threads
                try:
                    T.pick_variants(a.variants.split(",") if a.variants else None)
                except ValueError as e:
                    sys.exit(f"--variants: {e}")
                return T.compare_cli(default_threads(), max(4, a.sentences),
                                     a.rebuild, a.out or (data_home() / "bench"
                                                          / "timing"), ref,
                                     variants=(a.variants.split(",")
                                               if a.variants else None))
            return T.main_cli(threads, max(1, a.sentences), a.rebuild,
                              a.out or (data_home() / "bench" / "timing"), ref,
                              build_only=a.build_only)
        except RuntimeError as e:
            print(f"\n  {e}", file=sys.stderr)
            return 1

    if a.cmd == "bench-batch":
        from realme import batchbench as BB, bench as B
        from realme.adapters.base import AdapterUnavailable
        from realme.core.env import data_home
        try:
            sizes = [int(x) for x in a.sizes.replace(" ", "").split(",") if x]
            if not sizes or sizes[0] != 1 or any(x < 1 or x > 16 for x in sizes):
                raise ValueError
        except ValueError:
            sys.exit("--sizes starts at 1 and goes up to 16, like 1,2,4")
        ref, label = B.resolve_reference(a.reference)
        if ref is not None and Path(ref).is_file():
            from realme.core.media import ensure_reference_wav
            ref = ensure_reference_wav(Path(ref))
        print(f"  reference: {ref}  [{label}]")
        try:
            return BB.main_cli(sizes, max(2, a.sentences),
                               a.out or (data_home() / "bench" / "batch"), ref,
                               quant=a.quant, save=not a.no_save)
        except (AdapterUnavailable, RuntimeError) as e:
            print(f"\n  {e}", file=sys.stderr)
            return 1

    if a.cmd == "bench-parallel":
        from realme import bench as B, parbench as PB
        from realme.core.cpu import physical_cores
        from realme.core.env import data_home
        try:
            layouts = (PB.parse_layouts(a.layouts) if a.layouts
                       else PB.default_layouts(os.cpu_count() or 4,
                                               physical_cores()))
            units, _ms, speeds = B.load_units(a.script, mode="natural")
        except (ValueError, OSError) as e:
            sys.exit(str(e))
        lines = list(zip(units, speeds or [1.0] * len(units)))
        batch = max(1, min(16, a.batch))
        want = a.lines
        if batch > 1:
            want = max(want, 2 * batch * max(w for w, _ in layouts))
        while len(lines) < want:               # the built-in four, repeated
            lines += lines[:want - len(lines)]
        lines = lines[:max(1, want)]
        ref = None
        if a.engine != "selftest":
            ref, label = B.resolve_reference(a.reference)
            if ref is not None and Path(ref).is_file():
                from realme.core.media import ensure_reference_wav
                ref = ensure_reference_wav(Path(ref))
            print(f"  reference: {ref}  [{label}]")
            why = PB.can_start(a.engine, ref)
            if why:
                print(f"\n  {a.engine} cannot run on this computer, so there "
                      f"is nothing to measure here.\n\n  {why}\n\n  Run "
                      f"this on a computer where the Studio's free local clone "
                      f"works (Twin Setup -> Environment says so). Nothing "
                      f"was changed.", file=sys.stderr)
                return 1
        est = PB.estimate_s(lines, layouts, max(1, a.repeat))
        print(f"  layouts (workers x threads): "
              + ", ".join(f"{w}x{t}" for w, t in layouts))
        print(f"  this takes roughly {est / 60:.0f} minutes; leave the "
              f"computer alone while it runs.")
        if not a.yes and a.engine != "selftest":
            if input("  Continue? [y/N] ").strip().lower() not in ("y", "yes"):
                return 0
        out = a.out or (data_home() / "bench" / "parallel")
        r = PB.main(a.engine, lines, layouts, repeat=max(1, a.repeat),
                    outdir=out, reference=ref, batch=batch)
        return 0 if any(l["runs"] for l in r["layouts"]) else 1

    if a.cmd == "bench":
        from realme import bench as B
        from realme.core.env import data_home
        try:
            script, control_ms, speeds = B.load_units(
                a.script, minutes=a.minutes, mode=a.mode)
        except OSError as e:
            # Windows answers `open("...")` with "Permission denied", not "no
            # such file", so the raw errno reads like a locked file rather
            # than an invalid path. Show what was given and how it resolved.
            given = Path(a.script) if a.script else None
            sys.exit(f"Cannot read the script file.\n"
                     f"  you gave  : {a.script}\n"
                     f"  resolved  : {given.resolve() if given else '-'}\n"
                     f"  working in: {Path.cwd()}\n"
                     f"  ({e.__class__.__name__}: {e})\n"
                     f"  Give a path to a text file, or leave --script off to "
                     f"use the built-in sentences.")
        except ValueError as e:
            sys.exit(str(e))
        if a.lines:
            script = script[:a.lines]
            control_ms = control_ms[:a.lines]
            speeds = speeds[:a.lines]
        # `tuned` (the default) is the take adjusted in the Studio; `raw` is the
        # recording it was made from; anything else is a path.
        ref, ref_label = B.resolve_reference(a.reference)
        outdir = a.out or (data_home() / "bench")
        # Convert once, here, so the adapters, the pause pattern and the
        # scoring all see the same readable file.
        if ref is not None and Path(ref).is_file():
            from realme.core.media import ensure_reference_wav
            try:
                ref = ensure_reference_wav(Path(ref))
            except Exception as e:
                sys.exit(f"Could not read {ref}: {e}")
        if ref is not None and getattr(a, "ref_seconds", 0):
            from realme.core.media import trim_reference
            try:
                ref = trim_reference(Path(ref), a.ref_seconds)
                ref_label += f", first {a.ref_seconds:g}s"
            except Exception as e:
                sys.exit(f"Could not trim the reference: {e}")
        if a.rescore:
            try:
                print(f"  reference: {ref}  [{ref_label}]")
                B.rescore(outdir, ref)
            except ValueError as e:
                sys.exit(str(e))
            return 0
        engines = [n.strip() for n in a.engines.split(",") if n.strip()]
        est = B.estimate(script, engines)
        print(f"Benchmarking {len(engines)} engine(s): {', '.join(engines)}")
        print(f"  {est['utterances']} utterances, {est['words']} words, "
              f"about {est['audio_minutes']:.1f} min of narration each")
        # Say where the words came from: the built-in script, or the file
        # given with --script.
        where = str(a.script) if a.script else \
            "built-in epidemiology sentences (bench.DEFAULT_SCRIPT)"
        print(f"  script   : {where}")
        print(f"  split    : {a.mode}")
        print(f"  reference: {ref or '-'}  [{ref_label}]")
        print(f"  output   : {outdir}")
        # State the worst-case run time before starting; at these real-time
        # factors a long script across several engines takes a while.
        print(f"\n  This could take up to ~{est['worst_case_minutes']:.0f} "
              f"minutes. Use --minutes 1 for a quick read first.")
        if not a.yes and sys.stdin.isatty():
            try:
                if input("  Continue? [y/N] ").strip().lower() not in ("y", "yes"):
                    print("  Cancelled.")
                    return 0
            except (EOFError, KeyboardInterrupt):
                print("\n  Cancelled.")
                return 0
        print()
        repeat = max(1, int(getattr(a, "repeat", 1) or 1))
        runs, stopped = [], False
        for k in range(repeat):
            # Each repetition gets its own folder, so the audio can be listened
            # to run by run instead of the last one overwriting the rest.
            rdir = outdir if repeat == 1 else outdir / f"run{k + 1:02d}"
            if repeat > 1:
                print(f"  --- run {k + 1} of {repeat} -> {rdir}")
            results = []
            for n in engines:
                try:
                    results.append(B.run_engine(n, script, rdir, ref,
                                                threads=a.threads,
                                                speeds=speeds))
                except KeyboardInterrupt:
                    # Keep whatever finished and report the engines that
                    # completed before the interrupt.
                    print("\n  Interrupted - reporting what finished.")
                    stopped = True
                    break
            B.report(results, rdir, reference=ref, script_path=a.script,
                     mode=a.mode, ref_label=ref_label, control_ms=control_ms,
                     brief=(repeat > 1))
            print(f"  results  : {B.write_results(results, rdir, ref)}")
            runs.append(results)
            if stopped:
                break
        if len(runs) > 1:
            B.variance_report(runs, reference=ref)
        return 0 if any(r.ok for run in runs for r in run) else 1

    if a.cmd == "split":
        from realme.pipeline import split as SPL
        if not a.video.is_file():
            sys.exit(f"No such file: {a.video}")
        seg = a.segments or a.video.with_name(a.video.stem + "_segments.json")
        if not Path(seg).is_file():
            sys.exit(f"No cut list at {seg}.\n"
                     f"Every render writes <name>_segments.json beside the "
                     f"video; without it there is nothing that says where a "
                     f"slide begins.")
        try:
            cuts, meta = SPL.load_cuts(Path(seg))
        except ValueError as e:
            sys.exit(str(e))

        if not a.at.strip():
            # No split points: show what there is to split, and stop. The
            # author chooses the boundaries.
            print(f"{a.video.name}: {len(cuts)} slides, "
                  f"{meta['total_s'] / 60:.1f} minutes\n")
            print(f"  {'slide':>5}  {'starts':>9}  words")
            for c in cuts:
                m, sec = divmod(c["start_s"], 60)
                print(f"  {c['slide']:>5}  {int(m):>4}:{sec:05.2f}  {c['words']:>5}")
            print("\n  Choose where each part begins, then:")
            print(f"    realme split \"{a.video.name}\" --at 9,21 "
                  f"--titles \"Foundations|Bias|Design\"")
            return 0

        try:
            starts = [int(x) for x in a.at.replace(" ", "").split(",") if x]
        except ValueError:
            sys.exit(f"--at wants slide numbers, comma separated; got {a.at!r}")
        titles = [t.strip() for t in a.titles.split("|")] if a.titles else []
        try:
            plan = SPL.plan(cuts, starts, titles, stem=a.video.stem,
                            total_s=meta.get("total_s"))
        except ValueError as e:
            sys.exit(str(e))
        outdir = a.out or a.video.with_name(a.video.stem + "_parts")
        srt = a.srt or a.video.with_suffix(".srt")
        print(f"{plan.summary()}")
        SPL.execute(a.video, plan, outdir, mode=a.mode, cuts=cuts, render=meta,
                    srt=srt if Path(srt).is_file() else None, log=print)
        if a.projects:
            # The source project is the folder the video sits in: that is where
            # its slides and its narration are.
            SPL.to_projects(a.video, plan, outdir, a.video.parent, outdir,
                            log=print)
            try:
                outdir.rmdir()
            except OSError:
                pass
        print()
        for part in plan.parts:
            print(f"  {part.filename}  ({part.measured_s or 0:.1f}s, "
                  f"slides {part.first_slide}-{part.last_slide}, {part.how})"
                  + (f"  ! {part.note}" if part.note else ""))
        print(f"\n  {outdir}")
        return 0

    if a.cmd == "merge":
        from realme.pipeline import merge as MRG
        try:
            plan = MRG.plan(a.videos)
        except (ValueError, OSError) as e:
            sys.exit(str(e))
        print(plan.summary())
        for i in plan.inputs:
            print(f"  {i.path.name:<44} {i.seconds:>8.2f}s"
                  + (f"   ! {i.note}" if i.note else ""))
        if not plan.compatible:
            print("\n  differences:")
            for r in plan.reasons:
                print(f"    {r}")
        if a.check:
            return 0
        print()
        MRG.execute(plan, a.out, mode=a.mode, log=print)
        print(f"\n  {a.out}  ({plan.measured_s or 0:.1f}s, {plan.how})"
              + (f"\n  ! {plan.note}" if plan.note else ""))
        return 0

    if a.cmd == "splice":
        from realme.pipeline import splice as SP
        from realme.pipeline import slides as slides_mod
        for f in (a.video, a.segments, a.deck):
            if not f.is_file():
                sys.exit(f"No such file: {f}")
        a.out.mkdir(parents=True, exist_ok=True)
        print(f"Comparing {a.deck.name} against {a.video.name}")
        try:
            pdf = slides_mod.deck_to_pdf(a.deck, a.out)
            pages = slides_mod.rasterize(pdf, a.out / "slides")
            forced = [int(x) for x in a.rerecord.replace(" ", "").split(",") if x]
            r = SP.plan_from_video(a.video, a.segments, pages,
                                   also_rerecord=forced,
                                   workdir=a.out / "_work", log=print)
        except (ValueError, OSError) as e:
            sys.exit(str(e))
        print()
        for c_ in r["plan"].changes:
            where = (f"reuse {r['reuse'][c_.slide_number][0]:.1f}-"
                     f"{r['reuse'][c_.slide_number][1]:.1f}s"
                     if c_.slide_number in r["reuse"] else "RE-RECORD")
            print(f"  slide {c_.slide_number:>3}  {c_.status:<8} {where}"
                  + (f"   ({c_.note})" if c_.note else ""))
        if r["plan"].removed:
            print(f"\n  gone from the old deck: "
                  f"{', '.join(map(str, r['plan'].removed))}")
        n = len(r["to_render"])
        print(f"\n  {len(r['reuse'])} of {len(pages)} slides reusable, "
              f"{n} to re-record.")
        if n:
            print(f"  Write narration for slide(s) "
                  f"{', '.join(map(str, r['to_render']))}, then render them "
                  f"with the lecture\n  pipeline at "
                  f"{r['render'].get('width', '?')}x"
                  f"{r['render'].get('height', '?')} "
                  f"{r['render'].get('fps', '?')}fps to match.")
        print("\n  A small text edit is INSIDE the noise of video "
              "compression and is not\n  detected. Name those slides with "
              "--rerecord.")
        return 0

    if a.cmd == "blocked":
        from realme.security import report as security_report
        return security_report()

    if a.cmd == "setup":
        from realme.setup_check import report
        return report(a.voice, install=a.install)

    if a.cmd == "doctor":
        from realme.adapters.tts import REGISTRY
        from realme.adapters.script_writer import GeminiScriptWriter
        from realme.adapters.base import AdapterUnavailable
        from realme.core.media import require, MediaError
        from realme.pipeline import pdfdoc
        required = ["ffmpeg", "ffprobe"]
        optional = {"soffice": "only for .pptx decks; PDF decks need nothing",
                    "espeak-ng": "only needed for the lexicon's word/letters "
                                 "check"}
        if not pdfdoc.HAVE_MUPDF:
            required.append("pdftoppm")
        else:
            optional["pdftoppm"] = "not needed - PyMuPDF handles rasterization"
        print("Binaries:")
        for b in required:
            try:
                print(f"  [ok]   {b}: {require(b)}")
            except MediaError:
                print(f"  [MISS] {b}  <-- REQUIRED")
        for b, why in optional.items():
            try:
                require(b); print(f"  [ok]   {b}")
            except MediaError:
                print(f"  [--]   {b}: absent - {why}")
        from realme.pipeline import pdfdoc
        print(f"PDF backend: {pdfdoc.backend()}")
        print("Script writers:")
        from realme.adapters.script_writer_more import WRITERS as _MORE
        for nm, w in [("gemini", GeminiScriptWriter()),
                      *((k, c()) for k, c in _MORE.items())]:
            try:
                w.preflight(); print(f"  [ok]   {nm}")
            except AdapterUnavailable as e:
                print(f"  [MISS] {nm}: {e}")
        # Through the FACTORY, with the enrolled reference -- not `cls()`.
        #
        # Constructing adapters any other way would test a path no render
        # uses: a bare `cls()` has no reference clip, and a fault in the
        # factory would not show up here.
        from realme.adapters.factory import build as _build
        print("Voices (built the way a render builds them):")
        for nm in REGISTRY:
            try:
                eng = _build(nm, quiet=True)   # the column below says it
                eng.preflight()
                clip = getattr(eng, "reference_wav", None)
                how = (f"cloning from {Path(clip).name}" if clip
                       else "own voice" if not getattr(eng, "is_voice_clone", False)
                       else "NO REFERENCE - would not be your voice")
                print(f"  [ok]   {nm:16} {how}")
            except Exception as e:
                print(f"  [MISS] {nm:16} {str(e).splitlines()[0][:80]}")
        return 0

    if a.cmd == "lecture":
        from realme.pipeline.lecture import build_lecture
        r = build_lecture(a.deck, a.outdir, _writer(a.script, a), _tts(a.tts, a),
                          style=a.style, course_context=a.context, layout=a.layout,
                          width=a.width, height=a.height, signalling=a.signalling,
                          mode=a.prosody_mode,
                          pause_scale=(a.pause_scale if a.pause_scale is not None
                                       else _pause_scale()),
                          check_audio=not a.no_audio_check)
        print(json.dumps(r["report"], indent=2))
        _result(report=r["report"],
                warnings=r["report"].get("warnings", []),
                files={k: r[k] for k in ("master", "srt", "vtt", "chapters",
                                         "manifest") if r.get(k)})
        return 0 if r["report"]["verdict"].startswith("PASS") else 1

    if a.cmd == "dialogue":
        from realme.pipeline.dialogue import build_dialogue
        # The guest defaults to the draft voice, never to the instructor's
        # engine (`a.guest_tts or a.tts` would make `--tts qwen3cpp` clone
        # the instructor twice), and the female-guest default applies only
        # to piper. A dialogue needs two distinct voices.
        from realme.pipeline import casting
        guest_spec = a.guest_tts or "piper"
        if a.guest_voice and casting.split_spec(guest_spec)[1] is None:
            guest_spec = f"{guest_spec}:{a.guest_voice}"
        voices = casting.cast(a.tts, guest_spec, speed=a.speed)
        # Both speakers named before anything is rendered. Two identical lines
        # here means one voice playing both parts.
        for role, spec in (("instructor", a.tts), ("guest", guest_spec)):
            print(f"  {role:<11}: {casting.describe(spec, voices[role])}")
        r = build_dialogue(a.source, a.topic, a.outdir, _writer(a.script, a), voices,
                           mode=a.mode, turns=a.turns, turn_gap=a.turn_gap,
                           host_name=(a.host_name or _display_name()),
                           guest_name=casting.speaker_name(guest_spec,
                                                           voices["guest"]),
                           bookends=a.bookends)
        print(json.dumps(r["report"], indent=2))
        return 0 if r["report"]["verdict"].startswith("PASS") else 1

    if a.cmd == "studio":
        from realme.app.server import serve, is_loopback
        if not is_loopback(a.host) and not a.allow_network:
            # Refuse before anything is printed or opened, so no URL is shown
            # for a server that will not start.
            print(f"Refusing --host {a.host}: the Studio has no login, so "
                  f"that would put your lectures, cloned voice and API keys "
                  f"on the network. Leave --host at 127.0.0.1, or add "
                  f"--allow-network if you mean it.", file=sys.stderr)
            return 2
        if a.colab:
            try:
                from google.colab import output
                output.serve_kernel_port_as_window(a.port)
                print(f"Opening RealMe Studio on port {a.port} via the Colab port proxy.")
            except ImportError:
                print("Not running in Colab; falling back to localhost.")
        else:
            print(f"RealMe Studio -> http://{a.host}:{a.port}")
        serve(host=a.host, port=a.port, allow_network=a.allow_network)
        return 0

    if a.cmd == "lexicon" and a.action in ("add", "remove", "where"):
        from realme.text.lexicon import (Entry, load_user_entries,
                                         save_user_entries, user_lexicon_path)
        if a.action == "where":
            p = user_lexicon_path()
            print(f"  {p}  [{'present' if p.is_file() else 'not created yet'}]")
            print(f"  {len(load_user_entries())} entry(ies) of your own")
            return 0
        if not a.term:
            sys.exit(f"Which term?  realme lexicon {a.action} NHANES "
                     f"--respelling \"en-haynes\"")
        entries = [e for e in load_user_entries() if e.term != a.term]
        if a.action == "remove":
            save_user_entries(entries)
            print(f"Removed {a.term}. The built-in entry, if there is one, "
                  f"applies again.")
            return 0
        if not (a.respelling or a.ipa or a.arpabet or a.no_spell):
            sys.exit(
                "Say how it should sound, or say it needs nothing:\n"
                "  --respelling \"en-haynes\"   ordinary letters, every engine\n"
                "  --ipa ...                   for engines that take phonemes\n"
                "  --no-spell                  just stop it being spelled out")
        entries.append(Entry(term=a.term, ipa=a.ipa, arpabet=a.arpabet,
                             respelling=a.respelling, note=a.note,
                             case_sensitive=True))
        path = save_user_entries(entries)
        print(f"Saved {a.term} -> "
              f"{a.respelling or a.ipa or a.arpabet or '(not spelled out)'}")
        print(f"  {path}")
        print("  This file survives updates; the built-in list does not.")
        return 0

    if a.cmd == "check-audio":
        from realme.text.lexicon import Lexicon
        from realme.verify.acoustic import AcousticVerifier, summarize
        av = AcousticVerifier(_tts(a.tts, a), Lexicon(), Path("./_templates"))
        res = av.check_utterance(a.wav, a.terms, label=a.wav.name)
        for r in res:
            print(f"{r.term:12} intended={r.intended_distance:.3f} "
                  f"naive={r.wrong_distance:.3f} margin={r.margin:.3f}  {r.verdict}")
        s = summarize(res)
        print("\n" + s["verdict"])
        return 0 if not s["mispronounced"] else 1

    if a.cmd == "import-script":
        from realme.pipeline.script_import import (parse_slide_script, to_manifest,
                                                   ScriptImportError, looks_like_json,
                                                   manifest_from_json)
        raw = _TXT(a.path)
        if looks_like_json(raw, a.path.name):
            # A RealMe narration file: cue words, pace and pauses come too.
            try:
                m, n = manifest_from_json(raw, Path(a.project).name if a.project else "")
            except ScriptImportError as e:
                sys.exit(str(e))
            print(f"{n} slide{'s' if n != 1 else ''} (a RealMe narration file)")
            if a.project:
                out = Path(a.project) / "_work" / "manifest.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(m.model_dump_json(indent=2), encoding="utf-8")
                print(f"\nWrote {out}\n  now: realme lecture <deck> -o {a.project} "
                      f"--script imported")
            return 0
        try:
            imported = parse_slide_script(raw)
        except ScriptImportError as e:
            sys.exit(str(e))
        print(f"{len(imported.sections)} slides ({imported.marker_format} markers)")
        for sec in imported.sections:
            words = len(sec.script.split())
            print(f"  slide {sec.slide_number}: {sec.title or '(untitled)':30} "
                  f"{words:4} words")
        for w in imported.warnings:
            print(f"  ! {w}")
        if a.project:
            out = Path(a.project) / "_work" / "manifest.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(to_manifest(imported, Path(a.project).name)
                           .model_dump_json(indent=2), encoding="utf-8")
            print(f"\nWrote {out}\n  now: realme lecture <deck> -o {a.project} "
                  f"--script imported")
        return 0

    if a.cmd == "export-script":
        from realme.pipeline.script_import import to_script_text
        from realme.core.env import data_home
        src = Path(a.source)
        if src.is_file():
            manifest, pdir = src, (src.parent.parent if src.parent.name == "_work" else None)
        else:
            if not src.is_dir() and (data_home() / "projects" / src.name).is_dir():
                src = data_home() / "projects" / src.name
            manifest, pdir = src / "_work" / "manifest.json", src
        if not manifest.is_file():
            sys.exit(f"No narration found at {manifest}. Give the lecture's "
                     f"folder or its _work/manifest.json.")
        try:
            data = json.loads(_TXT(manifest))
            titles = []
            if pdir is not None:
                cache = pdir / "_work" / "slide_titles.json"
                if cache.is_file():
                    titles = json.loads(_TXT(cache)).get("titles", [])
            text, warns = to_script_text(data, titles)
        except (ValueError, KeyError) as e:
            sys.exit(f"{manifest} is not a lecture's narration: {e}")
        name = (pdir.name if pdir is not None else manifest.stem)
        out = a.out or Path(f"{name}_notes.md")
        out.write_text(text, encoding="utf-8")
        n = text.count("[[slide ")
        print(f"Wrote {out}: {n} slide{'s' if n != 1 else ''}")
        for w in warns:
            print(f"  ! {w}")
        _result(report={"slides": n}, warnings=list(warns), files={"notes": out})
        return 0

    if a.cmd == "preview":
        from realme.pipeline.speak import preview
        r = preview(a.text, _tts(a.tts, a), a.out, index=a.index)
        print(json.dumps(r, indent=2, ensure_ascii=False))
        _result(report=r, warnings=r.get("warnings", []), files={"audio": a.out})
        for w in r["warnings"]:
            print(f"  ! {w}", file=sys.stderr)
        return 0

    if a.cmd == "math":
        from realme.text.mathspeech import MathSpeech
        ms = MathSpeech(cache_path=a.cache)
        if a.pending:
            for e in ms.pending():
                print(f"{e['latex']}\n  -> {e['spoken']}\n")
            return 0
        if not a.latex:
            print(json.dumps(MathSpeech.available(), indent=2)); return 0
        spoken, approved = ms.speak(a.latex)
        print(spoken or "(could not convert)")
        if not approved:
            print("  [not yet approved]", file=sys.stderr)
        return 0

    if a.cmd == "lexicon":
        from realme.text.lexicon import Lexicon
        lx = Lexicon()
        if a.action == "check":
            probs = lx.check()
            print("\n".join(probs) if probs else
                  f"{len(lx.entries)} entries, all valid.")
            return 1 if probs else 0
        if a.action == "list":
            # Mark the user's own entries, to tell them apart from the ones
            # that ship with the project.
            for e in lx.entries.values():
                mine = "*" if e.term in lx.user_terms else " "
                print(f"{mine} {e.term:14} -> {e.spoken_fallback():22} {e.note}")
            if lx.user_terms:
                print(f"\n  * {len(lx.user_terms)} entry(ies) of your own, in "
                      f"{__import__('realme.text.lexicon', fromlist=['x']).user_lexicon_path()}")
            return 0
        emitters = {"pls": lx.to_pls, "mfa": lx.to_mfa_dict,
                    "espeak": lx.to_espeak_dict}
        print(emitters.get(a.engine, lx.to_espeak_dict)())
        return 0

    if a.cmd == "argue":
        from realme.pipeline.argue import Argument, reply
        writer = _writer("gemini", a)
        try:
            writer.preflight()
        except Exception as e:
            sys.exit(str(e))
        if a.resume:
            arg = Argument.load(a.resume)
            print(f"Resuming: {arg.topic} ({len(arg.moves)} moves)")
        else:
            if not a.topic:
                sys.exit("Give it a topic, or --resume a saved argument.")
            ctx = ""
            if a.source and a.source.exists():
                if a.source.suffix.lower() == ".pdf":
                    from pypdf import PdfReader
                    ctx = "\n".join((pg.extract_text() or "")
                                     for pg in PdfReader(str(a.source)).pages)
                else:
                    ctx = a.source.read_text(encoding="utf-8", errors="ignore")
            arg = Argument(topic=a.topic, stance=a.stance, context=ctx)
        a.outdir.mkdir(parents=True, exist_ok=True)
        slug = "".join(c if c.isalnum() else "_" for c in arg.topic).strip("_")[:50]
        path = a.outdir / f"{slug}.json"

        print(f"\nArguing about: {arg.topic}")
        print(f"Stance: {arg.stance}.  /stance <name>  /referee  /save  /render  /quit\n")
        while True:
            try:
                you = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not you:
                continue
            if you in ("/quit", "/q"):
                break
            if you == "/save":
                print("saved:", arg.save(path)); continue
            if you.startswith("/stance"):
                parts = you.split()
                if len(parts) > 1:
                    arg.stance = parts[1]
                    print(f"stance -> {arg.stance}")
                continue
            if you == "/referee":
                from realme.pipeline.argue import STANCES
                verdict = writer.chat(STANCES["referee"] + f"\n\nTopic: {arg.topic}",
                                      arg.history(), "Referee the exchange so far.")
                print(f"\nreferee> {verdict}\n"); continue
            if you == "/render":
                arg.save(path)
                print("Saved. Render to audio with:")
                print(f"  realme render-argument {path} --tts qwen3cpp "
                      f"--guest-tts piper")
                continue
            try:
                print(f"\nthem> {reply(arg, writer, you)}\n")
            except Exception as e:
                print(f"\n[error] {e}\n")
        arg.save(path)
        (path.with_suffix(".md")).write_text(arg.to_markdown(), encoding="utf-8")
        print(f"Saved {len(arg.moves)} moves to {path} and {path.with_suffix('.md')}")
        return 0

    if a.cmd == "render-argument":
        from realme.pipeline.argue import Argument
        from realme.pipeline import dialogue as dlg
        arg = Argument.load(a.argument)
        voices = {"instructor": _tts(a.tts, a),
                  "guest": _tts(a.guest_tts or a.tts, a)}
        out = dlg.render_script(arg.to_dialogue_script(), a.outdir, voices, log=print)
        print(json.dumps(out["report"], indent=2))
        return 0

    if a.cmd == "verify":
        from realme.core.media import probe
        info = probe(a.media)
        print(json.dumps({
            "duration_s": round(float(info["format"]["duration"]), 2),
            "size_mb": round(int(info["format"]["size"]) / 1e6, 2),
            "streams": [s["codec_type"] for s in info["streams"]],
            "codecs": [s.get("codec_name") for s in info["streams"]],
        }, indent=2))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
