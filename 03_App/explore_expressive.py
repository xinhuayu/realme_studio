#!/usr/bin/env python3
"""
Expressive-dialogue experiment runner. EXPLORATORY -- not part of the app.

Deliberately a standalone script rather than a `realme` subcommand: nothing on
a shipping path should have to change for an experiment to run, and nothing
here can break a lecture, a narration or a dialogue render.

  python explore_expressive.py --out runs/exp1                 # real voices
  python explore_expressive.py --out runs/exp1 --repeat 5
  python explore_expressive.py --out runs/exp1 --dry-run       # no engines needed

--dry-run substitutes a synthetic voice whose duration responds to pace the way
piper's length_scale does. It proves the harness end to end -- pacing, joining,
measuring, reporting, blinding -- on a machine with no TTS installed. It says
nothing about how anything sounds.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


class SyntheticVoice:
    """A stand-in whose duration responds to pace, as a real engine's does.

    Not a mock in the testing sense -- it renders real audio that the real
    measurement code reads. What it cannot do is sound like anything, which is
    why --dry-run reports numbers and refuses to draw a conclusion from them.
    """
    is_voice_clone = False

    def __init__(self, name: str, f0: float, wps: float = 3.1):
        self.name, self.model, self.f0, self.wps = name, f"tone{f0:.0f}", f0, wps

    def voice_fingerprint(self) -> str:
        return f"{self.name}:{self.model}"

    def synthesize(self, text, out_wav, voice=None, pace=1.0, **kw):
        import numpy as np
        from realme.expressive.wavio import write_wav
        sr = 22050
        words = max(1, len([w for w in text.split() if any(c.isalnum() for c in w)]))
        seconds = words / (self.wps * max(0.2, pace or 1.0))
        t = np.arange(int(sr * seconds)) / sr
        # A slow declination, so a turn is not a dead flat tone and the pitch
        # measurement has something lifelike to work on.
        f = self.f0 * 2 ** ((-0.6 * t / max(seconds, 1e-6)) / 12.0)
        ph = 2 * np.pi * np.cumsum(f) / sr
        y = sum(a * np.sin(k * ph) for k, a in ((1, 1.0), (2, .5), (3, .3), (4, .15)))
        env = np.clip(np.sin(np.pi * np.arange(y.size) / max(y.size, 1)) * 3, 0, 1)
        return write_wav(out_wav, 0.3 * env * y / 2, sr)


def wrong_interpreter_note() -> str:
    """The instruction to print when the engines are missing.

    Both this script and `verify_tree.py` need it, so it lives in
    `realme.core.env` and this is the one call that names the command worth
    repeating. Imported late: the whole point is that the interpreter may be
    one where importing anything from the package is what just failed.
    """
    try:
        from realme.core.env import wrong_interpreter_note as _note
    except Exception:
        import sys as _s
        return f"  this Python : {_s.executable}"
    return _note("03_App\\explore_expressive.py --out runs\\exp1 --repeat 3")


def real_voices(log, instructor=None, guest=None, registers=True):
    """The two dialogue voices, from your profile unless told otherwise.

    The overrides exist because comparing guests is the common reason to run
    this twice, and editing the saved profile to do it means the comparison
    changes your settings as a side effect.
    """
    from realme.pipeline import casting
    from realme.app.profile import Profile
    # core.env, not core.paths. Guessed wrong the first time and it failed on
    # the first real run; every other caller in the project uses this one.
    from realme.core.env import data_home
    profile = Profile(data_home() / "profile")
    a = profile.data.get("adapters", {})
    cast = casting.cast(instructor or a.get("tts", "qwen3cpp"),
                        guest or a.get("guest_tts", "piper"),
                        profile=profile, registers=registers)
    log(f"  instructor: {casting.describe(a.get('tts'), cast['instructor'])}")
    log(f"  guest     : {casting.describe(a.get('guest_tts'), cast['guest'])}")
    return cast


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="where to write renders")
    ap.add_argument("--repeat", type=int, default=3,
                    help="passes per condition; 1 is not enough to conclude "
                         "anything. With --tau the alpha = 0 passes double as "
                         "the noise floor every other cell is judged against.")
    ap.add_argument("--strength", type=float, default=1.0,
                    help="scales every stance multiplier; 0 reproduces the control")
    ap.add_argument("--dry-run", action="store_true",
                    help="synthetic voices; exercises the harness, judges nothing")
    ap.add_argument("--seed", type=int, default=None, help="blinding order")
    ap.add_argument("--audition", nargs="?", const="shortlist", default=None,
                    metavar="WHICH",
                    help="Instead of the stance experiment: render one passage "
                         "in several draft voices and report on them. WHICH is "
                         "'shortlist' (default), 'all', 'female', 'male', or a "
                         "comma-separated list of voice names.")
    ap.add_argument("--tau", action="store_true",
                    help="Instead of the stance experiment: sweep the emotion "
                         "directions extracted by `realme voice tau`, "
                         "rendering one sentence at several strengths along "
                         "each and measuring whether the engine responded.")
    ap.add_argument("--alpha", type=float, action="append", default=[],
                    metavar="A",
                    help="With --tau: a strength to render. Repeat for "
                         "several. Default sweeps a fixed set including 0, "
                         "which is the control.")
    ap.add_argument("--direction", default=None, metavar="NAMES",
                    help="With --tau: comma-separated directions to sweep "
                         "(default: all of them).")
    ap.add_argument("--no-renorm", action="store_true",
                    help="With --tau: leave the shifted embedding at whatever "
                         "length the arithmetic gives instead of rescaling it "
                         "to the enrolled one's. For comparing by ear.")
    ap.add_argument("--compare", action="store_true",
                    help="Render the same exchange three ways — plain, "
                         "register-switched, and shifted along the emotion "
                         "directions — and blind them, so the question 'is "
                         "tau better than the registers I already had?' is "
                         "answered by ear rather than by preference for the "
                         "newer thing.")
    ap.add_argument("--list-voices", action="store_true",
                    help="Print the draft voices that can be auditioned")
    ap.add_argument("--guest", default=None, metavar="SPEC",
                    help="Override the guest voice for this run only, e.g. "
                         "piper:en_GB-cori-high. Your saved profile is not "
                         "changed.")
    ap.add_argument("--instructor", default=None, metavar="SPEC",
                    help="Override your own voice for this run only, e.g. "
                         "piper:en_US-ryan-medium to hear the experiment "
                         "without waiting for the cloned engine.")
    a = ap.parse_args()

    from realme.expressive import audition as AUD
    if a.list_voices:
        for name, (sex, accent, label) in AUD.CANDIDATES.items():
            mark = " *" if name in AUD.SHORTLIST else "  "
            print(f"{mark}{name:<38} {sex:<7} {accent:<3} {label}")
        print("\n  * = the shortlist, which is what --audition uses by default")
        return 0

    if a.audition:
        which = a.audition.strip().lower()
        if which == "shortlist":
            voices = list(AUD.SHORTLIST)
        elif which == "all":
            voices = list(AUD.CANDIDATES)
        elif which in ("female", "male"):
            voices = [v for v, (sex, _, _) in AUD.CANDIDATES.items() if sex == which]
        else:
            voices = [v.strip() for v in a.audition.split(",") if v.strip()]
            unknown = [v for v in voices if v not in AUD.CANDIDATES]
            if unknown:
                print(f"Not in the catalogue: {', '.join(unknown)}\n"
                      f"Run --list-voices to see what there is.", file=sys.stderr)
                return 2
        print(f"Auditioning {len(voices)} voice(s). Already-installed ones are "
              f"not downloaded again.\n")
        heard = AUD.hear(voices, a.out, log=print)
        sheet = AUD.contact_sheet(heard, Path(a.out) / "all_voices.wav")
        print()
        print(AUD.report(heard))
        if sheet:
            print(f"\nAll of them in order, one file: {sheet}")
            print("  order: " + ", ".join(h.voice for h in heard if h.wav))
        return 0

    if a.compare:
        # The decision this whole line of work comes down to. StanceVoiced
        # already switches between the four recorded registers; tau's only
        # extra claim is that it can sit BETWEEN them and be dialled. That
        # claim is worth something only if the result is at least as good, so
        # the two are rendered from the same script, by the same voices, and
        # handed over unlabelled.
        from realme.core.env import data_home
        from realme.app.profile import Profile
        from realme.pipeline import casting
        from realme.expressive import tau as TAU
        from realme.expressive.experiment import run, Condition
        from realme.expressive.voices import StancePaced
        store = data_home() / "voice" / "registers" / "tau.json"
        if not store.is_file():
            print("No emotion directions yet: run `realme voice tau` first.",
                  file=sys.stderr)
            return 2
        if a.dry_run:
            print("--compare cannot be dry-run: both arms are about a cloned "
                  "voice.", file=sys.stderr)
            return 2
        try:
            # UNWRAPPED, deliberately. Each arm below applies exactly one kind
            # of control, and a control arm that arrived pre-controlled is not
            # a control.
            adapters = real_voices(print, instructor=a.instructor,
                                   guest=a.guest, registers=False)
        except Exception as e:
            print(f"Could not build the dialogue voices: {e}", file=sys.stderr)
            return 2
        prof = Profile(data_home() / "profile")
        alpha = a.alpha[0] if a.alpha else None

        def as_is(ad, _st):
            return ad

        def registers(ad, _st):
            # Pace AND register, which is what a podcast render does today.
            return StancePaced(casting.with_registers(ad, prof, log=None),
                               enabled=True, strength=1.0)

        def shifted(ad, _st):
            # Pace AND embedding shift. Never both register and shift: the
            # wrapper refuses that, for the reason in its docstring.
            #
            # --alpha 1.0 --no-renorm is the diagnostic that matters: at that
            # setting the vector handed over IS the register's own embedding,
            # so this arm and the `registers` arm should be indistinguishable.
            # If they are, the shift mechanism is sound and any loss at
            # intermediate alpha belongs to the interpolation. If they are
            # not, the loss is in the plumbing and no listening test of alpha
            # means anything yet.
            return StancePaced(casting.with_expression(
                ad, prof, alpha=alpha, renorm=not a.no_renorm, log=None),
                               enabled=True, strength=1.0)

        print()
        print(TAU.report(TAU.load(store)))
        print()
        try:
            res = run(adapters, a.out, repeat=a.repeat, seed=a.seed,
                      conditions=[
                          Condition("plain", 0.0, "no stance, no register, "
                                                  "no shift", wrap=as_is),
                          Condition("registers", 1.0, "pace + register switch "
                                                      "(what ships today)",
                                    wrap=registers),
                          Condition("shifted", 1.0, "pace + emotion direction",
                                    wrap=shifted)])
        except RuntimeError as e:
            print(f"\n{e}\n", file=sys.stderr)
            return 3
        print()
        print(res["report"])
        print()
        print(f"Listen to listen_A / listen_B / listen_C in {a.out} and write "
              f"down which you prefer\nBEFORE opening listening_key.json.")
        return 0

    if a.tau:
        # The sweep needs the REAL cloned engine: a direction is a point in
        # one encoder's space and the synthetic voice has no encoder at all.
        # Said plainly rather than by producing tones and a table.
        from realme.core.env import data_home
        from realme.expressive import tau as TAU
        store = data_home() / "voice" / "registers" / "tau.json"
        if not store.is_file():
            print(f"No emotion directions extracted yet. Record the registers, "
                  f"then:\n\n    realme voice registers --take ...\n"
                  f"    realme voice tau\n", file=sys.stderr)
            return 2
        ts = TAU.load(store)
        if a.dry_run:
            print("--tau cannot be dry-run: a direction only means anything to "
                  "the encoder that\nproduced it, and the synthetic voice has "
                  "none.", file=sys.stderr)
            return 2
        try:
            adapters = real_voices(print, instructor=a.instructor,
                                   guest=a.guest, registers=False)
        except Exception as e:
            print(f"Could not build your voice: {e}", file=sys.stderr)
            return 2
        eng = adapters["instructor"]
        alphas = tuple(a.alpha) if a.alpha else TAU.SWEEP_ALPHAS
        dirs = ([d.strip() for d in a.direction.split(",") if d.strip()]
                if a.direction else None)
        print()
        print(TAU.report(ts))
        n = len(dirs or ts.directions) * len(alphas) * max(1, a.repeat)
        print(f"\nSweeping {len(dirs or ts.directions)} direction(s) x "
              f"{len(alphas)} strengths x {a.repeat} draws = {n} renders.\n"
              f"Each one re-runs the speaker encoder on its own output, which "
              f"is what proves\nthe vector reached the decoder; the alpha = 0 "
              f"draws double as the noise floor.\n")
        try:
            heard = TAU.sweep(eng, ts, a.out, directions=dirs, alphas=alphas,
                              repeat=a.repeat, renorm=not a.no_renorm,
                              log=print)
        except (TypeError, KeyError, ValueError) as e:
            print(f"\n{e}\n", file=sys.stderr)
            return 3
        sheet = TAU.contact_sheet(heard, Path(a.out) / "all_alphas.wav")
        print()
        print(TAU.sweep_report(heard, ts))
        if sheet:
            print(f"\nAll of them in order, one file: {sheet}")
            print("  order: " + ", ".join(
                f"{h.direction} {h.alpha:+.2f}" for h in heard if h.wav))
        return 0

    try:
        import numpy  # noqa: F401
    except ImportError:
        print("This needs numpy, which the voice engines already require.\n"
              "  pip install numpy", file=sys.stderr)
        return 2
    from realme.expressive.experiment import run, Condition
    log = print
    if a.dry_run:
        log("DRY RUN: synthetic voices. The numbers describe the harness, not a voice.")
        adapters = {"instructor": SyntheticVoice("synth-host", 112.0),
                    "guest": SyntheticVoice("synth-guest", 178.0)}
    else:
        try:
            adapters = real_voices(log, instructor=a.instructor, guest=a.guest)
        except Exception as e:
            print(f"Could not build the dialogue voices: {e}\n"
                  f"Run `realme setup check`, or try --dry-run to exercise the "
                  f"harness without them.", file=sys.stderr)
            return 2

    if a.repeat < 2:
        log("NOTE: --repeat 1 cannot show whether an effect exceeds the noise.")
    try:
        res = run(adapters, a.out, repeat=a.repeat, seed=a.seed,
                  conditions=[Condition("flat", 0.0, "control"),
                              Condition("stance", a.strength, "stance pacing")])
    except RuntimeError as e:
        # A voice that cannot start. Reported as a sentence and an instruction,
        # not a traceback: the useful line is usually the last one and readers
        # should not have to dig for it.
        print(f"\n{e}\n", file=sys.stderr)
        if not a.dry_run:
            print(wrong_interpreter_note(), file=sys.stderr)
            print("\n  Or --dry-run to exercise the harness with no engines "
                  "at all.", file=sys.stderr)
        return 3
    print()
    print(res["report"])
    print()
    print(f"Listen to {a.out / 'listen_A.wav'} and {a.out / 'listen_B.wav'} before "
          f"reading {a.out / 'listening_key.json'}.")
    if a.dry_run:
        print("These were synthetic tones. Nothing here says whether stance pacing "
              "sounds better.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
