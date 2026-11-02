#!/usr/bin/env python3
"""
`realme bench-parallel`: does it measure what it says, and does it refuse to
call noise a result?

Runs the benchmark end to end on its stand-in engine (which takes a fixed
time per word, so two workers really are twice as fast), in separate
processes exactly as it runs the local clone. Then checks the verdict rules
on made-up timings, where the right answer is known.
"""
from __future__ import annotations
import json, os, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

_ISO = tempfile.mkdtemp(prefix="t_parbench_")
os.environ["REALME_HOME"] = str(Path(_ISO) / "home")
os.environ["HOME"] = _ISO
os.environ["USERPROFILE"] = _ISO

from t_util import check, summary          # noqa: E402


def lay(w, t, walls):
    from realme.parbench import Layout, Run
    return Layout(w, t, runs=[Run(w, t, i + 1, wall_s=x, audio_s=60.0)
                              for i, x in enumerate(walls)])


def main() -> int:
    from realme import parbench as PB

    print("layouts")
    check(PB.parse_layouts("1x6, 2x4,3x3") == [(1, 6), (2, 4), (3, 3)],
          "workers x threads are read as written")
    for bad in ("2x4,1x6", "1x6,abc", "1x0", ""):
        try:
            PB.parse_layouts(bad)
            ok = False
        except ValueError:
            ok = True
        check(ok, f"{bad!r} is refused with a reason")
    d = PB.default_layouts(16)
    check(d[0][0] == 1 and len(d) >= 3 and all(w * t <= 16 for w, t in d),
          "the default starts from one worker and fits the logical processors", d)
    check(PB.default_layouts(4)[0] == (1, 2), "and suits a small machine",
          PB.default_layouts(4))
    check(PB.default_layouts(12)[:3] == [(1, 6), (2, 3), (2, 6)],
          "on 6 cores: one engine of 6, the cores shared by two, two of 6",
          PB.default_layouts(12))
    mem = PB.available_memory_gb()
    check(mem is None or mem > 0, "free memory is read or admitted unknown", mem)

    print("\nthe verdict")
    v = PB.verdict([lay(1, 6, [300, 310]), lay(2, 4, [200, 205])])
    check(v["worth_it"] and v["best"] == "2 x 4", "clearly faster in every run",
          v)
    v = PB.verdict([lay(1, 6, [300, 200]), lay(2, 4, [210, 220])])
    check(not v["worth_it"] and "noise" in v["say"],
          "runs that overlap are called noise, not a result", v["say"])
    v = PB.verdict([lay(1, 6, [300, 302]), lay(2, 4, [280, 285])])
    check(v["worth_it"] and "REALME_RECORD_WORKERS 2" in v["say"],
          "a consistent gain of 5% or more is worth using, and says how", v["say"])
    v = PB.verdict([lay(1, 6, [300, 302]), lay(2, 4, [290, 293])])
    check(not v["worth_it"] and "1.05" in v["say"],
          "a consistent but smaller gain is not", v["say"])
    v = PB.verdict([lay(1, 6, [300, 302]), lay(2, 4, [330, 340])])
    check(not v["worth_it"] and v["best"] is None,
          "slower is reported as no help", v["say"])

    print("\nan engine that cannot start here")
    check(PB.can_start("selftest", None) == "", "the stand-in can start")
    why = PB.can_start("no-such-engine", None)
    check(bool(why), "one that cannot says why before anything is timed", why)
    # Free memory as if plentiful: on a computer with little free, every
    # layout would be skipped for memory before the engine is even tried.
    _free = PB.available_memory_gb
    PB.available_memory_gb = lambda: 64.0
    try:
        r = PB.main("no-such-engine", [("A sentence.", 1.0)] * 2, [(1, 1), (2, 1)],
                    repeat=1, outdir=Path(_ISO) / "nobench", reference=None,
                    log=lambda *_: None)
    finally:
        PB.available_memory_gb = _free
    check(r["layouts"] == [] and r["verdict"]["best"] is None,
          "and a run that cannot start one engine stops instead of trying "
          "every layout", r["verdict"])

    print("\nend to end, in separate processes")
    out = Path(_ISO) / "bench"
    script = [("Confounding is a property of the question you ask of the "
               "data, not of the data.", 1.0)] * 8
    r = PB.main("selftest", script, [(1, 2), (2, 2)], repeat=2, outdir=out,
                reference=None, log=lambda *_: None)
    runs = [run for l in r["layouts"] for run in l["runs"]]
    check(len(runs) == 4 and all(x["wall_s"] > 0 and not x["failed"] for x in runs),
          "every run of every layout finishes", runs)
    check(r["verdict"]["best"] == "2 x 2" and r["verdict"]["speedup"] > 1.4,
          "two workers that really are faster are found faster", r["verdict"])
    wavs = sorted(out.glob("2x2_run1_*.wav"))
    check(len(wavs) == len(script), "the audio of each layout is kept to listen to",
          len(wavs))
    saved = sorted(out.glob("parallel_*.json"))
    check(saved and json.loads(saved[-1].read_text("utf-8"))["verdict"]["best"]
          == "2 x 2", "and the results are saved")
    return summary()


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        import shutil
        shutil.rmtree(_ISO, ignore_errors=True)
