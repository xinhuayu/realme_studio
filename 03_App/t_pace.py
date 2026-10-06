#!/usr/bin/env python3
"""
The guest's pace: measured for the voice in hand, or measured now.

Written because Amy dragged. The cause was not one bug but a gap between two
that each looked right on its own: `voice pace` saved its numbers in one
shape, the per-voice lookup read another, and when the lookup honestly said
"I have never measured this voice" the caster applied a constant anyway --
0.72, which is slower than the ratio measured against a FASTER voice.

No engine here. Piper is stubbed, so this runs anywhere and still exercises
the real arithmetic and the real profile shapes.
"""
from __future__ import annotations
import json, os, sys, tempfile, types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Isolated from the real installation, deliberately and before any import
# that reads it.
#
# These suites build their own fake tree and fake data folder -- and then
# asked the REAL one anyway, because the code they exercise resolves those
# through `data_home()` and `find_tool()`, which read the environment at call
# time. On a clean machine both come back empty and the fixtures are used; on
# a working machine the test silently measured against the instructor's own
# profile and packaged his own engine folder. Eight checks failed on his
# machine and none in the sandbox, which is the worst way for a test to be
# wrong.
_ISO = tempfile.mkdtemp(prefix="t_pace_iso_")
os.environ["REALME_HOME"] = str(Path(_ISO) / "home")
os.environ["REALME_TOOLS"] = str(Path(_ISO) / "tools")

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}"
          + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


#: His profile, as `realme voice pace` actually left it: one nested record,
#: no top-level your_wpm, no per-voice table. The shape the lookup missed.
ORDINARY = {
    "draft_speed": 0.744,
    "draft_pace_measured": {"voice": "en_US-ryan-medium", "your_wpm": 160.3,
                            "voice_wpm": 215.4, "speed": 0.744,
                            "clamped": False},
}
#: The shape `realme voice pace --all` leaves.
FULL = {"your_wpm": 160.3,
        "draft_voice_wpm": {"en_US-ryan-medium": 215.4,
                            "en_US-amy-medium": 174.7}}


class FakeProfile:
    def __init__(self, data, text="one two three four five six seven eight "
                                  "nine ten eleven twelve thirteen"):
        self.data = dict(data)
        self.reference_text = text
        self.written = []

    def update(self, d):
        self.data.update(d)
        self.written.append(dict(d))


def main() -> int:
    from realme.adapters.tts import rates
    from realme.pipeline import casting

    print("\nwhere the numbers were saved")
    mine, table = rates(ORDINARY)
    check("your rate is found in the nested record", mine == 160.3, str(mine))
    check("so is the voice it was measured against",
          table.get("en_US-ryan-medium") == 215.4, str(table))
    mine2, table2 = rates(FULL)
    check("and the --all shape still works",
          mine2 == 160.3 and table2.get("en_US-amy-medium") == 174.7)
    both = dict(FULL); both.update(ORDINARY)
    _, t3 = rates(both)
    check("the per-voice table wins where the two overlap",
          t3["en_US-ryan-medium"] == 215.4)

    print("\nthe ratio is per voice")
    from realme.enrollment.pace import MIN_SPEED, MAX_SPEED

    def speed_from(data, voice):
        m, t = rates(data)
        if voice and m and t.get(voice):
            return round(max(MIN_SPEED, min(MAX_SPEED, m / t[voice])), 3)
        return None

    check("Ryan gets the ratio measured against Ryan",
          speed_from(ORDINARY, "en_US-ryan-medium") == 0.744,
          str(speed_from(ORDINARY, "en_US-ryan-medium")))
    check("Amy, never measured, gets nothing rather than Ryan's",
          speed_from(ORDINARY, "en_US-amy-medium") is None)
    amy = speed_from(FULL, "en_US-amy-medium")
    check("once measured, Amy gets her own",
          abs(amy - round(160.3 / 174.7, 3)) < 1e-9, str(amy))
    # The whole complaint, in one line: she was 27% slower than she should be.
    check("and it is faster than the constant she was getting",
          amy > casting.DIALOGUE_SPEED,
          f"{amy} vs {casting.DIALOGUE_SPEED}")

    print("\nan unmeasured voice is measured, not guessed at")
    calls = []

    def fake_calibrate(ref, text, *, voice=None, workdir=None, log=print):
        calls.append(voice)
        return {"voice": voice, "your_wpm": 160.3, "voice_wpm": 174.7,
                "speed": round(160.3 / 174.7, 3), "clamped": False}

    import realme.enrollment.pace as PACE
    real = PACE.calibrate
    PACE.calibrate = fake_calibrate
    # Temp, and removed: `shipped_files` walks the folder, so scratch written
    # beside the source ends up in the next release.
    import shutil, tempfile
    tmp = Path(tempfile.mkdtemp(prefix="t_pace_"))
    ref = tmp / "ref.wav"; ref.write_bytes(b"x")
    try:
        prof = FakeProfile({**ORDINARY, "voice_reference": str(ref)})
        said = []
        got = casting.guest_speed("en_US-amy-medium", prof, log=said.append)
        check("the voice is calibrated on the spot", calls == ["en_US-amy-medium"],
              str(calls))
        check("and the matched speed is used",
              abs(got - round(160.3 / 174.7, 3)) < 1e-9, str(got))
        saved = prof.data.get("draft_voice_wpm") or {}
        check("it is written back in the durable shape",
              prof.data.get("your_wpm") == 160.3
              and saved.get("en_US-amy-medium") == 174.7,
              json.dumps(saved))
        check("the older measurement is kept beside it",
              saved.get("en_US-ryan-medium") == 215.4, json.dumps(saved))
        check("and it says what it is doing", any("measuring" in s for s in said),
              str(said))

        # Second cast: the number is on file, so nothing renders again.
        calls.clear()
        check("a measured voice is never re-measured",
              casting.guest_speed("en_US-amy-medium", prof)
              == round(160.3 / 174.7, 3) and calls == [], str(calls))

        # Nothing to measure from is the only case the constant survives.
        bare = FakeProfile({}, text="")
        bare.data["voice_reference"] = None
        said2 = []
        check("with no enrolment to measure against, the constant is used",
              casting.guest_speed("en_US-amy-medium", bare, log=said2.append)
              == casting.DIALOGUE_SPEED)
        check("and the reason is printed, not swallowed",
              any("could not measure" in s for s in said2), str(said2))

        # A render must never die because a pace could not be measured.
        def explodes(*a, **k):
            raise RuntimeError("piper is not installed")
        PACE.calibrate = explodes
        prof2 = FakeProfile({**ORDINARY, "voice_reference": str(ref)})
        try:
            fell = (casting.guest_speed("en_GB-cori-high", prof2,
                                        log=lambda *_: None)
                    == casting.DIALOGUE_SPEED)
        except Exception as e:
            fell = False
            detail = f"raised {type(e).__name__}: {e}"
        else:
            detail = ""
        check("a failed measurement falls back instead of raising", fell, detail)
    finally:
        PACE.calibrate = real
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: {', '.join(FAILED)}")
        return 1
    print("all good")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
