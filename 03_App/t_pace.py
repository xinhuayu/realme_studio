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

    # ------------------------------------------------------------------
    # The 25% error: a re-recorded enrolment of a SHORTER passage, with the
    # previous passage's transcript still on disk. 163 words over 28.8 s of
    # speech measured 323 words a minute, the calibration asked for 1.846x,
    # the clamp turned that into 1.25, and 1.25 was stored as if measured.
    # Four things had to be true at once; each one is checked here.
    # ------------------------------------------------------------------
    print("\na transcript that is not the recording's is refused")
    import subprocess
    from realme.core.media import require as _req
    from realme.enrollment import pace as P
    from realme.app import profile as PROF

    box = Path(tempfile.mkdtemp(prefix="t_pace_wpm_"))
    try:
        # 20 s of unbroken tone: silencedetect finds nothing, so speech time
        # is the duration and the implied rate is exactly words / 20 * 60.
        tone = box / "tone.wav"
        subprocess.run([_req("ffmpeg"), "-y", "-loglevel", "error", "-f",
                        "lavfi", "-t", "20", "-i",
                        "sine=frequency=200:sample_rate=24000", str(tone)],
                       check=True)
        plain = " ".join(["word"] * 60)        # 180 wpm -- a person
        stale = " ".join(["word"] * 110)       # 330 wpm -- not a person
        check("words over speech time is what is measured",
              abs(P.words_per_minute(tone, plain) - 180.0) < 2.0,
              str(P.words_per_minute(tone, plain)))
        check("a lecturer's rate passes the band",
              P.check_human_wpm(180.0, tone, plain) == 180.0)
        try:
            P.check_human_wpm(330.0, tone, stale)
            said = ""
        except ValueError as e:
            said = str(e)
        check("a rate no one speaks at is refused", bool(said))
        check("and the refusal names both inputs, since one of them is wrong",
              "110 words" in said and "20 s of speech" in said, said[:160])

        # A stub engine, so this measures the arithmetic and not the network.
        class Stub:
            name = "stub"

            def __init__(self, seconds, rate_match=1.0):
                self.seconds, self.rate_match = seconds, rate_match
                self.measured_at = None
                self.renders = 0

            def preflight(self):
                pass

            def synthesize(self, text, out, pace=1.0, **kw):
                self.renders += 1
                self.measured_at = self.rate_match
                subprocess.run([_req("ffmpeg"), "-y", "-loglevel", "error",
                                "-f", "lavfi", "-t", str(self.seconds), "-i",
                                "sine=frequency=200:sample_rate=24000",
                                str(out)], check=True)
                return Path(out)

        import realme.adapters.factory as FACT
        real_build, built = FACT.build, {}

        def fake_build(name, **kw):
            return built["engine"]

        FACT.build = fake_build
        try:
            # 60 words in 20 s = 180 wpm from the engine, against a reference
            # measured at 180: no correction needed.
            built["engine"] = Stub(20.0)
            r = P.calibrate_engine("stub", tone, plain, workdir=box,
                                   log=lambda *_: None)
            check("a matched pair measures as no correction",
                  abs(r["pace"] - 1.0) < 0.03, str(r))
            check("and nothing is reported as clamped", r["clamped"] is False)

            # The failure as it happened: the engine is fine, the target is a
            # mismatched transcript. Refused BEFORE a paid render.
            built["engine"] = Stub(20.0)
            try:
                P.calibrate_engine("stub", tone, stale, workdir=box,
                                   log=lambda *_: None)
                refused = ""
            except ValueError as e:
                refused = str(e)
            check("a mismatched transcript is refused as a target",
                  bool(refused), refused[:120])
            check("before the engine is asked to render anything",
                  built["engine"].renders == 0,
                  f"{built['engine'].renders} renders")

            # An engine genuinely twice as slow: the correction needed is
            # outside what retiming can do without being heard, so it is a
            # failed measurement rather than a stored MAX_SPEED.
            built["engine"] = Stub(40.0)       # 90 wpm against a 180 target
            try:
                got = P.calibrate_engine("stub", tone, plain, workdir=box,
                                         log=lambda *_: None)
                saturated = f"stored {got['pace']}"
            except ValueError as e:
                saturated = ""
                message = str(e)
            check("a correction beyond the range is not stored as the clamp",
                  not saturated, saturated)
            check("and the refusal says what the two measurements were",
                  "180" in message and "90" in message, message[:160])

            # Measuring through the last correction compounds it.
            built["engine"] = Stub(20.0, rate_match=1.25)
            P.calibrate_engine("stub", tone, plain, workdir=box,
                               log=lambda *_: None)
            check("the engine is measured with its stored correction off",
                  built["engine"].measured_at == 1.0,
                  str(built["engine"].measured_at))
        finally:
            FACT.build = real_build

        print("\nthe transcript knows which recording it belongs to")
        prof = PROF.Profile(box / "profile")
        prof.store_audio(tone, "voice_reference")
        prof.set_reference_text(plain)
        check("a transcript saved for the recording on disk matches it",
              prof.transcript_matches_recording)
        check("and nothing is said about it", prof.transcript_note == "")

        other = box / "other.wav"
        subprocess.run([_req("ffmpeg"), "-y", "-loglevel", "error", "-f",
                        "lavfi", "-t", "12", "-i",
                        "sine=frequency=300:sample_rate=24000", str(other)],
                       check=True)
        prof.store_audio(other, "voice_reference")
        check("a new recording makes the transcript a transcript of something "
              "else", not prof.transcript_matches_recording)
        check("and that is said, with what to do about it",
              "actually read" in prof.transcript_note, prof.transcript_note)
        prof.set_reference_text(plain)
        check("re-saving the transcript pairs it with the new recording",
              prof.transcript_matches_recording)

        print("\nand a transcript corrected by hand is a transcript")
        # The .txt sits beside the audio so it can be opened and fixed, and
        # the design says the FILE wins over the JSON mirror. A hand edit
        # therefore has to be able to settle the pair -- otherwise doing the
        # right thing in Notepad leaves the pace permanently unmeasurable.
        import time
        hand = PROF.Profile(box / "hand")
        hand.store_audio(other, "voice_reference")
        hand.data["reference_audio_id"] = None
        hand.data["reference_text_id"] = None
        hand.save()
        txt = hand.reference_text_path
        txt.parent.mkdir(parents=True, exist_ok=True)
        wav_mtime = Path(hand.data["voice_reference"]).stat().st_mtime
        txt.write_text(plain + "\n", encoding="utf-8")
        os.utime(txt, (wav_mtime - 600, wav_mtime - 600))
        check("a transcript written before the recording cannot describe it",
              not hand.transcript_matches_recording)
        check("and the refusal says that, rather than blaming the engine",
              "before the enrolment" in hand.transcript_note,
              hand.transcript_note[:90])
        os.utime(txt, (wav_mtime + 5, wav_mtime + 5))
        check("one written after it is accepted, with no digests at all",
              hand.transcript_matches_recording)
        hand.note_transcript_pairing()
        check("and a measurement settles the pair by content from then on",
              bool(hand.data["reference_audio_id"])
              and hand.data["reference_audio_id"] == hand.data["reference_text_id"])
        os.utime(txt, (wav_mtime - 600, wav_mtime - 600))
        check("so the timestamps stop mattering once it is settled",
              hand.transcript_matches_recording)
        hand.store_audio(tone, "voice_reference")
        check("while a new recording unsettles it again, as it should",
              not hand.transcript_matches_recording)

        print("\na correction smaller than the noise is not a correction")
        # His case exactly: 172.5 words a minute of speech against the
        # clone's 175.1 -- 1.5%, where two takes of the same passage by the
        # same person differ by 3-5%. Retiming that costs high-frequency
        # detail to chase a difference nobody can hear.
        import realme.adapters.factory as F2
        real2 = F2.build
        F2.build = lambda name, **kw: built["engine"]
        try:
            built["engine"] = Stub(20.3)       # 177 wpm against a 180 target
            spoke = []
            tiny = P.calibrate_engine("stub", tone, plain, workdir=box,
                                      log=spoke.append)
            check("a 2% gap is reported as no correction",
                  tiny["pace"] == 1.0, str(tiny["pace"]))
            check("and it says what it measured, not just what it decided",
                  any("inside the" in m for m in spoke), str(spoke))
            check("the measured rates are still returned, to be argued with",
                  tiny["target_wpm"] > 0 and tiny["engine_wpm"] > 0, str(tiny))
            built["engine"] = Stub(23.0)       # 157 wpm -- 13%, real
            kept = P.calibrate_engine("stub", tone, plain, workdir=box,
                                      log=lambda *_: None)
            check("while a gap outside the spread is still corrected",
                  abs(kept["pace"] - 180.0 / 156.5) < 0.05, str(kept["pace"]))
        finally:
            F2.build = real2
    finally:
        shutil.rmtree(box, ignore_errors=True)

    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: {', '.join(FAILED)}")
        return 1
    print("all good")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
