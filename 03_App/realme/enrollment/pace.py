"""
Matching the draft voice's pace to the speaker's own.

Where this matters is DIALOGUE. Two voices alternate there, and one speaking
noticeably faster than the other is audible as a mismatch -- the guest sounds
like they are rushing you, or you sound like you are holding them up.

Where it does not matter is lecture and narration drafting. A draft is
scaffolding: it is replaced by the cloned voice before anything is published,
so matching its pace to yours buys nothing and makes every iteration longer.
Those keep the voice's own pace.

A constant cannot do this job: two people do not share a speaking rate, and
any number chosen by ear is about whoever was listening.

So: measure both, and divide.

  your rate     words in the enrolment transcript / seconds of SPEECH in the
                enrolment recording (silence removed -- pauses are rhythm, and
                counting them turns a thoughtful speaker into a slow one)

  piper's rate  the same text through piper at speed 1.0, measured the same way

  speed         your rate / piper's rate, clamped

Clamped because a wildly wrong transcript would otherwise produce a wildly
wrong speed, and a draft voice at 0.4 is not a draft voice.
"""
from __future__ import annotations
import re
import subprocess
from pathlib import Path

from realme.core.media import duration_of, require

#: Speech at -32 dB and gaps from 0.18 s: the same thresholds
#: `prosody.measure_reference_pauses` uses, so the two measurements describe
#: the same recording rather than two different ideas of silence.
NOISE_DB = "-32dB"
MIN_SILENCE_S = 0.18

MIN_SPEED, MAX_SPEED = 0.55, 1.25
FALLBACK_SPEED = 0.8


def silence_seconds(wav: Path) -> float:
    proc = subprocess.run(
        [require("ffmpeg"), "-hide_banner", "-i", str(wav),
         "-af", f"silencedetect=noise={NOISE_DB}:d={MIN_SILENCE_S}",
         "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    return sum(float(m) for m in
               re.findall(r"silence_duration:\s*([\d.]+)", proc.stderr))


def words_per_minute(wav: Path, text: str) -> float | None:
    """Articulation rate: words per minute of actual speech."""
    words = len([w for w in re.split(r"\s+", text.strip()) if w])
    if words < 12:
        return None            # too short to measure anything stable
    total = duration_of(wav)
    speech = total - silence_seconds(wav)
    if speech < 3.0:
        return None
    return words / speech * 60.0


def calibrate(reference_wav, reference_text: str, *, voice: str | None = None,
              workdir: Path | None = None, log=print) -> dict:
    """
    Measure both rates and return the speed that makes them agree.

    Renders a sample through piper, because piper's rate at speed 1.0 is a
    property of the voice file and is not written down anywhere. It costs a few
    seconds -- piper runs at about RTF 0.08 -- and it is the only way to get a
    number rather than an opinion.
    """
    reference_wav = Path(reference_wav)
    mine = words_per_minute(reference_wav, reference_text)
    if mine is None:
        raise ValueError(
            "Not enough to measure from: the enrolment transcript needs at "
            "least a dozen words and a few seconds of speech.\n"
            "    realme voice enroll <file> --transcript-file <txt>")

    # A sample, not the whole transcript: rate is stable well before a minute
    # of audio and this is a calibration, not a render.
    words = [w for w in re.split(r"\s+", reference_text.strip()) if w][:60]
    sample = " ".join(words)

    from realme.adapters.factory import build
    piper = build("piper", quiet=True, **({"model": voice} if voice else {}),
                  speed=1.0)
    piper.preflight()
    workdir = Path(workdir) if workdir else reference_wav.parent
    workdir.mkdir(parents=True, exist_ok=True)
    out = workdir / "_pace_calibration.wav"
    piper.synthesize(sample, out)
    theirs = words_per_minute(out, sample)
    if not theirs:
        raise ValueError("The calibration render was too short to measure.")

    raw = mine / theirs
    speed = max(MIN_SPEED, min(MAX_SPEED, raw))
    log(f"  you        {mine:5.1f} words/min of speech")
    log(f"  {piper.model:<22} {theirs:5.1f} words/min at speed 1.0")
    log(f"  speed      {speed:.3f}"
        + ("" if abs(speed - raw) < 1e-6 else f"  (clamped from {raw:.3f})"))
    return {"voice": piper.model, "your_wpm": round(mine, 1),
            "voice_wpm": round(theirs, 1), "speed": round(speed, 3),
            "clamped": abs(speed - raw) > 1e-6}


def calibrate_all(reference_wav, reference_text: str, *, voices=None,
                  workdir: Path | None = None, log=print) -> dict:
    """Measure YOUR rate once, and every offered voice's rate beside it.

    The quotient was never the durable thing. Your words per minute is a fact
    about you; each voice's is a fact about that voice; the speed that matches
    them is arithmetic, and doing it per voice at cast time is what stops a
    ratio measured against one speaker being applied to another.

    Piper runs at about RTF 0.08, so measuring three voices costs a few
    seconds.
    """
    from realme.adapters.tts import DRAFT_VOICES
    voices = list(voices or DRAFT_VOICES)
    table, mine, failed = {}, None, []
    for v in voices:
        try:
            r = calibrate(reference_wav, reference_text, voice=v,
                          workdir=workdir, log=lambda *_: None)
        except Exception as e:
            failed.append((v, str(e)))
            log(f"  {v:<24} could not be measured: {e}")
            continue
        mine = mine or r["your_wpm"]
        table[v] = r["voice_wpm"]
        log(f"  {v:<24} {r['voice_wpm']:5.1f} words/min at speed 1.0"
            f"   -> speed {r['speed']:.3f}"
            + ("  (clamped)" if r["clamped"] else ""))
    if not table:
        raise ValueError("No draft voice could be measured. Install them with "
                         "`realme engine install`.")
    log(f"  {'you':<24} {mine:5.1f} words/min of speech")
    return {"your_wpm": mine, "voice_wpm": table,
            "failed": [v for v, _ in failed]}


# ------------------------------------------------------- any engine, not piper
#
# `calibrate` above measures PIPER, because matching a draft voice to the
# instructor is what it was written for. A cloned voice was assumed not to need
# it: the clone speaks at the speaker's own rate by construction.
#
# That assumption holds only while the enrolment recording is at the speaker's
# own rate. Zero-shot cloning copies the reference's DELIVERY, not just its
# timbre -- so a reference read carefully for the microphone produces a clone
# that lectures carefully, forever, at a pace the speaker has never used in a
# classroom. Measured on one such pair: 193.8 words a minute of speech in the
# first enrolment, 183.1 in a more deliberate re-recording of the same passage.
# Five per cent sounds like nothing and is forty seconds added to a
# fifteen-minute lecture, in the direction people already complain about.
#
# The fix is the same arithmetic as for piper, with the engine swapped: measure
# the target rate, measure what the engine does at pace 1.0, divide. What it
# must NOT be is a number somebody picked by listening.

def engine_wpm(engine, sample: str, out_wav: Path, *, pace: float = 1.0) -> float | None:
    """Words per minute of speech this engine produces, measured.

    Nothing about an engine's natural rate is written down -- not for piper,
    whose rate is a property of the voice file, and not for a hosted model,
    whose rate is a property of whatever it was cloned from.
    """
    engine.synthesize(sample, Path(out_wav), pace=pace)
    return words_per_minute(Path(out_wav), sample)


def calibrate_engine(engine_name: str, reference_wav, reference_text: str, *,
                     target_wpm: float | None = None,
                     workdir: Path | None = None, words: int = 60,
                     log=print, **build_kw) -> dict:
    """
    The pace that makes `engine_name` speak at the target rate.

    `target_wpm` defaults to the enrolment recording's own articulation rate.
    Pass it explicitly when the enrolment is not how you actually lecture --
    a reference recorded slowly for clarity is the usual case, and calibrating
    against it would preserve exactly the thing you wanted corrected.
    """
    reference_wav = Path(reference_wav)
    mine = target_wpm or words_per_minute(reference_wav, reference_text)
    if not mine:
        raise ValueError(
            "Nothing to match: the enrolment needs a transcript of at least a "
            "dozen words.\n"
            "    realme voice enroll <file> --transcript-file <txt>")

    sample = " ".join([w for w in re.split(r"\s+", reference_text.strip())
                       if w][:words])
    from realme.adapters.factory import build
    engine = build(engine_name, quiet=True, **build_kw)
    engine.preflight()
    workdir = Path(workdir) if workdir else reference_wav.parent
    workdir.mkdir(parents=True, exist_ok=True)
    out = workdir / f"_pace_{engine_name.replace(':', '_')}.wav"
    theirs = engine_wpm(engine, sample, out, pace=1.0)
    if not theirs:
        raise ValueError("The calibration render was too short to measure.")

    raw = mine / theirs
    speed = max(MIN_SPEED, min(MAX_SPEED, raw))
    log(f"  target     {mine:5.1f} words/min of speech")
    log(f"  {engine_name:<22} {theirs:5.1f} words/min at pace 1.0")
    log(f"  pace       {speed:.3f}"
        + ("" if abs(speed - raw) < 1e-6 else f"  (clamped from {raw:.3f})"))
    return {"engine": engine_name, "target_wpm": round(mine, 1),
            "engine_wpm": round(theirs, 1), "pace": round(speed, 3),
            "clamped": abs(speed - raw) > 1e-6, "sample_wav": str(out)}


def reference_drift(reference_wav, reference_text: str,
                    previous_wpm: float | None) -> dict:
    """Has a re-recorded enrolment changed how fast you talk?

    Worth saying out loud at the moment of enrolment, because the clone
    inherits it and the person listening to the result months later will hear
    "the voice is slow" without connecting it to a recording session.
    """
    now = words_per_minute(Path(reference_wav), reference_text)
    out = {"wpm": round(now, 1) if now else None,
           "previous_wpm": previous_wpm, "note": ""}
    if now and previous_wpm:
        change = now / previous_wpm - 1.0
        out["change_pct"] = round(change * 100, 1)
        if abs(change) >= 0.03:
            faster = "faster" if change > 0 else "slower"
            out["note"] = (
                f"This recording is {abs(change) * 100:.0f}% {faster} than "
                f"your previous enrolment ({now:.0f} against "
                f"{previous_wpm:.0f} words a minute of speech). A cloned "
                f"voice copies the delivery it was cloned from, so every "
                f"lecture rendered from here will carry that. If it was not "
                f"deliberate, re-record at the pace you actually teach at; "
                f"if it was, nothing to do.")
    return out


# ------------------------------------------------- what a rendering actually did
#
# Duration alone cannot tell "it dropped a sentence" from "it said everything
# faster", and those want opposite responses: the first means use smaller
# calls, the second means keep the size and retime. Measured on one probe, the
# difference was invisible by duration and obvious by syllable count -- the
# long calls carried their content and simply hurried through it.
#
# Two further things duration gets wrong, both found the same way:
#
#   * Edge silence. Every call comes back with a little lead-in and tail --
#     0.27 s and 0.41 s on the run that prompted this. Comparing one long call
#     against the SUM of N short ones therefore compares it against N copies of
#     that padding: 7.5 seconds of phantom shortfall over eleven sentences,
#     which is most of an 11% "SHORT" verdict with nothing wrong at all.
#   * Pauses. A model given a whole paragraph places its own pauses; a model
#     given one sentence cannot. Speaking time is the comparable quantity.

def speech_profile(path) -> dict:
    """Speaking time, syllable nuclei and articulation rate for one file.

    Syllable nuclei rather than words, because there is no transcriber here
    and a peak in the smoothed energy envelope is a syllable to within a few
    per cent -- enough to tell a missing sentence (a tenth of the content)
    from a brisk one (a twentieth of the time).
    """
    import numpy as np
    from realme.verify.acoustic import read_wav
    sig, sr = read_wav(Path(path))
    sig = sig.astype(np.float64)
    if sig.size == 0:
        return {"duration_s": 0.0, "speech_s": 0.0, "syllables": 0,
                "articulation": 0.0, "lead_s": 0.0, "tail_s": 0.0}
    hop, win = int(sr * 0.01), int(sr * 0.02)
    n = max((len(sig) - win) // hop, 1)
    env = np.array([np.sqrt((sig[i * hop:i * hop + win] ** 2).mean())
                    for i in range(n)]) + 1e-12
    # Two thresholds, not one. A purely relative threshold has no opinion
    # about an empty file: digital silence is uniformly 1e-12, every frame is
    # 0 dB below the peak, and the whole clip measures as speech. The absolute
    # floor (-80 dBFS) is what makes "nothing here" an answer the measurement
    # can give. Found by a test asserting the obvious thing about a silent wav.
    peak = float(env.max())
    if peak < 1e-4:
        return {"duration_s": len(sig) / sr, "speech_s": 0.0, "syllables": 0,
                "articulation": 0.0, "lead_s": 0.0, "tail_s": 0.0}
    db = 20 * np.log10(env / peak)
    speech = (db > -35) & (env > 1e-4)
    idx = np.flatnonzero(speech)
    smooth = np.convolve(env, np.ones(5) / 5, mode="same")
    floor = np.percentile(smooth[speech], 35) if speech.any() else 0.0
    syllables = sum(1 for i in range(1, len(smooth) - 1)
                    if smooth[i] > smooth[i - 1] and smooth[i] >= smooth[i + 1]
                    and smooth[i] > floor and speech[i])
    speech_s = float(speech.sum()) * 0.01
    return {"duration_s": len(sig) / sr,
            "speech_s": speech_s,
            "syllables": syllables,
            "articulation": syllables / max(speech_s, 1e-9),
            "lead_s": float(idx[0]) * 0.01 if idx.size else 0.0,
            "tail_s": float(len(speech) - 1 - idx[-1]) * 0.01 if idx.size else 0.0}
