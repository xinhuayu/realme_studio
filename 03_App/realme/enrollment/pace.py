"""
Matching the draft voice's pace to the speaker's own.

This is used for DIALOGUE, where two voices alternate and a difference in
pace is audible as a mismatch.

Lecture and narration drafts keep the voice's own pace: a draft is replaced
by the cloned voice before anything is published.

Speaking rates differ between people, so the speed is measured, not fixed:

  your rate     words in the enrolment transcript / seconds of SPEECH in the
                enrolment recording (silence removed, so pauses do not count
                as slow speech)

  piper's rate  the same text through piper at speed 1.0, measured the same way

  speed         your rate / piper's rate, clamped

Clamped so that a wrong transcript cannot produce an extreme speed.
"""
from __future__ import annotations
import re
import subprocess
from pathlib import Path

from realme.core.media import duration_of, require

#: Speech at -32 dB and gaps from 0.18 s: the same thresholds
#: `prosody.measure_reference_pauses` uses, so both measurements agree on
#: what counts as silence.
NOISE_DB = "-32dB"
MIN_SILENCE_S = 0.18

MIN_SPEED, MAX_SPEED = 0.55, 1.25
FALLBACK_SPEED = 0.8

#: A measured correction this small is reported as no correction at all.
#:
#: The measurement is still made and printed. Two takes of the same passage
#: by the same person differ by 3-5%, so a smaller gap between a speaker and
#: their clone is within normal variation. Retiming has a cost: atempo removes
#: 2-11% of the energy above 4 kHz, which makes a voice sound smoothed.
RATE_DEADBAND = 0.03

#: What a person reading aloud can actually be, in words per minute of SPEECH
#: (pauses excluded, so these are higher than the familiar 110-160 figures for
#: conversation). A rate outside this band indicates a mismatch between the
#: recording and its transcript, for example a re-recorded enrolment whose
#: transcript still describes the earlier passage. Such a rate is refused
#: rather than clamped and stored.
HUMAN_WPM_MIN, HUMAN_WPM_MAX = 90.0, 260.0


def check_human_wpm(wpm: float, wav: Path, text: str) -> float:
    """`wpm` back, or a refusal that says which of the two inputs is wrong.

    The message carries the word count and the seconds of speech; the usual
    fix is to paste the transcript of the passage that was actually read.
    """
    if HUMAN_WPM_MIN <= wpm <= HUMAN_WPM_MAX:
        return wpm
    words = len([w for w in re.split(r"\s+", (text or "").strip()) if w])
    speech = duration_of(wav) - silence_seconds(wav)
    direction = "faster" if wpm > HUMAN_WPM_MAX else "slower"
    raise ValueError(
        f"The enrolment measures {wpm:.0f} words a minute of speech, which is "
        f"{direction} than anyone lectures ({HUMAN_WPM_MIN:.0f}-"
        f"{HUMAN_WPM_MAX:.0f} is the plausible band). That is a mismatch "
        f"between the recording and its transcript, not a speaking rate: "
        f"{words} words of text against {speech:.0f} s of speech in "
        f"{Path(wav).name}. Paste the transcript of the passage you actually "
        f"read, then measure again.")


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
    property of the voice file and is not recorded anywhere. It costs a few
    seconds; piper runs at about RTF 0.08.
    """
    reference_wav = Path(reference_wav)
    mine = words_per_minute(reference_wav, reference_text)
    if mine is None:
        raise ValueError(
            "Not enough to measure from: the enrolment transcript needs at "
            "least a dozen words and a few seconds of speech.\n"
            "    realme voice enroll <file> --transcript-file <txt>")
    mine = check_human_wpm(mine, reference_wav, reference_text)

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

    The rates are stored, not the quotient: the matching speed is computed
    per voice at cast time, so a ratio measured against one speaker is never
    applied to another.

    Piper runs at about RTF 0.08, so measuring three voices costs a few
    seconds.
    """
    from realme.adapters.tts import DRAFT_VOICES
    voices = list(voices or DRAFT_VOICES)
    # Checked once, before the loop. A stale transcript makes every voice
    # fail, and the collected failure below would point at the engines
    # instead of the text file.
    once = words_per_minute(Path(reference_wav), reference_text)
    if once:
        check_human_wpm(once, Path(reference_wav), reference_text)
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
# `calibrate` above measures PIPER, for matching a draft voice. A cloned voice
# can need it too: zero-shot cloning copies the reference's delivery as well
# as its timbre, so a reference read slowly for the microphone gives a clone
# that lectures slowly. A 5% difference adds about forty seconds to a
# fifteen-minute lecture.
#
# Same arithmetic as for piper, with the engine swapped: measure the target
# rate, measure what the engine does at pace 1.0, divide.

def engine_wpm(engine, sample: str, out_wav: Path, *, pace: float = 1.0) -> float | None:
    """Words per minute of speech this engine produces, measured.

    An engine's natural rate is not recorded anywhere: for piper it is a
    property of the voice file, for a hosted model of the clone's reference.
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
    against it would keep that slow pace.
    """
    reference_wav = Path(reference_wav)
    mine = target_wpm or words_per_minute(reference_wav, reference_text)
    if not mine:
        raise ValueError(
            "Nothing to match: the enrolment needs a transcript of at least a "
            "dozen words.\n"
            "    realme voice enroll <file> --transcript-file <txt>")
    if not target_wpm:
        mine = check_human_wpm(mine, reference_wav, reference_text)

    sample = " ".join([w for w in re.split(r"\s+", reference_text.strip())
                       if w][:words])
    from realme.adapters.factory import build
    engine = build(engine_name, quiet=True, **build_kw)
    # Measure the engine, not the engine plus the last correction.
    #
    # An adapter that declares `retimes_after` leaves the stretching to the
    # pipeline, so what `engine_wpm` gets back below is already the engine's
    # own rate. This stays as the guard for the other case: an engine that
    # applies a stored correction itself would be measured through it, and
    # each calibration would then multiply on top of the last.
    if (getattr(engine, "rate_match", 1.0) != 1.0
            and not getattr(engine, "retimes_after", False)):
        log(f"  (ignoring the stored correction {engine.rate_match:.3f} "
            f"while measuring)")
        engine.rate_match = 1.0
    engine.preflight()
    workdir = Path(workdir) if workdir else reference_wav.parent
    workdir.mkdir(parents=True, exist_ok=True)
    out = workdir / f"_pace_{engine_name.replace(':', '_')}.wav"
    theirs = engine_wpm(engine, sample, out, pace=1.0)
    if not theirs:
        raise ValueError("The calibration render was too short to measure.")

    raw = mine / theirs
    log(f"  target     {mine:5.1f} words/min of speech")
    log(f"  {engine_name:<22} {theirs:5.1f} words/min at pace 1.0")
    # A calibration outside the bounds is refused, not clamped. A clone speaks
    # at roughly the rate of its reference, so a correction near the bounds
    # means one of the two measurements is wrong, and the retiming would be
    # audible as an artefact anyway.
    if not MIN_SPEED <= raw <= MAX_SPEED:
        raise ValueError(
            f"Not stored: matching this would need a {raw:.3f}x retime, "
            f"outside the {MIN_SPEED:.2f}-{MAX_SPEED:.2f} range that can be "
            f"applied without being audible. The target measured "
            f"{mine:.0f} words a minute of speech and {engine_name} measured "
            f"{theirs:.0f}; a cloned voice should be within a few per cent of "
            f"its reference, so check that the transcript describes the "
            f"recording before trusting either number.")
    if abs(raw - 1.0) <= RATE_DEADBAND:
        log(f"  pace       1.000  (measured {raw:.3f}, inside the "
            f"{RATE_DEADBAND:.0%} take-to-take spread -- not retimed)")
        speed = 1.0
    else:
        speed = raw
        log(f"  pace       {speed:.3f}")
    return {"engine": engine_name, "target_wpm": round(mine, 1),
            "engine_wpm": round(theirs, 1), "pace": round(speed, 3),
            "clamped": False, "sample_wav": str(out)}


def reference_drift(reference_wav, reference_text: str,
                    previous_wpm: float | None) -> dict:
    """Has a re-recorded enrolment changed how fast you talk?

    Reported at enrolment, because the clone inherits the change in pace.
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
# calls, the second means keep the size and retime. Syllable count separates
# the two where duration cannot.
#
# Two further things duration gets wrong:
#
#   * Edge silence. Every call comes back with a short lead-in and tail
#     (a few tenths of a second each). Comparing one long call against the
#     SUM of N short ones counts N copies of that padding as a shortfall.
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
    # Two thresholds, not one. With a purely relative threshold, digital
    # silence (uniformly 1e-12) is 0 dB below its own peak and the whole clip
    # measures as speech. The absolute floor (-80 dBFS) lets a silent file
    # measure as silent.
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
