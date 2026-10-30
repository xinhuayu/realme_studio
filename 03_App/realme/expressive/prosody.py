"""
Measuring what a turn sounds like.

Four numbers per turn, chosen because they are what a listener reports when
they say a delivery is flat:

    pitch      median F0 and the spread around it, in semitones
    loudness   RMS energy, in dB
    rate       words per second of SPEECH, excluding the silence
    pause      the silence before the turn begins

Semitones rather than hertz for pitch, and dB rather than amplitude for
loudness, because both are perceptual ratios: a 20 Hz rise is large for a low
voice and inaudible for a high one, and the same is true of a fixed amplitude
step. Measuring in linear units makes two speakers incomparable and makes the
same speaker look more variable when they are simply louder.

Rate excludes silence deliberately. A turn with a long pause in the middle is
not a slow turn; it is a normal turn with a pause, and averaging over the
silence turns one into the other.

NUMPY AND THE STANDARD LIBRARY, NOTHING ELSE. The first version of this file
used librosa and soundfile, which are installed in the author's sandbox and not
on the machine this has to run on -- it failed on the first real attempt with
ModuleNotFoundError. An exploratory module is not worth a dependency, and the
project already owns everything needed: `verify.acoustic.read_wav` reads a WAV
to mono float, and `enrollment.voice` has been tracking pitch by
autocorrelation over the same 60-400 Hz band since enrollment was built. One
reader, one method.

Everything here is measurement only. Nothing in this module changes audio, and
nothing in the application calls it.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path

#: Voiced-frame floor and ceiling in hertz. The same band the enrollment
#: analysis uses, and wide enough for any adult speaker of either sex; a
#: tighter band would silently discard the ends of an expressive contour,
#: which is precisely what this is trying to see.
F0_MIN, F0_MAX = 60.0, 400.0

#: A frame quieter than this below the clip's loudest counts as silence.
#: -40 dB is the usual speech/silence line.
SILENCE_DB = -40.0

#: Analysis framing, in seconds. 40 ms holds at least two periods of the
#: lowest pitch in band, which is the minimum autocorrelation can work with.
FRAME_S, HOP_S = 0.040, 0.010

#: Autocorrelation peak height, relative to lag zero, for a frame to count as
#: voiced. 0.3 is what the enrollment analysis uses; below it the "pitch" is
#: the tracker finding structure in noise.
VOICING_FLOOR = 0.3


@dataclass
class TurnProsody:
    """What one rendered turn measured. All fields perceptual units."""
    seconds: float = 0.0
    speech_seconds: float = 0.0
    lead_silence_s: float = 0.0
    words: int = 0
    rate_wps: float = 0.0          # words per second of speech
    f0_median_hz: float = 0.0
    f0_spread_st: float = 0.0      # interquartile range, semitones
    f0_range_st: float = 0.0       # 10th-90th percentile, semitones
    voiced_fraction: float = 0.0
    energy_db: float = 0.0
    energy_spread_db: float = 0.0
    ok: bool = True
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _semitones(hi: float, lo: float) -> float:
    """Ratio of two frequencies in semitones. Zero when either is missing."""
    import math
    if hi <= 0 or lo <= 0:
        return 0.0
    return 12.0 * math.log2(hi / lo)


def _pitch_track(frames, sr: int):
    """F0 per frame by autocorrelation, NaN where the frame is not voiced.

    The same method the enrollment analysis uses, vectorised over frames. It is
    crude next to YIN and entirely consistent, which is what a comparison
    between two renders of the same words needs.
    """
    import numpy as np
    lo, hi = int(sr / F0_MAX), int(sr / F0_MIN)
    out = np.full(len(frames), np.nan, dtype=np.float64)
    if frames.size == 0 or hi <= lo or frames.shape[1] <= hi:
        return out
    x = frames - frames.mean(axis=1, keepdims=True)
    # Autocorrelation through the FFT: one transform per frame instead of an
    # O(n^2) correlate, which matters when a run measures a few hundred turns.
    n = 1 << int(np.ceil(np.log2(2 * x.shape[1])))
    spec = np.fft.rfft(x, n=n, axis=1)
    ac = np.fft.irfft(spec * np.conjugate(spec), n=n, axis=1)[:, :x.shape[1]]
    zero = ac[:, 0]
    band = ac[:, lo:hi]
    if band.size == 0:
        return out
    peak_at = band.argmax(axis=1) + lo
    peak = band.max(axis=1)
    good = (zero > 0) & (peak / np.maximum(zero, 1e-12) > VOICING_FLOOR)
    out[good] = sr / peak_at[good]
    return out


def measure(wav: Path, words: int = 0) -> TurnProsody:
    """
    Measure one rendered turn.

    Returns a result with ok=False and a note rather than raising: a run over
    forty turns should report the one that could not be measured and carry on,
    not lose the other thirty-nine.
    """
    import numpy as np
    try:
        from realme.verify.acoustic import read_wav
    except ImportError as e:                    # numpy missing: nothing works
        return TurnProsody(ok=False, note=f"cannot read audio: {e}")
    try:
        y, sr = read_wav(Path(wav))
    except Exception as e:                      # unreadable, truncated, absent
        return TurnProsody(ok=False, note=f"could not read {Path(wav).name}: {e}")
    if y.size == 0:
        return TurnProsody(ok=False, note=f"{Path(wav).name} is empty")

    p = TurnProsody(words=int(words))
    p.seconds = float(y.size / sr)

    frame, hop = max(8, int(sr * FRAME_S)), max(1, int(sr * HOP_S))
    if y.size < frame:
        p.ok = False
        p.note = f"{Path(wav).name} is shorter than one analysis frame"
        return p
    frames = np.lib.stride_tricks.sliding_window_view(y, frame)[::hop]
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
    peak = float(rms.max()) or 1e-12
    db = 20.0 * np.log10(np.maximum(rms, 1e-12) / peak)
    voiced_frames = db > SILENCE_DB
    frame_s = hop / sr
    p.speech_seconds = float(voiced_frames.sum() * frame_s)

    # Leading silence, which is how a turn can sound hesitant or eager before
    # a single word of it is heard.
    first = int(np.argmax(voiced_frames)) if voiced_frames.any() else len(db)
    p.lead_silence_s = float(first * frame_s)

    if p.speech_seconds > 0 and words:
        p.rate_wps = float(words / p.speech_seconds)

    # Loudness over the SPEECH only. Averaging in the silence makes a turn
    # with a long pause look quiet.
    if voiced_frames.any():
        loud = rms[voiced_frames]
        ref = float(np.sqrt(np.mean(loud ** 2)))
        p.energy_db = float(20.0 * np.log10(max(ref, 1e-12)))
        q = np.percentile(20.0 * np.log10(np.maximum(loud, 1e-12)), [10, 90])
        p.energy_spread_db = float(q[1] - q[0])

    f0 = _pitch_track(frames, sr)
    # Keep frames that are both voiced by the tracker and not silence; an
    # out-of-band reading inside a pause is the tracker guessing, and those
    # guesses are what make a steady voice look wild.
    good = voiced_frames & np.isfinite(f0)
    p.voiced_fraction = float(good.sum() / max(1, len(f0)))
    if good.sum() >= 8:
        vals = f0[good]
        p.f0_median_hz = float(np.median(vals))
        q1, q3 = np.percentile(vals, [25, 75])
        d1, d9 = np.percentile(vals, [10, 90])
        p.f0_spread_st = _semitones(q3, q1)
        p.f0_range_st = _semitones(d9, d1)
    else:
        p.note = "too few voiced frames to measure pitch"
    return p
