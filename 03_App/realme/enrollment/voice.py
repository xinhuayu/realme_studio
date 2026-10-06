"""
Voice enrollment: check a recording, then polish it — the "Better Me" idea.

The premise from the original design is right and worth keeping: nobody wants
their tired Tuesday-afternoon voice preserved for a whole semester. What you
want is *you*, on a good day, in a decent room. Not a different person —
recognisably you, with the room noise, the sibilance and the boominess taken
out and the level made consistent.

Where this differs from the original: it happens ONCE, on the reference clip,
before cloning. The cloning model then learns the polished voice and every
lecture inherits it for free. Polishing every render instead would be slower,
and would drift as the chain changed.

Everything here is ffmpeg. No extra dependency, nothing to download, and each
filter is a standard broadcast tool rather than anything exotic:

    highpass      remove desk rumble, aircon, footsteps, plosive thump
    afftdn        FFT denoise - takes out steady hiss and fan noise
    equalizer     cut boxiness ~300 Hz, add chest ~180 Hz, add clarity ~3 kHz
    deesser       tame harsh "s" - the single most fatiguing thing over 30 min
    acompressor   even out the loud and quiet parts of a sentence
    loudnorm      EBU R128 to -16 LUFS, the podcast/lecture standard
    alimiter      catch stray peaks without audible pumping
"""
from __future__ import annotations
import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

from realme.core.media import require, run, duration_of, probe

# Presets are deliberately conservative. A voice clone trained on an
# over-processed reference sounds processed in every lecture you ever make,
# and that is not undoable.
PRESETS: dict[str, dict] = {
    "off": {"label": "No processing", "chain": []},
    "clean": {
        "label": "Clean up only — remove noise and rumble, level it",
        "chain": ["highpass=f=75",
                  "afftdn=nf=-25",
                  "acompressor=threshold=-18dB:ratio=2.5:attack=8:release=180:makeup=1.5",
                  "loudnorm=I=-16:TP=-1.5:LRA=11",
                  "alimiter=limit=0.95"],
    },
    "natural": {
        "label": "Natural polish — clean up, soften sibilance, gentle warmth",
        "chain": ["highpass=f=75",
                  "afftdn=nf=-25",
                  "equalizer=f=300:t=q:w=1.2:g=-2.5",     # take out boxiness
                  "equalizer=f=180:t=q:w=1.0:g=1.5",      # a little chest
                  "equalizer=f=3200:t=q:w=1.4:g=1.8",     # clarity, not sizzle
                  "deesser=i=0.4",
                  "acompressor=threshold=-18dB:ratio=3:attack=6:release=160:makeup=2",
                  "loudnorm=I=-16:TP=-1.5:LRA=11",
                  "alimiter=limit=0.95"],
    },
    "warm": {
        "label": "Warm broadcast — fuller and closer, for a quiet room",
        "chain": ["highpass=f=70",
                  "afftdn=nf=-28",
                  "equalizer=f=320:t=q:w=1.3:g=-3",
                  "equalizer=f=150:t=q:w=0.9:g=2.5",
                  "equalizer=f=2600:t=q:w=1.5:g=2",
                  "equalizer=f=8000:t=h:w=2:g=1",
                  "deesser=i=0.5",
                  "acompressor=threshold=-20dB:ratio=3.5:attack=5:release=140:makeup=2.5",
                  "loudnorm=I=-16:TP=-1.5:LRA=9",
                  "alimiter=limit=0.95"],
    },
}


# ---------------------------------------------------------------- treatment
#
# Continuous tone controls, adopted from the real_voice_qwen3 project, whose
# bounds were arrived at by actually listening rather than by guessing. They are
# deliberately narrow: Qwen3-TTS (or whichever engine) remains the identity
# anchor, and these only move the tone colour slightly around your own voice.
#
# The presets above are named points in this same space, so you can start from
# one and then nudge a single control rather than choosing between four fixed
# options.

PACE_MIN, PACE_MAX = 0.94, 1.06
PITCH_MIN, PITCH_MAX = -0.75, 0.75          # semitones
WARMTH_MIN, WARMTH_MAX = -0.35, 0.35
CLARITY_MIN, CLARITY_MAX, CLARITY_NEUTRAL = 0.35, 0.65, 0.50
SMOOTHING_MIN, SMOOTHING_MAX = 0.0, 0.30
BRIGHTNESS_MIN, BRIGHTNESS_MAX = -0.35, 0.35
VIGOR_MIN, VIGOR_MAX = -0.30, 0.30
VIVIDNESS_MIN, VIVIDNESS_MAX = -0.35, 0.35
DEFAULT_WARMTH = 0.05


@dataclass(frozen=True)
class Treatment:
    """
    Fine tone adjustment applied to the reference before cloning.

    `clarity` is neutral at 0.50, not 0. Everything else is neutral at 0. The
    default adds only a very small warmth lift and does not pitch-shift or
    time-stretch at all.
    """
    pace: float = 1.0
    pitch_semitones: float = 0.0
    warmth: float = DEFAULT_WARMTH
    clarity: float = CLARITY_NEUTRAL
    smoothing: float = 0.0
    brightness: float = 0.0
    vigor: float = 0.0
    vividness: float = 0.0

    def validated(self) -> "Treatment":
        limits = {
            "pace": (self.pace, PACE_MIN, PACE_MAX),
            "pitch_semitones": (self.pitch_semitones, PITCH_MIN, PITCH_MAX),
            "warmth": (self.warmth, WARMTH_MIN, WARMTH_MAX),
            "clarity": (self.clarity, CLARITY_MIN, CLARITY_MAX),
            "smoothing": (self.smoothing, SMOOTHING_MIN, SMOOTHING_MAX),
            "brightness": (self.brightness, BRIGHTNESS_MIN, BRIGHTNESS_MAX),
            "vigor": (self.vigor, VIGOR_MIN, VIGOR_MAX),
            "vividness": (self.vividness, VIVIDNESS_MIN, VIVIDNESS_MAX),
        }
        for name, (value, lo, hi) in limits.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            if value < lo or value > hi:
                raise ValueError(f"{name} must be between {lo:g} and {hi:g}")
        return self

    def is_neutral(self) -> bool:
        return (self.pace == 1.0 and self.pitch_semitones == 0.0
                and self.warmth == 0.0 and self.clarity == CLARITY_NEUTRAL
                and self.smoothing == 0.0 and self.brightness == 0.0
                and self.vigor == 0.0 and self.vividness == 0.0)

    def filters(self, sample_rate: int = 24000) -> list[str]:
        """
        The ffmpeg chain for this treatment.

        Pitch and pace are decoupled, which is the part worth copying: shifting
        pitch with `asetrate` also changes the tempo, so `atempo` is given
        `pace / pitch_factor` to undo exactly that much. Change the pitch and
        the speaking rate stays put.
        """
        self.validated()
        pitch_factor = 2 ** (self.pitch_semitones / 12.0)
        tempo = self.pace / pitch_factor
        # The rate the caller is working at, not a constant. Resampling a
        # 48 kHz enrolment down to 24 here threw away the top octave silently.
        sr = int(sample_rate)
        chain = [
            f"aresample={sr}",
            f"asetrate={sr * pitch_factor:.6f}",
            f"aresample={sr}",
            f"atempo={tempo:.8f}",
            f"bass=g={2.0 * self.warmth:.3f}:f=180:w=0.6",
            f"equalizer=f=3200:t=q:w=1.2:g={2.0 * (self.clarity - CLARITY_NEUTRAL):.3f}",
            f"equalizer=f=4200:t=q:w=1.1:g={2.0 * self.brightness:.3f}",
            f"equalizer=f=1900:t=q:w=1.0:g={0.70 * self.vigor:.3f}",
            f"equalizer=f=2500:t=q:w=0.9:g={0.80 * self.vividness:.3f}",
            f"volume={10 ** ((0.75 * self.vigor) / 20.0):.6f}",
        ]
        if self.smoothing > 0.0 or self.vigor > 0.0:
            ratio = 1.0 + (0.8 * self.smoothing) + (0.25 * self.vigor)
            makeup = 1.0 + (1.5 * self.smoothing) + (0.12 * self.vigor)
            chain.append(f"acompressor=threshold=-24dB:ratio={ratio:.3f}:"
                         f"attack=18:release=140:makeup={makeup:.3f}")
        chain.append("alimiter=limit=0.98")
        return chain


# The named presets, expressed as points in the continuous space above, so the
# two systems agree instead of drifting apart.
PRESET_TREATMENTS: dict[str, Treatment] = {
    "off": Treatment(warmth=0.0, clarity=CLARITY_NEUTRAL),
    "clean": Treatment(warmth=0.0, clarity=CLARITY_NEUTRAL, smoothing=0.10),
    "natural": Treatment(warmth=0.10, clarity=0.55, smoothing=0.12, vividness=0.05),
    "warm": Treatment(warmth=0.22, clarity=0.53, smoothing=0.18,
                      brightness=0.08, vigor=0.10, vividness=0.12),
}


@dataclass
class VoiceReport:
    path: str
    duration_s: float
    sample_rate: int
    channels: int
    peak_db: float
    rms_db: float
    noise_floor_db: float
    snr_db: float
    clipped_samples: int
    problems: list[str]
    advice: list[str]
    verdict: str

    def ok(self) -> bool:
        return self.verdict.startswith("good") or self.verdict.startswith("usable")


def _astats(path: Path) -> dict:
    """Time-domain statistics straight from ffmpeg."""
    proc = subprocess.run(
        [require("ffmpeg"), "-hide_banner", "-i", str(path),
         "-af", "astats=metadata=1:reset=0", "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    text = proc.stderr
    out: dict[str, float] = {}
    for key, pat in (("peak_db", r"Peak level dB:\s*(-?[\d.]+|-inf)"),
                     ("rms_db", r"RMS level dB:\s*(-?[\d.]+|-inf)"),
                     ("rms_trough_db", r"RMS trough dB:\s*(-?[\d.]+|-inf)"),
                     ("flat", r"Flat factor:\s*([\d.]+)"),
                     ("peak_count", r"Number of samples:\s*(\d+)")):
        m = re.search(pat, text)
        if m:
            v = m.group(1)
            out[key] = -120.0 if v == "-inf" else float(v)
    m = re.search(r"Number of clipped samples:\s*(\d+)", text)
    out["clipped"] = float(m.group(1)) if m else 0.0
    return out


def analyze(path: Path) -> VoiceReport:
    """
    Judge a reference recording and say what to do about it, in plain words.

    The goal is not a score. It is: should you re-record, and if so, what should
    you change? A number nobody can act on is worse than useless here.
    """
    path = Path(path)
    info = probe(path)
    stream = next((s for s in info["streams"] if s.get("codec_type") == "audio"), {})
    sr = int(stream.get("sample_rate", 0) or 0)
    ch = int(stream.get("channels", 0) or 0)
    dur = duration_of(path)
    st = _astats(path)

    peak = st.get("peak_db", -120.0)
    rms = st.get("rms_db", -120.0)
    trough = st.get("rms_trough_db", -120.0)
    clipped = int(st.get("clipped", 0))
    # RMS trough approximates the quiet moments between phrases, i.e. the room.
    noise = trough
    snr = max(0.0, rms - noise) if noise > -119 else 60.0

    problems: list[str] = []
    advice: list[str] = []

    if dur < 8:
        problems.append(f"only {dur:.1f}s long")
        advice.append("Record 30-60 seconds. Most cloning services want at "
                      "least 10s and do better with more.")
    if sr and sr < 22050:
        problems.append(f"sample rate {sr} Hz")
        advice.append("Record at 44.1 kHz or 48 kHz. Low rates throw away the "
                      "high frequencies that make a voice sound like itself.")
    if peak > -0.5:
        problems.append(f"peaks at {peak:.1f} dBFS")
        advice.append("It is hitting the ceiling. Lower the input gain and "
                      "re-record - clipping cannot be repaired afterwards.")
    if clipped > 50:
        problems.append(f"{clipped} clipped samples")
        advice.append("Audible distortion. Re-record with the gain lower.")
    if peak < -24:
        problems.append(f"very quiet (peak {peak:.1f} dBFS)")
        advice.append("Move closer to the mic or raise the gain. Amplifying a "
                      "quiet take afterwards raises the room noise with it.")
    if snr < 18:
        problems.append(f"noisy (about {snr:.0f} dB signal-to-noise)")
        advice.append("There is a lot of room in this. Turn off fans and air "
                      "conditioning, close the window, move away from the "
                      "computer, and get closer to the mic. Soft furnishings "
                      "help more than anything you can buy.")
    if ch > 1:
        advice.append("Stereo will be mixed to mono - that is fine and expected.")

    if not problems:
        verdict = "good — clone from this"
    elif any("clipped" in p or "peaks at" in p for p in problems):
        verdict = "re-record — clipping cannot be fixed"
    elif snr < 12:
        verdict = "re-record — too much room noise to clone cleanly"
    else:
        verdict = "usable — polishing will help, but a better take would help more"

    return VoiceReport(path=str(path), duration_s=round(dur, 2), sample_rate=sr,
                       channels=ch, peak_db=round(peak, 1), rms_db=round(rms, 1),
                       noise_floor_db=round(noise, 1), snr_db=round(snr, 1),
                       clipped_samples=clipped, problems=problems,
                       advice=advice, verdict=verdict)


def _measured_loudnorm(src: Path, target_i: float = -16.0,
                       tp: float = -1.5, lra: float = 11.0) -> str:
    """
    Two-pass loudnorm: measure the file, then apply ONE fixed correction.

    Single-pass `loudnorm` is adaptive. It rides the level as the file plays,
    which on a lecture reference means the gain moves between sentences -- heard
    as "not steady across sentences", and reported that way. The whole point of
    a reference clip is that it is one consistent example of a voice.

    ffmpeg's own answer is the two-pass form: measure with `print_format=json`,
    feed the measurements back, and the second pass applies a single offset. If
    the measurement fails for any reason this returns the plain single-pass
    filter rather than nothing, because a slightly-riding reference beats no
    reference.
    """
    import json
    probe_chain = (f"loudnorm=I={target_i}:TP={tp}:LRA={lra}:"
                   f"print_format=json")
    proc = subprocess.run(
        [require("ffmpeg"), "-hide_banner", "-i", str(src),
         "-af", probe_chain, "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    try:
        blob = proc.stderr[proc.stderr.rindex("{"):proc.stderr.rindex("}") + 1]
        m = json.loads(blob)
        return (f"loudnorm=I={target_i}:TP={tp}:LRA={lra}:"
                f"measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
                f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:"
                f"offset={m['target_offset']}:linear=true:print_format=summary")
    except (ValueError, KeyError):
        return f"loudnorm=I={target_i}:TP={tp}:LRA={lra}"


def sibilance_ratio(path: Path) -> float:
    """Share of voiced energy between 5 and 9 kHz. A rough sibilance meter."""
    import numpy as np
    from realme.verify.acoustic import read_wav
    sig, sr = read_wav(Path(path))
    if sig.size < sr // 2 or sr < 12000:
        return 0.0
    frame = 2048
    n = (sig.size // frame) * frame
    frames = sig[:n].reshape(-1, frame) * np.hanning(frame)
    spec = np.abs(np.fft.rfft(frames, axis=1))
    freqs = np.fft.rfftfreq(frame, 1 / sr)
    total = spec.sum()
    if total <= 0:
        return 0.0
    band = spec[:, (freqs >= 5000) & (freqs <= 9000)].sum()
    return float(band / total)


def repairs_for(src: Path) -> list[tuple[str, str]]:
    """
    Only the corrections this recording actually needs, with the reason.

    The chain used to be fixed: every take was denoised and de-essed whether or
    not it had hiss or sibilance to remove. On a clean recording a denoiser has
    no noise to find and takes detail instead -- which is heard as smearing, and
    was, by the person whose voice it was.
    """
    out = [("highpass=f=75", "removed rumble below 75 Hz")]
    try:
        snr = analyze(src).snr_db
    except Exception:
        snr = 0.0
    if snr and snr < 30:
        out.append(("afftdn=nf=-25", f"denoised (only {snr:.0f} dB SNR)"))
    try:
        sib = sibilance_ratio(src)
    except Exception:
        sib = 0.0
    if sib > 0.06:
        out.append(("deesser=i=0.35", f"softened sibilance ({sib * 100:.0f}% of energy at 5-9 kHz)"))
    return out


def treat(src: Path, dst: Path, treatment: Treatment,
          *, sample_rate: int | None = None, repair: bool = True) -> Path:
    """
    Apply a continuous Treatment, optionally after the corrective clean-up.

    `repair` runs the fix-the-recording half first (rumble, hiss, sibilance,
    level); the Treatment is the taste half on top. They are separable because
    they answer different questions — "is this recording usable" versus "do I
    want to sound slightly warmer".

    Three things here were wrong, and a listener found all three by preferring
    the untreated take:

    * `loudnorm` ran single-pass, which is adaptive -- the gain moved between
      sentences. It is measured first now, and applied as one fixed offset.
    * `afftdn` denoised every recording, including clean ones, where there is
      nothing to remove and the only audible effect is smearing. It now runs
      only when the recording actually needs it.
    * the output was forced to 24 kHz, so a 48 kHz enrolment lost its top octave
      on the way in for no reason anyone had asked for. The source rate is kept
      unless a caller names one.
    """
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if sample_rate is None:
        try:
            stream = next(s for s in probe(src)["streams"]
                          if s.get("codec_type") == "audio")
            sample_rate = int(stream.get("sample_rate") or 24000)
        except Exception:
            sample_rate = 24000

    chain: list[str] = []
    applied: list[str] = []
    if repair:
        for f, why in repairs_for(src):
            chain.append(f)
            applied.append(why)
        chain.append(_measured_loudnorm(src))
        applied.append("levelled to a fixed target (measured first)")
    chain += treatment.filters(sample_rate=sample_rate)
    run([require("ffmpeg"), "-y", "-i", str(src), "-af", ",".join(chain),
         "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(dst)],
        "apply voice treatment")
    treat.last_applied = applied      # what was done, for a caller to print
    return dst


def beautify(src: Path, dst: Path, preset: str = "natural",
             *, sample_rate: int | None = None) -> Path:
    """
    Apply the chosen preset and write a mono reference.

    **Through `treat`, which is the one chain.** `PRESET_TREATMENTS` says of
    itself that the presets are "expressed as points in the continuous space
    above, so the two systems agree instead of drifting apart" -- and then this
    function used a second, hand-written chain in `PRESETS` and they drifted
    anyway. That second chain is what produced the reference a listener rejected:
    a 3:1 compressor with a 6 ms attack, a denoiser on a recording with 56 dB of
    signal-to-noise, and a single-pass `loudnorm` riding the level from sentence
    to sentence. `PRESETS` keeps its labels; the audio comes from one place now.

    Mono because every cloning backend wants it. The sample rate follows the
    source unless a caller names one -- the engines resample to whatever they
    need, and doing it twice only loses.
    """
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}. Choose from: {', '.join(PRESETS)}")
    src, dst = Path(src), Path(dst)
    if preset == "off":
        # "No processing" means no processing. This used to run the file through
        # the filter chain anyway -- zero-gain EQ, a resample and a limiter --
        # which is nearly transparent and still not what the label promises. A
        # WAV is copied; anything else is converted, because the engines cannot
        # read AAC and that is a format change, not a treatment.
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix.lower() == ".wav":
            shutil.copy2(src, dst)
        else:
            from realme.core.media import to_reference_wav
            to_reference_wav(src, dst)
        treat.last_applied = ["copied unchanged"]
        return dst
    return treat(src, dst, PRESET_TREATMENTS[preset],
                 sample_rate=sample_rate, repair=True)


def compare(src: Path, outdir: Path, presets: list[str] | None = None,
            *, seconds: float = 12.0) -> dict[str, Path]:
    """
    Render a short excerpt through each preset so they can be heard back to back.

    Choosing a voice by reading filter names is hopeless; choosing it by
    listening takes fifteen seconds.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    excerpt = outdir / "_excerpt.wav"
    run([require("ffmpeg"), "-y", "-i", str(src), "-t", str(seconds),
         "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(excerpt)],
        "trim excerpt")
    out: dict[str, Path] = {}
    for name in (presets or list(PRESETS)):
        out[name] = beautify(excerpt, outdir / f"preset_{name}.wav", name)
    return out


CONSENT_SCRIPT = (
    "I am {name}. I am the owner of this voice, and I consent to a synthetic "
    "voice model of it being created and used for my own teaching and academic "
    "work. Today's date is {date}."
)


def consent_text(name: str, date: str) -> str:
    """
    The wording to read for the consent recording.

    Google's Chirp 3 requires its own exact sentence and will reject anything
    else; this is for your own records and for institutions that ask. Keep it
    with the reference audio — it is the artifact that answers the question
    later, and it costs ten seconds now.
    """
    return CONSENT_SCRIPT.format(name=name or "the speaker", date=date)


# ------------------------------------------------- does it still sound like me
#
# Adopted from real_voice_qwen3. This answers a different question from
# `realme/verify/acoustic.py`: that one asks "did the engine say the right
# word", this asks "does the clone still sound like the person". Both are
# needed, and neither substitutes for the other.
#
# The framing from the original is kept deliberately: this is descriptive
# acoustic resemblance, NOT biometric speaker verification. It is a tuning aid.
# A high score does not prove identity and a low one does not disprove it.

SIGNATURE_FEATURES = (
    # (name, scale at which a difference costs full marks, weight)
    ("pitch_median_hz", 120.0, 0.24),
    ("pitch_variation_semitones", 5.0, 0.16),
    ("spectral_centroid_hz", 1500.0, 0.18),
    ("dynamic_range_db", 10.0, 0.14),
    ("zero_crossing_rate", 0.04, 0.10),
    ("rms_dbfs", 12.0, 0.18),
)


#: Every signature is measured at ONE sample rate, whatever the file's own.
#:
#: Two of these six features are defined in terms of the sample rate and are
#: therefore meaningless across files that differ in it. Spectral centroid is an
#: energy-weighted mean over the spectrum, and the spectrum runs to Nyquist -- so
#: a 48 kHz file is averaged over twice the bandwidth of a 24 kHz one and reads
#: far brighter for that reason alone. Zero-crossing rate is counted per sample,
#: so it halves when the rate halves.
#:
#: Found the hard way: comparing one enrolled clip at 48 kHz against its own
#: tuned copy at 24 kHz returned a centroid similarity of 0.0 and an overall
#: 61.6%, for two files that are the same recording. Every earlier comparison
#: between engines that emit different rates -- 16 kHz from kNN-VC, 22.05 from
#: piper, 24 from qwen3cpp -- carried some of the same distortion.
ANALYSIS_RATE = 22050


def _to_analysis_rate(sig, sr: int):
    """Band-limit and resample in the frequency domain -- no aliasing, no SciPy."""
    import numpy as np
    if sr == ANALYSIS_RATE or sig.size == 0:
        return sig, sr
    n_out = int(round(sig.size * ANALYSIS_RATE / sr))
    if n_out < 2:
        return sig, sr
    spec = np.fft.rfft(sig)
    keep = min(len(spec), n_out // 2 + 1)
    out = np.fft.irfft(spec[:keep], n=n_out) * (n_out / sig.size)
    return out.astype(np.float64), ANALYSIS_RATE


def acoustic_signature(path: Path) -> dict:
    """Descriptive features of a recording, for resemblance comparison."""
    import numpy as np
    from realme.verify.acoustic import read_wav
    sig, sr = read_wav(Path(path))
    if sig.size == 0:
        return {}
    sig, sr = _to_analysis_rate(sig, sr)
    frame = int(sr * 0.04)
    hop = int(sr * 0.02)
    if len(sig) < frame * 2:
        return {}
    frames = np.lib.stride_tricks.sliding_window_view(sig, frame)[::hop]
    rms = np.sqrt((frames ** 2).mean(axis=1)) + 1e-12
    voiced = frames[rms > np.percentile(rms, 55)]
    if voiced.size == 0:
        voiced = frames

    # Pitch by autocorrelation over the voiced frames; crude but consistent,
    # which is all a comparison needs.
    pitches = []
    lo, hi = int(sr / 400), int(sr / 60)
    for f in voiced[: min(len(voiced), 400)]:
        f = f - f.mean()
        ac = np.correlate(f, f, mode="full")[len(f) - 1:]
        if ac[0] <= 0:
            continue
        seg = ac[lo:hi]
        if seg.size == 0:
            continue
        peak = int(np.argmax(seg)) + lo
        if ac[peak] / ac[0] > 0.3:
            pitches.append(sr / peak)
    pitch_median = float(np.median(pitches)) if pitches else None
    if pitches and pitch_median:
        semis = 12 * np.log2(np.array(pitches) / pitch_median)
        pitch_var = float(np.percentile(semis, 90) - np.percentile(semis, 10))
    else:
        pitch_var = None

    spec = np.abs(np.fft.rfft(voiced * np.hanning(frame), axis=1))
    freqs = np.fft.rfftfreq(frame, 1 / sr)
    centroid = float((spec * freqs).sum() / (spec.sum() + 1e-12))
    db = 20 * np.log10(rms)
    return {
        "pitch_median_hz": pitch_median,
        "pitch_variation_semitones": pitch_var,
        "spectral_centroid_hz": centroid,
        "dynamic_range_db": float(np.percentile(db, 95) - np.percentile(db, 10)),
        "zero_crossing_rate": float((np.diff(np.sign(sig)) != 0).mean()),
        "rms_dbfs": float(20 * np.log10(np.sqrt((sig ** 2).mean()) + 1e-12)),
    }


def compare_signatures(reference: dict, candidate: dict) -> dict:
    """
    Transparent, per-feature resemblance. Every contribution is shown.

    NOT biometric verification. Use it to tell whether a treatment change or a
    different engine drifted away from your voice, not to prove who is speaking.
    """
    features: dict[str, dict] = {}
    weighted = 0.0
    total_weight = 0.0
    for name, scale, weight in SIGNATURE_FEATURES:
        a, b = reference.get(name), candidate.get(name)
        if a is None or b is None:
            continue
        diff = abs(float(a) - float(b))
        sim = max(0.0, 1.0 - min(1.0, diff / scale))
        features[name] = {"reference": round(float(a), 4),
                          "candidate": round(float(b), 4),
                          "difference": round(diff, 4),
                          "similarity": round(sim, 4)}
        weighted += sim * weight
        total_weight += weight
    score = None if total_weight == 0 else round(weighted / total_weight, 4)
    return {
        "method": "local-acoustic-signature",
        "score": score,
        "score_percent": None if score is None else round(score * 100, 1),
        "feature_count": len(features),
        "features": features,
        "interpretation": ("Descriptive acoustic resemblance only; "
                           "not biometric speaker verification."),
    }


# --- Reference transcript -----------------------------------------------------
#
# Qwen3 clones two ways at once: a speaker embedding from the audio, and
# in-context learning from the reference clip paired with its exact words. The
# second is what carries prosody, and it is also why the transcript has to match
# the recording word for word -- including the false starts you left in.
#
# It is prefilled as context on *every* utterance, so a long reference is paid
# for on every line of every lecture. Longer is not better past a point.

REF_SECONDS_IDEAL = (10, 45)
REF_SECONDS_MAX = 90


def read_transcript(value: str | Path) -> tuple[str, str]:
    """
    Accept either the words themselves or a file containing them.

    Returns (text, source). A transcript long enough to be annoying to paste is
    exactly the case where it should live in a file, so this takes both rather
    than making anyone escape quotes around three paragraphs.
    """
    raw = str(value)
    candidate = Path(raw.strip().strip('"').strip("'"))
    try:
        is_file = candidate.is_file()
    except OSError:
        is_file = False
    if is_file:
        from realme.core.textio import read_text
        text = read_text(candidate)
        source = str(candidate)
    else:
        text, source = raw, "the command line"
    # Line breaks are layout, not speech. Collapse them so the engine sees one
    # continuous utterance -- a transcript split across lines would otherwise
    # carry newlines into the prompt.
    return " ".join(text.split()), source


def transcript_fit(text: str, audio_seconds: float) -> list[str]:
    """
    Does the transcript plausibly match the recording? Advice, not a verdict.

    A mismatch here is the most common cause of a clone that sounds nothing like
    the reference, and it is silent: the engine accepts whatever it is given.
    """
    notes: list[str] = []
    words = len(text.split())
    if not words:
        return ["No transcript. Qwen3 clones better with the exact words; "
                "without them you get the speaker embedding alone."]
    # 130-160 wpm is ordinary reading aloud. Well outside that means the text
    # and the audio are probably not the same material.
    if audio_seconds > 0:
        wpm = words / (audio_seconds / 60.0)
        if wpm > 220:
            notes.append(
                f"The transcript is {words} words for {audio_seconds:.0f}s of "
                f"audio ({wpm:.0f} wpm). That is faster than anyone reads -- is "
                f"this the transcript of a longer recording?")
        elif wpm < 80:
            notes.append(
                f"Only {words} words for {audio_seconds:.0f}s of audio "
                f"({wpm:.0f} wpm). If part of the recording is not transcribed, "
                f"the clone will be conditioned on a mismatch.")
    if audio_seconds > REF_SECONDS_MAX:
        notes.append(
            f"This reference is {audio_seconds:.0f}s. It is prefilled as "
            f"context on every utterance, so a long one slows down every line "
            f"of every lecture without cloning better. "
            f"{REF_SECONDS_IDEAL[0]}-{REF_SECONDS_IDEAL[1]}s of clean, steady "
            f"reading is the sweet spot -- and if you shorten the audio, "
            f"shorten the transcript to match exactly.")
    return notes

# --- Steadying synthesized output ------------------------------------------
#
# Distinct from `Treatment`, which shapes the REFERENCE before cloning. This
# runs on what an engine produced, and it exists because of two measurements on
# converted audio.
#
# First, it comes out pinned to the ceiling: peaks at 0.00 dBFS on six of seven
# utterances, with zero clipped samples. Not distortion, but no headroom either,
# and no headroom is how a lecture picks up inter-sample clipping later, in
# whatever encoder ships it.
#
# Second, the within-phrase range was 35.9 dB where the speaker's own recording
# sits at 38.3 -- but the loud moments were arriving faster than a listener
# wants to ride, which is what "dynamic should be a bit controlled and steady"
# means in practice.
#
# The chain is deliberately FIXED rather than adaptive. `loudnorm` and
# `dynaudnorm` measure the clip and choose their own gain, so the same sentence
# rendered twice can come out at two levels, and utterances drift apart from
# each other over a lecture. Fixed threshold, fixed makeup, fixed ceiling: the
# same input always gives the same output.
#
# Measured over seven utterances, `steady` moved RMS -15.7 -> -16.7 dB (the
# speaker's own recording is -16.7), peak 0.0 -> -1.3 dBFS, and within-phrase
# range 35.9 -> 30.9 dB.

STEADY_PRESETS: dict[str, str | None] = {
    "off": None,
    "light": ("acompressor=threshold=-18dB:ratio=2.5:attack=25:release=280,"
              "volume=4dB,alimiter=limit=0.89"),
    "steady": ("acompressor=threshold=-20dB:ratio=3:attack=20:release=250,"
               "volume=5dB,alimiter=limit=0.89"),
    "tight": ("acompressor=threshold=-22dB:ratio=4:attack=15:release=220,"
              "volume=7dB,alimiter=limit=0.89"),
}


def steady_output(src: Path, dst: Path | None = None,
                  preset: str = "steady") -> Path:
    """
    Even out a synthesized utterance and give it back its headroom.

    `preset="off"` returns the file untouched, so a caller can pass the setting
    through without branching on it.
    """
    src = Path(src)
    chain = STEADY_PRESETS.get(preset, STEADY_PRESETS["steady"])
    if chain is None:
        if dst and Path(dst) != src:
            shutil.copy2(src, dst)
            return Path(dst)
        return src
    out = Path(dst) if dst else src
    tmp = out.with_suffix(".steady.tmp.wav")
    run([require("ffmpeg"), "-y", "-loglevel", "error", "-i", str(src),
         "-af", chain, "-c:a", "pcm_s16le", str(tmp)], "steadying output")
    tmp.replace(out)
    return out
