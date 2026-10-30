"""
Writing a WAV with the standard library, for the experiment's own files.

Reading goes through `verify.acoustic.read_wav`, which the project already
owns. Writing had nowhere to go: the adapters each write their own output and
none of them exposes a plain "save this array" helper. This is that, in twelve
lines of `wave`, so the harness needs no audio library of its own.

PCM 16-bit mono, which is what every engine in this project already emits and
what `read_wav` reads back without surprises.
"""
from __future__ import annotations
import wave
from pathlib import Path


def write_wav(path, signal, sr: int) -> Path:
    """Write mono float in [-1, 1] as 16-bit PCM."""
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.asarray(signal, dtype=np.float64).reshape(-1)
    # Clip rather than normalise. Normalising would quietly rescale a clip
    # that was rendered too loud, and loudness is one of the things being
    # measured -- the instrument must not adjust what it is reading.
    a = np.clip(a, -1.0, 1.0)
    pcm = (a * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())
    return path


def resample(signal, sr_in: int, sr_out: int):
    """Frequency-domain resample, the same approach `enrollment.voice` uses.

    Only needed when joining turns from two engines that disagree on rate --
    piper is 22.05 kHz and the cloned voice 24 kHz, and concatenating those
    without conversion plays one of them at the wrong speed.
    """
    import numpy as np
    a = np.asarray(signal, dtype=np.float64).reshape(-1)
    if sr_in == sr_out or a.size == 0:
        return a
    n_out = int(round(a.size * sr_out / sr_in))
    if n_out < 2:
        return a
    spec = np.fft.rfft(a)
    keep = min(len(spec), n_out // 2 + 1)
    return np.fft.irfft(spec[:keep], n=n_out) * (n_out / a.size)
