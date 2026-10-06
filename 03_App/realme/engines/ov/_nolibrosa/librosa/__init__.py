"""
Enough of librosa for OpenVoice's converter, and nothing more.

**Why this exists.** librosa depends on numba, numba ships llvmlite, and
llvmlite ships a native `_helperlib` DLL. On a machine with Windows Application
Control switched on, importing librosa dies with:

    ImportError: DLL load failed while importing _helperlib:
    An Application Control policy has blocked this file.

That is the same policy that blocked `qwen3-tts-cli.exe`, and there is no
per-application allowlist to add to. But OpenVoice's tone colour converter uses
librosa for exactly one thing -- reading an audio file and resampling it -- and
soundfile plus torchaudio do that with no native code that anything objects to.

So this is a stand-in, put on `sys.path` **only when the real librosa cannot be
imported**. If librosa works, it is used. Nothing here overrides a working
install.

Deliberately not a reimplementation: `filters.mel` raises rather than returning
something plausible. The converter never calls it -- it works on linear
spectrograms -- and a wrong mel filterbank would be a silent, subtle wrongness
in the output, which is the failure mode this project refuses everywhere else.
"""
from __future__ import annotations
import numpy as np
import soundfile as sf

from . import filters, util  # noqa: F401  (imported for `librosa.filters.mel`)

__version__ = "0.0.0+realme-shim"


def load(path, sr=None, mono=True, offset=0.0, duration=None, dtype=None,
         res_type=None, **kw):
    """
    `librosa.load` for the cases OpenVoice actually uses: a path, a target rate.

    Anything soundfile cannot decode (a phone's .m4a, most often) is converted
    through ffmpeg first, which is already a hard dependency of this project.
    """
    try:
        data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception:
        from pathlib import Path
        from realme.core.media import to_reference_wav
        import tempfile
        tmp = Path(tempfile.mkdtemp()) / (Path(path).stem + ".wav")
        data, file_sr = sf.read(str(to_reference_wav(Path(path), tmp)),
                                dtype="float32", always_2d=True)

    y = data.mean(axis=1) if (mono and data.shape[1] > 1) else data[:, 0]
    if offset:
        y = y[int(offset * file_sr):]
    if duration:
        y = y[:int(duration * file_sr)]
    if sr and int(sr) != int(file_sr):
        import torch
        import torchaudio
        y = torchaudio.functional.resample(
            torch.from_numpy(np.ascontiguousarray(y)),
            int(file_sr), int(sr)).numpy()
        file_sr = int(sr)
    return y.astype("float32"), int(file_sr)
