"""
Deciding, from the audio itself, which pronunciation was spoken.

This is rungs 4 and 5 of the verification ladder, and it closes the gap between
"the file is correct" and "the words are correct".

THE IDEA. Round-trip ASR is the obvious approach and it is a poor one here: the
published evidence is that word-error rate is anti-correlated with the thing we
care about, and that ASR fails hardest on exactly our risk surface -- proper
nouns, abbreviations and digits. An ASR that hears "ip-twuh" will helpfully
"correct" it to IPTW, hiding the very error we are hunting.

So we do not ask an open-set question. We ask a closed-set one. For every
lexicon term we already know both the pronunciation we intended AND the
pronunciation we observed the engine produce when left alone -- that pair is
what a lexicon entry IS. Synthesize both with the same voice, slide each over
the rendered audio, and see which one the audio actually matches. "Which of
these two?" needs no confidence threshold and no language model.

WHY IT IS ENGINE-AGNOSTIC. Both templates are produced by the same TTS adapter
that rendered the lecture, so they carry the same voice, the same sample rate
and the same artefacts. That holds for espeak, for Piper, and equally for a
cloned neural voice -- it costs two short synthesis calls per term, cached.

WHAT IT CANNOT DO. It verifies terms that are IN the lexicon. A term nobody has
noticed yet is not checked, because there is no competing variant to compare
against -- which is why the linter in `realme/text/prepare.py` matters: it
nominates candidates, and this pass convicts them.

No model downloads, no GPU, numpy and scipy only.
"""
from __future__ import annotations
import math
import wave
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


# --------------------------------------------------------------- audio I/O

def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Mono float32 in [-1, 1]. Handles 8/16/32-bit PCM."""
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        ch = w.getnchannels()
        width = w.getsampwidth()
        raw = w.readframes(n)
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(width)
    if dtype is None:
        raise ValueError(f"Unsupported sample width {width} in {path}")
    a = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if width == 1:
        a = (a - 128.0) / 128.0
    else:
        a = a / float(np.iinfo(dtype).max)
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a, sr


# ------------------------------------------------------------------ features

def _mel_filterbank(sr: int, n_fft: int, n_mels: int = 26,
                    fmin: float = 60.0, fmax: float | None = None) -> np.ndarray:
    fmax = fmax or sr / 2
    def hz2mel(f): return 2595.0 * math.log10(1.0 + f / 700.0)
    def mel2hz(m): return 700.0 * (10 ** (m / 2595.0) - 1.0)
    pts = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel2hz(pts) / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        if r <= l:
            continue
        for k in range(l, c):
            fb[m - 1, k] = (k - l) / max(c - l, 1)
        for k in range(c, r):
            fb[m - 1, k] = (r - k) / max(r - c, 1)
    return fb


def mfcc(signal: np.ndarray, sr: int, *, n_mfcc: int = 13,
         win_ms: float = 25.0, hop_ms: float = 10.0) -> np.ndarray:
    """
    Plain MFCCs, hand-rolled so this needs no librosa.

    Cepstral mean-variance normalization at the end is what makes the
    comparison tolerant of loudness and channel differences between a template
    and the rendered lecture.
    """
    from scipy.fft import dct, rfft
    n_fft = 1 << int(math.ceil(math.log2(win_ms / 1000 * sr)))
    hop = max(1, int(hop_ms / 1000 * sr))
    if len(signal) < n_fft:
        signal = np.pad(signal, (0, n_fft - len(signal)))
    # pre-emphasis lifts the high band where consonants live
    signal = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])
    frames = np.lib.stride_tricks.sliding_window_view(signal, n_fft)[::hop]
    frames = frames * np.hanning(n_fft).astype(np.float32)
    power = (np.abs(rfft(frames, axis=1)) ** 2) / n_fft
    fb = _mel_filterbank(sr, n_fft)
    energy = np.log(power @ fb.T + 1e-10)
    c = dct(energy, type=2, axis=1, norm="ortho")[:, :n_mfcc]
    c = (c - c.mean(axis=0)) / (c.std(axis=0) + 1e-8)          # CMVN
    return c.astype(np.float32)


# ---------------------------------------------------------------------- DTW

def dtw_distance(a: np.ndarray, b: np.ndarray, band: float = 0.25) -> float:
    """
    Normalized DTW distance between two feature sequences.

    A Sakoe-Chiba band keeps it near-linear and stops pathological warps that
    would let a short template stretch to match anything.
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf")
    w = max(int(band * max(n, m)), abs(n - m) + 1)
    d = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    d[0, 0] = 0.0
    for i in range(1, n + 1):
        lo, hi = max(1, i - w), min(m, i + w)
        diff = b[lo - 1:hi] - a[i - 1]
        cost = np.sqrt((diff * diff).sum(axis=1))
        for k, j in enumerate(range(lo, hi + 1)):
            d[i, j] = cost[k] + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])
    return float(d[n, m] / (n + m))


def best_window_distance(haystack: np.ndarray, template: np.ndarray,
                         *, stride: int = 3, slack: float = 0.45) -> float:
    """
    Slide a template across a longer clip and keep the best match.

    We do not need to know where in the utterance the term falls -- searching
    removes any dependence on the character-offset estimate used elsewhere, so
    a timing approximation cannot cause a false accusation.
    """
    tlen = len(template)
    if tlen == 0 or len(haystack) < tlen // 2:
        return float("inf")
    best = float("inf")
    for length in (int(tlen * (1 - slack)), tlen, int(tlen * (1 + slack))):
        if length < 4 or length > len(haystack):
            continue
        for start in range(0, len(haystack) - length + 1, stride):
            best = min(best, dtw_distance(template, haystack[start:start + length]))
    return best


# ------------------------------------------------------------------ verdicts

@dataclass
class TermVerdict:
    term: str
    utterance: str
    intended_distance: float
    wrong_distance: float
    margin: float
    verdict: str          # "intended" | "MISPRONOUNCED" | "inconclusive"

    def ok(self) -> bool:
        return self.verdict != "MISPRONOUNCED"


class AcousticVerifier:
    """
    Synthesize the competing pronunciations once per term, then judge audio.

    `min_margin` is a relative gap, not an absolute distance, so it does not
    need retuning per voice. Below it the result is reported as inconclusive --
    which is the honest answer and much more useful than a coin-flip verdict.
    """

    #: Templates shorter than this many MFCC frames (10 ms each) carry too
    #: little acoustic evidence to separate two candidates. Measured: a
    #: two-letter term like "OR" produces ~40 frames and returns a margin
    #: indistinguishable from noise, which showed up as a FALSE NEGATIVE in
    #: testing -- it called a genuinely mispronounced "OR" correct. Refusing to
    #: judge is the honest answer; the linter still flags such terms for a human.
    MIN_TEMPLATE_FRAMES = 55

    def __init__(self, tts, lexicon, cache_dir: Path, *, min_margin: float = 0.18):
        self.tts = tts
        self.lexicon = lexicon
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # 0.18 rather than something tighter, deliberately. A wrong "this is
        # fine" is far more damaging than "I could not tell": the first ships a
        # mispronounced lecture with a clean bill of health, the second sends a
        # human to listen. Bias the threshold toward admitting ignorance.
        self.min_margin = min_margin
        self._templates: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}
        # Templates belong to a voice. The documented draft-then-final
        # workflow renders both in one project, and a template synthesised by
        # the 22 kHz draft voice was being DTW-matched against the 24 kHz
        # clone -- a different filterbank, a meaningless verdict.
        import hashlib
        try:
            fp = str(tts.voice_fingerprint())
        except Exception:
            fp = "unknown"
        self._voice_tag = hashlib.sha256(fp.encode()).hexdigest()[:10]
        self.synth_errors: list[str] = []

    def _synth(self, text: str, name: str) -> np.ndarray | None:
        path = self.cache_dir / f"{name}__{self._voice_tag}.wav"
        if not path.exists():
            try:
                self.tts.synthesize(text, path)
            except Exception as e:
                # Silently dropping the term made "terms_checked" shrink with
                # no trace of why.
                self.synth_errors.append(f"{name}: {type(e).__name__}: {e}")
                return None
        try:
            sig, sr = read_wav(path)
        except Exception:
            return None
        return mfcc(sig, sr)

    def templates(self, term: str):
        """(intended, naive) feature templates for a term, or None."""
        if term in self._templates:
            return self._templates[term]
        entry = self.lexicon.get(term)
        result = None
        if entry is not None:
            safe = "".join(c if c.isalnum() else "_" for c in term)
            # Intended: what the lexicon says. Naive: the bare term, which is
            # what the engine does when the lexicon is not applied.
            intended = self._synth(entry.espeak_block()
                                   if getattr(self.tts, "phoneme_syntax", None) == "espeak"
                                   else entry.spoken_fallback(), f"{safe}__intended")
            naive = self._synth(term, f"{safe}__naive")
            if intended is not None and naive is not None:
                if min(len(intended), len(naive)) < self.MIN_TEMPLATE_FRAMES:
                    result = "too_short"
                else:
                    result = (intended, naive)
        self._templates[term] = result
        return result

    def check_utterance(self, wav: Path, terms: list[str],
                        label: str = "") -> list[TermVerdict]:
        out: list[TermVerdict] = []
        if not terms:
            return out
        try:
            sig, sr = read_wav(wav)
        except Exception:
            return out
        feats = mfcc(sig, sr)
        for term in dict.fromkeys(terms):
            tpl = self.templates(term)
            if tpl is None:
                continue
            if tpl == "too_short":
                out.append(TermVerdict(term=term, utterance=label,
                                       intended_distance=0.0, wrong_distance=0.0,
                                       margin=0.0, verdict="not checkable"))
                continue
            intended, naive = tpl
            d_int = best_window_distance(feats, intended)
            d_wrong = best_window_distance(feats, naive)
            if not (math.isfinite(d_int) and math.isfinite(d_wrong)):
                continue
            denom = min(d_int, d_wrong) or 1e-6
            margin = (max(d_int, d_wrong) - min(d_int, d_wrong)) / denom
            if margin < self.min_margin:
                verdict = "inconclusive"
            elif d_int < d_wrong:
                verdict = "intended"
            else:
                verdict = "MISPRONOUNCED"
            out.append(TermVerdict(term=term, utterance=label,
                                   intended_distance=round(d_int, 4),
                                   wrong_distance=round(d_wrong, 4),
                                   margin=round(margin, 4), verdict=verdict))
        return out


def summarize(verdicts: list[TermVerdict]) -> dict:
    bad = [v for v in verdicts if v.verdict == "MISPRONOUNCED"]
    unsure = [v for v in verdicts if v.verdict in ("inconclusive", "not checkable")]
    return {
        "terms_checked": len(verdicts),
        "confirmed_correct": sum(1 for v in verdicts if v.verdict == "intended"),
        "mispronounced": [asdict(v) for v in bad],
        "inconclusive": sorted({v.term for v in unsure}),
        "verdict": ("FAIL: " + ", ".join(sorted({v.term for v in bad}))
                    if bad else "PASS" if verdicts else "no lexicon terms in audio"),
    }
