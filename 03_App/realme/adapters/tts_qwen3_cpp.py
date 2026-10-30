"""
Qwen3-TTS through qwen3-tts.cpp — the same model, no Python at inference time.

Same weights, same architecture, same voice. What changes is what runs them:
GGML/C++ instead of PyTorch, reading a GGUF conversion of the safetensors you
already have. That means no torch import per render and a much smaller resident
footprint, which on a CPU laptop is most of where the time goes.

**Treat this as an experiment until you have measured it on your own machine.**
The project's headline "4.07x faster" was measured on Apple Silicon with CoreML
and Metal doing part of the work (`docs/benchmark_pytorch_vs_cpp.json` records
`"hardware": "Darwin / arm"`). Its own CPU figure, on a Ryzen 5 3600 with four
threads, is a real-time factor of 1.94 -- faster than PyTorch there, but nothing
like four times. `realme bench` exists to settle that question with numbers from
your hardware rather than someone else's.

Two things this adapter cannot do that the worker can:

  * **No transcript conditioning.** The CLI clones from reference audio alone
    (an ECAPA-TDNN x-vector); there is no way to pass the reference transcript.
    The Python worker uses audio *plus* its exact transcript, which produces a
    closer clone. This is a quality/speed trade, not a free win.
  * **No profile cache.** Each call is a fresh process that re-encodes the
    reference. RealMe's own utterance ledger still prevents re-synthesis of
    unchanged text, so this costs once per utterance, not once per render.
"""
from __future__ import annotations
import hashlib
import os
import subprocess
from pathlib import Path

from realme.adapters.base import BaseTTS, AdapterUnavailable
from realme.core.media import probe, retime


def cpp_home() -> Path:
    env = os.environ.get("REALME_QWEN3_CPP")
    if env:
        return Path(env)
    from realme.engines.cpp import cpp_home as _h
    return _h()


class Qwen3CppTTS(BaseTTS):
    name = "qwen3-tts-cpp"
    is_voice_clone = True
    accepts_reference_per_call = True
    is_placeholder = False
    # Same model as the worker, so the same answer: no phoneme interface, and
    # the lexicon falls back to respellings.
    phoneme_syntax = None

    def __init__(self, home: Path | None = None, *,
                 reference_wav: Path | None = None,
                 threads: int = 4,
                 temperature: float = 0.9,
                 top_k: int = 50,
                 repetition_penalty: float = 1.05,
                 max_tokens: int = 4096):
        self.home = Path(home) if home else cpp_home()
        self.reference_wav = Path(reference_wav) if reference_wav else None
        self.threads = int(threads)
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.repetition_penalty = float(repetition_penalty)
        self.max_tokens = int(max_tokens)

    # ----------------------------------------------------------------- paths
    @property
    def cli(self) -> Path:
        exe = "qwen3-tts-cli" + (".exe" if os.name == "nt" else "")
        return self.home / "build" / exe

    @property
    def models(self) -> Path:
        return self.home / "models"

    def voice_fingerprint(self) -> str:
        """
        Must differ from the worker's fingerprint even for the same reference.

        Two engines running the same weights still produce different audio --
        different sampler, no transcript conditioning here. If they shared a
        fingerprint the utterance ledger would serve one engine's cached audio
        for the other's render, which is the v2 bug all over again.
        """
        ref = "none"
        if self.reference_wav and self.reference_wav.exists():
            h = hashlib.sha256(self.reference_wav.read_bytes()).hexdigest()[:12]
            ref = f"{self.reference_wav.name}:{h}"
        return f"{self.name}:{ref}:t{self.temperature}:k{self.top_k}"

    # ------------------------------------------------------------- lifecycle
    def preflight(self) -> None:
        from realme.engines.cpp import (binary_state, DEFENDER_ADVICE,
                                        APPCONTROL_ADVICE)
        state, detail = binary_state(self.home)
        if state == "missing":
            raise AdapterUnavailable(
                f"qwen3-tts.cpp is not built. Expected {self.cli}\n"
                f"    Build it:  realme engine cpp")
        if state == "blocked":
            # Do NOT suggest rebuilding. A rebuild produces another unsigned
            # binary that gets blocked identically, which wastes twenty minutes
            # and teaches the wrong lesson.
            raise AdapterUnavailable(
                "The C++ engine is built, but Windows will not run it.\n  "
                + detail + "\n\n"
                + DEFENDER_ADVICE.format(build=self.cli.parent))
        if state == "appcontrol":
            raise AdapterUnavailable(
                "The C++ engine is built, but Windows Application Control "
                "will not run it.\n  " + detail + "\n\n" + APPCONTROL_ADVICE)
        if state == "broken":
            raise AdapterUnavailable(f"qwen3-tts.cpp will not start.\n  {detail}")
        # The CLI takes a directory and looks for exact filenames inside it, so
        # check for the files. A models/ folder that exists but is empty
        # otherwise fails deep inside the binary with an unreadable error.
        missing = []
        if not any((self.models / f"qwen3-tts-0.6b-{q}.gguf").exists()
                   for q in ("q8_0", "f16")):
            missing.append("qwen3-tts-0.6b-f16.gguf (or -q8_0)")
        if not (self.models / "qwen3-tts-tokenizer-f16.gguf").exists():
            missing.append("qwen3-tts-tokenizer-f16.gguf")
        if missing:
            raise AdapterUnavailable(
                f"Converted weights missing from {self.models}:\n"
                + "\n".join(f"    {m}" for m in missing)
                + "\n    Convert them:  realme engine cpp convert")

    def synthesize(self, text, out_wav, voice=None, pace=1.0):
        self.preflight()
        out_wav = Path(out_wav)
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        cmd = [str(self.cli), "-m", str(self.models), "-t", text,
               "-o", str(out_wav), "-j", str(self.threads),
               "--temperature", str(self.temperature),
               "--top-k", str(self.top_k),
               "--repetition-penalty", str(self.repetition_penalty),
               "--max-tokens", str(self.max_tokens)]
        ref = voice or self.reference_wav
        # `voice=` is a path from a caller that may never have seen the
        # factory. Convert here too: one function, and every route through it.
        if ref is not None:
            from realme.core.media import ensure_reference_wav
            ref = ensure_reference_wav(Path(ref))
        if ref:
            cmd += ["-r", str(ref)]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc.returncode != 0 or not out_wav.exists():
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]
            raise AdapterUnavailable(
                "qwen3-tts-cli failed:\n  " + "\n  ".join(tail))
        # The CLI has no pace flag. Rather than silently ignore the request --
        # which would desynchronise every slide downstream -- apply it here with
        # the same pitch-preserving filter the beautifier uses.
        if abs(pace - 1.0) > 1e-3:
            retime(out_wav, pace)
        probe(out_wav)
        return out_wav
