"""
qwen3-tts.cpp in-process, through its C API — no child process at all.

This runs under Windows Application Control, and it is also faster than the
CLI.

**Why it gets past the block.** Application Control policies (Smart App Control,
WDAC) refuse to *launch* untrusted executables. `qwen3-tts-cli.exe` is a fresh,
unsigned binary, so it is refused. But the same build also produces
`qwen3tts.dll`, a complete C API over the identical code, and loading a library
into an already-trusted process is a different operation from starting a new
one. Python is signed and trusted; it does the running, and the DLL is just
data it maps. The code is the same code that was compiled; the policy gates
what starts, and nothing new starts.

It may still be refused if the policy enforces DLL rules too. That is a
configuration difference, and the adapter reports it plainly rather than
guessing.

**Why it is faster.** The CLI is a process per utterance: it reloads 2.2 GB of
weights and re-encodes the reference speaker every single time. This holds the
model open for the life of the process and extracts the speaker embedding
**once**, then reuses it. Upstream's own profiling puts the speaker encoder at
the largest single cost of a cloned utterance, so skipping it per-utterance is
not a marginal saving. Like the Python worker it is a server, so a benchmark of
the two measures synthesis rather than startup.
"""
from __future__ import annotations
import ctypes
import hashlib
import os
import struct
import wave
from ctypes import (c_char_p, c_float, c_int32, c_void_p, POINTER, Structure,
                    byref)
from pathlib import Path

from realme.adapters.base import BaseTTS, AdapterUnavailable
from realme.core.media import probe, retime

SAMPLE_RATE = 24000
#: Every language the weights speak, with the id the engine expects (the same
#: ten `main.cpp` handles). An unknown code falls back to English with a
#: warning (see `_params`).
LANGUAGE_IDS = {
    "en": 2050,   # English
    "de": 2053,   # German
    "es": 2054,   # Spanish
    "zh": 2055,   # Chinese
    "ja": 2058,   # Japanese
    "fr": 2061,   # French
    "ko": 2064,   # Korean
    "ru": 2069,   # Russian
    "it": 2070,   # Italian
    "pt": 2071,   # Portuguese
}


class Params(Structure):
    _fields_ = [("max_audio_tokens", c_int32), ("temperature", c_float),
                ("top_p", c_float), ("top_k", c_int32), ("n_threads", c_int32),
                ("repetition_penalty", c_float), ("language_id", c_int32)]


class Audio(Structure):
    _fields_ = [("samples", POINTER(c_float)), ("n_samples", c_int32),
                ("sample_rate", c_int32)]


#: Threads one engine uses at most. Measured on a 6-core machine with the
#: count applied: 6 threads 3.45 times real time, 8 -> 3.72, 12 -> 3.55. Past
#: one per core there is nothing to gain, and 6 is where the engine stops
#: getting faster; a machine with more cores than this runs more engines
#: instead (`pipeline/prerecord.py`).
MAX_ENGINE_THREADS = 6


def default_threads() -> int:
    """One per core, at most MAX_ENGINE_THREADS, at least 2.

    REALME_ENGINE_THREADS overrides it, for a machine where measuring
    (`realme engine-timing`) says otherwise.
    """
    raw = os.environ.get("REALME_ENGINE_THREADS", "").strip()
    if raw.isdigit() and 1 <= int(raw) <= 64:
        return int(raw)
    from realme.core.cpu import physical_cores
    return max(2, min(MAX_ENGINE_THREADS, physical_cores()))


def library_path(home: Path) -> Path:
    name = ("qwen3tts.dll" if os.name == "nt"
            else "libqwen3tts.dylib" if os.uname().sysname == "Darwin"
            else "libqwen3tts.so")
    for d in (home / "build", home / "build" / "Release"):
        p = d / name
        if p.is_file():
            return p
    return home / "build" / name


def _bind(lib: ctypes.CDLL) -> None:
    """
    Declare every signature.

    ctypes defaults to an int return, which silently truncates a 64-bit pointer
    to 32 bits and crashes a long way from the cause -- so every function is
    declared, not just the ones with awkward arguments.

    A missing symbol here is not a subtle problem: on Windows a DLL exports
    nothing unless it was told to, and the whole API is absent at once.
    """
    try:
        lib.qwen3_tts_create
    except AttributeError:
        raise AdapterUnavailable(
            "The shared library loaded, but exports no functions.\n"
            "  On Windows a DLL must declare its exports at build time, and "
            "this one was\n  built without that flag. Rebuild "
            "just the C++ side:\n\n"
            "    realme engine cpp --cpp-action build --force\n\n"
            "  ggml is not rebuilt, so this is minutes rather than the full "
            "build.") from None
    lib.qwen3_tts_default_params.argtypes = [POINTER(Params)]
    lib.qwen3_tts_default_params.restype = None
    lib.qwen3_tts_create.argtypes = [c_char_p, c_int32]
    lib.qwen3_tts_create.restype = c_void_p
    lib.qwen3_tts_is_loaded.argtypes = [c_void_p]
    lib.qwen3_tts_is_loaded.restype = ctypes.c_int
    lib.qwen3_tts_destroy.argtypes = [c_void_p]
    lib.qwen3_tts_destroy.restype = None
    lib.qwen3_tts_free_audio.argtypes = [POINTER(Audio)]
    lib.qwen3_tts_free_audio.restype = None
    lib.qwen3_tts_get_error.argtypes = [c_void_p]
    lib.qwen3_tts_get_error.restype = c_char_p
    lib.qwen3_tts_synthesize.argtypes = [c_void_p, c_char_p, POINTER(Params)]
    lib.qwen3_tts_synthesize.restype = POINTER(Audio)
    lib.qwen3_tts_synthesize_with_voice_file.argtypes = [
        c_void_p, c_char_p, c_char_p, POINTER(Params)]
    lib.qwen3_tts_synthesize_with_voice_file.restype = POINTER(Audio)
    lib.qwen3_tts_extract_embedding_file.argtypes = [
        c_void_p, c_char_p, POINTER(c_float), c_int32]
    lib.qwen3_tts_extract_embedding_file.restype = c_int32
    lib.qwen3_tts_synthesize_with_embedding.argtypes = [
        c_void_p, c_char_p, POINTER(c_float), c_int32, POINTER(Params)]
    lib.qwen3_tts_synthesize_with_embedding.restype = POINTER(Audio)
    # RealMe's batch entry point (engines/cpp.py, apply_batch_patch).
    # Optional: older builds lack it.
    fn = getattr(lib, BATCH_FN, None)
    if fn is not None:
        fn.argtypes = [c_void_p, POINTER(c_char_p), c_int32, POINTER(c_float),
                       c_int32, POINTER(Params), POINTER(POINTER(Audio))]
        fn.restype = c_int32


#: Which copy of the voice model to use: "f16" (the default, the model as
#: converted) or "q8_0" (RealMe's 8-bit copy in models_q8/, made by
#: `realme engine cpp --cpp-action quantize`). Set with
#: `realme key REALME_QWEN3CPP_QUANT q8_0`.
QUANT_ENV = "REALME_QWEN3CPP_QUANT"


def quant_setting(home: Path | None = None) -> str:
    """The model lectures use: REALME_QWEN3CPP_QUANT when set (Twin Setup,
    `realme key`), else the 8-bit model where it has been made (a build or
    rebuild makes it), else the 16-bit one. The 8-bit model is preferred:
    1.35 against 1.51 seconds of work per second of speech on a 6-core PC,
    and not told apart in listening."""
    v = os.environ.get(QUANT_ENV, "").strip().lower()
    if v:
        return "q8_0" if v.startswith("q8") else "f16"
    return "q8_0" if q8_ready(home) else "f16"


def q8_ready(home: Path | None = None) -> bool:
    try:
        from realme.engines.cpp import Q8_DIR, Q8_FILE, TOKENIZER_FILE, cpp_home
    except ImportError:
        return False
    d = Path(home or cpp_home()) / Q8_DIR
    return (d / Q8_FILE).is_file() and (d / TOKENIZER_FILE).is_file()


#: Sentences per pass where this computer has not measured its own
#: (`_Bench_Batch.bat`): what measures best on a 6-core PC --
#: 16-bit 1.51 s of work per second of speech at 8 (2.54 one at a time),
#: 8-bit 1.35 at 4 (1.77). The sound is identical whatever the size; only the
#: memory differs, and the prerecorder lowers it where memory is short.
DEFAULT_BATCH = {"f16": 8, "q8_0": 4}


def batch_measured() -> dict:
    """What `bench-batch` saved on this computer: {"f16": 4, "q8_0": 2}."""
    import json
    try:
        from realme.core.env import data_home
        saved = json.loads((data_home() / BATCH_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError, ImportError):
        return {}
    if not isinstance(saved, dict):
        return {}
    if "batch" in saved:                       # an older, flat format
        saved = {saved.get("quant") or "f16": saved}
    out = {}
    for q, v in saved.items():
        try:
            out[q] = int((v or {}).get("batch"))
        except (TypeError, ValueError, AttributeError):
            pass
    return out


#: The engine's vocoder switches RealMe turns on unless they are set: the
#: one-pass snake with RealMe's sine, and the 32-bit vocoder -- in its
#: copy-free form ("2") where the engine was built with it, else "1". On a
#: 6-core PC "1" took the vocoder from 0.85 to 0.55 seconds of work per
#: second of speech, the samples 62 dB below the speech apart and not heard
#: in listening; "2" gives the same samples (about 120 dB apart) in about a
#: third less time and a third of the working memory. A value set in the
#: shell, in the settings file (`realme key`) or in Twin Setup ("0" for the
#: original) wins. Read once per process by the engine, so a change takes
#: effect in a new process.
VOCODER_DEFAULTS = {"QWEN3_TTS_FAST_SNAKE": "1", "QWEN3_TTS_VOCODER_F32": "1"}
#: Which of them this process set itself (not the user's own values).
DEFAULTED: set = set()


def vocoder_defaults(home: Path | None = None) -> dict:
    """VOCODER_DEFAULTS for the engine in `home` (default: the installed one)."""
    out = dict(VOCODER_DEFAULTS)
    try:
        from realme.engines.cpp import vocoder2_built
        if vocoder2_built(home):
            out["QWEN3_TTS_VOCODER_F32"] = "2"
    except Exception:
        pass
    return out


#: The longest sentence the 32-bit vocoder ("1") takes, in codes (12.5 a
#: second). It needs about 1.8 times the original's working memory (a
#: 20-second sentence 4.6 GB at its peak against 2.5; 30 s about 6 against
#: 3.7), on top of the engine and several sentences per pass, so by the
#: machine's memory (a "16 GB" machine reports a little under 16): 32 GB or
#: more, 30 s; 16 GB or more, 20 s; less, 10 s. Longer sentences are decoded
#: the original way. The copy-free form ("2") needs less memory than the
#: original at any length and has no limit. A value set in the shell or
#: settings file wins.
FRAMES_ENV = "QWEN3_TTS_VOCODER_F32_MAX_FRAMES"
FRAMES_BY_MEMORY = ((31.0, 375), (15.0, 250), (0.0, 125))


def vocoder_frames_default() -> str:
    from realme.parbench import total_memory_gb
    total = total_memory_gb() or 0.0
    return str(next(f for gb, f in FRAMES_BY_MEMORY if total >= gb))


def apply_vocoder_defaults(home: Path | None = None) -> None:
    from realme.core import env as _env
    for k, v in vocoder_defaults(home).items():
        if not os.environ.get(k):
            os.environ[k] = v
            DEFAULTED.add(k)
            # RealMe's own value, like one from the settings file: a later
            # re-read of `.env` replaces it, and it is not "set in the shell".
            _env._OURS.add(k)
    if os.environ.get("QWEN3_TTS_VOCODER_F32") == "1" and not os.environ.get(FRAMES_ENV):
        os.environ[FRAMES_ENV] = vocoder_frames_default()
        DEFAULTED.add(FRAMES_ENV)
        _env._OURS.add(FRAMES_ENV)


def vocoder_setting() -> str:
    """"fast" (both on, the default), "original" (both "0"), or "mixed"."""
    snake = os.environ.get("QWEN3_TTS_FAST_SNAKE") or VOCODER_DEFAULTS["QWEN3_TTS_FAST_SNAKE"]
    voc = os.environ.get("QWEN3_TTS_VOCODER_F32") or VOCODER_DEFAULTS["QWEN3_TTS_VOCODER_F32"]
    if snake != "0" and voc in ("1", "2"):
        return "fast"
    if snake == "0" and voc not in ("1", "2"):
        return "original"
    return "mixed"


def batch_built_at(home: Path) -> bool:
    """Was the engine in `home` built with the batch entry point, for the
    CPU? Read from its build record, so asking does not load the model."""
    import json
    try:
        info = json.loads((Path(home) / "build_info.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    # Batches were checked on the CPU backend only; a GPU build records
    # one sentence at a time.
    return bool(info.get("batch")) and (info.get("gpu") or "off") in (
        "off", "none", "cpu", "")


def batch_default(quant: str | None = None) -> int:
    """Sentences per pass with nothing set: measured here, else DEFAULT_BATCH."""
    q = quant or quant_setting()
    return batch_measured().get(q) or DEFAULT_BATCH.get(q, 1)


#: Several sentences in one pass (see engines/qwen3cpp_batch).
BATCH_FN = "qwen3_tts_synthesize_batch_with_embedding"
#: Where `realme bench-batch` saves how many sentences per pass to use.
BATCH_FILE = "qwen3_batch.json"


class Qwen3DllTTS(BaseTTS):
    name = "qwen3-tts-cpp (in-process)"
    is_voice_clone = True
    #: `synthesize(voice=...)` really does switch the reference per call.
    #: The register wrapper checks this rather than assuming every cloning
    #: engine can, because most of the others silently ignore `voice`.
    accepts_reference_per_call = True
    #: Which languages this adapter can be asked for, per utterance.
    speaks_languages = tuple(sorted(LANGUAGE_IDS))
    is_placeholder = False
    phoneme_syntax = None

    #: The engine's sampling controls, and their environment overrides.
    #:
    #: These are the only "voice modifiers" it has. The VOICE comes from the
    #: speaker embedding and cannot be adjusted; what these change is the
    #: DELIVERY -- how much the model varies from one rendering to the next, and
    #: how tightly it is held to the likeliest continuation.
    #:
    #:   temperature        0 is deterministic and flat; 0.9 is the default;
    #:                      higher is more expressive and more likely to drift
    #:   top_k / top_p      how much of the distribution may be sampled from.
    #:                      Lower is safer and more monotone
    #:   repetition_penalty guards the stutter-and-loop failure. Below 1.0
    #:                      invites it; far above flattens prosody
    #:
    #: There is no pitch, speed or emphasis here. Speed is the pipeline's
    #: (`--piper-speed`, `[[slow]]`), pitch is a treatment on the reference.
    ENV_DEFAULTS = {
        "temperature": ("REALME_TTS_TEMPERATURE", float, 0.9),
        "top_k": ("REALME_TTS_TOP_K", int, 50),
        "top_p": ("REALME_TTS_TOP_P", float, 1.0),
        "repetition_penalty": ("REALME_TTS_REPETITION_PENALTY", float, 1.05),
    }

    @classmethod
    def _default(cls, name):
        env, cast, fallback = cls.ENV_DEFAULTS[name]
        raw = os.environ.get(env, "")
        try:
            return cast(raw) if raw else fallback
        except ValueError:
            return fallback

    def __init__(self, home: Path | None = None, *,
                 reference_wav: Path | None = None,
                 threads: int = 0, temperature: float | None = None,
                 top_k: int | None = None, top_p: float | None = None,
                 repetition_penalty: float | None = None,
                 max_tokens: int = 4096, language: str = "en",
                 backend: str = "auto"):
        from realme.engines.cpp import cpp_home
        self.home = Path(home) if home else cpp_home()
        #: The model, decided once for this engine: a render keeps the model
        #: (and the takes' fingerprint) it started with, whatever is chosen or
        #: made meanwhile. Pickled with it, so the recording processes agree.
        self.quant = quant_setting(self.home)
        self.reference_wav = Path(reference_wav) if reference_wav else None
        # The thread count the engine runs its calculations on. Upstream
        # stores the count but never gives it to GGML; RealMe's build passes
        # it on. Measured on a 12-thread, 6-core machine (`realme
        # engine-timing`): 4 -> 4.36 times real time, 6 -> 3.45, 8 -> 3.72,
        # 12 -> 3.55. So one thread per core, which is half the logical
        # processors on any machine with two per core; 6 at most. Older
        # engine builds ignore it and stay on 4 (`_Rebuild_Voice_Engine.bat`
        # updates them).
        self.threads = int(threads) or default_threads()
        self.temperature = float(self._default("temperature")
                                 if temperature is None else temperature)
        self.top_k = int(self._default("top_k") if top_k is None else top_k)
        self.top_p = float(self._default("top_p") if top_p is None else top_p)
        self.repetition_penalty = float(
            self._default("repetition_penalty")
            if repetition_penalty is None else repetition_penalty)
        self.max_tokens, self.language = int(max_tokens), language
        self.backend = (backend or "auto").lower()
        self._lib = None
        self._handle = None
        self._embedding = None
        self._embedding_for = None
        self._embeddings: dict = {}
        #: An emotion direction to add to the speaker embedding, or None.
        #: None is the default and leaves the enrolled voice unchanged;
        #: nothing sets it unless `realme.expressive` is explicitly wired in.
        self._expression = None

    #: Several copies may record at once, each in its own process with its
    #: own model (`pipeline/prerecord.py`). Measured a quarter faster with two.
    parallel_safe = True

    def applies_threads(self) -> bool:
        """Was this engine built to use its thread count? Older builds run
        on 4 threads whatever is asked."""
        from realme.engines.cpp import threads_applied
        return threads_applied(self.home)

    #: Set at construction (see __init__); decided on first use for an
    #: engine made another way.
    quant: str | None = None

    def _quant(self) -> str:
        if self.quant is None:
            self.quant = quant_setting(getattr(self, "home", None))
        return self.quant

    def __getstate__(self):
        """Settings only. The library and the loaded model belong to one
        process; a copy loads its own. The speaker embedding is re-read from
        its disk cache, so the encoder does not run again."""
        state = dict(self.__dict__)
        state.update(_lib=None, _handle=None, _embedding=None,
                     _embedding_for=None, _embeddings={})
        return state

    def _built_gpu(self) -> str:
        """Which backend this binary was compiled with, from the build record."""
        import json
        try:
            return json.loads(
                (self.home / "build_info.json").read_text(encoding="utf-8")).get("gpu", "off")
        except Exception:
            return "off"

    def voice_fingerprint(self) -> str:
        ref = "none"
        if self.reference_wav and self.reference_wav.exists():
            h = hashlib.sha256(self.reference_wav.read_bytes()).hexdigest()[:12]
            ref = f"{self.reference_wav.name}:{h}"
        # Every setting that changes the audio, or the render ledger serves a
        # take made with different ones.
        fp = (f"qwen3cpp-dll:{ref}:t{self.temperature:g}:k{self.top_k}"
              f":p{self.top_p:g}:r{self.repetition_penalty:g}")
        # The 8-bit model is a different model: its takes key differently,
        # and the 16-bit ones are still there if it is switched back.
        if self._quant() == "q8_0":
            fp = f"{fp}:q8_0"
        # A shifted embedding is a different voice and must key differently,
        # or the ledger serves the unshifted take it rendered earlier. The
        # wrapper that sets expressions labels itself too; the redundancy
        # costs at most a re-render, never a wrong cache hit.
        if self._expression is not None:
            vec, alpha = self._expression
            h = hashlib.sha256(
                ",".join(f"{v:.6g}" for v in vec).encode()).hexdigest()[:8]
            fp = f"{fp}:x{h}@{alpha:g}"
        return fp

    # ------------------------------------------------------------- lifecycle
    def _load(self) -> None:
        if self._lib is not None:
            return
        lib_path = library_path(self.home)
        if not lib_path.is_file():
            raise AdapterUnavailable(
                f"The C++ shared library is not there: {lib_path}\n"
                f"    Build it:  realme engine cpp")
        apply_vocoder_defaults(self.home)   # before the engine reads them
        if os.name == "nt":
            # ggml.dll and friends sit beside it and are NOT on PATH. Without
            # this the load fails with a bare "DLL load failed" that names the
            # library we asked for rather than the dependency it could not find.
            os.add_dll_directory(str(lib_path.parent))

        # Runtime backend selection.
        #
        # Set only when we know better than the default. Upstream's `auto`
        # picks by device TYPE -- integrated GPU, discrete, accelerator, then
        # CPU -- which a Vulkan device already satisfies, so a Vulkan build
        # needs no variable at all. `cuda` is different: it matches on the
        # backend registry NAME, so it must be asked for explicitly, and it is
        # NVIDIA-only.
        # Read by the engine when it creates its backend -- once per process,
        # so it is set before the library is loaded. Older engine builds
        # ignore QWEN3_TTS_THREADS.
        os.environ["QWEN3_TTS_THREADS"] = str(self.threads)
        if self.backend and self.backend != "auto":
            os.environ["QWEN3_TTS_BACKEND"] = self.backend
        elif self._built_gpu() == "cuda":
            os.environ["QWEN3_TTS_BACKEND"] = "cuda"
        try:
            self._lib = ctypes.CDLL(str(lib_path))
        except OSError as e:
            if getattr(e, "winerror", None) == 4551:
                raise AdapterUnavailable(
                    "Windows Application Control refuses to load the C++ "
                    "library as well as the executable.\n  " + str(lib_path)
                    + "\n\n  The policy is enforcing DLL rules, so the "
                    "in-process route is closed too.\n  Use the Python engine "
                    "(qwen3); it does the same work.") from e
            raise AdapterUnavailable(
                f"Could not load {lib_path.name}: {e}\n"
                f"  If this names a missing dependency, the ggml DLLs may not "
                f"be beside it.") from e
        _bind(self._lib)

    def _models_dir(self, make: bool = False) -> Path:
        if self._quant() == "q8_0":
            from realme.engines.cpp import Q8_DIR, Q8_FILE, TOKENIZER_FILE
            d = self.home / Q8_DIR
            need = [n for n in (Q8_FILE, TOKENIZER_FILE) if not (d / n).is_file()]
            why = ""
            if (need and make
                    and (self.home / "models" / "qwen3-tts-0.6b-f16.gguf").is_file()):
                # Chosen before a build made it (a package from another
                # computer, say): made here, once, by this engine's library.
                # Only when recording starts -- never in a status check.
                from realme.engines.cpp import make_q8
                try:
                    make_q8(self.home, log=lambda m: print(m, flush=True))
                except Exception as e:
                    why = f" Making it failed: {e}"
                need = [n for n in (Q8_FILE, TOKENIZER_FILE) if not (d / n).is_file()]
            if need:
                raise AdapterUnavailable(
                    f"The 8-bit model is not in {d} ({QUANT_ENV}=q8_0 asks for "
                    f"it).{why}\n    Make it:  realme engine cpp --cpp-action "
                    f"quantize\n    Or go back to the 16-bit model:  realme key "
                    f"{QUANT_ENV} f16")
            return d
        d = self.home / "models"
        need = [n for n in ("qwen3-tts-tokenizer-f16.gguf",) if not (d / n).is_file()]
        if not any((d / f"qwen3-tts-0.6b-{q}.gguf").is_file() for q in ("q8_0", "f16")):
            need.append("qwen3-tts-0.6b-f16.gguf")
        if need:
            raise AdapterUnavailable(
                f"Converted weights missing from {d}:\n"
                + "\n".join(f"    {n}" for n in need)
                + "\n    Convert them:  realme engine cpp --cpp-action convert")
        return d

    def check(self) -> None:
        """The library binds and the weights are converted -- nothing loaded.

        Everything `preflight` does except `qwen3_tts_create`, which is the
        expensive part and the only part a status check does not need.
        """
        self._load()
        try:
            self._models_dir()
        except AdapterUnavailable:
            # 8-bit chosen but not made yet: it is made from the 16-bit model
            # at the first recording, so that is ready enough.
            if not (self._quant() == "q8_0"
                    and (self.home / "models" / "qwen3-tts-0.6b-f16.gguf").is_file()):
                raise

    def preflight(self) -> None:
        self._load()
        models = self._models_dir(make=True)
        if self._handle is None:
            # Load once, for the life of the process. Announced because it is
            # slow and silence looks like a hang, and so the log shows the
            # model is loaded only once.
            import sys as _sys, time as _t
            built = self._built_gpu()
            asked = os.environ.get("QWEN3_TTS_BACKEND", "auto")
            print(f"  [qwen3cpp] loading the model into memory (once) "
                  f"[built for: {built}, backend: {asked}] ...",
                  file=_sys.stderr, flush=True)
            t0 = _t.perf_counter()
            try:
                handle = self._lib.qwen3_tts_create(str(models).encode("utf-8"),
                                                    self.threads)
            except OSError as e:
                # An access violation inside the library reaches Python as a
                # bare OSError naming an address, which tells the reader
                # nothing. The two causes worth naming are a GPU backend the
                # driver cannot serve, and a weights file that did not copy
                # intact -- both common, both cheap to test.
                raise AdapterUnavailable(
                    f"The engine crashed while loading the model "
                    f"({e.__class__.__name__}: {e}).\n"
                    f"    built for: {built}, backend asked for: {asked}\n"
                    f"    Two things to try, cheapest first:\n"
                    f"      realme engine cpp --cpp-action selftest\n"
                    f"        checks the weights are intact and loads the "
                    f"library in a separate process\n"
                    + (f"      realme engine cpp --cpp-action build --force "
                       f"--gpu off\n"
                       f"        a {built.upper()} build that crashes on load "
                       f"is usually the driver, not the model\n"
                       if built not in ("off", "?", "") else "")) from e
            print(f"  [qwen3cpp] loaded in {_t.perf_counter() - t0:.1f}s; "
                  f"it stays open for every later utterance",
                  file=_sys.stderr, flush=True)
            if not handle:
                raise AdapterUnavailable(
                    f"The engine loaded but could not open the models in {models}")
            self._handle = handle
        if not self._lib.qwen3_tts_is_loaded(self._handle):
            raise AdapterUnavailable("Models did not finish loading")

    def _params(self, language: str | None = None) -> Params:
        p = Params()
        self._lib.qwen3_tts_default_params(byref(p))
        p.max_audio_tokens = self.max_tokens
        p.temperature = self.temperature
        p.top_k = self.top_k
        p.top_p = self.top_p
        p.n_threads = self.threads
        p.repetition_penalty = self.repetition_penalty
        code = (language or self.language or "en").lower().replace("_", "-")[:2]
        if code not in LANGUAGE_IDS:
            # Loudly, not as English. A language the weights cannot speak is a
            # setting that did not take, and rendered as English it sounds
            # like the engine mispronouncing rather than like a wrong code.
            import sys as _sys
            print(f"  [qwen3cpp] no voice for language {code!r}; using English."
                  f" Supported: {', '.join(sorted(LANGUAGE_IDS))}",
                  file=_sys.stderr, flush=True)
        p.language_id = LANGUAGE_IDS.get(code, 2050)
        return p

    def _speaker_embedding(self, ref: Path):
        """
        Extract once, reuse for every utterance.

        The speaker encoder is the single largest cost in a cloned utterance
        upstream measured, and it produces the same vector every time from the
        same clip. Re-running it per line is pure waste.
        """
        key = str(ref.resolve())
        # Cached per REFERENCE, not one slot.
        #
        # Registers alternate -- a pressing turn, then a conceding one -- and
        # a single slot would re-run the speaker encoder on every switch,
        # which is the most expensive thing in a cloned utterance. Four
        # registers cost four extractions for the whole dialogue instead of
        # one per turn.
        hit = self._embeddings.get(key)
        if hit is not None:
            return hit
        if self._embedding is not None and self._embedding_for == key:
            return self._embedding
        import sys as _sys, time as _t
        # Disk first. The encoder takes a minute or two on CPU and returns
        # the same 1024 floats every time from the same clip and the same
        # weights, so it is not paid again on every restart (it would land on
        # the first Render). Same idea as the kNN-VC matching set, cached the
        # same way.
        cached = self._embedding_from_disk(ref)
        if cached is not None:
            self._embedding, self._embedding_for = cached, key
            self._embeddings[key] = cached
            return cached
        size = 4096
        buf = (c_float * size)()
        t0 = _t.perf_counter()
        n = self._lib.qwen3_tts_extract_embedding_file(
            self._handle, str(ref).encode("utf-8"), buf, size)
        took = _t.perf_counter() - t0
        if n <= 0:
            # The fallback re-encodes the reference clip on EVERY utterance,
            # which nearly halves throughput, so say so.
            print(f"  [qwen3cpp] the speaker encoder would not return an "
                  f"embedding (returned {n}) -- falling back to re-encoding "
                  f"{ref.name} on every utterance, which is much slower",
                  file=_sys.stderr, flush=True)
            return None
        self._embedding = (buf, n)
        self._embedding_for = key
        self._embeddings[key] = (buf, n)
        self._embedding_to_disk(ref, buf, n)
        print(f"  [qwen3cpp] speaker embedding from {ref.name}: "
              f"{n} floats in {took:.1f}s (kept on disk; later runs reuse it)",
              file=_sys.stderr, flush=True)
        return self._embedding

    # ---------------------------------------------------------- embedding cache
    def _embedding_cache_path(self, ref: Path) -> Path:
        """One file per (clip contents, speaker-encoder weights).

        Keyed on the CONTENTS of the reference, not its path: enrolment
        always writes the same filename, so a path key would serve the old
        voice after re-recording. The model files are in the key because a different encoder produces
        a different vector from the same clip.
        """
        import hashlib
        h = hashlib.sha256(ref.read_bytes()).hexdigest()[:16]
        models = self.home / "models"
        stamp = hashlib.sha256(
            "|".join(sorted(f"{q.name}:{q.stat().st_size}"
                            for q in models.glob("*.gguf"))).encode()
        ).hexdigest()[:8]
        return self.home / "embeddings" / f"{h}-{stamp}.f32"

    def _embedding_from_disk(self, ref: Path):
        from array import array
        try:
            p = self._embedding_cache_path(ref)
            if not p.is_file():
                return None
            a = array("f")
            with p.open("rb") as fh:
                a.frombytes(fh.read())
            n = len(a)
            if not 0 < n <= 4096:
                return None
            buf = (c_float * 4096)()
            for i, v in enumerate(a):
                buf[i] = v
            import sys as _sys
            print(f"  [qwen3cpp] speaker embedding for {ref.name} read from "
                  f"disk ({n} floats); the encoder did not have to run",
                  file=_sys.stderr, flush=True)
            return (buf, n)
        except (OSError, ValueError):
            # A truncated or unreadable cache is not worth a failed render.
            return None

    def _embedding_to_disk(self, ref: Path, buf, n: int) -> None:
        from array import array
        try:
            p = self._embedding_cache_path(ref)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_bytes(array("f", buf[:n]).tobytes())
            tmp.replace(p)
        except OSError as e:
            import sys as _sys
            print(f"  [qwen3cpp] could not keep the speaker embedding "
                  f"({e}); it will be recomputed next time",
                  file=_sys.stderr, flush=True)

    def warm_voice(self) -> None:
        """Have the enrolled voice's embedding ready before it is asked for.

        Called by the Studio's background warm-up so the minute of speaker
        encoding does not land on the first Render.
        """
        ref = self.reference_wav
        if ref and Path(ref).is_file():
            from realme.core.media import ensure_reference_wav
            self._speaker_embedding(ensure_reference_wav(Path(ref)))

    def set_expression(self, vector=None, alpha: float = 0.0) -> None:
        """
        Speak with a shifted speaker embedding until told otherwise.

        `vector` is a COMPLETE embedding -- base plus alpha times a direction,
        already blended and already renormalised by `realme.expressive.tau` --
        not a delta. The arithmetic lives there, with the tests, where it can
        be checked without a model loaded; this end only hands over floats.

        Passing None restores the enrolled voice exactly. That is the state
        the adapter is in unless something sets it, so the lecture, narration
        and ordinary dialogue paths use the unshifted embedding.
        """
        if vector is None or not len(vector):
            self._expression = None
            return
        self._expression = ([float(v) for v in vector], float(alpha))

    def _shifted(self, buf, n):
        """The embedding to actually synthesise with.

        Returns the cached buffer UNCHANGED when no expression is set -- the
        same object, not a copy -- so the untouched path costs nothing and
        cannot drift.

        When one is set it builds a NEW buffer. Writing into the cached one
        would carry whatever the last turn asked for into every later
        utterance.
        """
        if self._expression is None:
            return buf, n
        vec, _alpha = self._expression
        if len(vec) != int(n):
            raise AdapterUnavailable(
                f"the emotion direction has {len(vec)} values but this "
                f"engine's speaker embedding is {int(n)}. They came from "
                f"different encoders; re-run `realme voice tau`.")
        out = (c_float * int(n))()
        for i in range(int(n)):
            out[i] = vec[i]
        return out, n

    def _write_wav(self, audio, out_wav: Path) -> None:
        n = int(audio.contents.n_samples)
        rate = int(audio.contents.sample_rate) or SAMPLE_RATE
        if n <= 0:
            raise AdapterUnavailable("The engine returned no audio")
        src = audio.contents.samples
        pcm = bytearray(n * 2)
        # float32 [-1,1] -> int16, clamped. Clamping matters: a sample slightly
        # over 1.0 wraps to full-scale negative and becomes an audible click.
        struct.pack_into(f"<{n}h", pcm, 0,
                         *(max(-32768, min(32767, int(src[i] * 32767.0)))
                           for i in range(n)))
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out_wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(bytes(pcm))

    def synthesize(self, text, out_wav, voice=None, pace=1.0, language=None):
        self.preflight()
        out_wav = Path(out_wav)
        params = self._params(language)
        ref = Path(voice) if voice else self.reference_wav
        # `voice=` is a path from a caller that may never have seen the
        # factory. Convert here too: one function, and every route through it.
        if ref is not None:
            from realme.core.media import ensure_reference_wav
            ref = ensure_reference_wav(Path(ref))

        audio = None
        try:
            if ref and Path(ref).is_file():
                emb = self._speaker_embedding(Path(ref))
                if emb is not None:
                    buf, n = self._shifted(*emb)
                    audio = self._lib.qwen3_tts_synthesize_with_embedding(
                        self._handle, text.encode("utf-8"), buf, n, byref(params))
                else:
                    if self._expression is not None:
                        # The fingerprint says this take is shifted; the
                        # voice-file path cannot shift anything. Fail rather
                        # than render it flat and cache it as shifted.
                        raise AdapterUnavailable(
                            "An expression shift was requested but the engine "
                            "could not extract a speaker embedding, so it "
                            "cannot be applied. Check the DLL exports "
                            "qwen3_tts_speaker_embedding.")
                    audio = self._lib.qwen3_tts_synthesize_with_voice_file(
                        self._handle, text.encode("utf-8"),
                        str(ref).encode("utf-8"), byref(params))
            else:
                audio = self._lib.qwen3_tts_synthesize(
                    self._handle, text.encode("utf-8"), byref(params))
            if not audio:
                err = (self._lib.qwen3_tts_get_error(self._handle) or b"").decode(
                    "utf-8", "replace")
                raise AdapterUnavailable(
                    f"Synthesis failed: {err or 'no detail from the engine'}")
            self._write_wav(audio, out_wav)
        finally:
            if audio:
                self._lib.qwen3_tts_free_audio(audio)

        if abs(pace - 1.0) > 1e-3:
            retime(out_wav, pace)
        probe(out_wav)
        return out_wav

    # ------------------------------------------------------------- batches
    def batch_built(self) -> bool:
        """Was this engine built with the batch entry point? Read from its
        build record, so asking does not load the model."""
        return batch_built_at(self.home)

    def synthesize_batch(self, items, voice=None) -> list:
        """Several sentences in one pass: `items` are (text, out_wav, pace,
        language) and each lands where `synthesize` would have put it.

        One pass reads the engine's weights once for all of them, which is
        where its time goes. Sentences of different languages go in separate
        passes. Without a speaker embedding, or in an engine without the
        batch entry point, they are recorded one by one instead.
        """
        self.preflight()
        items = [(t, Path(o), float(p or 1.0), lang) for t, o, p, lang in items]
        fn = getattr(self._lib, BATCH_FN, None)
        ref = Path(voice) if voice else self.reference_wav
        if ref is not None:
            from realme.core.media import ensure_reference_wav
            ref = ensure_reference_wav(Path(ref))
        emb = (self._speaker_embedding(Path(ref))
               if ref and Path(ref).is_file() else None)
        if fn is None or emb is None or len(items) < 2:
            return [self.synthesize(t, o, voice=voice, pace=p, language=lang)
                    if lang else self.synthesize(t, o, voice=voice, pace=p)
                    for t, o, p, lang in items]
        groups: dict = {}
        for it in items:
            groups.setdefault(it[3] or None, []).append(it)
        if len(groups) > 1:
            out = {}
            for g in groups.values():
                for path in self.synthesize_batch(g, voice=voice):
                    out[str(path)] = path
            return [out[str(o)] for _, o, _, _ in items]
        lang = items[0][3]
        params = self._params(lang)
        buf, n = self._shifted(*emb)
        k = len(items)
        texts = (c_char_p * k)(*[t.encode("utf-8") for t, _, _, _ in items])
        audio = (POINTER(Audio) * k)()
        got = fn(self._handle, texts, k, buf, n, byref(params), audio)
        try:
            if got != k:
                err = (self._lib.qwen3_tts_get_error(self._handle) or b"").decode(
                    "utf-8", "replace")
                raise AdapterUnavailable(
                    f"Batch synthesis failed: {err or 'no detail from the engine'}")
            for i, (_, o, _, _) in enumerate(items):
                self._write_wav(audio[i], o)
        finally:
            for i in range(k):
                if audio[i]:
                    self._lib.qwen3_tts_free_audio(audio[i])
        for _, o, p, _ in items:
            if abs(p - 1.0) > 1e-3:
                retime(o, p)
            probe(o)
        return [o for _, o, _, _ in items]

    def close(self) -> None:
        if self._handle and self._lib:
            self._lib.qwen3_tts_destroy(self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
