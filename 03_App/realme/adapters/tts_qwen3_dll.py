"""
qwen3-tts.cpp in-process, through its C API — no child process at all.

Written to get past Windows Application Control, and it turns out to be the
better design anyway.

**Why it gets past the block.** Application Control policies (Smart App Control,
WDAC) refuse to *launch* untrusted executables. `qwen3-tts-cli.exe` is a fresh,
unsigned binary, so it is refused. But the same build also produces
`qwen3tts.dll`, a complete C API over the identical code, and loading a library
into an already-trusted process is a different operation from starting a new
one. Python is signed and trusted; it does the running, and the DLL is just
data it maps. This is not a bypass of anything -- the code is the same code you
compiled, the policy's job is to gate what starts, and nothing new starts.

It may still be refused if the policy enforces DLL rules too. That is a
configuration difference, and the adapter reports it plainly rather than
guessing.

**Why it is faster.** The CLI is a process per utterance: it reloads 2.2 GB of
weights and re-encodes the reference speaker every single time. This holds the
model open for the life of the process and extracts the speaker embedding
**once**, then reuses it. Upstream's own profiling puts the speaker encoder at
the largest single cost of a cloned utterance, so skipping it per-utterance is
not a marginal saving. It also makes the comparison against the Python worker
honest: both are now servers, and the benchmark measures synthesis rather than
startup.
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
#: Every language the weights speak, with the id the engine expects.
#:
#: This held three. The model and `main.cpp` have always handled ten, so seven
#: were unreachable through the Python adapter and an unknown code silently
#: became English -- which sounds like a mispronunciation rather than like a
#: setting that did not take.
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
            "this one was\n  built before RealMe passed that flag. Rebuild "
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
        self.reference_wav = Path(reference_wav) if reference_wav else None
        # 4 was upstream's default, chosen for a 4-core machine. Code
        # generation is the dominant cost and it scales with cores, so leaving
        # this at 4 on an 8- or 16-core laptop throws away most of the machine.
        #
        # Capped because past a point memory bandwidth, not cores, is the
        # limit and more threads start to cost.
        #
        # This cap was briefly raised to 12 on the strength of a four-sentence
        # bench on a Core Ultra 7 155H that read 3.29 / 2.95 / 2.87 at 8 / 12 /
        # 16 threads -- a tidy monotonic curve. Rerun with six sentences it came
        # back 3.54 at 6 threads and 3.63 at 12: the ordering reversed. The
        # engine samples at temperature 0.9 with no seed, so each of those
        # numbers was one draw, and the gap between conditions was smaller than
        # the gap between two draws of the same condition. The curve was noise
        # wearing the shape of a result, so the change is withdrawn.
        #
        # 6, chosen by the listener rather than the clock -- which is the right
        # way round here, because the clock has nothing to say. Measured steady
        # RTF at 6, 8, 12 and 16 threads on a 155H spread less than the spread
        # between two runs at the SAME thread count, and the ordering reversed
        # when the script grew from four sentences to six. When every option
        # costs the same, preference decides, and 6 was preferred twice.
        #
        # Raise it with --threads if a machine ever shows a real difference;
        # `realme bench --repeat 3` is what would establish one.
        self.threads = int(threads) or max(1, min(6, (os.cpu_count() or 4)))
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
        #: None is the shipping state and produces the identical bytes this
        #: adapter produced before the field existed; nothing sets it unless
        #: `realme.expressive` is explicitly wired in.
        self._expression = None

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
        # A shifted embedding is a different voice and must key differently,
        # or the ledger serves the unshifted take it rendered earlier. The
        # wrapper that sets expressions labels itself too; two labels for one
        # change costs a re-render at worst, an unnoticed cache hit at best,
        # and only one of those is a bug.
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

    def _models_dir(self) -> Path:
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
        self._models_dir()

    def preflight(self) -> None:
        self._load()
        models = self._models_dir()
        if self._handle is None:
            # Load once, for the life of the process. Announced because it is
            # slow and silence looks like a hang -- and because seeing it
            # exactly once is how you confirm the model is not being reloaded.
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
        # One slot was right while there was one voice. Registers alternate --
        # a pressing turn, then a conceding one -- and a single slot re-runs
        # the speaker encoder on every switch, which is the most expensive
        # thing in a cloned utterance. Four registers cost four extractions
        # for the whole dialogue instead of one per turn.
        hit = self._embeddings.get(key)
        if hit is not None:
            return hit
        if self._embedding is not None and self._embedding_for == key:
            return self._embedding
        import sys as _sys, time as _t
        # Disk first. The encoder takes a minute or two on CPU and returns
        # the same 1024 floats every time from the same clip and the same
        # weights, so paying it again on every restart is pure waste -- and
        # it is paid at the first Render, which is exactly when somebody is
        # waiting. Same idea as the kNN-VC matching set, cached the same way.
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
            # The fallback re-encodes the reference clip on EVERY utterance.
            # It used to happen silently, which is the worst way for it to
            # happen: throughput drops by most of a factor of two and nothing
            # says why, so the cost gets blamed on cloning itself. Say it.
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
        voice after re-recording -- the same fault the render ledger had.
        The model files are in the key because a different encoder produces
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
        and ordinary dialogue paths are byte-identical to before.
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
        would poison every later utterance with whatever the last turn asked
        for, and the symptom would be a dialogue that grows steadily more
        insistent for no visible reason.
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
                        # voice-file path cannot shift anything. Rendering it
                        # flat and caching it as shifted is the one outcome
                        # worse than failing.
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

    def close(self) -> None:
        if self._handle and self._lib:
            self._lib.qwen3_tts_destroy(self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
