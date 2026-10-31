#!/usr/bin/env python3
"""
qwen3_tts_server.py -- a local voice server for realme-qwen3-tts.cpp.

Speaks the OpenAI speech API, so any tool that can use OpenAI's text-to-speech
(or "an OpenAI-compatible voice server") can use a voice cloned on your own
computer instead: nothing uploaded, nothing to pay.

    POST /v1/audio/speech   {"model": "realme-qwen3-tts", "input": "Hello.", "voice": "realme",
                             "response_format": "wav", "speed": 1.0, "language": "en"}
                            -> audio: mp3 (OpenAI's default), wav, flac, opus,
                               aac or pcm (raw 24 kHz s16le). wav and pcm need
                               nothing else; the others need ffmpeg on the
                               PATH, and come back as wav without it.
    GET  /v1/models         the model (realme-qwen3-tts), as OpenAI lists models
    GET  /v1/voices         the voices it can speak in
    GET  /health            "ok", and what is loaded

Needs only Python 3.9+ (no packages) and the engine, built from this
repository (build.bat / build.sh) or in a server folder's bin/. Voices are WAV recordings of a speaker,
10-60 seconds of ordinary speech, in a folder (`voices/` by default): the file
`voices/realme.wav` is the voice "realme", the default. Each is turned into a
speaker embedding once and kept beside it (`voices/.embeddings/`), so later
starts are quick.

Settings are in `voice_server.json` beside server/ (host, port, voice,
api_key, batch, pause, model_aliases); a command-line option overrides it.

    python server/qwen3_tts_server.py                       # voice_server.json
    python server/qwen3_tts_server.py --port 5050           # the port a tool expects

The port is whatever the tool using it expects (8800 if nothing says
otherwise). Listens on 127.0.0.1 only unless the host says otherwise; an
api_key makes it ask for "Authorization: Bearer <key>", as OpenAI's API does.

The model is "realme-qwen3-tts", and requests naming any other model are
refused. A tool that can only send another name (say "tts-1") can be let in by
listing that name in model_aliases; none is accepted by default.

Long input is split into sentences, which are recorded several at a time in
one pass of the engine (--batch, default 4) -- the same voice as one at a
time (sample for sample with sampling off), faster -- and joined with a short
pause. Requests that arrive together for the same voice, language and
temperature are recorded in the same pass.

The engine's speed switches are turned on unless set in the environment
already: QWEN3_TTS_FAST_SNAKE=1, QWEN3_TTS_VOCODER_F32=3 (RealMe's
vocoder kernel; 2, the copy-free 32-bit vocoder, with an engine built before
it), and QWEN3_TTS_THREADS. --exact leaves them off, for upstream's numbers.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import io
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import wave
from array import array
from ctypes import POINTER, Structure, byref, c_char_p, c_float, c_int32, c_void_p
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
#: The model a request names: this, or a name added in voice_server.json
#: ("model_aliases", say "tts-1" for a tool that cannot be told otherwise).
MODEL_ID = "realme-qwen3-tts"
RATE = 24000
LANGUAGES = {"en": 2050, "de": 2053, "es": 2054, "zh": 2055, "ja": 2058,
             "fr": 2061, "ko": 2064, "ru": 2069, "it": 2070, "pt": 2071}
F16, Q8, TOK = ("qwen3-tts-0.6b-f16.gguf", "qwen3-tts-0.6b-q8_0.gguf",
                "qwen3-tts-tokenizer-f16.gguf")


# ------------------------------------------------------------------ the engine

class Params(Structure):
    _fields_ = [("max_audio_tokens", c_int32), ("temperature", c_float),
                ("top_p", c_float), ("top_k", c_int32), ("n_threads", c_int32),
                ("repetition_penalty", c_float), ("language_id", c_int32)]


class Audio(Structure):
    _fields_ = [("samples", POINTER(c_float)), ("n_samples", c_int32),
                ("sample_rate", c_int32)]


def find_library(given: str | None) -> Path:
    if given:
        return Path(given).resolve()
    names = ("qwen3tts.dll", "libqwen3tts.so", "libqwen3tts.dylib")
    for d in (ROOT / "build", ROOT / "build" / "Release", ROOT / "bin", ROOT):
        for n in names:
            if (d / n).is_file():
                return (d / n).resolve()
    raise SystemExit("The engine library was not found in build/ or bin/. Build "
                     "it first (build.bat or build.sh), or say where: --library")


def find_models(given: str | None) -> Path:
    if given:
        return Path(given).resolve()
    for d in (ROOT / "models_q8", ROOT / "models"):
        if (d / TOK).is_file() and ((d / Q8).is_file() or (d / F16).is_file()):
            return d.resolve()
    raise SystemExit("No models found in models_q8/ or models/ (see README, "
                     "Models), or say where: --models")


def physical_cores() -> int:
    n = os.cpu_count() or 2
    return max(1, n // 2) if n >= 4 else n


def total_memory_gb() -> float | None:
    try:
        if os.name == "nt":
            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            ms = MS()
            ms.dwLength = ctypes.sizeof(MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                return ms.ullTotalPhys / 1e9
            return None
        if sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                 text=True, timeout=5).stdout.strip()
            return int(out) / 1e9 if out else None
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024 / 1e9
    except Exception:
        return None
    return None


def has_vocoder_kernel(library: Path | None) -> bool:
    """Was this engine library built with RealMe's vocoder kernel
    (QWEN3_TTS_VOCODER_F32=3)? Its start-up message is in the file; an
    engine built before it would read "3" as off."""
    try:
        return library is not None and b"vocoder 3: on" in Path(library).read_bytes()
    except OSError:
        return False


def set_engine_defaults(exact: bool, threads: int | None, library: Path | None = None) -> dict:
    """The engine reads its switches once, when it loads. A value already in
    the environment wins."""
    want = {"QWEN3_TTS_THREADS": str(threads or min(6, physical_cores()))}
    if not exact:
        gb = total_memory_gb() or 0.0
        # RealMe's vocoder kernel ("3"), or the copy-free 32-bit vocoder
        # ("2") in an engine without it: no length limit, as both need less
        # memory than the original at any length.
        want.update({"QWEN3_TTS_FAST_SNAKE": "1",
                     "QWEN3_TTS_VOCODER_F32": "3" if has_vocoder_kernel(library) else "2"})
        if os.environ.get("QWEN3_TTS_VOCODER_F32") == "1":
            # The older 32-bit vocoder, if asked for: its longest sentence
            # (codes, 12.5 a second) 30 s with 32 GB, 20 s with 16, 10 s below.
            want["QWEN3_TTS_VOCODER_F32_MAX_FRAMES"] = (
                "375" if gb >= 31 else "250" if gb >= 15 else "125")
    for k, v in want.items():
        os.environ.setdefault(k, v)
    return {k: os.environ[k] for k in want}


class Engine:
    """One loaded engine; used from one thread at a time."""

    def __init__(self, library: Path, models: Path, max_tokens: int = 4096):
        if os.name == "nt":
            os.add_dll_directory(str(library.parent))
        else:
            # GGML's libraries beside the engine (a bundle's bin/): loaded
            # first, so the engine finds them wherever it was built.
            for stem in ("libggml-base", "libggml-cpu", "libggml"):
                for p in sorted(library.parent.glob(stem + ".so*")) + \
                         sorted(library.parent.glob(stem + ".dylib")):
                    try:
                        ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
                        break
                    except OSError:
                        pass
        self.lib = L = ctypes.CDLL(str(library))
        L.qwen3_tts_default_params.argtypes = [POINTER(Params)]
        L.qwen3_tts_create.argtypes = [c_char_p, c_int32]
        L.qwen3_tts_create.restype = c_void_p
        L.qwen3_tts_synthesize_with_embedding.argtypes = [
            c_void_p, c_char_p, POINTER(c_float), c_int32, POINTER(Params)]
        L.qwen3_tts_synthesize_with_embedding.restype = POINTER(Audio)
        L.qwen3_tts_extract_embedding_file.argtypes = [
            c_void_p, c_char_p, POINTER(c_float), c_int32]
        L.qwen3_tts_extract_embedding_file.restype = c_int32
        L.qwen3_tts_free_audio.argtypes = [POINTER(Audio)]
        L.qwen3_tts_get_error.argtypes = [c_void_p]
        L.qwen3_tts_get_error.restype = c_char_p
        self.batch_fn = getattr(L, "qwen3_tts_synthesize_batch_with_embedding", None)
        if self.batch_fn is not None:
            self.batch_fn.argtypes = [c_void_p, POINTER(c_char_p), c_int32,
                                      POINTER(c_float), c_int32, POINTER(Params),
                                      POINTER(POINTER(Audio))]
            self.batch_fn.restype = c_int32
        self.models = models
        self.threads = int(os.environ.get("QWEN3_TTS_THREADS") or 4)
        self.handle = L.qwen3_tts_create(str(models).encode("utf-8"), self.threads)
        if not self.handle:
            raise SystemExit(f"The engine could not load the models in {models}.")
        self.max_tokens = max_tokens

    def error(self) -> str:
        e = self.lib.qwen3_tts_get_error(self.handle)
        return e.decode("utf-8", "replace") if e else "no detail from the engine"

    def params(self, language: str, temperature: float | None) -> Params:
        p = Params()
        self.lib.qwen3_tts_default_params(byref(p))
        p.max_audio_tokens = self.max_tokens
        p.n_threads = self.threads
        p.language_id = LANGUAGES.get(language, 2050)
        if temperature is not None:
            p.temperature = float(temperature)
        return p

    def embedding(self, wav: Path) -> array:
        buf = (c_float * 4096)()
        n = self.lib.qwen3_tts_extract_embedding_file(
            self.handle, str(wav).encode("utf-8"), buf, 4096)
        if n <= 0:
            raise RuntimeError(f"no speaker embedding from {wav.name}: {self.error()}")
        return array("f", buf[:n])

    @staticmethod
    def _take(audio) -> array:
        a = audio.contents
        return array("f", ctypes.cast(a.samples, POINTER(c_float * a.n_samples)).contents)

    def speak(self, texts: list[str], emb: array, language: str,
              temperature: float | None) -> list[array]:
        """Several sentences, in one pass when the engine can."""
        cemb = (c_float * len(emb))(*emb)
        p = self.params(language, temperature)
        if self.batch_fn is not None and len(texts) > 1:
            arr = (c_char_p * len(texts))(*[t.encode("utf-8") for t in texts])
            out = (POINTER(Audio) * len(texts))()
            if self.batch_fn(self.handle, arr, len(texts), cemb, len(emb),
                             byref(p), out) == len(texts):
                res = [self._take(out[i]) for i in range(len(texts))]
                for i in range(len(texts)):
                    self.lib.qwen3_tts_free_audio(out[i])
                return res
            # A pass the engine would not take (a GPU build, low-memory
            # mode): one at a time instead.
        res = []
        for t in texts:
            a = self.lib.qwen3_tts_synthesize_with_embedding(
                self.handle, t.encode("utf-8"), cemb, len(emb), byref(p))
            if not a:
                raise RuntimeError(self.error())
            res.append(self._take(a))
            self.lib.qwen3_tts_free_audio(a)
        return res


# ------------------------------------------------------------------ the voices

class Voices:
    """`<name>.wav` in a folder is the voice <name>; `<name>.f32` (1024 raw
    float32) is a ready-made speaker embedding. Embeddings of WAVs are kept
    in `.embeddings/`, keyed by the recording's contents."""

    def __init__(self, folder: Path, engine: Engine, log=print):
        self.folder, self.engine, self.log = folder, engine, log
        self.cache: dict[str, array] = {}
        self.mu = threading.Lock()

    def names(self) -> list[str]:
        if not self.folder.is_dir():
            return []
        return sorted({p.stem for p in self.folder.iterdir()
                       if p.suffix.lower() in (".wav", ".f32") and p.is_file()})

    def get(self, name: str) -> array:
        if not re.fullmatch(r"[A-Za-z0-9 _.\-]{1,64}", name or ""):
            raise KeyError(name)
        with self.mu:
            if name in self.cache:
                return self.cache[name]
            f32 = self.folder / f"{name}.f32"
            wav = self.folder / f"{name}.wav"
            if f32.is_file():
                emb = array("f")
                emb.frombytes(f32.read_bytes())
            elif wav.is_file():
                key = hashlib.sha256(wav.read_bytes()).hexdigest()[:16]
                kept = self.folder / ".embeddings" / f"{name}-{key}.f32"
                if kept.is_file():
                    emb = array("f")
                    emb.frombytes(kept.read_bytes())
                else:
                    t0 = time.perf_counter()
                    emb = self.engine.embedding(wav)
                    kept.parent.mkdir(parents=True, exist_ok=True)
                    kept.write_bytes(emb.tobytes())
                    self.log(f"  voice {name!r}: embedding made in "
                             f"{time.perf_counter() - t0:.1f}s and kept")
            else:
                raise KeyError(name)
            if not 0 < len(emb) <= 4096:
                raise KeyError(name)
            self.cache[name] = emb
            return emb


# ------------------------------------------------------------------ text, audio

# After . ! ? and a space; after the Chinese and Japanese full stops with or
# without one.
_SENT = re.compile(r"(?<=[.!?])\s+|(?<=[\u3002\uff01\uff1f])\s*")


def sentences(text: str, longest: int = 300) -> list[str]:
    """Sentences, and over-long ones cut at commas or spaces, so no single
    recording runs past the engine's comfortable length."""
    out = []
    for s in _SENT.split(" ".join(text.split())):
        s = s.strip()
        while len(s) > longest:
            cut = max(s.rfind(", ", 0, longest), s.rfind("; ", 0, longest))
            if cut < longest // 3:
                cut = s.rfind(" ", 0, longest)
            if cut <= 0:
                cut = longest
            out.append(s[:cut + 1].strip())
            s = s[cut + 1:].strip()
        if s:
            out.append(s)
    return out


def to_s16(samples: array) -> bytes:
    return array("h", (max(-32768, min(32767, int(round(v * 32767.0))))
                       for v in samples)).tobytes()


def wav_bytes(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)
    return buf.getvalue()


def change_speed(pcm: bytes, speed: float) -> bytes:
    """Faster or slower without changing the pitch, with ffmpeg's atempo when
    it is on the PATH; otherwise unchanged."""
    if abs(speed - 1.0) < 1e-3 or not shutil.which("ffmpeg"):
        return pcm
    chain, s = [], speed
    while s > 2.0:
        chain.append("atempo=2.0")
        s /= 2.0
    while s < 0.5:
        chain.append("atempo=0.5")
        s /= 0.5
    chain.append(f"atempo={s:.4f}")
    r = subprocess.run(["ffmpeg", "-v", "error", "-f", "s16le", "-ar", str(RATE),
                        "-ac", "1", "-i", "pipe:0", "-filter:a", ",".join(chain),
                        "-f", "s16le", "pipe:1"], input=pcm, capture_output=True)
    return r.stdout if r.returncode == 0 and r.stdout else pcm


#: Formats made by ffmpeg from the WAV, when it is on the PATH.
ENCODED = {"mp3": ("mp3", "audio/mpeg", ["-c:a", "libmp3lame", "-b:a", "128k"]),
           "flac": ("flac", "audio/flac", ["-c:a", "flac"]),
           "opus": ("ogg", "audio/ogg", ["-c:a", "libopus", "-b:a", "64k"]),
           "aac": ("adts", "audio/aac", ["-c:a", "aac", "-b:a", "128k"])}


def encode(pcm: bytes, fmt: str) -> tuple[bytes, str, str]:
    """(body, content type, format given). WAV when ffmpeg is missing or
    fails: a client asking for MP3 (OpenAI's default) still gets sound."""
    if fmt == "pcm":
        return pcm, "audio/pcm", "pcm"
    if fmt in ENCODED and shutil.which("ffmpeg"):
        mux, ctype, args = ENCODED[fmt]
        r = subprocess.run(["ffmpeg", "-v", "error", "-f", "s16le", "-ar", str(RATE),
                            "-ac", "1", "-i", "pipe:0", *args, "-f", mux, "pipe:1"],
                           input=pcm, capture_output=True)
        if r.returncode == 0 and r.stdout:
            return r.stdout, ctype, fmt
    return wav_bytes(pcm), "audio/wav", "wav"


# ------------------------------------------------------------------ the queue

class Job:
    def __init__(self, texts, voice, language, temperature):
        self.texts, self.voice = texts, voice
        self.language, self.temperature = language, temperature
        self.done = threading.Event()
        self.result: list[array] | None = None
        self.error: str | None = None


class Recorder(threading.Thread):
    """The one thread that uses the engine. Sentences waiting for the same
    voice, language and temperature go into one pass, up to `batch`."""

    def __init__(self, engine: Engine, voices: Voices, batch: int, log=print):
        super().__init__(daemon=True)
        self.engine, self.voices, self.batch, self.log = engine, voices, max(1, batch), log
        self.q: "queue.Queue[Job]" = queue.Queue()

    def submit(self, job: Job) -> Job:
        self.q.put(job)
        return job

    def run(self):
        while True:
            first = self.q.get()
            group = [first]
            time.sleep(0.02)       # let requests that arrive together meet
            rest = []
            while True:
                try:
                    j = self.q.get_nowait()
                except queue.Empty:
                    break
                same = (j.voice, j.language, j.temperature) == \
                       (first.voice, first.language, first.temperature)
                (group if same else rest).append(j)
            for j in rest:
                self.q.put(j)
            try:
                emb = self.voices.get(first.voice)
                pending = [(j, i) for j in group for i in range(len(j.texts))]
                outs: dict[tuple[int, int], array] = {}
                for k in range(0, len(pending), self.batch):
                    part = pending[k:k + self.batch]
                    got = self.engine.speak([j.texts[i] for j, i in part], emb,
                                            first.language, first.temperature)
                    for (j, i), a in zip(part, got):
                        outs[(id(j), i)] = a
                for j in group:
                    j.result = [outs[(id(j), i)] for i in range(len(j.texts))]
            except KeyError:
                for j in group:
                    j.error = f"unknown voice {first.voice!r}"
            except Exception as e:
                for j in group:
                    j.error = str(e)
            for j in group:
                j.done.set()


# ------------------------------------------------------------------ HTTP

def make_handler(recorder: Recorder, voices: Voices, info: dict,
                 api_key: str | None, default_voice: str | None, pause_s: float,
                 aliases: list[str] | None = None):
    accepted = {MODEL_ID, *(aliases or [])}

    class Handler(BaseHTTPRequestHandler):
        server_version = "realme-qwen3-tts/1.3"

        def log_message(self, fmt, *args):
            sys.stderr.write(f"  {self.address_string()} {fmt % args}\n")

        def _json(self, code: int, obj) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, code: int, message: str, kind: str = "invalid_request_error"):
            self._json(code, {"error": {"message": message, "type": kind}})

        def _authorised(self) -> bool:
            if not api_key:
                return True
            if self.headers.get("Authorization", "") == f"Bearer {api_key}":
                return True
            self._error(401, "Missing or wrong API key.", "authentication_error")
            return False

        def do_GET(self):
            if not self._authorised():
                return
            path = self.path.split("?")[0].rstrip("/")
            if path in ("/health", ""):
                self._json(200, {"status": "ok", **info, "voices": voices.names()})
            elif path == "/v1/models":
                self._json(200, {"object": "list", "data": [
                    {"id": m, "object": "model", "owned_by": "local"}
                    for m in [MODEL_ID, *sorted(accepted - {MODEL_ID})]]})
            elif path == "/v1/voices":
                self._json(200, {"object": "list", "data": [
                    {"id": n, "object": "voice"} for n in voices.names()]})
            else:
                self._error(404, f"No such path: {path}")

        def do_POST(self):
            if not self._authorised():
                return
            if self.path.split("?")[0].rstrip("/") != "/v1/audio/speech":
                self._error(404, f"No such path: {self.path}")
                return
            try:
                n = int(self.headers.get("Content-Length") or 0)
                req = json.loads(self.rfile.read(min(n, 2_000_000)) or b"{}")
            except (ValueError, json.JSONDecodeError):
                self._error(400, "The body is not JSON.")
                return
            text = str(req.get("input") or "").strip()
            voice = str(req.get("voice") or default_voice or "")
            fmt = str(req.get("response_format") or "mp3").lower()
            language = str(req.get("language") or "en").lower().split("-")[0]
            try:
                speed = float(req.get("speed") or 1.0)
                temperature = (None if req.get("temperature") is None
                               else float(req["temperature"]))
            except (TypeError, ValueError):
                self._error(400, "speed and temperature are numbers.")
                return
            model = str(req.get("model") or MODEL_ID)
            if model not in accepted:
                self._error(400, f"model {model!r}: this server is "
                                 f"{', '.join(sorted(accepted))} (more names can be "
                                 f"added as model_aliases in voice_server.json).",
                            "invalid_request_error")
                return
            if not text:
                self._error(400, "input is empty.")
                return
            if fmt not in ("wav", "pcm", *ENCODED):
                self._error(400, f"response_format {fmt!r}: one of wav, pcm, "
                                 f"{', '.join(ENCODED)}.")
                return
            if language not in LANGUAGES:
                self._error(400, f"language {language!r}: one of {', '.join(LANGUAGES)}.")
                return
            if not 0.25 <= speed <= 4.0:
                self._error(400, "speed is between 0.25 and 4.0.")
                return
            if voice not in voices.names():
                self._error(400, f"unknown voice {voice!r}; voices: "
                                 f"{', '.join(voices.names()) or 'none'}")
                return
            t0 = time.perf_counter()
            job = recorder.submit(Job(sentences(text), voice, language, temperature))
            job.done.wait()
            if job.error:
                self._error(500, job.error, "server_error")
                return
            gap = array("f", [0.0]) * int(pause_s * RATE)
            joined = array("f")
            for k, a in enumerate(job.result):
                if k:
                    joined.extend(gap)
                joined.extend(a)
            pcm = change_speed(to_s16(joined), speed)
            body, ctype, given = encode(pcm, fmt)
            secs = len(pcm) / 2 / RATE
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("X-Audio-Format", given)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Audio-Seconds", f"{secs:.2f}")
            self.send_header("X-Work-Seconds", f"{time.perf_counter() - t0:.2f}")
            self.end_headers()
            self.wfile.write(body)

    return Handler


#: The settings file beside server/ (`voice_server.json`), and what holds
#: where neither it nor the command line says otherwise.
CONFIG_FILE = ROOT / "voice_server.json"
CONFIG_DEFAULTS = {"host": "127.0.0.1", "port": 8800, "voice": "realme",
                   "api_key": "", "batch": 4, "pause": 0.25,
                   "model_aliases": []}


def load_config(path: Path) -> dict:
    cfg = dict(CONFIG_DEFAULTS)
    if path.is_file():
        try:
            got = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise SystemExit(f"{path} is not valid JSON: {e}")
        cfg.update({k: v for k, v in got.items() if k in CONFIG_DEFAULTS})
    return cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG_FILE),
                    help="settings file (default: voice_server.json beside server/)")
    ap.add_argument("--library", help="the engine library (default: found in build/ or bin/)")
    ap.add_argument("--models", help="model folder (default: models_q8/, else models/)")
    ap.add_argument("--voices", default=str(ROOT / "voices"),
                    help="folder of voices: <name>.wav or <name>.f32 (default: voices/)")
    ap.add_argument("--voice", help="the voice when a request names none (default: realme)")
    ap.add_argument("--host", help="address to listen on (default: 127.0.0.1, this computer only)")
    ap.add_argument("--port", type=int, help="port (default 8800): whatever the tool expects")
    ap.add_argument("--api-key", help="ask for this key (Authorization: Bearer ...)")
    ap.add_argument("--batch", type=int, help="sentences per pass (default 4; 1 = one at a time)")
    ap.add_argument("--threads", type=int, help="CPU threads (default: one per core, up to 6)")
    ap.add_argument("--pause", type=float,
                    help="seconds of silence between sentences (default 0.25)")
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="longest sentence, in codes at 12.5 a second (default 4096)")
    ap.add_argument("--exact", action="store_true",
                    help="leave the fast snake and the 32-bit vocoder off")
    a = ap.parse_args(argv)
    cfg = load_config(Path(a.config))
    for key in ("host", "port", "voice", "api_key", "batch", "pause"):
        if getattr(a, key) is not None:
            cfg[key] = getattr(a, key)

    library, models = find_library(a.library), find_models(a.models)
    switches = set_engine_defaults(a.exact, a.threads, library)
    voices_dir = Path(a.voices).resolve()
    voices_dir.mkdir(parents=True, exist_ok=True)
    print(f"  engine:  {library}")
    print(f"  models:  {models}")
    print(f"  voices:  {voices_dir}")
    print("  " + ", ".join(f"{k}={v}" for k, v in switches.items()))
    engine = Engine(library, models, a.max_tokens)
    voices = Voices(voices_dir, engine)
    names = voices.names()
    if not names:
        print(f"\n  No voices yet: put a WAV recording of a speaker (10-60 s of "
              f"ordinary speech) in {voices_dir}, e.g. realme.wav, and restart.")
    for n in names:                       # embeddings now, not at the first request
        try:
            voices.get(n)
        except Exception as e:
            print(f"  voice {n!r} could not be read: {e}")
    default_voice = cfg["voice"] if cfg["voice"] in names else (names[0] if names else None)
    aliases = [str(m) for m in cfg.get("model_aliases") or []]
    recorder = Recorder(engine, voices, int(cfg["batch"]))
    recorder.start()
    info = {"model": MODEL_ID, "model_aliases": aliases, "models": str(models),
            "batch": int(cfg["batch"]), "engine_batch": engine.batch_fn is not None,
            **switches}
    handler = make_handler(recorder, voices, info, cfg["api_key"] or None,
                           default_voice, float(cfg["pause"]), aliases)
    host, port = str(cfg["host"]), int(cfg["port"])
    if host not in ("127.0.0.1", "localhost", "::1") and not cfg["api_key"]:
        print("  WARNING: listening beyond this computer with no api_key.")
    httpd = ThreadingHTTPServer((host, port), handler)
    print(f"\n  Ready: http://{host}:{port}/v1/audio/speech  model {MODEL_ID}, "
          f"voices: {', '.join(names) or 'none'}  (Ctrl+C stops it)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
