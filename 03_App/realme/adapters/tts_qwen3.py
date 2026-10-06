"""
Qwen3-TTS running locally — the adapter that drives your own worker.

This does not reimplement anything. It launches the worker from
`real_voice_qwen3` and speaks its loopback protocol, so RealMe inherits, for
free, everything already built and audited there: lazy model loading, the
cached clone prompt, the per-profile synthesis cache, bounded codec-token
budgets, warm-up, cancellation, and the CPU thread tuning.

Why this is the right default for a laptop:

  * It runs on CPU. Four threads on a four-core machine is the tuned setting.
  * Your voice never leaves the machine. No API key, no upload, no per-hour cost.
  * Apache 2.0 on the model.
  * Cloning conditions on a reference clip **plus its exact transcript**, which
    is a real constraint worth knowing before you record: you must type out
    precisely what you said. It also makes the clone noticeably better than
    audio alone.

Point `REALME_QWEN3_ROOT` at the real_voice_qwen3 folder and this adapter
finds the venv, worker, model and profile store underneath it.
"""
from __future__ import annotations
import hashlib
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from realme.adapters.base import BaseTTS, AdapterUnavailable
from realme.core.media import probe

DEFAULT_MODEL = "Qwen3-TTS-12Hz-0.6B-Base"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class Qwen3LocalTTS(BaseTTS):
    name = "qwen3-tts-local"
    is_voice_clone = True
    is_placeholder = False
    # Qwen3-TTS has no documented phoneme interface, so the lexicon falls back
    # to respellings ("I P T W"), which still fix the pronunciation.
    phoneme_syntax = None

    def __init__(self, root: Path | None = None, *,
                 reference_wav: Path | None = None,
                 reference_text: str = "",
                 language: str = "en-US",
                 model: str = DEFAULT_MODEL,
                 device: str = "auto",
                 threads: int = 4,
                 profile_name: str = "realme"):
        raw_root = str(root or os.environ.get("REALME_QWEN3_ROOT", "")).strip()
        # Path("") is the current directory and would silently pass an is_dir()
        # check, producing a confusing "worker not found" instead of "you have
        # not told me where the project is".
        self.root = Path(raw_root).expanduser() if raw_root else None
        self.reference_wav = Path(reference_wav) if reference_wav else None
        self.reference_text = (reference_text
                               or os.environ.get("REALME_QWEN3_REF_TEXT", "")).strip()
        self.language = language
        self.model = model
        self.device = device
        self.threads = threads
        self.profile_name = profile_name
        self._proc: subprocess.Popen | None = None
        self._port: int | None = None
        self._token: str | None = None
        self._profile_id: str | None = None

    # ------------------------------------------------------------- locations
    # ---------------------------------------------------------- self-contained
    @staticmethod
    def home() -> Path:
        """
        Wherever the engine actually is -- the shared resolver, not a private one.

        This used to compute its own answer by counting `.parent`s from this
        file, which resolved to `03_App/tools/qwen3`. That was the old install
        location. Once the payload moved to the project root, this adapter alone
        kept looking in the empty folder and reported the engine missing, while
        piper and qwen3cpp -- which ask `realme.core.paths` -- found it. Three
        resolvers, three answers. Now there is one.
        """
        from realme.engines.install import engine_home
        return engine_home()

    @property
    def _external(self) -> bool:
        """True when pointed at a separate real_voice_qwen3 style install."""
        return self.root is not None

    @property
    def python(self) -> Path:
        if self._external:
            venv = self.root / ".qwen3-tts-venv"
        else:
            venv = self.home() / "venv"
        return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    @property
    def worker(self) -> Path:
        # Always the vendored copy: it is the version this adapter was written
        # against, and an external project may have drifted.
        return Path(__file__).resolve().parents[1] / "engines" / "qwen3" / "worker.py"

    @property
    def model_path(self) -> Path:
        if self._external:
            return self.root / "runtime" / "qwen3-tts-models" / self.model
        return self.home() / "models" / self.model

    @property
    def profile_root(self) -> Path:
        if self._external:
            return self.root / "runtime" / "qwen3-tts-profiles"
        return self.home() / "profiles"

    @property
    def cache_root(self) -> Path:
        """
        Where numba and librosa keep their compile caches.

        Every other path here has an external/self-contained pair. This one did
        not: `_start` reached for `self.root` directly, which is None unless
        REALME_QWEN3_ROOT points at an external project. Nobody noticed because
        the engine never got as far as starting -- preflight failed first, for
        unrelated reasons, in every code path that could have reached it.
        """
        if self._external:
            return self.root / "runtime" / "qwen3-tts-model-cache"
        return self.home() / "cache"

    def voice_fingerprint(self) -> str:
        ref = self.reference_wav.name if self.reference_wav else "none"
        h = hashlib.sha256(self.reference_text.encode()).hexdigest()[:8]
        return f"{self.name}:{self.model}:{ref}:{h}"

    # ------------------------------------------------------------- lifecycle
    PATHS = ("python", "worker", "model_path", "profile_root", "cache_root")

    def _resolve_paths(self) -> None:
        """
        Force every path property to evaluate, here, where the error is legible.

        This class has two modes -- self-contained and external -- and each path
        needs a branch for both. One of them did not have it, and the failure
        surfaced mid-render as "unsupported operand type(s) for /: 'NoneType'
        and 'str'", which names neither the property nor the mode. Touching them
        all in preflight turns that into a sentence.
        """
        for name in self.PATHS:
            try:
                value = getattr(self, name)
            except TypeError as e:
                raise AdapterUnavailable(
                    f"Internal: {name!r} does not resolve when the engine is "
                    f"{'external' if self._external else 'self-contained'} "
                    f"({e}). This is a bug in RealMe, not your setup.") from e
            if value is None:
                raise AdapterUnavailable(f"Internal: {name!r} resolved to None")

    def preflight(self) -> None:
        self._resolve_paths()
        if self.root is not None and not self.root.is_dir():
            raise AdapterUnavailable(
                f"REALME_QWEN3_ROOT points at {self.root}, which does not exist. "
                f"Unset it to use the built-in engine under {self.home()}.")
        missing = [(label, path) for label, path in
                   (("python runtime", self.python), ("worker", self.worker),
                    ("model weights", self.model_path)) if not path.exists()]
        if missing:
            detail = "\n".join(f"    {label}: {path}" for label, path in missing)
            raise AdapterUnavailable(
                "The local Qwen3 voice engine is not installed yet. Missing:\n"
                f"{detail}\n\n"
                "  realme engine install                 (downloads it)\n"
                "  realme engine install --from <folder> (copies an existing one)")
        if not self.reference_wav or not self.reference_wav.exists():
            raise AdapterUnavailable(
                "No voice reference yet. Enrol a recording (any format):\n"
                '  realme voice enroll my_take.m4a --transcript "what you said"'
                "\n  or in the Studio: Twin Setup -> Your voice.")
        if not self.reference_text:
            beside = self.reference_wav.with_suffix(".txt")
            # Show what IS in the folder. "File not found" when the user can see
            # the file is not a diagnosis -- and the usual cause is a name that
            # looks right in Explorer and is not: voice_reference.txt.txt, which
            # Notepad produces when "Save as type" is left on Text Documents,
            # and which Explorer displays as "voice_reference.txt".
            listing = ""
            try:
                names = sorted(p.name for p in beside.parent.iterdir()
                               if p.is_file())
                if names:
                    listing = ("\n\n  What is actually in that folder:\n"
                               + "\n".join(f"    {n}" for n in names)
                               + "\n\n  If you see a name ending .txt.txt, that "
                                 "is Windows hiding the real\n  extension. "
                                 "Rename it, or turn on File name extensions in "
                                 "Explorer's\n  View menu to see what is there.")
            except OSError:
                pass
            raise AdapterUnavailable(
                "Reference recorded, but no transcript. Qwen3-TTS clones from "
                "the audio PLUS its exact words, so it needs both.\n\n"
                "  Expected, exactly:\n"
                f"    {beside}" + listing
                + "\n\n  Or let RealMe write it:  "
                  "realme voice transcript my_script.txt")

    def _start(self) -> None:
        if self._proc and self._proc.poll() is None:
            return
        self._port = _free_port()
        self._token = f"rvy_{secrets.token_urlsafe(36)}"
        cache = self.cache_root
        cache.mkdir(parents=True, exist_ok=True)
        self.profile_root.mkdir(parents=True, exist_ok=True)
        cmd = [str(self.python), str(self.worker),
               "--engine", "qwen3-tts",
               "--host", "127.0.0.1", "--port", str(self._port),
               "--token", self._token,
               "--model-path", str(self.model_path),
               "--profile-root", str(self.profile_root),
               "--model-cache-root", str(cache),
               "--device", self.device,
               "--max-threads", str(self.threads),
               "--parent-pid", str(os.getpid())]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT)
        deadline = time.time() + 120
        while time.time() < deadline:
            if self._proc.poll() is not None:
                out = (self._proc.stdout.read() or b"").decode("utf-8", "replace")
                raise AdapterUnavailable(
                    f"Qwen3 worker exited immediately:\n{out[-800:]}")
            try:
                self._request("GET", "/health")
                return
            except Exception:
                time.sleep(0.5)
        raise AdapterUnavailable("Qwen3 worker did not become healthy in 120s.")

    def _request(self, method: str, path: str, body: dict | None = None,
                 timeout: float = 900.0) -> dict:
        url = f"http://127.0.0.1:{self._port}{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method, headers={
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read().decode()
            return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:400]
            raise RuntimeError(f"Qwen3 worker HTTP {e.code} on {path}: {detail}") from e

    def _ensure_profile(self) -> str:
        if self._profile_id:
            return self._profile_id
        self._start()
        digest = hashlib.sha256()
        digest.update(self.reference_wav.read_bytes())
        digest.update(self.reference_text.encode())
        pid = f"realme_{digest.hexdigest()[:28]}"
        directory = self.profile_root / pid
        directory.mkdir(parents=True, exist_ok=True)
        dest = directory / "reference-01.wav"
        if not dest.exists():
            shutil.copy(self.reference_wav, dest)
        self._request("POST", "/profiles", {
            "profile_id": pid,
            "display_name": self.profile_name,
            "sample_files": [dest.name],
            "sample_hashes": [hashlib.sha256(dest.read_bytes()).hexdigest()],
            "labels": {
                "reference_transcript": self.reference_text,
                "reference_language": self.language,
                "reference_audio_strategy": "realme-beautified",
            },
        })
        self._profile_id = pid
        return pid

    def warmup(self) -> None:
        """Load weights without touching a voice. Worth calling early."""
        self._start()
        self._request("POST", "/warmup", {})

    # ------------------------------------------------------------ synthesis
    def synthesize(self, text, out_wav, voice=None, pace=1.0):
        self.preflight()
        pid = self._ensure_profile()
        out_wav = Path(out_wav)
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        result = self._request("POST", "/synthesize", {
            "profile_id": pid,
            "text": text,
            "operation_id": f"realme-{secrets.token_hex(6)}",
            "language_code": self.language,
            "speed": float(pace),
            "prosody_mode": "natural",
            "processing_mode": "fast",
            "variation_scale": 0.40,
        })
        rel = Path(str(result.get("audio_file") or ""))
        if not rel.parts or rel.is_absolute() or ".." in rel.parts:
            raise RuntimeError(f"Qwen3 worker returned an unusable path: {rel!r}")
        produced = (self.profile_root / rel).resolve()
        produced.relative_to(self.profile_root.resolve())
        if not produced.is_file():
            raise RuntimeError("Qwen3 worker produced no audio file.")
        shutil.move(str(produced), str(out_wav))
        probe(out_wav)
        return out_wav

    def close(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                self._request("POST", "/shutdown", {}, timeout=10)
            except Exception:
                self._proc.terminate()
