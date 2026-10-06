"""Authenticated loopback worker for the local Qwen3-TTS adapter.

This file is intentionally dependency-light at import time. The Qwen package
is imported only after its isolated Python environment is explicitly enabled
and a local model directory is supplied.
The main application never imports this module.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import threading
import time
from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote

PROFILE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,128}$")
OPERATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,180}$")
ENGINE_MODULES = {"qwen3-tts": "qwen_tts"}
SUPPORTED_LANGUAGES = {
    "qwen3-tts": ("EN", "ZH", "JA", "KO", "DE", "FR", "RU", "PT", "ES", "IT"),
}
QWEN_LANGUAGE_NAMES = {
    "en": "English",
    "en-us": "English",
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ja": "Japanese",
    "ko": "Korean",
    "auto": "auto",
}


def _activate_shared_sox() -> None:
    """Expose the one shared SoX installation to Qwen's Python wrapper."""

    common_tools_root = Path(__file__).resolve().parents[2] / "_common_tools"
    sox_root = common_tools_root / "sox"
    if not sox_root.is_dir():
        return
    candidates = sorted(
        (
            candidate
            for candidate in sox_root.iterdir()
            if candidate.is_dir() and (candidate / "sox.exe").is_file()
        ),
        key=lambda candidate: candidate.name,
        reverse=True,
    )
    if not candidates:
        return
    sox_bin = str(candidates[0])
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if sox_bin not in path_entries:
        os.environ["PATH"] = os.pathsep.join([sox_bin, *[entry for entry in path_entries if entry]])


_activate_shared_sox()


class OperationCancelled(RuntimeError):
    """Raised when synthesis is cancelled at a safe worker boundary."""


def _contained(root: Path, value: Path) -> Path:
    resolved_root = root.resolve()
    resolved = value.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("Path leaves the managed Qwen3-TTS profile root.") from exc
    return resolved


def _write_wav(path: Path, audio: Any, sample_rate: int) -> None:
    import numpy as np
    import soundfile as sf

    array = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
    array = np.asarray(array)
    if array.ndim == 2:
        # Torch audio convention is channels x samples; generated TTS is mono.
        if array.shape[0] <= 4 and array.shape[1] > array.shape[0]:
            array = array[0]
        else:
            array = array[:, 0]
    array = np.asarray(array, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise ValueError("Qwen3-TTS returned empty audio.")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(array, -1.0, 1.0), int(sample_rate), format="WAV", subtype="PCM_16")


class Qwen3TTSEngine:
    def __init__(
        self,
        *,
        engine: str,
        model_path: Path,
        profile_root: Path,
        model_cache_root: Path,
        device: str,
        max_threads: int,
        non_streaming_mode: bool,
        synthesis_cache_enabled: bool,
        synthesis_cache_max_entries: int,
    ) -> None:
        if engine not in ENGINE_MODULES:
            raise ValueError(f"Unsupported Qwen3-TTS engine: {engine}")
        self.engine = engine
        self.model_path = model_path.resolve()
        self.profile_root = profile_root.resolve()
        self.model_cache_root = model_cache_root.resolve()
        self.requested_device = device
        self.max_threads = max(1, min(int(max_threads), 8))
        self.non_streaming_mode = bool(non_streaming_mode)
        self.synthesis_cache_enabled = bool(synthesis_cache_enabled)
        self.synthesis_cache_max_entries = max(
            1, min(int(synthesis_cache_max_entries), 5_000)
        )
        self.profile_root.mkdir(parents=True, exist_ok=True)
        self.model_cache_root.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(self.model_cache_root / "huggingface"))
        os.environ.setdefault("TORCH_HOME", str(self.model_cache_root / "torch"))
        os.environ.setdefault("NUMBA_CACHE_DIR", str(self.model_cache_root / "numba"))
        Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        self._lock = threading.RLock()
        self._model: Any = None
        self._device = "cpu"
        self._qwen_prompts: dict[str, Any] = {}
        self._operation_cancel_lock = threading.Lock()
        self._operation_cancel_events: dict[str, threading.Event] = {}

    def _resolve_device(self) -> str:
        if self.requested_device != "auto":
            return self.requested_device
        try:
            import torch

            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _check_package(self) -> None:
        module = ENGINE_MODULES[self.engine]
        if importlib.util.find_spec(module) is None:
            raise RuntimeError(
                f"{module} is not installed in this isolated Qwen3-TTS environment."
            )
        if not self.model_path.is_dir():
            raise RuntimeError("The configured local model directory is missing.")

    def health(self) -> dict[str, Any]:
        self._check_package()
        self._device = self._resolve_device()
        return {
            "status": "ready",
            "engine": self.engine,
            "device": self._device,
            "supported_languages": list(SUPPORTED_LANGUAGES[self.engine]),
            "languages": list(SUPPORTED_LANGUAGES[self.engine]),
            "processing_location": "local",
            "models_loaded_lazily": True,
            "model_loaded": self._model is not None,
            "max_threads": self.max_threads,
            "non_streaming_mode": self.non_streaming_mode,
            "synthesis_cache_enabled": self.synthesis_cache_enabled,
            "synthesis_cache_max_entries": self.synthesis_cache_max_entries,
            "no_auto_download": True,
        }

    def _configure_torch(self) -> None:
        try:
            import torch

            torch.set_num_threads(self.max_threads)
            if self._device == "cpu":
                torch.set_num_interop_threads(1)
        except (ImportError, RuntimeError):
            pass

    def _load_qwen(self) -> Any:
        if self._model is None:
            self._device = self._resolve_device()
            self._configure_torch()
            # qwen-tts 0.1.x exposes Qwen3TTSModel (the README's older
            # shorthand Qwen3TTS is not exported by the installed package).
            from qwen_tts import Qwen3TTSModel

            kwargs: dict[str, Any] = {}
            if self._device != "cpu":
                kwargs["device_map"] = self._device
            self._model = Qwen3TTSModel.from_pretrained(str(self.model_path), **kwargs)
        return self._model

    def warmup(self) -> dict[str, Any]:
        """Load the local model without reading a user profile or voice sample."""

        with self._lock:
            self._check_package()
            started = time.perf_counter()
            already_loaded = self._model is not None
            self._load_qwen()
            return {
                "status": "ready",
                "engine": self.engine,
                "device": self._device,
                "model": self.model_path.name,
                "model_loaded": True,
                "already_loaded": already_loaded,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "max_threads": self.max_threads,
                "non_streaming_mode": self.non_streaming_mode,
                "synthesis_cache_enabled": self.synthesis_cache_enabled,
            }

    def _profile_directory(self, profile_id: str) -> Path:
        if not PROFILE_ID_PATTERN.fullmatch(profile_id):
            raise ValueError("Invalid Qwen3-TTS profile id.")
        return _contained(self.profile_root, self.profile_root / profile_id)

    def _sample_path(self, profile_id: str, sample_files: list[str]) -> Path:
        directory = self._profile_directory(profile_id)
        if not sample_files:
            raise ValueError("At least one reference recording is required.")
        path = _contained(directory, directory / Path(sample_files[0]).name)
        if not path.is_file():
            raise FileNotFoundError("A managed reference recording is missing.")
        return path

    def create_profile(
        self,
        *,
        profile_id: str,
        sample_files: list[str],
        display_name: str,
        sample_hashes: list[str],
        labels: Mapping[str, Any],
    ) -> dict[str, Any]:
        with self._lock:
            self._check_package()
            directory = self._profile_directory(profile_id)
            sample = self._sample_path(profile_id, sample_files)
            sample_text = str(
                labels.get("reference_transcript") or labels.get("sample_text") or ""
            ).strip()[:5_000]
            sample_language = str(
                labels.get("reference_language") or labels.get("sample_language") or "en-US"
            )
            if self.engine == "qwen3-tts" and not sample_text:
                raise ValueError("Qwen3-TTS voice cloning requires the reference transcript.")
            cached_prompt = False
            if self.engine == "qwen3-tts":
                model = self._load_qwen()
                prompt = model.create_voice_clone_prompt(
                    ref_audio=str(sample),
                    ref_text=sample_text,
                    x_vector_only_mode=False,
                )
                import torch

                torch.save(prompt, directory / "voice-clone-prompt.pt")
                self._qwen_prompts[profile_id] = prompt
                cached_prompt = True
            manifest = {
                "profile_id": profile_id,
                "display_name": display_name[:200],
                "engine": self.engine,
                "sample_files": sample_files,
                "sample_hashes": sample_hashes,
                "sample_text": sample_text,
                "reference_transcript": sample_text,
                "sample_text_present": bool(sample_text),
                "sample_language": sample_language,
                "reference_language": sample_language,
                "reference_transcript_source": str(
                    labels.get("reference_transcript_source")
                    or "user_supplied_recording_text"
                )[:100],
                "reference_audio_strategy": str(
                    labels.get("reference_audio_strategy")
                    or "provider_reference_audio"
                )[:100],
                "reference_tuning": (
                    dict(labels.get("reference_tuning"))
                    if isinstance(labels.get("reference_tuning"), Mapping)
                    else {}
                ),
                "cached_voice_prompt": cached_prompt,
                "clone_conditioning": "cached_voice_clone_prompt" if cached_prompt else None,
                "x_vector_only_mode": False,
                "reference_transcript_required": self.engine == "qwen3-tts",
                "processing_location": "local",
                "created_at": int(time.time()),
            }
            temporary = directory / "profile-manifest.tmp"
            temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temporary, directory / "profile-manifest.json")
            return {**manifest, "engine": self.engine, "cached_voice_prompt": cached_prompt}

    def _qwen_prompt(self, profile_id: str, sample: Path, sample_text: str) -> Any:
        prompt = self._qwen_prompts.get(profile_id)
        if prompt is not None:
            return prompt
        import torch

        prompt_file = self._profile_directory(profile_id) / "voice-clone-prompt.pt"
        if prompt_file.is_file():
            try:
                prompt = torch.load(str(prompt_file), map_location=self._device, weights_only=False)
            except TypeError:
                prompt = torch.load(str(prompt_file), map_location=self._device)
        else:
            if not sample_text:
                raise ValueError("The saved Qwen3-TTS profile has no reference transcript.")
            prompt = self._load_qwen().create_voice_clone_prompt(
                ref_audio=str(sample),
                ref_text=sample_text,
                x_vector_only_mode=False,
            )
        self._qwen_prompts[profile_id] = prompt
        return prompt

    @staticmethod
    def _dynamic_token_limit(text: str) -> tuple[int, float]:
        """Estimate a conservative 12 Hz codec-token ceiling from authored text."""

        latin_words = len(re.findall(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?", text))
        cjk_characters = sum(
            "\u3400" <= character <= "\u4dbf"
            or "\u4e00" <= character <= "\u9fff"
            or "\u3040" <= character <= "\u30ff"
            or "\uac00" <= character <= "\ud7af"
            for character in text
        )
        other_characters = sum(
            character.isalpha() and not character.isascii() for character in text
        ) - cjk_characters
        estimated_seconds = max(
            2.0,
            latin_words / 2.15
            + cjk_characters / 3.6
            + max(0, other_characters) / 10.0,
        )
        # 12 codec tokens/second, generous delivery headroom, and a fixed EOS
        # allowance. Round upward to a stable 64-token boundary.
        estimated_tokens = estimated_seconds * 12.0 * 1.75 + 96
        rounded = int(math.ceil(estimated_tokens / 64.0) * 64)
        return max(256, min(rounded, 8_192)), round(estimated_seconds, 2)

    @classmethod
    def _qwen_generation_options(
        cls,
        body: Mapping[str, Any],
        text: str,
        *,
        non_streaming_mode: bool,
    ) -> dict[str, Any]:
        """Return bounded generation options supported by Qwen Base models.

        Qwen3-TTS Base is an in-context voice-clone model. It does not expose
        the application's direct pitch/warmth controls; those are applied to
        the temporary reference before this worker creates its clone prompt.
        Keep the worker's model-facing generation options small and explicit so
        an overly large request cannot consume unbounded CPU/RAM.
        """

        delivery = body.get("delivery_plan")
        delivery = delivery if isinstance(delivery, Mapping) else {}
        estimated_limit, estimated_seconds = cls._dynamic_token_limit(text)
        raw_limit = delivery.get("max_new_tokens", body.get("max_new_tokens"))
        if raw_limit is None:
            max_new_tokens = estimated_limit
            token_limit_source = "text_estimate"
        else:
            try:
                max_new_tokens = int(raw_limit)
            except (TypeError, ValueError):
                max_new_tokens = estimated_limit
                token_limit_source = "text_estimate"
            else:
                token_limit_source = "explicit"
        max_new_tokens = max(256, min(max_new_tokens, 8192))
        raw_variation = body.get("variation_scale", 0.40)
        try:
            variation = float(raw_variation)
        except (TypeError, ValueError):
            variation = 0.40
        variation = max(0.30, min(variation, 0.60))
        processing_mode = str(body.get("processing_mode") or "fast").strip().lower()
        if processing_mode not in {"fast", "fresh"}:
            raise ValueError("processing_mode must be fast or fresh.")
        # The UI keeps the original app's conservative 30–60% control. Map it
        # to Qwen's standard generation temperature without exposing unbounded
        # model kwargs to the worker.
        temperature = 0.65 + ((variation - 0.30) / 0.30) * 0.45
        return {
            "max_new_tokens": max_new_tokens,
            "temperature": round(temperature, 3),
            "top_p": 0.90,
            "non_streaming_mode": non_streaming_mode,
            "token_limit_source": token_limit_source,
            "estimated_speech_seconds": estimated_seconds,
            "processing_mode": processing_mode,
        }

    @staticmethod
    def _model_generation_options(options: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: options[key]
            for key in ("max_new_tokens", "temperature", "top_p")
        }

    def _profile_fingerprint(self, manifest: Mapping[str, Any]) -> str:
        model_file = self.model_path / "model.safetensors"
        model_identity = {
            "name": self.model_path.name,
            "size": model_file.stat().st_size if model_file.is_file() else None,
            "modified_ns": model_file.stat().st_mtime_ns if model_file.is_file() else None,
        }
        value = {
            "schema": "real-voice-of-you.qwen-profile-fingerprint.v1",
            "engine": self.engine,
            "model": model_identity,
            "sample_hashes": list(manifest.get("sample_hashes") or []),
            "reference_transcript": str(
                manifest.get("reference_transcript") or manifest.get("sample_text") or ""
            ),
            "reference_language": str(
                manifest.get("reference_language") or manifest.get("sample_language") or ""
            ),
            "reference_tuning": manifest.get("reference_tuning") or {},
        }
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _synthesis_cache_key(
        self,
        *,
        profile_id: str,
        manifest: Mapping[str, Any],
        text: str,
        language: str,
        generation_options: Mapping[str, Any],
    ) -> str:
        value = {
            "schema": "real-voice-of-you.qwen-synthesis-cache.v1",
            "profile_id": profile_id,
            "profile_fingerprint": self._profile_fingerprint(manifest),
            "text": text,
            "language": language,
            "generation_options": dict(generation_options),
        }
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _prune_synthesis_cache(self, directory: Path) -> None:
        files = sorted(
            directory.glob("*.wav"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        for path in files[self.synthesis_cache_max_entries :]:
            path.unlink(missing_ok=True)

    @staticmethod
    def _valid_cached_wav(path: Path) -> bool:
        if not path.is_file() or path.stat().st_size <= 44:
            return False
        with path.open("rb") as source:
            return source.read(4) == b"RIFF"

    def _synthesize_qwen(
        self,
        body: Mapping[str, Any],
        sample: Path,
        sample_text: str,
        language: str,
        generation_options: Mapping[str, Any],
    ) -> tuple[Any, int, dict[str, Any]]:
        model = self._load_qwen()
        prompt = self._qwen_prompt(str(body["profile_id"]), sample, sample_text)
        wavs, sample_rate = model.generate_voice_clone(
            text=str(body["text"]),
            language=language,
            voice_clone_prompt=prompt,
            non_streaming_mode=bool(generation_options["non_streaming_mode"]),
            **self._model_generation_options(generation_options),
        )
        return (
            wavs[0] if isinstance(wavs, (list, tuple)) else wavs,
            int(sample_rate),
            generation_options,
        )

    def _register_operation(self, operation_id: str) -> None:
        with self._operation_cancel_lock:
            if operation_id in self._operation_cancel_events:
                raise ValueError(f"Operation already active: {operation_id}")
            self._operation_cancel_events[operation_id] = threading.Event()

    def _unregister_operation(self, operation_id: str) -> None:
        with self._operation_cancel_lock:
            self._operation_cancel_events.pop(operation_id, None)

    def _check_operation_cancelled(self, operation_id: str) -> None:
        with self._operation_cancel_lock:
            event = self._operation_cancel_events.get(operation_id)
        if event is not None and event.is_set():
            raise OperationCancelled("Local Qwen3-TTS synthesis was cancelled by the user.")

    def cancel(self, operation_id: str) -> bool:
        with self._operation_cancel_lock:
            matches = [
                event
                for active_id, event in self._operation_cancel_events.items()
                if active_id == operation_id
                or active_id.startswith(f"{operation_id}-segment-")
            ]
            for event in matches:
                event.set()
        return bool(matches)

    def synthesize(self, body: Mapping[str, Any]) -> dict[str, Any]:
        operation_id = str(body.get("operation_id") or "")
        if not OPERATION_ID_PATTERN.fullmatch(operation_id):
            return self._synthesize_impl(body)
        self._register_operation(operation_id)
        try:
            return self._synthesize_impl(body)
        finally:
            self._unregister_operation(operation_id)

    def _synthesize_impl(self, body: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            profile_id = str(body.get("profile_id") or "")
            operation_id = str(body.get("operation_id") or "")
            if not OPERATION_ID_PATTERN.fullmatch(operation_id):
                raise ValueError("Invalid synthesis operation id.")
            text = str(body.get("text") or "").strip()
            if not text:
                raise ValueError("Narration text is required.")
            self._check_operation_cancelled(operation_id)
            directory = self._profile_directory(profile_id)
            manifest_path = directory / "profile-manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError("The Qwen3-TTS voice profile is incomplete.")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            sample = self._sample_path(profile_id, list(manifest.get("sample_files") or []))
            sample_text = str(manifest.get("sample_text") or "")
            self._device = self._resolve_device()
            self._configure_torch()
            output = _contained(
                self.profile_root,
                self.profile_root / "_operations" / f"{operation_id.replace(':', '-')}.wav",
            )
            requested_language = str(body.get("language_code") or "auto").strip().lower()
            language = QWEN_LANGUAGE_NAMES.get(
                requested_language,
                QWEN_LANGUAGE_NAMES.get(requested_language.split("-", 1)[0], "auto"),
            )
            generation_options = self._qwen_generation_options(
                body,
                text,
                non_streaming_mode=self.non_streaming_mode,
            )
            cache_key = self._synthesis_cache_key(
                profile_id=profile_id,
                manifest=manifest,
                text=text,
                language=language,
                generation_options=generation_options,
            )
            cache_directory = _contained(directory, directory / "synthesis-cache")
            cache_path = _contained(cache_directory, cache_directory / f"{cache_key}.wav")
            cache_eligible = (
                self.synthesis_cache_enabled
                and generation_options["processing_mode"] == "fast"
            )
            cache_hit = (
                cache_eligible
                and self._valid_cached_wav(cache_path)
            )
            if cache_hit:
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(cache_path, output)
                os.utime(cache_path, None)
            else:
                audio, sample_rate, generation_options = self._synthesize_qwen(
                    body,
                    sample,
                    sample_text,
                    language,
                    generation_options,
                )
                self._check_operation_cancelled(operation_id)
                if cache_eligible:
                    cache_directory.mkdir(parents=True, exist_ok=True)
                    temporary_cache = cache_path.with_suffix(".wav.tmp")
                    _write_wav(temporary_cache, audio, sample_rate)
                    os.replace(temporary_cache, cache_path)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(cache_path, output)
                    self._prune_synthesis_cache(cache_directory)
                else:
                    _write_wav(output, audio, sample_rate)
            self._check_operation_cancelled(operation_id)
            return {
                "profile_id": profile_id,
                "operation_id": operation_id,
                "audio_file": output.relative_to(self.profile_root).as_posix(),
                "output_format": "wav",
                "media_type": "audio/wav",
                "engine": self.engine,
                "model": self.model_path.name,
                "language": body.get("language_code") or "auto",
                "device": self._device,
                "profile_reused": True,
                "clone_conditioning": "cached_voice_clone_prompt",
                "reference_transcript_required": True,
                "reference_transcript_present": bool(
                    manifest.get("reference_transcript") or sample_text
                ),
                "reference_language": manifest.get(
                    "reference_language", manifest.get("sample_language", "en-US")
                ),
                "reference_audio_strategy": manifest.get("reference_audio_strategy"),
                "reference_tuning": manifest.get("reference_tuning", {}),
                "x_vector_only_mode": False,
                "generation_options": generation_options,
                "synthesis_cache_hit": cache_hit,
                "synthesis_cache_key": cache_key if cache_eligible else None,
                "synthesis_cache_eligible": cache_eligible,
                "processing_mode": generation_options["processing_mode"],
                "model_warm": self._model is not None,
                "speech_pattern_received": bool(body.get("speech_pattern")),
                "delivery_plan_received": bool(body.get("delivery_plan")),
                "speech_pattern_applied": False,
                "delivery_plan_applied": False,
                "control_note": (
                    "Qwen3-TTS Base applies voice identity through a cached clone prompt "
                    "built from the locally beautified reference audio. Pace and script "
                    "delivery controls are applied after generation by the main application; "
                    "identity shaping is not applied a second time."
                ),
            }

    def delete_profile(self, profile_id: str) -> dict[str, Any]:
        with self._lock:
            directory = self._profile_directory(profile_id)
            existed = directory.is_dir()
            if existed:
                shutil.rmtree(directory)
            self._qwen_prompts.pop(profile_id, None)
            gc.collect()
            return {"profile_id": profile_id, "profile_deleted": existed}


class WorkerServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], engine: Qwen3TTSEngine, token: str):
        super().__init__(address, WorkerHandler)
        self.engine = engine
        self.token = token


class WorkerHandler(BaseHTTPRequestHandler):
    server: WorkerServer

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def _authorized(self) -> bool:
        return self.headers.get("Authorization") == f"Bearer {self.server.token}"

    def _json_body(self) -> dict[str, Any]:
        size = int(self.headers.get("Content-Length") or 0)
        if size <= 0 or size > 2_000_000:
            raise ValueError("Invalid JSON request size.")
        value = json.loads(self.rfile.read(size).decode("utf-8"))
        if not isinstance(value, dict):
            raise TypeError("JSON request must be an object.")
        return value

    def _send(self, status: int, value: dict[str, Any]) -> None:
        encoded = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _run(self, operation: Any) -> None:
        if not self._authorized():
            self._send(HTTPStatus.UNAUTHORIZED, {"error": "Unauthorized local worker request."})
            return
        try:
            self._send(HTTPStatus.OK, operation())
        except FileNotFoundError as exc:
            self._send(HTTPStatus.NOT_FOUND, {"error": str(exc)})
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": str(exc)})
        except OperationCancelled as exc:
            self._send(
                HTTPStatus.CONFLICT,
                {
                    "error": "Synthesis cancelled.",
                    "error_code": "cancelled",
                    "detail": str(exc),
                },
            )
        except Exception as exc:  # noqa: BLE001 - isolate ML failures at the HTTP boundary
            self._send(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "Local Qwen3-TTS processing failed.", "detail": str(exc)[:500]},
            )

    def do_GET(self) -> None:
        if self.path != "/health":
            self._send(HTTPStatus.NOT_FOUND, {"error": "Not found."})
            return
        self._run(self.server.engine.health)

    def do_POST(self) -> None:
        if self.path == "/shutdown":
            if not self._authorized():
                self._send(HTTPStatus.UNAUTHORIZED, {"error": "Unauthorized local worker request."})
                return
            self._send(HTTPStatus.OK, {"status": "shutting_down"})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        try:
            body = self._json_body()
        except Exception as exc:  # noqa: BLE001
            self._send(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": str(exc)})
            return
        if self.path == "/profiles":
            self._run(
                lambda: self.server.engine.create_profile(
                    profile_id=str(body.get("profile_id") or ""),
                    sample_files=[str(item) for item in body.get("sample_files") or []],
                    display_name=str(body.get("display_name") or ""),
                    sample_hashes=[str(item) for item in body.get("sample_hashes") or []],
                    labels=body.get("labels") if isinstance(body.get("labels"), Mapping) else {},
                )
            )
            return
        if self.path == "/warmup":
            self._run(self.server.engine.warmup)
            return
        if self.path == "/synthesize":
            self._run(lambda: self.server.engine.synthesize(body))
            return
        if self.path == "/cancel":
            operation_id = str(body.get("operation_id") or "")
            if not OPERATION_ID_PATTERN.fullmatch(operation_id):
                self._send(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": "Invalid synthesis operation id."})
                return
            self._run(
                lambda: {
                    "operation_id": operation_id,
                    "cancel_requested": self.server.engine.cancel(operation_id),
                }
            )
            return
        self._send(HTTPStatus.NOT_FOUND, {"error": "Not found."})

    def do_DELETE(self) -> None:
        if not self.path.startswith("/profiles/"):
            self._send(HTTPStatus.NOT_FOUND, {"error": "Not found."})
            return
        profile_id = unquote(self.path.rsplit("/", 1)[-1])
        self._run(lambda: self.server.engine.delete_profile(profile_id))


def main() -> int:
    parser = argparse.ArgumentParser(description="Local Qwen3-TTS worker")
    parser.add_argument("--engine", choices=sorted(ENGINE_MODULES), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--model-cache-root", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-threads", type=int, default=4)
    parser.add_argument(
        "--non-streaming-mode",
        dest="non_streaming_mode",
        action="store_true",
    )
    parser.add_argument(
        "--streaming-input-mode",
        dest="non_streaming_mode",
        action="store_false",
    )
    parser.add_argument("--synthesis-cache", dest="synthesis_cache", action="store_true")
    parser.add_argument("--no-synthesis-cache", dest="synthesis_cache", action="store_false")
    parser.add_argument("--synthesis-cache-max-entries", type=int, default=512)
    parser.add_argument("--parent-pid", type=int, default=0)
    parser.set_defaults(non_streaming_mode=True, synthesis_cache=True)
    args = parser.parse_args()
    engine = Qwen3TTSEngine(
        engine=args.engine,
        model_path=args.model_path,
        profile_root=args.profile_root,
        model_cache_root=args.model_cache_root,
        device=args.device,
        max_threads=args.max_threads,
        non_streaming_mode=args.non_streaming_mode,
        synthesis_cache_enabled=args.synthesis_cache,
        synthesis_cache_max_entries=args.synthesis_cache_max_entries,
    )
    server = WorkerServer((args.host, args.port), engine, args.token)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
