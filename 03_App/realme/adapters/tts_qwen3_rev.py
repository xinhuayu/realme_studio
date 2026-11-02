"""realme-rev-qwen3-tts: RealMe's own local voice (engines/rev.py)."""
import hashlib
import os
from pathlib import Path

from realme.adapters.base import AdapterUnavailable
from realme.adapters.tts_qwen3_dll import Qwen3DllTTS, _engine_call, batch_built_at
from realme.engines.rev import (NAME, MASK, MODEL_NAMES, engine_home, selection, cpu_settings,
                                environment, model_identity, sound_switches, batch_default,
                                library_folder)
from realme.engines.rev_build import DLLS


class RealmeRevTTS(Qwen3DllTTS):
    name = NAME
    log_name = NAME
    fingerprint_name = NAME

    def __init__(self, home=None, *, reference_wav=None, threads=0,
                 temperature=None, top_k=None, top_p=None,
                 repetition_penalty=None, max_tokens=4096, language='en',
                 backend='cpu'):
        home = Path(home) if home else engine_home()
        chosen = selection(home)
        super().__init__(home, reference_wav=reference_wav,
                         threads=threads or chosen['threads'], temperature=temperature,
                         top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
                         max_tokens=max_tokens, language=language, backend='cpu')
        # Earlier Studio base classes have no seed setting. Own it here so the
        # adapter has the same policy as the standalone bindings.
        raw_seed = os.environ.get('QWEN3_TTS_SEED', '')
        self.seed = None
        if raw_seed.isascii() and raw_seed.isdecimal():
            try:
                value = int(raw_seed)
                if value <= 0xffffffff:
                    self.seed = value
            except ValueError:
                pass
        self.quant = 'q8_0'
        self.models = chosen['models']
        self.cpu_enabled = chosen['enabled']
        self.cpu_mask = MASK
        self.cp_steps = chosen['cp_steps']
        self.library_directory = library_folder(home)
        self._identities = None
        # Cache keys must change when a rebuild changes the actual producer.
        from realme.engines.rev import sha
        self.library_identity = hashlib.sha256(''.join(
            sha(self.library_directory/name) if (self.library_directory/name).is_file() else 'missing:'+name
            for name in DLLS).encode()).hexdigest()

    def _model_identities(self):
        # Read when a take or an embedding needs it, not on a status check.
        if self._identities is None:
            self._identities = model_identity(self.home, self.models)
        return self._identities

    def _engine_environment(self):
        return cpu_settings(self.threads, getattr(self, 'seed', None), self.cp_steps)

    def _library_path(self):
        return self.library_directory/'qwen3tts.dll'

    def _models_dir(self, make=False):
        if not all((self.models/name).is_file() for name in MODEL_NAMES):
            raise AdapterUnavailable(f'{NAME}: the 8-bit model files are missing in {self.models}')
        return self.models

    def check(self):
        # Status refresh must not load the model.
        if not self._library_path().is_file():
            raise AdapterUnavailable(f'{NAME}: not built yet; run _Rebuild_Voice_Engine.bat')
        for name in ['ggml.dll', 'ggml-base.dll', 'ggml-cpu.dll']:
            if not (self.library_directory / name).is_file():
                raise AdapterUnavailable(f'{NAME}: missing {name}; run _Rebuild_Voice_Engine.bat')
        self._models_dir()

    def preflight(self):
        from realme.adapters.tts_qwen3_dll import _ENGINE_CALL_LOCK
        with _ENGINE_CALL_LOCK:
            super().preflight()
            super().set_cpu_options(self.cpu_mask, self.cpu_enabled)

    @_engine_call
    def set_cpu_options(self, mask=0, enabled=0):
        # preflight used to overwrite caller changes at the next synthesis.
        super().set_cpu_options(mask, enabled)
        self.cpu_mask, self.cpu_enabled = mask, enabled

    def applies_threads(self):
        # This source applies its thread count (whichever build is loaded):
        # one engine per that many cores, not two old-style 4-thread ones.
        return True

    def batch_built(self):
        # Its own build keeps the record beside the library; the previous
        # engine's build (the fallback), one folder up.
        for folder in (self.library_directory, self.library_directory.parent, self.home):
            if (folder/'build_info.json').is_file():
                return batch_built_at(folder, 'cpu')
        return False

    def batch_default(self):
        # Measured here by "Measure this computer", else 16 a pass with 32 GB
        # or more, else 8.
        return batch_default(self.home)

    def voice_fingerprint(self):
        fingerprint = super().voice_fingerprint()
        # Older Studio also omitted sampled seeds from its take identity.
        seed = getattr(self, 'seed', None)
        if self.temperature > 0 and seed is not None and f's{seed}' not in fingerprint.split(':'):
            fingerprint += f':s{seed}'
        models = hashlib.sha256(''.join(self._model_identities()[n] for n in MODEL_NAMES).encode()).hexdigest()
        with environment(self._engine_environment()):
            switches = sound_switches()
        # Not the thread count: the samples are the same on any number of
        # threads, and a recording engine copy given its share of the cores
        # must be recognised as the same voice.
        return (f'{fingerprint}:models{models}'
                f':cpu{self.cpu_mask}.{self.cpu_enabled}:dll{self.library_identity}{switches}')

    def _embedding_cache_path(self, ref):
        digest = hashlib.sha256(Path(ref).read_bytes()).hexdigest()[:16]
        encoder = self._model_identities()[MODEL_NAMES[0]]
        return self.home / 'embeddings' / f'{digest}-{encoder[:16]}.f32'
