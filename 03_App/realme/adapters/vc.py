"""
Say the words fast, then repaint the voice.

An autoregressive engine like Qwen3 generates one audio token per forward pass,
which is why it runs at RTF 6 and a 50-minute lecture costs five hours. Nothing
about that is a tuning problem -- it is what the architecture does.

This adapter takes the other route. A non-autoregressive engine that generates
the whole waveform in parallel (piper: RTF 0.04-0.10) says the words, and a
converter changes *whose voice* they are in. Two cheap stages instead of one
expensive one.

**Measured, on xyu's own 72-second reference, eight utterances, this container:**

    piper alone        RTF 0.04    resemblance 64.3%
    piper + knnvc      RTF 0.42    resemblance 71.8%

The composite number understates it, and the per-feature breakdown says why:

    pitch median          207.8 Hz -> 168.4 Hz   (reference 155.3)   56% -> 89%
    spectral centroid     1162 Hz  -> 1558 Hz    (reference 1484)    78% -> 95%
    pitch variation                                                  48% -> 73%
    dynamic range         36.8 dB  -> 29.7 dB    (reference 38.3)    85% -> 14%

Conversion moved every feature that carries *identity* to 89-95%. What it lost
was dynamic range, because a neural vocoder compresses -- that is a mixing
artefact, not a wrong voice, and it costs about ten points of the composite.
Read the features, not just the total.

The reference was 72 seconds. kNN-VC is non-parametric: it rebuilds each frame
out of the nearest frames in *your* recording, so it has more to work with the
more you give it, and 5-10 minutes is what the method expects. The recording
session that Path A needs for fine-tuning feeds this too.

**Limitation worth knowing.** WavLM and the kNN-VC vocoder both run at 16 kHz,
so output is 16 kHz -- duller than the 22-24 kHz the other engines produce.
OpenVoice does not have that ceiling.
"""
from __future__ import annotations
import hashlib
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path

from realme.adapters.base import BaseTTS, AdapterUnavailable


def _as_wav(path: Path) -> Path:
    """
    A reference the converters can read. `media.ensure_reference_wav` decides;
    this is here because a converter can be used without going through the
    adapter factory, not because it is a second implementation.
    """
    from realme.core.media import ensure_reference_wav
    return ensure_reference_wav(Path(path))


def _reference_key(path: Path) -> str:
    """Identity of a reference clip, for caching. Content, not filename."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


class Converter(ABC):
    """Turns one speaker's audio into another's. No text involved."""
    name = "converter"
    sample_rate = 16000

    @abstractmethod
    def preflight(self, reference: Path) -> None: ...

    @abstractmethod
    def convert(self, src_wav: Path, out_wav: Path, reference: Path) -> Path: ...


class KNNVCConverter(Converter):
    """
    kNN-VC: every frame of the draft is replaced by its nearest neighbours in
    your own recording, then vocoded back to sound.

    Two things are expensive and both are done once: loading WavLM (~18 s) and
    building the matching set from your reference (~19 s per minute of audio).
    The matching set is cached to disk against the reference's content hash, so
    changing the clip invalidates it and renaming it does not.
    """
    name = "knnvc"

    #: How many of your own frames get averaged into each output frame.
    #:
    #: This is the smoothing control. 4 is upstream's default and the safe
    #: choice; 1 copies the single nearest frame and keeps the most detail.
    #: Measured on eight utterances against a 72-second reference, every axis
    #: that relates to "over-smoothed" improved monotonically as it came down:
    #:
    #:     topk   resemblance   pitch variation   dynamic range
    #:     4         71.1%          9.26              24.7 dB
    #:     2         71.3%          9.90              25.3 dB
    #:     1         72.4%         10.43              25.8 dB
    #:     you                     11.81              38.3 dB
    #:
    #: The default stays at 4 because a metric cannot hear roughness, and a
    #: small matching set has fewer good neighbours to fall back on. Try
    #: `--vc-topk 1` and trust your ears over the table.
    DEFAULT_TOPK = 4

    def __init__(self, topk: int | None = None, device: str = "cpu",
                 vad_trigger_level: float = 0.0):
        import os as _os
        try:
            env_topk = int(_os.environ.get("REALME_VC_TOPK", "") or 0)
        except ValueError:
            env_topk = 0
        topk = topk or env_topk or self.DEFAULT_TOPK
        self.topk = int(topk)
        self.device = device
        self.vad_trigger_level = float(vad_trigger_level)
        self._knn = None
        self._matching = None
        self._matching_key = None

    # -- loading ------------------------------------------------------------
    def _patch_audio_io(self) -> None:
        """
        torchaudio 2.9+ routes load/save through torchcodec, which is a separate
        install and usually absent. Found by running this, not by reading it:
        the first end-to-end attempt died on `torchaudio.load`. The audio here
        is plain 16-bit PCM, so soundfile reads it directly.
        """
        import torch
        import torchaudio
        try:
            import torchcodec  # noqa: F401
            return
        except ImportError:
            pass
        try:
            import soundfile as sf
        except ImportError as e:
            raise AdapterUnavailable(
                "Voice conversion needs soundfile (or torchcodec) to read WAVs. "
                "Run: realme engine vc --install") from e

        def _load(path, normalize=True, **kw):
            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
            return torch.from_numpy(data.T.copy()), sr

        def _save(path, tensor, sr, **kw):
            sf.write(str(path), tensor.squeeze(0).cpu().numpy(), sr)

        torchaudio.load, torchaudio.save = _load, _save

    def weights_present(self) -> None:
        """The files are on disk. No torch, no model load.

        Everything `_load` checks before it starts importing -- which is the
        part a status page needs and the only part it can act on.
        """
        from realme.engines.vc_install import knnvc_status
        st = knnvc_status()
        if not st["ready"]:
            missing = [n for n, f in st["files"].items() if not f["complete"]]
            raise AdapterUnavailable(
                f"kNN-VC weights are not installed "
                f"({', '.join(missing) or 'incomplete'}). "
                f"Run: realme engine vc --install")

    def _load(self):
        if self._knn is not None:
            return self._knn
        try:
            import torch
        except ImportError as e:
            raise AdapterUnavailable(
                "Voice conversion needs PyTorch. Run: realme engine vc --install"
            ) from e
        from realme.engines.vc_install import knnvc_source, knnvc_status
        st = knnvc_status()
        if not st["ready"]:
            missing = [n for n, f in st["files"].items() if not f["complete"]]
            raise AdapterUnavailable(
                f"kNN-VC weights are not installed ({', '.join(missing) or 'torch'}). "
                f"Run: realme engine vc --install")
        self._patch_audio_io()

        # The vendored source uses top-level imports, kept identical to upstream
        # so a future diff means something. Put its directory on the path rather
        # than rewriting it.
        src = str(knnvc_source())
        if src not in sys.path:
            sys.path.insert(0, src)
        import json
        # Vendored upstream code, kept byte-identical so a future diff against
        # the original means something -- which means its two bits of noise
        # cannot be edited out at the source.
        #
        #   torch: `weight_norm` is deprecated in favour of
        #          `parametrizations.weight_norm`. A warning about a torch API
        #          used by kNN-VC's HiFi-GAN, in code this project does not
        #          own and does not intend to modernise.
        #   hifigan: prints "Removing weight norm..." from inside
        #          `remove_weight_norm()`.
        #
        # Neither is actionable by anyone reading it, and both land in the
        # middle of a render where a real message would be missed among them.
        # Silenced here, narrowly: one warning category, one module's stdout,
        # for the duration of the load only.
        import warnings, contextlib, io
        from wavlm.WavLM import WavLM, WavLMConfig
        from hifigan.models import Generator as HiFiGAN
        from hifigan.utils import AttrDict
        from matcher import KNeighborsVC

        home = Path(st["home"])
        cfg = AttrDict(json.loads(
            (knnvc_source() / "hifigan" / "config_v1_wavlm.json").read_text(encoding="utf-8")))
        with warnings.catch_warnings(), \
                contextlib.redirect_stdout(io.StringIO()):
            warnings.filterwarnings(
                "ignore", category=FutureWarning,
                message=r".*weight_norm.*is deprecated.*")
            gen = HiFiGAN(cfg).to(self.device)
            sd = torch.load(home / "prematch_g.pt", map_location=self.device,
                            weights_only=False)
            gen.load_state_dict(sd["generator"])
            gen.eval()
            gen.remove_weight_norm()

        ck = torch.load(home / "WavLM-Large.pt", map_location="cpu",
                        weights_only=False)
        wavlm = WavLM(WavLMConfig(ck["cfg"]))
        wavlm.load_state_dict(ck["model"])
        wavlm.eval().to(self.device)

        self._knn = KNeighborsVC(wavlm, gen, cfg, self.device)
        return self._knn

    # -- the matching set ---------------------------------------------------
    def _matching_set(self, reference: Path):
        import torch
        reference = _as_wav(Path(reference))
        key = _reference_key(reference)
        if self._matching is not None and self._matching_key == key:
            return self._matching
        from realme.engines.vc_install import vc_home
        cache = vc_home() / "cache" / f"knnvc_{key}.pt"
        knn = self._load()
        if cache.is_file():
            try:
                self._matching = torch.load(cache, map_location=self.device,
                                            weights_only=False)
                self._matching_key = key
                return self._matching
            except Exception:
                cache.unlink(missing_ok=True)   # a bad cache is not a failure
        m = knn.get_matching_set([str(reference)],
                                 vad_trigger_level=self.vad_trigger_level)
        cache.parent.mkdir(parents=True, exist_ok=True)
        try:
            torch.save(m, cache)
        except Exception:
            pass
        self._matching, self._matching_key = m, key
        return m

    # -- the contract -------------------------------------------------------
    def preflight(self, reference: Path) -> None:
        if not (reference and Path(reference).is_file()):
            raise AdapterUnavailable(
                "Voice conversion needs your voice reference. Run: realme voice enroll")
        self._load()
        _as_wav(Path(reference))       # fail here, not once per utterance

    def convert(self, src_wav: Path, out_wav: Path, reference: Path) -> Path:
        import torchaudio
        knn = self._load()
        matching = self._matching_set(Path(reference))
        query = knn.get_features(str(src_wav))
        out = knn.match(query, matching, topk=self.topk)
        Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(out_wav), out[None].cpu(), self.sample_rate)
        return Path(out_wav)


def _librosa_works() -> bool:
    """
    Can librosa do the two things OpenVoice needs of it?

    **Importing librosa is not the test**, which is what the first attempt at
    this got wrong. librosa 0.10 loads its submodules lazily, so `import
    librosa` succeeds on a machine where librosa is unusable, and the failure
    arrives later at `from librosa.filters import mel` -- that is the line that
    reaches numba, llvmlite, and the native DLL Windows Application Control
    blocks. The guard has to touch the same door the code will.
    """
    try:
        import librosa  # noqa: F401
        from librosa.filters import mel  # noqa: F401   the numba path
        return True
    except Exception:
        return False


def _use_librosa_shim(shim_dir: Path) -> bool:
    """
    Put RealMe's minimal librosa in front, if the real one cannot work here.

    Purging `sys.modules` first is the part that is easy to miss: the lazy
    top-level import SUCCEEDS, so a broken `librosa` is already sitting in
    `sys.modules`, and every later `import librosa` -- including the one inside
    OpenVoice -- gets that object back without consulting `sys.path` at all.
    Adding the shim to the path does nothing until the corpse is removed.
    """
    if _librosa_works():
        return False
    for name in [m for m in list(sys.modules)
                 if m == "librosa" or m.startswith("librosa.")]:
        del sys.modules[name]
    shim = str(shim_dir)
    if shim not in sys.path:
        sys.path.insert(0, shim)
    return True


class OpenVoiceConverter(Converter):
    """
    OpenVoice v2's tone colour converter: one global embedding for your voice,
    a flow model that repaints the draft with it.

    Lighter than kNN-VC in every dimension -- 30 MB of weights against 1.3 GB,
    seconds of reference audio instead of minutes, 24 kHz output instead of 16.

    **Not measured.** The container this was written in cannot reach the host
    that serves OpenVoice's weights, so unlike kNN-VC there is no number here
    that came from running it. Bench it before believing anything about it.
    """
    name = "openvoice"
    sample_rate = 24000

    #: Flow temperature. Lower is more deterministic and cleaner; higher moves
    #: further toward the target timbre and brings more warble with it. 0.3 is
    #: upstream's default. `--vc-tau` tunes it.
    DEFAULT_TAU = 0.3

    def __init__(self, device: str = "cpu", tau: float | None = None):
        import os as _os
        try:
            env_tau = float(_os.environ.get("REALME_VC_TAU", "") or 0)
        except ValueError:
            env_tau = 0.0
        self.device = device
        self.tau = float(tau or env_tau or self.DEFAULT_TAU)
        self._converter = None
        self._se = None
        self._se_key = None

    def weights_present(self) -> None:
        from realme.engines.vc_install import openvoice_status
        if not openvoice_status()["ready"]:
            raise AdapterUnavailable(
                "OpenVoice is not installed. Run: realme engine vc --install "
                "--vc-backend openvoice")

    def _load(self):
        if self._converter is not None:
            return self._converter
        from realme.engines.vc_install import openvoice_source, openvoice_status
        st = openvoice_status()
        if not st["ready"]:
            raise AdapterUnavailable(
                "OpenVoice is not installed. Run: realme engine vc --install "
                "--vc-backend openvoice")
        _use_librosa_shim(openvoice_source() / "_nolibrosa")
        src = str(openvoice_source())
        if src not in sys.path:
            sys.path.insert(0, src)
        from openvoice.api import OpenVoiceBaseClass, ToneColorConverter

        class _NoWatermark(ToneColorConverter):
            """
            ToneColorConverter without the watermarker.

            Upstream's own `enable_watermark=False` does not work: its
            `__init__(*args, **kwargs)` forwards every kwarg to the base class,
            which accepts only `config_path` and `device`, so passing it raises
            `TypeError: OpenVoiceBaseClass.__init__() got an unexpected keyword
            argument 'enable_watermark'`. Leaving it out instead makes the
            constructor `import wavmark` and download a second model, to stamp
            an inaudible watermark into a lecture nobody is going to contest.

            So the base initialiser is called directly and the two attributes
            the rest of the class needs are set here. `add_watermark` already
            returns the audio untouched when the model is None, so nothing else
            has to change.
            """

            def __init__(self, config_path, device="cpu"):
                OpenVoiceBaseClass.__init__(self, config_path, device=device)
                self.watermark_model = None
                self.version = getattr(self.hps, "_version_", "v1")

            def load_ckpt(self, ckpt_path):
                """
                `weights_only=False`, explicitly.

                torch 2.6 flipped that default to True, and a checkpoint
                carrying anything but tensors then fails with an unpickling
                error that reads like a corrupt file. This project has already
                been bitten by it once, loading kNN-VC's weights. The file comes
                from the repository we chose; the risk this flag guards against
                is not the risk we have.
                """
                import torch
                sd = torch.load(ckpt_path, map_location=self.device,
                                weights_only=False)
                missing, unexpected = self.model.load_state_dict(
                    sd["model"], strict=False)
                if unexpected:
                    print(f"  openvoice: {len(unexpected)} unexpected keys "
                          f"in the checkpoint (usually harmless)")

        ckpt = Path(st["checkpoint"]).parent
        conv = _NoWatermark(str(ckpt / "config.json"), device=self.device)
        conv.load_ckpt(str(ckpt / "checkpoint.pth"))
        self._converter = conv
        return conv

    def _embedding(self, reference: Path):
        reference = _as_wav(Path(reference))
        key = _reference_key(reference)
        if self._se is not None and self._se_key == key:
            return self._se
        conv = self._load()
        # extract_se directly, rather than se_extractor.get_se: the latter pulls
        # in whisper to segment the clip, which is a large dependency for a step
        # a single clean reference clip does not need.
        self._se = conv.extract_se([str(reference)])
        self._se_key = key
        return self._se

    def preflight(self, reference: Path) -> None:
        if not (reference and Path(reference).is_file()):
            raise AdapterUnavailable(
                "Voice conversion needs your voice reference. Run: realme voice enroll")
        self._load()

    def convert(self, src_wav: Path, out_wav: Path, reference: Path) -> Path:
        conv = self._load()
        src_se = conv.extract_se([str(src_wav)])
        Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
        conv.convert(audio_src_path=str(src_wav), src_se=src_se,
                     tgt_se=self._embedding(Path(reference)),
                     output_path=str(out_wav), tau=self.tau, message=None)
        return Path(out_wav)


BACKENDS = {"knnvc": KNNVCConverter, "openvoice": OpenVoiceConverter}


class VoiceConversionTTS(BaseTTS):
    """
    A base engine for the words, a converter for the voice.

    The base adapter is built through `adapters.factory`, not constructed here.
    That is not fussiness: four call sites once each built adapters their own
    way and each was wrong differently, and this would have been the fifth.
    """
    is_voice_clone = True
    is_placeholder = False

    def __init__(self, base: str = "piper", backend: str = "knnvc",
                 reference_wav: Path | None = None, polish: str | None = None,
                 **kw):
        if backend not in BACKENDS:
            raise AdapterUnavailable(
                f"Unknown converter '{backend}'. Known: {', '.join(BACKENDS)}")
        self.base_name = base
        self.backend_name = backend
        self.reference_wav = Path(reference_wav) if reference_wav else None
        import os as _os
        #: Compression and headroom on the converter's output. Both converters
        #: hand back audio pinned to 0.0 dBFS with dynamics that arrive faster
        #: than a listener wants to ride; see `voice.STEADY_PRESETS`.
        self.polish = (polish or _os.environ.get("REALME_VC_POLISH")
                       or "steady")
        self.converter = BACKENDS[backend](**kw)
        self._base = None
        self.name = f"{base} + {backend} (converted)"

    @property
    def phoneme_syntax(self):
        """Whatever the base engine understands -- conversion is text-blind."""
        try:
            return self.base().phoneme_syntax
        except Exception:
            return None

    def base(self):
        if self._base is None:
            from realme.adapters.factory import build
            self._base = build(self.base_name)
        return self._base

    def voice_fingerprint(self) -> str:
        ref = _reference_key(self.reference_wav) if (
            self.reference_wav and self.reference_wav.is_file()) else "noref"
        # Everything that changes the audio, or the ledger serves a cached take
        # made with different settings.
        # Distinct prefixes: `name[0]` gave topk and tau the same letter, so a
        # topk of 4 and a tau of 4 would have shared a cache entry.
        knobs = []
        for name, prefix in (("topk", "k"), ("tau", "t")):
            v = getattr(self.converter, name, None)
            if v is not None:
                knobs.append(f"{prefix}{v:g}")
        if self.polish and self.polish != "off":
            knobs.append(self.polish)
        return (f"{self.base().voice_fingerprint()}+{self.backend_name}"
                f"{('.' + '.'.join(knobs)) if knobs else ''}:{ref}")

    def check(self) -> None:
        """Could this run -- without loading WavLM or a vocoder.

        `preflight` loads the converter's models (~18 s and a few hundred
        megabytes of torch). A status page asked it on every refresh, which
        is why starting the Studio printed two weight_norm warnings and sat
        there. This asks the questions that actually decide the answer.
        """
        self.base().check()
        if not (self.reference_wav and Path(self.reference_wav).is_file()):
            raise AdapterUnavailable(
                "Voice conversion needs your voice reference. "
                "Run: realme voice enroll")
        ready = getattr(self.converter, "weights_present", None)
        if callable(ready):
            ready()

    def preflight(self) -> None:
        self.base().preflight()
        self.converter.preflight(self.reference_wav)

    def synthesize(self, text, out_wav, voice=None, pace=1.0):
        import tempfile
        self.preflight()
        out_wav = Path(out_wav)
        with tempfile.TemporaryDirectory() as td:
            draft = Path(td) / "draft.wav"
            self.base().synthesize(text, draft, voice=voice, pace=pace)
            self.converter.convert(draft, out_wav, self.reference_wav)
        if self.polish and self.polish != "off":
            from realme.enrollment.voice import steady_output
            steady_output(out_wav, out_wav, self.polish)
        return out_wav
