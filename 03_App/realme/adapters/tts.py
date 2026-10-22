"""
TTS adapters. Each is small on purpose: the pipeline depends only on BaseTTS,
so swapping Google -> ElevenLabs -> self-hosted is a one-line config change.
"""
from __future__ import annotations
import os, shutil, subprocess
from pathlib import Path
from realme.adapters.base import BaseTTS, AdapterUnavailable
from realme.core.media import run, require, probe


class EspeakTTS(BaseTTS):
    phoneme_syntax = "espeak"
    """
    Local, offline, zero-dependency speech. Intelligible but obviously robotic.
    Purpose: prove the pipeline end-to-end on any machine with no API key and
    no GPU. It is a PLACEHOLDER voice and is stamped as such.
    """
    name = "espeak-ng (placeholder voice)"
    is_voice_clone = False
    is_placeholder = True

    def preflight(self) -> None:
        if not shutil.which("espeak-ng"):
            raise AdapterUnavailable("espeak-ng not installed (`apt-get install espeak-ng`).")

    def synthesize(self, text, out_wav, voice=None, pace=1.0):
        out_wav = Path(out_wav); out_wav.parent.mkdir(parents=True, exist_ok=True)
        wpm = int(155 * pace)
        run([require("espeak-ng"), "-v", voice or "en-us+m3", "-s", str(wpm),
             "-p", "42", "-g", "6", "-w", str(out_wav), text],
            f"espeak-ng synth ({len(text)} chars)")
        probe(out_wav)  # assert we produced something decodable
        return out_wav


class GoogleChirp3CloneTTS(BaseTTS):
    phoneme_syntax = None   # uses uploaded custom_pronunciations, not inline
    """
    Google Cloud TTS - Chirp 3 'Instant Custom Voice'.
    Clones from ~10s of reference audio plus a mandatory spoken consent clip.
    Access is allowlist-gated: you must be approved by Google before this works.
    """
    name = "google-chirp3-instant-custom-voice"
    is_voice_clone = True

    def __init__(self, voice_clone_key: str | None = None, language: str = "en-US"):
        self.voice_clone_key = voice_clone_key or os.environ.get("REALME_GOOGLE_VOICE_KEY")
        self.language = language

    def voice_fingerprint(self) -> str:
        return f"{self.name}:{self.language}:{(self.voice_clone_key or '')[:16]}"

    def preflight(self) -> None:
        try:
            from google.cloud import texttospeech  # noqa: F401
        except ImportError as e:
            raise AdapterUnavailable(
                "pip install google-cloud-texttospeech, and set up ADC "
                "(`gcloud auth application-default login`)."
            ) from e
        if not self.voice_clone_key:
            raise AdapterUnavailable(
                "No voice clone key. Create one via Chirp 3 Instant Custom Voice "
                "(allowlist required) and set REALME_GOOGLE_VOICE_KEY."
            )

    def synthesize(self, text, out_wav, voice=None, pace=1.0):
        from google.cloud import texttospeech as tts
        client = tts.TextToSpeechClient()
        resp = client.synthesize_speech(
            input=tts.SynthesisInput(text=text),
            voice=tts.VoiceSelectionParams(
                language_code=self.language,
                voice_clone=tts.VoiceCloneParams(voice_cloning_key=self.voice_clone_key),
            ),
            audio_config=tts.AudioConfig(
                audio_encoding=tts.AudioEncoding.LINEAR16,
                sample_rate_hertz=24000,
                speaking_rate=pace,
            ),
        )
        out_wav = Path(out_wav); out_wav.parent.mkdir(parents=True, exist_ok=True)
        out_wav.write_bytes(resp.audio_content)
        probe(out_wav)
        return out_wav


class ElevenLabsTTS(BaseTTS):
    """Hosted cloning with a commercial licence on paid tiers. No allowlist."""
    name = "elevenlabs"
    is_voice_clone = True

    def __init__(self, voice_id: str | None = None, model: str = "eleven_multilingual_v2"):
        self.voice_id = voice_id or os.environ.get("REALME_ELEVEN_VOICE_ID")
        self.api_key = os.environ.get("ELEVENLABS_API_KEY")
        self.model = model

    def voice_fingerprint(self) -> str:
        return f"{self.name}:{self.model}:{self.voice_id}"

    def preflight(self) -> None:
        if not self.api_key:
            raise AdapterUnavailable("Set ELEVENLABS_API_KEY.")
        if not self.voice_id:
            raise AdapterUnavailable("Set REALME_ELEVEN_VOICE_ID to your cloned voice.")

    def synthesize(self, text, out_wav, voice=None, pace=1.0):
        import json, urllib.request
        vid = voice or self.voice_id
        req = urllib.request.Request(
            f"https://api.elevenlabs.io/v1/text-to-speech/{vid}?output_format=pcm_24000",
            data=json.dumps({"text": text, "model_id": self.model}).encode(),
            headers={"xi-api-key": self.api_key, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=180) as r:
            pcm = r.read()
        from realme.core.media import write_wav
        return write_wav(Path(out_wav), pcm, 24000)


class ChatterboxTTS(BaseTTS):
    """
    Self-hosted MIT-licensed cloning (Resemble AI Chatterbox). Runs on a Colab
    T4 or a local GPU; the Nano variant runs on CPU. Use this when you want the
    voice model to never leave your own machine.
    """
    name = "chatterbox-mit"
    is_voice_clone = True

    def __init__(self, reference_wav: Path | None = None, device: str = "cuda"):
        self.reference_wav = Path(reference_wav) if reference_wav else None
        self.device = device
        self._model = None

    def voice_fingerprint(self) -> str:
        ref = Path(self.reference_wav).name if self.reference_wav else "none"
        return f"{self.name}:{ref}"

    def preflight(self) -> None:
        try:
            import chatterbox  # noqa: F401
        except ImportError as e:
            raise AdapterUnavailable("pip install chatterbox-tts") from e
        if not self.reference_wav or not self.reference_wav.exists():
            raise AdapterUnavailable("Provide a 5-10s clean reference WAV of the instructor.")

    def synthesize(self, text, out_wav, voice=None, pace=1.0):
        import torchaudio
        from chatterbox.tts import ChatterboxTTS as _CB
        if self._model is None:
            self._model = _CB.from_pretrained(device=self.device)
        wav = self._model.generate(text, audio_prompt_path=str(self.reference_wav))
        out_wav = Path(out_wav); out_wav.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(out_wav), wav, self._model.sr)
        probe(out_wav)
        return out_wav


#: The draft voices kept installed, so the Studio can offer a choice. Piper is
#: the *draft* voice -- the one you iterate script, pacing and slide timing
#: against before paying for a real render -- and which of these is more
#: comfortable to work against is a matter of taste, not of correctness.
#: The draft voice's own pace. It is a draft: the words, the ordering and the
#: slide timing are what it is for, and it is replaced by the cloned voice
#: before anything is published. `realme voice pace` measures a ratio against
#: your own speaking rate; that ratio is applied to DIALOGUE, where the two
#: voices alternate and a mismatch is audible, and not to lecture drafting.
NATURAL_SPEED = 1.0

#: The stock draft voices the Studio offers, and the ones `realme engine
#: install` fetches. Every entry costs ~60 MB on a fresh install, so this is a
#: chosen shortlist rather than the whole piper catalogue -- `explore_expressive
#: --list-voices` shows what else can be auditioned.
#:
#: Chosen by listening, on the same passage, not by tier or by label.
#:
#: Lessac was the original female voice and is gone: it was heard as mechanical
#: and as English-English, which it is not -- a US voice whose stress is so even
#: that listeners place it in the wrong country. Amy replaced it on a
#: side-by-side.
#:
#: Cori is here for range rather than as a second-best Amy. Two US voices give a
#: podcast one accent and two timbres; an American and an English speaker give
#: it two of each, which is most of what makes an exchange sound like two
#: people. Cori is the high tier because that is the only tier it is published
#: at -- and the tier is model size, not a ranking. Amy at medium beat Lessac at
#: medium, which was a difference of speaker, not of tier.
DRAFT_VOICES = {
    "en_US-amy-medium": "Amy — female, US",
    "en_US-ryan-medium": "Ryan — male, US",
    "en_GB-cori-high": "Cori — female, UK",
}


def measured_speed(voice: str | None = None) -> float | None:
    """
    The ratio `realme voice pace` measured between your rate and the draft
    voice's, or None if it has never been run.

    Deliberately NOT the default everywhere. Matching the draft voice to the
    instructor's pace matters where the two voices alternate and a mismatch is
    audible as one speaker rushing the other -- that is a dialogue. A lecture
    draft is scaffolding: it is replaced by the cloned voice before anything is
    published, and slowing it only makes the iteration loop longer.

    So dialogue casting asks for this; narration and lecture drafting use the
    voice's own pace.

    PER VOICE, because the ratio is not a property of you alone. It is your
    rate divided by ONE PARTICULAR VOICE's rate at speed 1.0, and piper voices
    differ: applying a ratio measured against Lessac to Amy slows Amy by
    whatever Amy is already slower, on top of the intended match. The symptom
    is a guest who drags, and nothing says why.

    The calibration has always recorded `your_wpm` and the voice's `voice_wpm`
    -- it just threw them away and kept the quotient. They are kept now, one
    entry per voice, and the speed for a given voice is worked out from them.
    An older profile that has only the scalar still works: that number is
    returned for the voice it was measured against, and for a voice that has
    never been measured this returns None rather than a ratio belonging to a
    different speaker.
    """
    try:
        from realme.core.env import data_home
        from realme.app.profile import Profile
        data = Profile(data_home() / "profile").data
        got = speed_from(data, voice)
        if got is not None:
            return got
        saved = data.get("draft_speed")
        if not saved:
            return None
        measured_for = (data.get("draft_pace_measured") or {}).get("voice")
        if voice and measured_for and voice != measured_for:
            # Measured against a different voice. Returning it would be a
            # guess wearing a measurement's clothes.
            return None
        return float(saved)
    except Exception:
        return None


def speed_from(data: dict, voice: str | None) -> float | None:
    """The matched speed for `voice` from an ALREADY-LOADED profile, or None.

    Split out from `measured_speed` because that one reads the profile off
    disk, and a caller holding a profile object would otherwise be answered
    from a different copy of it. That is not hypothetical: `guest_speed` wrote
    a fresh measurement into the profile it was handed and then asked
    `measured_speed`, which read the file, missed the write, and measured the
    same voice again on the next call.
    """
    mine, table = rates(data or {})
    if not (voice and mine and table.get(voice)):
        return None
    from realme.enrollment.pace import MIN_SPEED, MAX_SPEED
    raw = float(mine) / float(table[voice])
    return round(max(MIN_SPEED, min(MAX_SPEED, raw)), 3)


def rates(data: dict) -> tuple:
    """(your words/min, {voice: its words/min}), from wherever they were saved.

    Two commands write these and they wrote them in two places. `voice pace
    --all` writes `your_wpm` and a `draft_voice_wpm` table; plain `voice pace`
    writes one nested `draft_pace_measured` record and nothing else. The
    per-voice lookup above read only the first pair, so on a profile
    calibrated the ordinary way it never fired -- the numbers were sitting in
    the file, in the other shape, and the guest fell through to a constant.

    Both shapes are read here, so one function decides what "has this been
    measured?" means and the answer does not depend on which command ran.
    """
    one = data.get("draft_pace_measured") or {}
    mine = data.get("your_wpm") or one.get("your_wpm")
    table = dict(data.get("draft_voice_wpm") or {})
    if one.get("voice") and one.get("voice_wpm"):
        table.setdefault(one["voice"], one["voice_wpm"])
    return mine, table


def draft_speed() -> float:
    """The draft voice's own pace. 1.0 is the voice as trained."""
    import os
    try:
        env = float(os.environ.get("REALME_PIPER_SPEED", "") or 0)
    except ValueError:
        env = 0.0
    return env or NATURAL_SPEED


def draft_voice(explicit: str | None = None) -> str:
    """
    Which piper voice to speak with. One order of precedence, everywhere.

    Explicit argument, then `REALME_PIPER_VOICE` (what `--piper-voice` sets),
    then the Studio's saved choice, then the default. The profile is consulted
    last of the three so a command-line override still wins for one run without
    changing what the Studio does tomorrow.
    """
    if explicit:
        return explicit
    env = os.environ.get("REALME_PIPER_VOICE")
    if env:
        return env
    try:
        from realme.core.env import data_home
        from realme.app.profile import Profile
        saved = Profile(data_home() / "profile").data.get("draft_voice")
        if saved:
            return str(saved)
    except Exception:
        pass
    return PiperTTS.DEFAULT_VOICE


class PiperTTS(BaseTTS):
    """
    Piper: the right draft/preview voice, and strictly better than espeak-ng.

    Sounds far more natural, runs faster than real time on a CPU, and keeps the
    same espeak-ng phoneme back end -- so `[[ ]]` injection and any espeak
    dictionary work carry over unchanged. Voices are MIT; the Piper code is
    GPL-3, which binds you only if you redistribute a modified Piper. Rendering
    audio with it puts no obligation on the audio.

    It does not clone. Use it while writing, then switch engines to record.
    """
    name = "piper (draft voice)"
    is_voice_clone = False
    is_placeholder = True

    #: **No phoneme injection.** This said "espeak", on the reasoning that piper
    #: uses espeak-ng as its phonemiser so `[[ ]]` blocks would carry over. They
    #: do not, and the failure is silent.
    #:
    #: espeak's `[[ ]]` takes espeak's own mnemonics -- `[[w,aI'oU'V]]` for
    #: WIOA. Piper does not interpret them: it phonemises text to IPA and looks
    #: each symbol up in the voice's map. That map holds ʌ, ɛ, ɪ, ʊ and has no
    #: V, E, I, U and no brackets, so every mnemonic character is dropped with a
    #: "Missing phoneme from id map" line on stderr that nothing surfaces.
    #: Six of them appeared while rendering one three-minute script.
    #:
    #: With no phoneme syntax the pipeline hands piper the RESPELLING instead --
    #: "why-oh-uh", "I P T W" -- which phonemises normally and lands in the map.
    phoneme_syntax = None

    #: Overridden by REALME_PIPER_VOICE, which is how a voice fine-tuned on your
    #: own corpus becomes the default everywhere -- render, bench, Studio, and
    #: inside `piper+knnvc` -- from one setting instead of four call sites.
    #:
    #: A male US voice, not lessac, and `medium` rather than `high`.
    #:
    #: Conversion transfers TIMBRE and keeps the source's pace, rhythm and
    #: accent, so whatever is wrong with the base voice survives into the
    #: converted output. Measured on the same eight utterances, converted
    #: against the same reference:
    #:
    #:     base            draft ->  converted pitch    centroid    RTF
    #:     lessac (f)                168.4 Hz  (89%)    1558 (95%)  0.42
    #:     ryan-high (m)             156.9 Hz  (98%)    1421 (94%)  0.72
    #:     ryan-medium (m)           155.3 Hz (100%)    1474 (99%)  0.44
    #:                               you: 155.3 Hz                  1484
    #:
    #: `high` costs 4x the synthesis time of `medium` and buys nothing, because
    #: kNN-VC's vocoder resamples to 16 kHz and throws the extra away.
    DEFAULT_VOICE = "en_US-ryan-medium"

    def __init__(self, model: str | None = None, speed: float | None = None):
        self.model = draft_voice(model)
        try:
            env_speed = float(os.environ.get("REALME_PIPER_SPEED", "") or 0)
        except ValueError:
            env_speed = 0.0
        #: 1.0 is the voice's own pace; 0.9 is a tenth slower. Piper thinks in
        #: length_scale, which is the reciprocal and reads backwards to everyone.
        #:
        #: The default is 0.8 because piper's own pace is faster than a lecture
        #: wants. That came from a listener twice -- 1.0 was "very fast", 0.9
        #: was still "hard to keep up" -- and the measurement agreed on the way
        #: down: slowing ryan-high from 1.0 to 0.9 moved converted resemblance
        #: from 65.4% to 70.1%. Conversion inherits pace from the base voice, so
        #: this is not a preview-only nicety. `--piper-speed` tunes it.
        self.speed = float(speed) if speed else (env_speed or draft_speed())
        self._voice = None

    def voice_fingerprint(self) -> str:
        # Speed changes the audio, so it belongs in the fingerprint: without it
        # the render ledger would serve yesterday's faster take from cache.
        return f"{self.name}:{self.model}:{self.speed:g}"

    def _voice_file(self) -> str:
        """
        Where the .onnx actually is.

        `realme engine install` puts voices in tools/piper so they sit with the
        rest of the installed payload rather than wherever the terminal happened
        to be. A file in the working directory still wins if one is there, so a
        hand-downloaded voice keeps working.
        """
        from realme.engines.install import engine_home, piper_home
        local = Path(f"{self.model}.onnx")
        if local.exists():
            return str(local)
        bundled = piper_home(engine_home()) / f"{self.model}.onnx"
        # The BUNDLED path when neither exists, not the working-directory one.
        # `realme engine install` writes to tools/piper, so naming a
        # cwd-relative file in the "not here" message sends the reader to look
        # in the wrong place -- or, worse, makes them think the voice is
        # expected somewhere that depends on where they were standing.
        return str(bundled)

    def preflight(self) -> None:
        """
        Three separate things can be missing, and they used to report as two
        vague sentences: "the draft voice is not installed" and "the voice is
        not downloaded". Neither distinguishes

          * the piper package absent from THIS interpreter -- the common case,
            because piper installed into a different environment is invisible
            here and the machine looks, correctly, like it has piper;
          * the .onnx model file missing;
          * the .onnx.json config beside it missing -- piper needs BOTH, and a
            voice copied by hand usually arrives without the json.

        Each now names itself, with the path that was actually looked at.
        """
        import sys as _sys
        try:
            import piper  # noqa: F401
        except ImportError as e:
            raise AdapterUnavailable(
                f"The piper package is not importable by this Python.\n"
                f"    interpreter: {_sys.executable}\n"
                f"    fix        : realme engine install\n"
                f"    or         : \"{_sys.executable}\" -m pip install piper-tts\n"
                f"    If piper is installed elsewhere, it is in a different "
                f"environment than RealMe is running in.") from e

        onnx = Path(self._voice_file())
        cfg = Path(f"{onnx}.json")
        if not onnx.is_file():
            raise AdapterUnavailable(
                f"The {self.model} voice file is not here.\n"
                f"    looked for : {onnx}\n"
                f"    also tried : {Path.cwd() / (self.model + '.onnx')}\n"
                f"    fix        : realme engine install")
        if not cfg.is_file():
            raise AdapterUnavailable(
                f"The {self.model} voice is missing its config.\n"
                f"    model      : {onnx}\n"
                f"    looked for : {cfg}\n"
                f"    A piper voice is TWO files: the .onnx and the .onnx.json "
                f"beside it.\n"
                f"    fix        : realme engine install")

        from piper import PiperVoice
        try:
            if self._voice is None:
                self._voice = PiperVoice.load(str(onnx))
        except Exception as e:
            raise AdapterUnavailable(
                f"Both files are present but piper could not load the voice "
                f"({e.__class__.__name__}: {str(e)[:120]}).\n"
                f"    model: {onnx}\n"
                f"    A truncated download is the usual cause; delete the two "
                f"files and run: realme engine install") from e

    def synthesize(self, text, out_wav, voice=None, pace=1.0):
        import wave
        self.preflight()
        out_wav = Path(out_wav); out_wav.parent.mkdir(parents=True, exist_ok=True)

        # Write the WAV header ourselves rather than letting piper do it.
        #
        # `synthesize_wav` sets the format when it emits its FIRST audio chunk.
        # Text it renders as nothing -- a fragment that is all punctuation, a
        # stray symbol, a line the phonemiser empties -- yields no chunks, the
        # header is never written, and closing the file raises
        # "wave.Error: # channels not specified". The error names the file
        # format, so it reads like a bug in the writer rather than what it is:
        # one unpronounceable line in the middle of a script.
        rate = int(getattr(getattr(self._voice, "config", None),
                           "sample_rate", 22050))
        with wave.open(str(out_wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            # `pace` was accepted and silently ignored here until a listener
            # said the draft voice was too fast and there was nothing to turn.
            # length_scale is piper's control and runs the other way: bigger is
            # slower, so it is the reciprocal of speed.
            scale = 1.0 / (self.speed * (pace or 1.0))
            syn = None
            if abs(scale - 1.0) > 1e-3:
                try:
                    from piper import SynthesisConfig
                    syn = SynthesisConfig(length_scale=scale)
                except Exception:
                    syn = None          # older piper: no config, keep going
            try:
                self._voice.synthesize_wav(text, w, syn_config=syn,
                                           set_wav_format=False)
            except TypeError:
                # Older piper builds have neither argument; they will set the
                # format again, harmlessly, over the values above.
                self._voice.synthesize_wav(text, w)

        info = probe(out_wav)
        if float(info.get("format", {}).get("duration", 0) or 0) <= 0.0:
            raise AdapterUnavailable(
                f"The draft voice produced no audio for: {text[:80]!r}\n"
                "  Usually a line with nothing pronounceable in it.")
        return out_wav


class IndexTTSAdapter(BaseTTS):
    """
    IndexTTS 2.5 -- the best fit for this project among self-hosted cloners.

    ~6 GB VRAM (fits a free-Colab T4 with room to spare), and the clearest
    pronunciation syntax of any engine: `<word|AY1 P IY1 T IY1>` accepting CMU
    phonemes directly. Licence permits commercial use below 100M MAU, which a
    university is not near. Its documented weakness is honest: long text is
    split and concatenated with no prosody modelling across the boundary, which
    is exactly why this pipeline chunks at the utterance level anyway.
    """
    name = "indextts-2.5"
    is_voice_clone = True
    phoneme_syntax = "indextts"

    def __init__(self, reference_wav: Path | None = None, device: str = "cuda"):
        self.reference_wav = Path(reference_wav) if reference_wav else None
        self.device = device
        self._model = None

    def voice_fingerprint(self) -> str:
        ref = Path(self.reference_wav).name if self.reference_wav else "none"
        return f"{self.name}:{ref}"

    def preflight(self) -> None:
        try:
            import indextts  # noqa: F401
        except ImportError as e:
            raise AdapterUnavailable(
                "pip install indextts (see github.com/index-tts/index-tts)") from e
        if not self.reference_wav or not self.reference_wav.exists():
            raise AdapterUnavailable("Provide a clean reference WAV of the instructor.")

    def synthesize(self, text, out_wav, voice=None, pace=1.0):
        from indextts.infer_v2 import IndexTTS2
        if self._model is None:
            self._model = IndexTTS2(device=self.device)
        out_wav = Path(out_wav); out_wav.parent.mkdir(parents=True, exist_ok=True)
        self._model.infer(spk_audio_prompt=str(self.reference_wav), text=text,
                          output_path=str(out_wav))
        probe(out_wav)
        return out_wav


def _qwen3():
    from realme.adapters.tts_qwen3 import Qwen3LocalTTS
    return Qwen3LocalTTS


class _LazyQwen3:
    """Deferred import so a missing Qwen3 install cannot break the registry."""
    is_voice_clone = True
    is_placeholder = False
    phoneme_syntax = None

    def __new__(cls, *a, **kw):
        return _qwen3()(*a, **kw)


class _LazyQwen3CppDll:
    """
    The C++ engine in-process, via its C API. The default for `qwen3cpp`.

    Preferred over the CLI for two independent reasons: it loads the model once
    instead of per utterance, and it does not start a new process -- which is
    what Windows Application Control blocks.
    """
    is_voice_clone = True
    is_placeholder = False
    phoneme_syntax = None

    def __new__(cls, *a, **kw):
        from realme.adapters.tts_qwen3_dll import Qwen3DllTTS
        return Qwen3DllTTS(*a, **kw)


class _LazyQwen3Cpp:
    """Same deferral, for the C++ engine: an unbuilt binary must not stop the
    registry from importing, or `realme doctor` could never report on it."""
    is_voice_clone = True
    is_placeholder = False
    phoneme_syntax = None

    def __new__(cls, *a, **kw):
        from realme.adapters.tts_qwen3_cpp import Qwen3CppTTS
        return Qwen3CppTTS(*a, **kw)


class _LazyVC:
    """
    Base engine plus converter, deferred.

    Importing this eagerly would drag PyTorch into every `realme --help`, and
    the registry has to stay listable on a machine where nothing is installed.
    """
    is_voice_clone = True
    is_placeholder = False
    phoneme_syntax = None          # piper's front end; see PiperTTS

    base = "piper"
    backend = "knnvc"

    def __new__(cls, *a, **kw):
        from realme.adapters.vc import VoiceConversionTTS
        kw.setdefault("base", cls.base)
        kw.setdefault("backend", cls.backend)
        return VoiceConversionTTS(*a, **kw)


class _LazyPiperKnnVC(_LazyVC):
    """piper says the words, kNN-VC repaints them. The measured fast path."""
    base, backend = "piper", "knnvc"


class _LazyPiperOpenVoice(_LazyVC):
    """Same shape, OpenVoice's tone-colour converter. Not measured here."""
    base, backend = "piper", "openvoice"


class _LazyGemini:
    """
    Gemini TTS with a replicated voice, deferred like the rest.

    Nothing to import that could fail -- the adapter is urllib and base64 --
    but it is written the same way so the registry stays uniform and so the
    profile is not read at import time.
    """
    is_voice_clone = True
    is_placeholder = False
    phoneme_syntax = None

    def __new__(cls, *a, **kw):
        from realme.adapters.tts_gemini import GeminiTTS
        return GeminiTTS(*a, **kw)


#: The engines this project builds, tests and ships an installer for.
#:
#: Everything here is either local (piper, the Qwen3 C++ engine, the voice
#: converters) or a system binary, and `realme engine install` can produce
#: all of it. The Studio probes these on startup because the answer is
#: actionable: a missing one has a button or a command that fixes it.
#: `gemini-tts` is here, and it is the first entry that is not local. It
#: qualifies on the stated criterion -- a failed check is actionable, and says
#: which of the three things is missing (key, enrolled voice, paid-tier
#: acknowledgement) -- but it is the one engine whose cost is not only time.
#: Order matters: this is the order the Studio lists them in, and the first
#: is what a new profile gets. gemini-tts leads as of October 2026 -- it is
#: the better-sounding clone and it needs no GPU -- with qwen3cpp immediately
#: behind it as the local one that costs nothing and works offline.
DEVELOPED = ("gemini-tts", "qwen3cpp", "qwen3cpp-cli", "qwen3", "piper")

#: Engines RealMe can drive but does not set up, with what each one needs.
#:
#: They stay in the REGISTRY -- `--tts chatterbox` works, and somebody who
#: has one of these installed should be able to use it. What stopped is
#: PROBING them on every status check: importing a half-installed torch stack
#: or asking about a key nobody set produced a row of red pills for engines
#: this project never promised, on every page load.
NOT_DEVELOPED = {
    "elevenlabs": "a paid API; needs ELEVENLABS_API_KEY and a cloned voice id",
    "google_chirp3": "Chirp 3 Instant Custom Voice; allowlisted Google Cloud "
                     "project and ADC",
    "chatterbox": "pip install chatterbox-tts, plus torch",
    "indextts": "pip install indextts, plus torch",
}

#: Retired September 2026, and why.
#:
#: `piper+knnvc` and `piper+openvoice` converted a piper draft into the
#: instructor's timbre; they were measured (65-80% resemblance) and slower
#: than the C++ engine that now does the job directly, and each dragged a
#: torch stack into every startup. `espeak` was the robotic placeholder from
#: before any voice existed. None of them is offered, checked or installed
#: any more. The classes and the vendored converter source are still in the
#: tree, unreferenced, if this is ever reversed.
#:
#: espeak-ng the BINARY is a different thing and stays: the lexicon asks it
#: for phoneme mnemonics to decide whether a term reads as a word or as
#: letters (see realme/text/lexicon.py).
RETIRED = {
    "piper+knnvc": "voice conversion; the C++ engine clones directly now",
    "piper+openvoice": "voice conversion; never measured as well as kNN-VC",
    "espeak": "the robotic placeholder from before voice cloning worked",
}

REGISTRY = {
    "qwen3": _LazyQwen3,
    "qwen3cpp": _LazyQwen3CppDll,
    "qwen3cpp-cli": _LazyQwen3Cpp,
    "piper": PiperTTS,
    "indextts": IndexTTSAdapter,
    "google_chirp3": GoogleChirp3CloneTTS,
    "elevenlabs": ElevenLabsTTS,
    "chatterbox": ChatterboxTTS,
    "gemini-tts": _LazyGemini,
}
