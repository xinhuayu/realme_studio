"""
The instructor profile -- the 'bake once' half of the architecture.

Holds the things that should be captured once and reused for a whole term:
the voice reference, the recorded consent, the teaching-voice notes, the
course context, and which adapters are bound. Everything biometric stays in
this directory; nothing here is uploaded anywhere except by the adapter you
explicitly select.
"""
from __future__ import annotations
import copy, json, shutil
from datetime import datetime, timezone
from pathlib import Path
from realme.core.textio import read_text

DEFAULTS = {
    "profile_id": "instructor",
    "display_name": "",
    "course_context": "",
    "style_notes": "",
    # qwen3cpp for recording, piper for drafting. Measured, not assumed: on a
    # 14-utterance passage the C++ engine was twice as fast as the Python worker
    # per second of audio AND closer to the enrolled voice (84.5% vs 65.5%),
    # while the worker produced unstable pitch and over-long output.
    # No `gemini_model`. The model is not a property of your voice profile,
    # and storing it only created a second place for it to go stale: the
    # Studio's menu and the CLI both read the code default (or
    # REALME_GEMINI_MODEL, or --model), never this file, so a saved id sat
    # here pointing at a retired preview, influencing nothing and confusing
    # anyone who read it.
    # The default for a profile that does not exist yet. An existing profile
    # is NEVER migrated onto this: moving someone from a local engine to a
    # hosted one would start sending their voice and their lectures to a
    # company, and start charging them, on the strength of a software update.
    # That is a choice, and it is made in Twin Setup.
    "adapters": {"script_writer": "gemini",
                 "tts": "gemini-tts",
                 # The guest is a DRAFT voice, and a different one from the
                 # instructor's. It said "qwen3cpp", which made a debate two
                 # copies of the same cloned person.
                 "guest_tts": "piper:en_US-amy-medium",
                 "draft_tts": "piper"},
    "video": {"width": 1920, "height": 1080, "fps": 30, "layout": "slide_only"},
    "signalling": "highlight",
    # How a slide is split for synthesis. "stable" renders each slide as ONE
    # call: about 7% cheaper per second of audio, because the reference context
    # is prefilled per call regardless of how little you then ask it to say.
    # The trade is granularity -- fixing one word re-renders the whole slide.
    # Only safe because a slide is 1-3 minutes and Qwen3-TTS-12Hz caps a single
    # call at 4096 frames, which is 5.7 minutes.
    "prosody_mode": "natural",
    #: Scales the silence inserted between delivery units, 0.85-1.25.
    #: 1.0 is the measured default; lower tightens a lecture that drags.
    #: `prosody_mode: "stable"` removes inserted gaps entirely.
    "pause_scale": 1.0,
    "disclosure": "This lecture was narrated with a synthetic voice model of the "
                  "instructor, from a script the instructor wrote and approved.",
    "consent_recorded_at": None,
    "voice_reference": None,
    # Qwen3 conditions the clone on the reference clip AND its exact
    # transcript. Without the text the clone is measurably worse, so it
    # is part of the profile rather than an environment variable.
    "voice_reference_text": "",
    #: register name -> treated .wav. Several takes of the same voice,
    #: levelled together so the difference between them is delivery.
    "voice_registers": {},
    "consent_recording": None,
    #: Words per minute of SPEECH in the enrolment recording, measured when it
    #: was enrolled. Kept so that a later re-recording can be compared against
    #: it: a clone copies the delivery it was cloned from, and a reference read
    #: more carefully than usual is otherwise an invisible change that shows up
    #: months later as "the voice sounds slow".
    "reference_wpm": None,
    #: Gemini TTS, which is a cloud engine and therefore carries more state
    #: than a local one: WHICH voice was enrolled, in whose project, when it
    #: expires, and whether the person was told what leaves the machine.
    #:
    #: `consent_recording` above is RealMe's own consent artifact, in the
    #: project's wording. Google requires its own exact sentence and rejects a
    #: paraphrase, so the clip it was enrolled with is recorded separately
    #: rather than overwriting the one your institution would want to see.
    "gemini_voice": {
        "voice": "",               # voice_... or voicekey_...
        "model": "gemini-3.8-flash-tts",
        "stored": True,            # False = a 7-day voicekey_
        "created_at": None,
        "consent_recording": None,
        "paid_tier_ack": False,
        #: Measured pace correction for the cloned voice, from
        #: `realme voice gemini --pace`. A clone speaks at the pace of the
        #: clip it was cloned from, which is the pace of a recording session
        #: and not necessarily the pace of a lecture. None = no correction.
        "rate_match": None,
        #: Characters per API call. None = the engine's own measured default
        #: (738, see adapters/tts_gemini.MEASURED_MAX_CHARS); 0 = fall back to
        #: the pipeline's 260. Any other number should be one that
        #: probe_gemini_tts.py measured on this account -- the documented API
        #: limit is a different and much larger number, and putting that here
        #: is how a lecture gets rendered in calls that hurry.
        "max_chars": None,
    },
}


class Profile:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "baked_assets").mkdir(exist_ok=True)
        self.path = self.root / "profile.json"
        self.data = (json.loads(read_text(self.path)) if self.path.exists()
                     else copy.deepcopy(DEFAULTS))
        # Backfill NESTED defaults too. `setdefault` only ever filled missing
        # top-level keys, so a profile saved before a sub-key existed never
        # gained it: a profile written when `adapters` held two entries kept
        # two entries forever, and every later addition was invisible to it.
        for k, v in DEFAULTS.items():
            if isinstance(v, dict) and isinstance(self.data.get(k), dict):
                for sk, sv in v.items():
                    self.data[k].setdefault(sk, sv)
            else:
                self.data.setdefault(k, v)
        if self._migrate():
            self.save()

    #: Engines that were retired, and what replaced them. A saved profile keeps
    #: pointing at a name long after the project stopped recommending it --
    #: `qwen3` (the Python worker) measured 65.5% resemblance against
    #: qwen3cpp's 80.5% and is several times slower, yet a profile written
    #: before the C++ engine existed still selected it, silently, in every
    #: dropdown.
    RETIRED_ENGINES = {"qwen3": "qwen3cpp", "qwen3cpp-cli": "qwen3cpp",
                       # Retired September 2026. A profile still naming one
                       # would select an engine the registry no longer has.
                       "piper+knnvc": "qwen3cpp", "piper+openvoice": "qwen3cpp",
                       "espeak": "piper"}

    def _migrate(self) -> bool:
        """Bring a profile written by an older version up to date. Returns
        True if anything changed, so the caller knows to write it back."""
        changed = False
        ad = self.data.get("adapters")
        if isinstance(ad, dict):
            for slot in ("tts", "guest_tts", "draft_tts"):
                cur = ad.get(slot)
                new = self.RETIRED_ENGINES.get(cur)
                if new:
                    ad[slot] = new
                    changed = True
                    import sys as _sys
                    print(f"  [profile] {slot}: '{cur}' is retired -> '{new}'",
                          file=_sys.stderr, flush=True)
            # A guest that clones YOUR voice is not a guest. This was the
            # stored default before the podcast had a second speaker worth the
            # name, and a saved profile still carries it into every dialogue.
            if ad.get("guest_tts") in ("qwen3cpp", "qwen3", "chatterbox",
                                       "indextts"):
                ad["guest_tts"] = "piper:en_US-amy-medium"
                changed = True
                import sys as _sys
                print(f"  [profile] guest_tts was a clone of your own voice "
                      f"-> {ad['guest_tts']}", file=_sys.stderr, flush=True)
            # A voice that is no longer offered. Removing one from the list
            # does not remove it from a profile that chose it, and the result
            # is a Studio dropdown with nothing selected and a greeting that
            # calls the guest by its file name. Moved to the default, out loud
            # -- the voice may still be on disk, so this is a change of
            # preference and the person should know it happened.
            guest = ad.get("guest_tts") or ""
            if guest.startswith("piper:"):
                from realme.adapters.tts import DRAFT_VOICES
                model = guest.split(":", 1)[1]
                if model and model not in DRAFT_VOICES:
                    ad["guest_tts"] = DEFAULTS["adapters"]["guest_tts"]
                    changed = True
                    import sys as _sys
                    print(f"  [profile] the guest voice '{model}' is no longer "
                          f"one of the offered voices -> "
                          f"{ad['guest_tts']}", file=_sys.stderr, flush=True)
            # Drop a stored script model rather than migrating it forward.
            # Nothing reads it -- the menu and the CLI take the code default,
            # REALME_GEMINI_MODEL or --model -- so the only thing a saved id
            # could do was go stale and mislead.
            stored = ad.pop("gemini_model", None)
            if stored is not None:
                changed = True
                import sys as _sys
                print(f"  [profile] dropped the saved script model "
                      f"'{stored}'; the model comes from the Lecture tab, "
                      f"--model, or REALME_GEMINI_MODEL in your .env",
                      file=_sys.stderr, flush=True)
        return changed

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def update(self, patch: dict) -> None:
        for k, v in patch.items():
            if isinstance(v, dict) and isinstance(self.data.get(k), dict):
                self.data[k].update(v)
            else:
                self.data[k] = v
        self.save()

    def store_audio(self, src: Path, kind: str) -> Path:
        """
        kind is 'voice_reference', 'consent_recording' or 'gemini_consent'.

        Converted on the way in, never copied as-is. Keeping the uploaded
        container meant an iPhone .m4a was stored as .m4a and handed straight to
        the engines, where the C++ one rejects it as "Not a RIFF file" and
        libsndfile has no AAC decoder at all. One conversion here; nothing
        downstream has to know what was uploaded.
        """
        from realme.core.media import to_reference_wav
        dst = self.root / "baked_assets" / f"{kind}.wav"
        to_reference_wav(Path(src), dst)

        # Keep the Studio's workbench copy in step.
        #
        # There are two ways in: `realme voice enroll` writes here directly,
        # while the Studio uploads to <data>/voice/reference_raw.wav and bakes
        # FROM there. Those never shared a source, so enrolling from the command
        # line and later pressing "Save this voice" in the Studio would have
        # baked an older recording over the newer one -- silently, because both
        # files are called the right thing in the right place.
        if kind == "voice_reference":
            raw = self.root.parent / "voice" / "reference_raw.wav"
            raw.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dst, raw)
        if kind == "gemini_consent":
            # Lives inside the engine's own block rather than at the top
            # level: it is Google's wording, for Google, and it should not be
            # mistaken for the consent artifact an institution would read.
            g = dict(self.data.get("gemini_voice") or {})
            g["consent_recording"] = str(dst)
            self.data["gemini_voice"] = g
            self.save()
            return dst
        self.data[kind] = str(dst)
        if kind == "consent_recording":
            self.data["consent_recorded_at"] = datetime.now(timezone.utc).isoformat()
        self.save()
        return dst

    # --- the reference transcript -------------------------------------------
    #
    # Stored as a plain .txt BESIDE the audio, not only inside profile.json.
    #
    # Two files that must correspond -- a clip and the words spoken in it --
    # belong together, where you can see them, open them in Notepad and fix a
    # typo. A transcript buried in a JSON blob is a transcript nobody checks
    # against the recording, and a mismatch there silently degrades every clone.
    #
    # The JSON field is kept in step as a mirror. If the two ever disagree the
    # FILE wins, because the file is the one a human edited.

    @property
    def reference_text_path(self) -> Path:
        return self.root / "baked_assets" / "voice_reference.txt"

    @property
    def reference_text(self) -> str:
        f = self.reference_text_path
        if f.is_file():
            text = " ".join(read_text(f).split())
            if text:
                return text
        return (self.data.get("voice_reference_text") or "").strip()

    def set_reference_text(self, text: str) -> Path:
        text = " ".join(str(text).split())
        f = self.reference_text_path
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text + "\n", encoding="utf-8")
        self.update({"voice_reference_text": text})
        return f

    @property
    def ready_for_cloning(self) -> bool:
        return bool(self.data.get("voice_reference") and self.data.get("consent_recording"))
