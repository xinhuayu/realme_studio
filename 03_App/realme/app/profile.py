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


def audio_id(path: Path) -> str:
    """A short digest of a recording's bytes: which clip this is, not where.

    Content, never mtime or path. The baked file is always called
    `voice_reference.wav` and is rewritten in place on every enrolment, so a
    name tells you nothing about which recording it holds.
    """
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


DEFAULTS = {
    "profile_id": "instructor",
    "display_name": "",
    "course_context": "",
    "style_notes": "",
    # No `gemini_model`: the model is not a property of the voice profile.
    # The Studio's menu and the CLI read the code default (or
    # REALME_GEMINI_MODEL, or --model), never this file.
    # The default for a profile that does not exist yet. An existing profile
    # is NEVER migrated onto this: moving from a local engine to a hosted one
    # sends the voice and lectures to a third party and incurs charges, so it
    # is left to the user in Twin Setup.
    "adapters": {"script_writer": "gemini",
                 "tts": "gemini-tts",
                 # The guest is a DRAFT voice, and a different one from the
                 # instructor's, so a debate has two distinct speakers.
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
    #: Saved speed per voice, as a factor: {"piper:en_US-amy-medium": 1.15,
    #: "gemini-tts": 0.97}. Applied wherever that voice speaks -- lecture,
    #: narration, podcast, explainer, previews -- after synthesis, so a
    #: change never re-records anything. Absent means 1.0.
    "voice_speeds": {},
    "disclosure": "This lecture was narrated with a synthetic voice model of the "
                  "instructor, from a script the instructor wrote and approved.",
    "consent_recorded_at": None,
    "voice_reference": None,
    # Qwen3 conditions the clone on the reference clip AND its exact
    # transcript. Without the text the clone is measurably worse, so it
    # is part of the profile rather than an environment variable.
    "voice_reference_text": "",
    #: Which recording the transcript describes.
    #:
    #: `reference_audio_id` is a digest of the enrolment recording now on
    #: disk; `reference_text_id` is the digest of the recording that was on
    #: disk when the transcript was last saved. Equal means the two describe
    #: each other. Every pace measurement in the system is words of text
    #: divided by seconds of speech, so when they disagree, every one of those
    #: numbers is wrong, though it may still look plausible.
    "reference_audio_id": None,
    "reference_text_id": None,
    #: register name -> treated .wav. Several takes of the same voice,
    #: levelled together so the difference between them is delivery.
    "voice_registers": {},
    "consent_recording": None,
    #: Words per minute of SPEECH in the enrolment recording, measured when it
    #: was enrolled. Kept so that a later re-recording can be compared against
    #: it: a clone copies the delivery it was cloned from, so a reference read
    #: more slowly than usual makes the clone slower too.
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
        #: The voice id `rate_match` was measured for. A correction belongs to
        #: one clone and must not be applied to a re-enrolled voice. Unset
        #: means unverifiable,
        #: which `tts_gemini.effective_rate_match` treats as "do not apply".
        "rate_match_for": None,
        #: Characters per API call. None = the engine's own measured default
        #: (738, see adapters/tts_gemini.MEASURED_MAX_CHARS); 0 = fall back to
        #: the pipeline's 260. Any other number should be one that
        #: probe_gemini_tts.py measured on this account. The documented API
        #: limit is much larger; calls that long make the voice hurry.
        "max_chars": None,
    },
}


#: What a saved setting may be. Checked when the Studio saves the profile,
#: so a bad value is rejected there rather than failing a later render.
VIDEO_LIMITS = {"width": (160, 7680), "height": (120, 4320), "fps": (1, 120)}
LAYOUTS = ("slide_only", "pip", "side_by_side")
SIGNALLING = ("highlight", "underline", "focus", "off")
PROSODY_MODES = ("natural", "sentence", "stable", "expressive")
#: The range `prosody.gap_ms` honours; outside it the value is clamped there.
PAUSE_SCALE = (0.85, 1.25)
VOICE_SPEED = (0.70, 1.30)        # casting.VOICE_SPEED_RANGE
_TEXT_KEYS = ("display_name", "course_context", "style_notes", "disclosure")


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) \
        and v == v and v not in (float("inf"), float("-inf"))


def validate_patch(patch: dict) -> None:
    """Raise ValueError naming every bad value in a profile update.

    Only the settings a render depends on are checked; keys not listed here
    pass through as they always have.
    """
    if not isinstance(patch, dict):
        raise ValueError("The profile update must be a JSON object.")
    bad: list[str] = []
    video = patch.get("video")
    if video is not None:
        if not isinstance(video, dict):
            bad.append("video must be an object")
        else:
            for k, (lo, hi) in VIDEO_LIMITS.items():
                if k not in video:
                    continue
                v = video[k]
                if not _number(v) or (k != "fps" and v != int(v)):
                    bad.append(f"video.{k} must be a whole number, not {v!r}")
                elif not lo <= v <= hi:
                    bad.append(f"video.{k} must be {lo}-{hi}, not {v}")
                elif k in ("width", "height") and int(v) % 2:
                    # H.264 in 4:2:0 refuses an odd dimension outright.
                    bad.append(f"video.{k} must be even, not {v}")
            if "layout" in video and video["layout"] not in LAYOUTS:
                bad.append(f"video.layout must be one of {', '.join(LAYOUTS)}")
    if "pause_scale" in patch:
        v = patch["pause_scale"]
        lo, hi = PAUSE_SCALE
        if not _number(v) or not lo <= v <= hi:
            bad.append(f"pause_scale must be a number {lo}-{hi}, not {v!r}")
    if "signalling" in patch and patch["signalling"] not in SIGNALLING:
        bad.append(f"signalling must be one of {', '.join(SIGNALLING)}")
    if "prosody_mode" in patch and patch["prosody_mode"] not in PROSODY_MODES:
        bad.append(f"prosody_mode must be one of {', '.join(PROSODY_MODES)}")
    if "voice_speeds" in patch:
        vs = patch["voice_speeds"]
        lo, hi = VOICE_SPEED
        if not isinstance(vs, dict) or not all(
                isinstance(k, str) and _number(v) and lo <= v <= hi
                for k, v in vs.items()):
            bad.append(f"voice_speeds must map voice names to factors "
                       f"{lo}-{hi}")
    if "adapters" in patch:
        ad = patch["adapters"]
        if not isinstance(ad, dict) or not all(
                isinstance(v, str) for v in ad.values()):
            bad.append("adapters must map slots to engine names")
    for k in _TEXT_KEYS:
        if k in patch and not isinstance(patch[k], str):
            bad.append(f"{k} must be text")
    if "draft_voice" in patch and not isinstance(patch["draft_voice"],
                                                 (str, type(None))):
        bad.append("draft_voice must be a voice name")
    if bad:
        raise ValueError("Not saved: " + "; ".join(bad) + ".")


class Profile:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "baked_assets").mkdir(exist_ok=True)
        self.path = self.root / "profile.json"
        self.data = (json.loads(read_text(self.path)) if self.path.exists()
                     else copy.deepcopy(DEFAULTS))
        # Backfill NESTED defaults too, so a profile saved before a sub-key
        # existed (e.g. a new `adapters` slot) gains it.
        for k, v in DEFAULTS.items():
            if isinstance(v, dict) and isinstance(self.data.get(k), dict):
                for sk, sv in v.items():
                    self.data[k].setdefault(sk, sv)
            else:
                self.data.setdefault(k, v)
        if self._migrate():
            self.save()

    #: Engine names an older profile may hold, and the current engine each
    #: maps to. None of these is in the registry.
    RETIRED_ENGINES = {"qwen3": "qwen3cpp", "qwen3cpp-cli": "qwen3cpp",
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
            # A guest must not clone the instructor's voice. An older profile
            # may hold a cloning engine here; reset it to the draft default.
            if ad.get("guest_tts") in ("qwen3cpp", "qwen3", "chatterbox",
                                       "indextts"):
                ad["guest_tts"] = "piper:en_US-amy-medium"
                changed = True
                import sys as _sys
                print(f"  [profile] guest_tts was a clone of your own voice "
                      f"-> {ad['guest_tts']}", file=_sys.stderr, flush=True)
            # A saved guest voice that is not in DRAFT_VOICES would leave the
            # Studio dropdown empty and the guest named by its file name.
            # Move it to the default and say so: the voice may still be on
            # disk, so this changes a preference the user made.
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
            # REALME_GEMINI_MODEL or --model.
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

        Converted on the way in, never copied as-is: the C++ engine rejects
        a non-RIFF file such as .m4a, and libsndfile has no AAC decoder. One
        conversion here; nothing downstream has to know what was uploaded.
        """
        from realme.core.media import to_reference_wav
        dst = self.root / "baked_assets" / f"{kind}.wav"
        to_reference_wav(Path(src), dst)

        # Keep the Studio's workbench copy in step.
        #
        # There are two ways in: `realme voice enroll` writes here directly,
        # while the Studio uploads to <data>/voice/reference_raw.wav and bakes
        # FROM there. Copying here keeps the two in step, so "Save this voice"
        # in the Studio after a command-line enrolment does not bake an older
        # recording over the newer one.
        if kind == "voice_reference":
            raw = self.root.parent / "voice" / "reference_raw.wav"
            raw.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dst, raw)
            self.data["reference_audio_id"] = audio_id(dst)
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
    # The clip and the words spoken in it are kept side by side so the
    # transcript can be opened in a text editor and checked against the
    # recording. A mismatch degrades every clone.
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

    @property
    def transcript_matches_recording(self) -> bool:
        """Can the transcript on file describe the recording on file?

        Two ways to be satisfied, because there are two legitimate ways the
        transcript gets written.

        Through the app, `set_reference_text` records which recording was on
        disk at the time, and the two digests settle it outright.

        By hand is the other way: the transcript lives beside the audio as a
        plain .txt so it can be opened and corrected, and where the file and
        the JSON mirror disagree the FILE wins. Nothing in the bytes of a text
        file can prove which clip it belongs to, so what is checked is order
        -- a transcript last written BEFORE the recording cannot be a
        transcript of it.

        What neither way catches is a transcript of the right length and the
        wrong words. That is what `pace.check_human_wpm` is for: it refuses an
        implied rate no one speaks at, whatever the provenance.
        """
        a = self.data.get("reference_audio_id") or ""
        t = self.data.get("reference_text_id") or ""
        if a and t:
            return a == t
        txt = self.reference_text_path
        wav = Path(self.data.get("voice_reference") or "")
        if txt.is_file() and wav.is_file():
            # One second of slack: the two are often written in the same
            # action, and filesystem timestamps are not ordered more finely
            # than that everywhere.
            return txt.stat().st_mtime >= wav.stat().st_mtime - 1.0
        return False

    def note_transcript_pairing(self) -> None:
        """Record that the transcript and the recording go together.

        Called once a rate has been measured from the pair and found
        plausible, so that the mtime reasoning above is needed only until the
        first measurement. After that the pair is settled by content, and
        re-recording breaks it again -- which is correct.
        """
        a = self.data.get("reference_audio_id")
        if not a:
            ref = self.data.get("voice_reference")
            if not (ref and Path(ref).is_file()):
                return
            a = audio_id(Path(ref))
        self.update({"reference_audio_id": a, "reference_text_id": a})

    @property
    def transcript_note(self) -> str:
        """Empty when the pair checks out; otherwise what to do about it."""
        if self.transcript_matches_recording or not self.reference_text:
            return ""
        return ("The transcript on file was last written before the enrolment "
                "recording now on disk, so it describes an earlier take. "
                "Every speaking rate in RealMe is words of text over seconds "
                "of speech, so measuring from this pair would be arithmetic "
                "on mismatched inputs. Put the text of the passage you "
                "actually read in "
                f"{self.reference_text_path.name} (Twin Setup, Notepad, or "
                "`realme voice enroll <file> --transcript-file <txt>`).")

    def set_reference_text(self, text: str) -> Path:
        text = " ".join(str(text).split())
        f = self.reference_text_path
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text + "\n", encoding="utf-8")
        self.update({"voice_reference_text": text,
                     "reference_text_id": self.data.get("reference_audio_id")})
        return f

    @property
    def ready_for_cloning(self) -> bool:
        return bool(self.data.get("voice_reference") and self.data.get("consent_recording"))
