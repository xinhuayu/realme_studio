"""
Gemini TTS with a replicated (cloned) voice.

The first cloning engine in RealMe that is NOT on your machine, which is the
whole of what is different about it. Everything it speaks, and the clip it was
cloned from, leaves the computer. That is a trade, not an upgrade, and the
adapter is written to keep the trade visible rather than convenient:

  * `check()` refuses unless the project has been acknowledged as a PAID
    Gemini project. On the free tier Google's terms say human reviewers may
    read API input and output and that the content is used to improve Google
    products, and tell you not to submit personal information. A recording of
    your own voice and the text of your lectures is exactly that. On the paid
    tier Google states prompts and responses are not used for improvement and
    are retained only briefly for abuse detection. This is a one-line
    acknowledgement, not a technical barrier -- there is no API that reports
    your billing tier -- so it is written down where it will be read.
  * A replicated voice lives in Google's project for a year from last use
    (`voice_...`), or seven days (`voicekey_...`). It is not yours in the way
    the Qwen3 weights on your disk are yours. `realme voice gemini --status`
    says which, and when it expires.

Two things about delivery are different from every local engine here, and both
are handled rather than hidden:

  1. **There is no numeric speaking-rate field.** Gemini steers delivery with a
     natural-language style string. RealMe's pace is a measured float that goes
     into the render ledger's key, and "speaking slightly slowly" is not a
     float. So pace is applied AFTERWARDS, by ffmpeg, exactly as it already is
     for every engine whose CLI has no speed control (`core.media.retime`). The
     style string carries manner -- not rate -- and a request for a pace beyond
     what retiming does cleanly adds a rate word as well, and says so.
  2. **It is happy with far more text per call than a local autoregressive
     engine.** The model's documented input limit is 8,192 tokens and its
     output limit 16,384 (about eight minutes of speech). That does not mean a
     long call SOUNDS better: `pipeline/speak.py` chunks because engines drift,
     and the one public long-form benchmark finds models that advertise
     single-pass long generation degrade worse than chunked alternatives. So
     this adapter declares `max_chars_per_call`, the pipeline honours it, and
     the number comes from `probe_gemini_tts.py` measuring THIS model on YOUR
     text -- not from this docstring and not from Google's limit. Until it is
     measured the pipeline default stands.

API shape, for the next person reading this against Google's docs:
  create a voice   POST /v1beta/voices       (source_audio + consent_audio)
  speak            POST /v1beta/interactions (generation_config.speech_config)
"""
from __future__ import annotations
import base64, json, os, time, urllib.error, urllib.request
from pathlib import Path

from realme.adapters.base import BaseTTS, AdapterUnavailable

API_ROOT = "https://generativelanguage.googleapis.com/v1beta"

#: The models that can speak with a replicated voice.
MODELS = ("gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts")
DEFAULT_MODEL = "gemini-3.8-flash-tts"

#: Output is 24 kHz mono 16-bit, which is what the rest of the pipeline uses.
SAMPLE_RATE = 24000

#: Dollars per million audio output tokens, and how many tokens a second of
#: speech bills at. Both halves double on 1 January 2027 and this is the one
#: place to change them.
#:
#: 32 rather than the 25 on Google's pricing page: measurement across the 3.x
#: models reports about 32, and the difference is 28% of the bill. The token
#: count Google returns with each response is used when it is there; this is
#: the fallback, and anything derived from it is called an estimate.
PRICE_PER_MTOK = {"gemini-3.8-flash-tts": 9.0, "gemini-3.8-flash-lite-tts": 6.0}
TOKENS_PER_SECOND = 32.0

#: Characters per call, measured rather than chosen -- and the default,
#: because the pipeline's 260 is tuned for local autoregressive engines that
#: drift past twenty seconds, which this one does not.
#:
#: Where 738 comes from. A ladder of calls from 113 to 1345 characters, each a
#: prefix of the same passage, every size compared against the same sentences
#: synthesized one at a time:
#:
#:     chars   speaking time   content (syllables)
#:       598       0.95x             1.01x        intact
#:       738       0.95x             0.96x        intact      <- here
#:       839       0.92x             0.92x        rushed by 8%
#:      1345       0.91x             0.92x        rushed by 9%
#:
#: Nothing is dropped at any size -- the syllables are all there. What a long
#: call does is say them faster, and that is audible as muddiness. 738 is the
#: last size that is not hurried, confirmed on a second take and by listening.
#:
#: It is NOT rescued by retiming a larger size. The hurry varies with the text
#: -- a list of short clauses is not a paragraph of prose -- so a single
#: correction cannot track it, and an unpredictable 8% is worse than a
#: predictable extra API call. Measure your own with probe_gemini_tts.py; this
#: is the default, not a constant of nature.
MEASURED_MAX_CHARS = 738

#: Google requires this sentence, in the speaker's own voice, before it will
#: create a replicated voice -- their wording, not ours, and it is rejected if
#: it is paraphrased. RealMe's own consent recording (enrollment/voice.py) says
#: something broader and is kept for your institution; this one is for Google.
#: Two clips, because one sentence cannot serve two readers.
CONSENT_SENTENCE = (
    "I am the owner of this voice and I consent to Google using this voice "
    "to create a synthetic voice model."
)

#: Why you are being asked to confirm the tier. Printed, not buried.
TIER_NOTICE = (
    "Gemini TTS sends your voice clip and every sentence of every lecture to "
    "Google.\n"
    "  On the FREE tier, Google's API terms say human reviewers may read API "
    "input and\n"
    "  output, that the content is used to improve Google products, and that "
    "you should\n"
    "  not submit personal information. Your voice is personal information.\n"
    "  On the PAID tier, Google states prompts and responses are not used to "
    "improve\n"
    "  its products and are logged only briefly for abuse detection.\n"
    "  Nothing in the API reports which tier you are on, so RealMe asks once:\n"
    "    realme voice gemini --acknowledge-paid\n"
    "  or set REALME_GEMINI_PAID=1."
)


def api_key(explicit: str | None = None) -> str:
    """The key, from the environment or from wherever RealMe keeps .env files.

    The fallback is not belt-and-braces; it is the whole point. A key normally
    lives in a .env file, and `core.env.load()` is what puts it into the
    environment -- the Studio calls it at import, the CLI calls it in `main`.
    A standalone script calls neither, so `probe_gemini_tts.py` told someone
    who had set a key perfectly well that there was no key, and sent him
    looking for a file that was exactly where it should be.

    Loading here means every caller gets the same answer: the Studio, the CLI,
    the probe, and whatever is written next.
    """
    key = explicit or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        try:
            from realme.core.env import load as _load_env
            _load_env()
        except Exception:
            pass
        key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        # "No key" is a dead end unless it says where it looked -- the same
        # reasoning as the Studio's doctor, which lists these paths already.
        where = ""
        try:
            from realme.core.env import candidates
            rows = []
            for c in candidates():
                try:
                    if not c.is_file():
                        state = "not there"
                    else:
                        from realme.core.textio import read_text
                        state = ("has GEMINI_API_KEY"
                                 if "GEMINI_API_KEY" in read_text(c)
                                 else "exists, but no GEMINI_API_KEY in it")
                except OSError as e:
                    state = f"unreadable ({e})"
                rows.append(f"      {c}  --  {state}")
            where = "\n  Looked in:\n" + "\n".join(rows)
        except Exception:
            pass
        raise AdapterUnavailable(
            "No GEMINI_API_KEY. It is the same key the script writer uses; "
            "put it in a .env\n  as GEMINI_API_KEY=... or run `realme key "
            "gemini <key>`." + where)
    return key


def _post(path: str, body: dict, key: str, *, timeout: int = 300,
          tries: int = 4, log=None) -> dict:
    """POST JSON, with the same retry manners as the script writer.

    429 and 5xx are retried with backoff; 4xx is not, because asking a
    malformed question again gets the same answer more slowly.
    """
    data = json.dumps(body).encode("utf-8")
    last = None
    for attempt in range(tries):
        req = urllib.request.Request(
            f"{API_ROOT}/{path.lstrip('/')}", data=data,
            headers={"Content-Type": "application/json", "x-goog-api-key": key})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:600]
            last = f"HTTP {e.code}: {detail}"
            if e.code not in (429, 500, 502, 503, 504) or attempt == tries - 1:
                raise AdapterUnavailable(f"Gemini TTS refused: {last}") from e
        except urllib.error.URLError as e:
            last = str(e.reason)
            if attempt == tries - 1:
                raise AdapterUnavailable(
                    f"Could not reach the Gemini API: {last}") from e
        wait = 2 ** attempt
        if log:
            log(f"  gemini: {last} -- retrying in {wait}s")
        time.sleep(wait)
    raise AdapterUnavailable(f"Gemini TTS failed: {last}")


def _get(path: str, key: str, *, timeout: int = 60) -> dict:
    req = urllib.request.Request(f"{API_ROOT}/{path.lstrip('/')}",
                                 headers={"x-goog-api-key": key})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:400]
        raise AdapterUnavailable(f"HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise AdapterUnavailable(
            f"Could not reach the Gemini API: {e.reason}") from e


def require_wav(path: Path) -> Path:
    """Refuse anything that is not already a RIFF WAV, in RealMe's words.

    Checked before the clip is measured or trimmed, not after. Trimming an
    .m4a first meant ffprobe got there before this did, and the message a
    phone recording produced was "moov atom not found" -- true, useless, and
    about a file the Studio would have converted on the way in.
    """
    path = Path(path)
    if path.read_bytes()[:4] != b"RIFF":
        raise AdapterUnavailable(
            f"{path.name} is not a RIFF WAV. Enrol it through RealMe "
            f"(`realme voice enroll <file>`), which converts to 24 kHz mono "
            f"WAV first.")
    return path


def _inline_audio(path: Path) -> dict:
    """A WAV as Google wants it inline: mime type plus base64."""
    raw = require_wav(path).read_bytes()
    return {"mime_type": "audio/wav", "data": base64.b64encode(raw).decode()}


# ----------------------------------------------- what Google checks before it clones
#
# Google runs a speaker-verification check between the two clips and refuses
# with "Voice mismatch detected" if they do not pass it. Its own best-practice
# note is the whole explanation: "Record both source_audio and consent_audio on
# the same microphone in the same acoustic setting so the speaker verification
# check succeeds reliably."
#
# That is easy to violate without noticing. The first attempt here used a
# 62-second reference recorded on a good microphone and a consent sentence
# recorded on an iPhone. Same person, same room, obviously the same voice to a
# listener -- and measurably not the same recording: spectral centroid 1593 Hz
# against 1076 Hz, a third of the brightness gone to a different capsule and a
# different distance. Verification compares the spectral envelope, so it said
# "different speaker" and was, in its own terms, right.
#
# So the two length rules Google documents are enforced here, before the
# upload, and the acoustic distance is MEASURED and reported rather than
# discovered as an HTTP 500 three minutes later.

#: Google documents 10-30 seconds for the reference clip.
SOURCE_MIN_S, SOURCE_MAX_S = 10.0, 30.0
#: What an over-long reference is trimmed to. Inside the window with room to
#: spare, because `duration_of` and Google's own measurement need not agree to
#: the millisecond.
TRIM_TO_S = 25.0


def first_sound(path: Path, *, floor_db: float = -45.0) -> float:
    """Seconds of leading silence, so a trim does not start on room tone."""
    from realme.core.media import require, run
    try:
        proc = run([require("ffmpeg"), "-i", str(path), "-af",
                    f"silencedetect=noise={floor_db}dB:d=0.25", "-f", "null",
                    "-"], "find the first word")
    except Exception:
        return 0.0          # a trim from zero is fine; this is a nicety
    text = proc.stderr.decode("utf-8", "replace")
    first = 0.0
    for line in text.splitlines():
        if "silence_end:" in line:
            try:
                first = float(line.split("silence_end:")[1].split()[0])
            except (IndexError, ValueError):
                pass
            break
    # Never skip more than a couple of seconds on the strength of a heuristic.
    return first if 0.0 < first <= 2.0 else 0.0


def prepare_source(src: Path, out: Path, *, seconds: float = TRIM_TO_S,
                   log=None) -> Path:
    """
    The reference clip, inside the length Google accepts.

    A 62-second clip is a BETTER reference for Qwen3, which has no such limit
    and conditions on the whole thing. So this trims a copy for Google and
    leaves the enrolled reference alone; the two engines want different things
    from the same recording and neither should be degraded for the other.

    No processing beyond the cut: a trimmed excerpt is still the same
    microphone in the same room, which is the property that matters here.
    """
    from realme.core.media import MediaError, duration_of, probe, require, run
    src, out = require_wav(src), Path(out)
    try:
        dur = duration_of(src)
    except MediaError as e:
        raise AdapterUnavailable(f"{src.name} could not be read: {e}") from e
    if dur < SOURCE_MIN_S:
        raise AdapterUnavailable(
            f"{src.name} is {dur:.1f}s. Google needs at least "
            f"{SOURCE_MIN_S:.0f} seconds of speech to clone from.")
    if dur <= SOURCE_MAX_S:
        return src
    start = first_sound(src)
    out.parent.mkdir(parents=True, exist_ok=True)
    run([require("ffmpeg"), "-y", "-ss", f"{start:.3f}", "-t", f"{seconds:.3f}",
         "-i", str(src), "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a",
         "pcm_s16le", str(out)], "trim the reference for Google")
    probe(out)
    if log:
        log(f"  the reference is {dur:.0f}s; Google takes "
            f"{SOURCE_MIN_S:.0f}-{SOURCE_MAX_S:.0f}s, so {seconds:.0f}s of it "
            f"was sent (your enrolled clip is untouched)")
    return out


def consent_check(reference: Path, consent: Path) -> dict:
    """
    How alike are the two RECORDINGS -- before Google is asked.

    Returns the project's own descriptive acoustic comparison, which is not
    speaker verification and does not pretend to be. It cannot tell you
    Google will accept the pair. It can tell you the two clips were made on
    different equipment, which is the one cause worth catching early because
    it is invisible to the person who made them: they heard themselves say
    both sentences.
    """
    from realme.enrollment import voice as V
    from realme.core.media import duration_of
    out = {"ok": True, "notes": [], "score": None,
           "reference_s": round(duration_of(reference), 1),
           "consent_s": round(duration_of(consent), 1)}
    if out["consent_s"] < SOURCE_MIN_S:
        out["notes"].append(
            f"The consent clip is {out['consent_s']:.1f}s. Google documents "
            f"10-30 seconds for these recordings; read the sentence a little "
            f"more slowly, or say it twice.")
    try:
        a = V.acoustic_signature(Path(reference))
        b = V.acoustic_signature(Path(consent))
        cmp_ = V.compare_signatures(a, b)
        out["score"] = cmp_.get("score")
        out["reference_signature"], out["consent_signature"] = a, b
        ca, cb = a.get("spectral_centroid_hz"), b.get("spectral_centroid_hz")
        if ca and cb and (max(ca, cb) / min(ca, cb)) > 1.2:
            out["notes"].append(
                f"The two clips have very different timbre "
                f"({ca:.0f} Hz against {cb:.0f} Hz of spectral centroid). "
                f"That is a different microphone or a different distance from "
                f"it, not a different voice -- but Google's speaker check "
                f"compares exactly this, and it is what makes it refuse.")
        pa, pb = a.get("pitch_median_hz"), b.get("pitch_median_hz")
        if pa and pb and (max(pa, pb) / min(pa, pb)) > 1.15:
            out["notes"].append(
                f"The pitch differs more than one person's normally does "
                f"between takes ({pa:.0f} Hz against {pb:.0f} Hz).")
    except Exception as e:                 # never block an upload on analysis
        out["notes"].append(f"(could not compare the recordings: {e})")
    out["ok"] = not out["notes"]
    return out


#: What Google's refusals mean, in words that say what to do. The raw text is
#: a 500 wrapping an INVALID_ARGUMENT wrapping a protobuf type-URL complaint,
#: and the sentence that matters is four lines in.
REFUSALS = (
    ("voice mismatch",
     "Google compared the two recordings and decided they are different "
     "speakers.\n"
     "  Almost always this is equipment, not identity: its own guidance is to "
     "record\n"
     "  both clips on the SAME microphone in the SAME room, one after the "
     "other.\n"
     "  A good desk mic for the reference and a phone for the consent "
     "sentence fails\n"
     "  this check even though both are obviously you."),
    ("consent flow failed",
     "The consent clip was not accepted. The sentence must be recited exactly:"
     "\n"
     f"    {CONSENT_SENTENCE}\n"
     "  No name, no date, no paraphrase -- RealMe's own consent recording "
     "says more\n"
     "  than this and is rejected for saying it."),
    ("too short",
     "One of the clips is shorter than Google accepts. Both want 10-30 "
     "seconds."),
    ("too long",
     "One of the clips is longer than Google accepts. The reference wants "
     "10-30 seconds."),
)


def explain(detail: str) -> str:
    """A refusal in words that name the fix, with the original kept."""
    low = (detail or "").lower()
    for needle, advice in REFUSALS:
        if needle in low:
            return advice
    return ""


# ------------------------------------------------------------- voice creation

def create_voice(reference_wav: Path, consent_wav: Path, *,
                 key: str | None = None, model: str = DEFAULT_MODEL,
                 display_name: str = "RealMe instructor",
                 store: bool = True, trim: bool = True, log=None) -> dict:
    """
    Enrol a replicated voice and return {"id"|"key", "expires", "raw"}.

    `store=True` gives a persistent `voice_...` (200 per project, expiring a
    year after last use). `store=False` gives a `voicekey_...` that lives seven
    days -- which is the right choice for a one-off, and the wrong one for a
    course you re-render in March.

    `trim` sends a 25-second excerpt when the enrolled reference is longer
    than Google's 30-second maximum, rather than failing on a clip that is
    exactly right for the local engine.
    """
    k = api_key(key)
    reference_wav = require_wav(reference_wav)
    require_wav(consent_wav)
    if trim:
        reference_wav = prepare_source(
            reference_wav,
            reference_wav.with_name(reference_wav.stem + "_for_google.wav"),
            log=log)
    body = {
        "store": bool(store),
        "voice": {
            "model": model,
            "type": "replicated",
            "display_name": display_name,
            "replicated": {
                "source_audio": _inline_audio(reference_wav),
                "consent_audio": _inline_audio(consent_wav),
            },
        },
    }
    try:
        resp = _post("voices", body, k, timeout=300)
    except AdapterUnavailable as e:
        # Google's refusal arrives as a 500 wrapping an INVALID_ARGUMENT
        # wrapping a protobuf type-URL complaint, with the sentence that
        # matters four lines down. Lead with what to do; keep the original.
        advice = explain(str(e))
        raise AdapterUnavailable(f"{advice}\n\n  Google said: {e}"
                                 if advice else str(e)) from e
    made = voice_from(resp)
    if not made["voice"]:
        raise AdapterUnavailable(
            "The voice was not created and the response did not say why: "
            f"{json.dumps(resp)[:400]}")
    return {"voice": made["voice"],
            "stored": made["stored"] if made["stored"] is not None else bool(store),
            "model": made["model"] or model,
            "ttl": made["ttl"],
            "expires": made["expires"],
            "raw": resp}


def voice_from(resp: dict) -> dict:
    """The identifier out of a voices response, whatever shape it arrives in.

    This was written to one documented shape -- {"replicated_voice": {"id":…}}
    -- and the API returned another: the fields flat at the top level, with
    `name`/`id` and an `expire_time`. The adapter then reported "the voice was
    not created" about a voice that HAD been created, charged for, and left
    sitting in the project with nothing in RealMe pointing at it. Two of them,
    before anyone noticed.

    So the rule here is the one the audio parser already follows: look for the
    thing in every shape it could plausibly be in, and only then say it is
    absent. An identifier that starts with voice_ or voicekey_ is unambiguous,
    which makes a tolerant search safe rather than sloppy.
    """
    for wrapper in ("replicated_voice", "replicatedVoice", "voice"):
        inner = resp.get(wrapper)
        if isinstance(inner, dict):
            resp = {**resp, **inner}
    ident = ""
    for field in ("id", "key", "name", "voice", "voice_id", "voiceId"):
        v = resp.get(field)
        if isinstance(v, str) and v.strip():
            # "voices/voice_abc" is how a resource name is spelled.
            cand = v.rsplit("/", 1)[-1]
            if cand.startswith(("voice_", "voicekey_")):
                ident = cand
                break
            ident = ident or cand
    expires = (resp.get("expire_time") or resp.get("expireTime") or "")
    stored = None
    if ident.startswith("voicekey_"):
        stored = False
    elif ident.startswith("voice_"):
        stored = True
    ttl = (f"{expires[:10]}" if expires else
           ("1 year from last use" if stored else "7 days"))
    model = (resp.get("model") or "").rsplit("/", 1)[-1]
    return {"voice": ident, "expires": expires, "stored": stored,
            "model": model, "ttl": ttl}


def get_voice(voice: str, *, key: str | None = None) -> dict:
    """Does this voice still exist? Cheap, and costs no audio."""
    return _get(f"voices/{voice}", api_key(key))


def list_voices(*, key: str | None = None) -> list[dict]:
    """Every replicated voice in the project, newest-looking first.

    The recovery path for a voice that exists at Google and not here -- which
    is a state this adapter has already produced once, by mis-reading a
    creation response. Something has to be able to find them again, or the
    only remedy is paying to make another.
    """
    resp = _get("voices", api_key(key))
    rows = (resp.get("voices") or resp.get("replicated_voices")
            or resp.get("replicatedVoices") or [])
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        made = voice_from(r)
        if made["voice"]:
            out.append({**made, "display_name": r.get("display_name")
                        or r.get("displayName") or "", "raw": r})
    return out


def delete_voice(voice: str, *, key: str | None = None) -> None:
    req = urllib.request.Request(f"{API_ROOT}/voices/{voice}", method="DELETE",
                                 headers={"x-goog-api-key": api_key(key)})
    try:
        urllib.request.urlopen(req, timeout=60).read()
    except urllib.error.HTTPError as e:
        raise AdapterUnavailable(f"Could not delete {voice}: HTTP {e.code}") from e


# -------------------------------------------------------------- the adapter

def _audio_from(resp: dict) -> bytes:
    """
    The audio out of an interactions response, wherever it is.

    Written as a search rather than a path because the response shape is
    documented loosely (`steps[].content[].data`) and a key rename upstream
    should cost a louder error than `KeyError: 'steps'` three frames deep.
    """
    found: list[str] = []

    def walk(node):
        if isinstance(node, dict):
            d = node.get("data")
            if isinstance(d, str) and len(d) > 64:
                found.append(d)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(resp)
    if not found:
        raise AdapterUnavailable(
            "Gemini returned no audio. The response was: "
            f"{json.dumps(resp)[:500]}")
    return b"".join(base64.b64decode(f) for f in found)


def _usage_from(resp: dict) -> dict:
    for k in ("usage_metadata", "usageMetadata", "usage"):
        u = resp.get(k)
        if isinstance(u, dict):
            return u
    return {}


class GeminiTTS(BaseTTS):
    """Gemini 3.8 Flash TTS speaking in a voice replicated from your clip."""

    name = "gemini-tts"
    is_voice_clone = True
    is_placeholder = False
    phoneme_syntax = None        # no inline phoneme syntax; respellings only

    #: It detects the input language itself and speaks 130+. Listed so that a
    #: [[fr-FR]] block is passed through rather than silently read in English;
    #: the list is the common teaching set, not the full 130.
    speaks_languages = ("en", "fr", "de", "es", "it", "pt", "nl", "pl", "ru",
                        "ar", "hi", "bn", "ja", "ko", "zh", "vi", "th", "tr",
                        "id", "uk", "sv", "da", "no", "fi", "el", "he", "fa",
                        "ta", "te", "ur", "sw")

    def __init__(self, voice: str | None = None, *, model: str | None = None,
                 api_key: str | None = None, style: str = "",
                 max_chars: int | None = None, profile=None,
                 rate_match: float | None = None,
                 reference_wav: Path | None = None, log=None):
        self.voice = voice or os.environ.get("REALME_GEMINI_VOICE", "")
        self.model = model or os.environ.get("REALME_GEMINI_TTS_MODEL",
                                             DEFAULT_MODEL)
        self._key = api_key
        self.style = style or os.environ.get("REALME_GEMINI_STYLE", "")
        self.reference_wav = reference_wav
        self.log = log
        #: What the last call cost, for the Studio's per-slide totals.
        self.last_usage: dict = {}
        #: Audio tokens billed so far by this adapter, and whether every one
        #: of them was counted by Google rather than inferred from duration.
        #: Reported as money by `spend_usd`, which the renderer prints after
        #: each slide. A number nobody can see is a number nobody can budget
        #: against -- this engine is the first one here that costs money per
        #: sentence, and a lecture should say what it spent while it spends it.
        self.audio_tokens: float = 0.0
        self.all_metered: bool = True
        self._profile = profile

        if profile is None and not self.voice:
            try:
                from realme.core.env import data_home
                from realme.app.profile import Profile
                profile = Profile(data_home() / "profile")
                self._profile = profile
            except Exception:
                profile = None
        if profile is not None:
            g = (profile.data.get("gemini_voice") or {}) if hasattr(profile, "data") else {}
            self.voice = self.voice or g.get("voice", "")
            self.model = model or g.get("model") or self.model
            if max_chars is None:
                max_chars = g.get("max_chars")
            if rate_match is None:
                rate_match = g.get("rate_match")

        #: A cloned voice speaks at the pace of the clip it was cloned from.
        #: Where that is not the pace you teach at, this is the measured
        #: correction (`realme voice gemini --pace`), not a guess. 1.0 means
        #: nothing is being corrected.
        env_rate = os.environ.get("REALME_GEMINI_RATE")
        if rate_match is None and env_rate:
            rate_match = float(env_rate)
        self.rate_match = float(rate_match) if rate_match else 1.0

        env_max = os.environ.get("REALME_GEMINI_MAX_CHARS")
        if max_chars is None and env_max:
            max_chars = int(env_max)
        #: `None` means "nobody has said", which takes the measured default
        #: above rather than the pipeline's 260. An explicit 0 means "use the
        #: pipeline's chunking after all", which is how you opt out.
        if max_chars is None:
            self.max_chars_per_call = MEASURED_MAX_CHARS
        elif int(max_chars) <= 0:
            self.max_chars_per_call = None
        else:
            self.max_chars_per_call = int(max_chars)

        #: Set by `_watch_pace` after each take; read and cleared by
        #: `last_rush`. One take's verdict must never be read as the next
        #: one's, so it is cleared at the start of every synthesis.
        self._last_rush: float | None = None
        #: Articulation rate of each chunk this adapter has spoken. A hurried
        #: chunk is the one failure here that nothing else would notice: the
        #: text is all present, so no length check fires, and it is only
        #: audible to someone listening to the whole lecture. Measured per
        #: chunk and reported when one is out of line with the rest.
        self._articulation: list[float] = []

    # -- identity ----------------------------------------------------------

    def voice_fingerprint(self) -> str:
        # The voice id AND the model AND the style. The ledger keys on this,
        # and all three change the audio for identical text. Leaving the model
        # out would have served Flash-Lite audio for a Flash render after a
        # one-word config change, silently, and reported a clean run.
        # The rate correction belongs here and not only in the pace argument:
        # `speak.py` puts the requested pace in the key, but the correction is
        # applied inside this adapter, so changing it would otherwise re-serve
        # audio at the old speed and report a clean run.
        return (f"{self.name}:{self.model}:{self.voice}:{self.style}"
                f":r{self.rate_match:.3f}")

    def last_rush(self) -> float | None:
        """See `BaseTTS.last_rush`. Measured, and cleared as it is read."""
        rush, self._last_rush = self._last_rush, None
        return rush

    def voice_note(self) -> str:
        """The line `factory.announce` prints before a render.

        This engine's clone is a row in Google's project, not a file on this
        disk, so the usual reasoning -- "is there a reference clip?" -- gets
        the answer wrong in both directions.
        """
        if not self.voice:
            return ("WARNING: no replicated voice enrolled, so there is "
                    "nothing of yours to speak with. Fix with: "
                    "realme voice gemini --enroll")
        where = ("Google, expiring a year after last use"
                 if self.voice.startswith("voice_")
                 else "Google, expiring 7 days after it was made")
        rate = ("" if abs(self.rate_match - 1.0) < 1e-3 else
                f", retimed to {self.rate_match:.3f} to match your own pace")
        return (f"cloning from {self.voice} on {self.model}, hosted at "
                f"{where}{rate}")

    # -- availability ------------------------------------------------------

    def paid_acknowledged(self) -> bool:
        if os.environ.get("REALME_GEMINI_PAID") == "1":
            return True
        p = self._profile
        if p is not None and hasattr(p, "data"):
            return bool((p.data.get("gemini_voice") or {}).get("paid_tier_ack"))
        return False

    def check(self) -> None:
        """Could this run? No network, no audio charge, no model load."""
        api_key(self._key)
        if self.model not in MODELS:
            raise AdapterUnavailable(
                f"{self.model} is not a Gemini TTS model. Use one of: "
                f"{', '.join(MODELS)}")
        if not self.voice:
            raise AdapterUnavailable(
                "No replicated voice yet. Record the consent sentence and "
                "enrol it:  realme voice gemini --enroll")
        if not self.paid_acknowledged():
            raise AdapterUnavailable(TIER_NOTICE)

    def preflight(self) -> None:
        """Be ready to speak. One cheap GET; it synthesizes nothing."""
        self.check()
        if self.voice.startswith("voicekey_"):
            # A stateless key is not registered server-side, so there is
            # nothing to GET. It either works at the first call or it has
            # expired -- seven days, and the error will say so.
            return
        try:
            get_voice(self.voice, key=self._key)
        except AdapterUnavailable as e:
            raise AdapterUnavailable(
                f"The replicated voice {self.voice} could not be fetched "
                f"({e}). A stored voice expires a year after its last use; "
                f"re-enrol with `realme voice gemini --enroll`.") from e

    # -- speaking ----------------------------------------------------------

    def _style_for(self, pace: float) -> str:
        """
        The style string. Manner only -- rate is handled by retiming.

        Gemini has no numeric rate, and a style string is not reproducible:
        the same words can land differently between calls, which would make
        the render ledger's pace key a lie. So the deterministic part of pace
        is done with ffmpeg afterwards. The one case where the style string
        has to help is a pace so far from 1.0 that retiming would be audible
        as an artefact rather than as a speed.
        """
        parts = [p for p in (self.style,) if p]
        pace = pace * self.rate_match
        if pace <= 0.78:
            parts.append("speaking slowly and deliberately")
        elif pace >= 1.28:
            parts.append("speaking briskly")
        return ", ".join(parts)

    def synthesize(self, text: str, out_wav, voice=None, pace: float = 1.0,
                   language: str | None = None):
        self.check()
        out_wav = Path(out_wav)
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        said = (text or "").strip()
        if not said:
            raise AdapterUnavailable("Nothing to say: the text was empty.")
        self._last_rush = None

        content = {"type": "text", "text": said}
        style = self._style_for(pace)
        if style:
            content["annotations"] = [{"type": "speech_metadata",
                                       "style": style}]
        body = {
            "model": self.model,
            "input": [{"type": "user_input", "content": [content]}],
            "response_format": {"type": "audio", "mime_type": "audio/wav",
                                "sample_rate": SAMPLE_RATE},
            "generation_config": {
                "speech_config": [{"voice": voice or self.voice}]},
        }
        t0 = time.perf_counter()
        resp = _post("interactions", body, api_key(self._key), log=self.log)
        audio = _audio_from(resp)
        usage = _usage_from(resp)
        self.last_usage = {**usage,
                           "wall_s": round(time.perf_counter() - t0, 2),
                           "chars": len(said)}

        # Both shapes are accepted on purpose: the unary default is a RIFF WAV,
        # the streaming default is headerless l16 at the same rate, and an
        # upstream change between them should not produce a file that every
        # later ffmpeg step rejects with "Invalid data found".
        if audio[:4] == b"RIFF":
            out_wav.write_bytes(audio)
        else:
            from realme.core.media import write_wav
            write_wav(out_wav, audio, SAMPLE_RATE)

        from realme.core.media import duration_of, probe, retime
        probe(out_wav)                    # decodable, or we do not pretend
        self._bill(usage, out_wav)
        self._watch_pace(out_wav, said)
        effective = pace * self.rate_match
        if abs(effective - 1.0) >= 1e-3:
            retime(out_wav, effective)
        return out_wav

    def _bill(self, usage: dict, wav: Path) -> None:
        """Add this call to the running total, metered if Google said so."""
        tokens = None
        for field in ("candidates_token_count", "candidatesTokenCount",
                      "output_token_count", "outputTokenCount",
                      "total_token_count", "totalTokenCount"):
            value = usage.get(field)
            if isinstance(value, (int, float)) and value > 0:
                tokens = float(value)
                break
        if tokens is None:
            self.all_metered = False
            try:
                from realme.core.media import duration_of
                tokens = duration_of(wav) * TOKENS_PER_SECOND
            except Exception:
                return
        self.audio_tokens += tokens

    @property
    def spend_usd(self) -> float:
        """What this adapter has spent so far. An estimate either way.

        Metered token counts make it a good one -- the arithmetic is Google's
        published rate against Google's own count -- but it is still not an
        invoice, and nothing that displays it should imply otherwise.
        """
        rate = PRICE_PER_MTOK.get(self.model, 9.0)
        return self.audio_tokens / 1e6 * rate

    #: How much faster than this voice's own median a chunk may be before it
    #: is re-recorded in smaller pieces. The measured spread BETWEEN sizes was
    #: 8-9% and take-to-take variation at one size is a couple of per cent, so
    #: 10% is outside the noise and is the point at which a listener called it
    #: muddy. One number, used both to report and to act -- a warning
    #: threshold and an action threshold that differ is a way to be told about
    #: something nothing then does anything about.
    RUSH_TOLERANCE = 1.10
    #: Chunks needed before a median means anything. The first few of a
    #: lecture cannot be judged, and are not.
    RUSH_MIN_SAMPLES = 5

    def _watch_pace(self, wav: Path, text: str) -> None:
        """Say so when one chunk comes back hurried.

        Chunk size is chosen to avoid this, but the hurry depends on the text
        as well as its length -- a run of short clauses is not a paragraph --
        so it cannot be ruled out by a setting. The alternative to measuring
        is finding it while watching the finished lecture.
        """
        if os.environ.get("REALME_GEMINI_NO_RATE_CHECK") == "1":
            return
        try:
            from realme.enrollment.pace import speech_profile
            rate = speech_profile(wav)["articulation"]
            if rate <= 0:
                return
            if len(self._articulation) >= self.RUSH_MIN_SAMPLES:
                ordered = sorted(self._articulation)
                median = ordered[len(ordered) // 2]
                if median > 0 and rate > median * self.RUSH_TOLERANCE:
                    self._last_rush = rate / median
            # The rate of a take that is about to be thrown away still
            # belongs in the baseline: it is how fast this voice can go, and
            # excluding it would make the median drift toward the calm takes
            # and call ordinary chunks hurried.
            self._articulation.append(rate)
        except Exception:
            pass          # a measurement must never break a render
