"""
RealMe Studio -- local web app.

Runs on your own machine or inside a Colab CPU runtime. Serves on 127.0.0.1 by
default and never opens a tunnel: in Colab, reach it through the built-in port
proxy (`google.colab.output.serve_kernel_port_as_window(8000)`) rather than a
share link, which keeps you clear of the "remote proxies" clause in Colab's
terms of service.
"""
from __future__ import annotations
import json, os, re, shutil, threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel, Field

from realme.core.env import load as _load_env
_load_env()

from realme.app.jobs import JobRunner
from realme.app.profile import Profile
from realme.adapters.base import AdapterUnavailable
from realme.adapters.tts import (REGISTRY as TTS_REGISTRY,
                                 DEVELOPED, NOT_DEVELOPED)
from realme.adapters.script_writer import GeminiScriptWriter, PlaceholderScriptWriter
from realme.core.media import require, MediaError
from realme.core.textio import read_text
from realme.pipeline import lecture as lec
from realme.pipeline import dialogue as dlg
from realme.pipeline import slides as slides_mod
from realme.pipeline import pdfdoc

HERE = Path(__file__).parent
DATA = Path(os.environ.get("REALME_HOME", Path.home() / "RealMeStudio"))
UPLOADS, PROJECTS = DATA / "uploads", DATA / "projects"
for d in (UPLOADS, PROJECTS):
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="RealMe Studio")

#: Keep every finished lecture as its video and text files only, without
#: being asked (see pipeline/lean.py). Off on the desktop: the working audio
#: is what makes a one-sentence fix cost one sentence, so slimming a lecture
#: down is a button you press when it is finished. REALME_LEAN=1 turns it on.
LEAN = os.environ.get("REALME_LEAN") == "1"

MB, GB = 1 << 20, 1 << 30
#: The most one upload may be, by what it is. A file is copied to disk in
#: chunks and refused (413) the moment it passes its cap; it is never held in
#: memory whole. Videos are the large case; text files get a small cap.
UPLOAD_LIMITS = {"video": 2 * GB, "deck": 1 * GB, "audio": 500 * MB,
                 "document": 200 * MB, "text": 20 * MB}
_UPLOAD_WHAT = {"video": "a video", "deck": "a slide deck",
                "audio": "a recording", "document": "a document",
                "text": "a text file"}
#: The largest request accepted at all: the biggest single file plus room
#: for an .srt and the form fields.
MAX_REQUEST_BYTES = UPLOAD_LIMITS["video"] + 64 * MB
_CHUNK = 1 * MB


def _too_big(file, kind: str) -> HTTPException:
    limit = UPLOAD_LIMITS[kind]
    size = f"{limit / GB:.0f} GB" if limit >= GB else f"{limit // MB} MB"
    return HTTPException(
        413, f"{file.filename or 'That file'} is larger than the {size} "
             f"allowed for {_UPLOAD_WHAT[kind]}. Nothing was saved.")


async def save_upload(file: UploadFile, dest: Path, kind: str) -> Path:
    """Copy an upload to `dest` in chunks, refusing it past its size cap.

    Written to a `.part` file and moved into place only when complete, so a
    refused or broken upload never replaces a file that was already there.
    """
    limit = UPLOAD_LIMITS[kind]
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    size = 0
    try:
        with open(part, "wb") as out:
            while True:
                chunk = await file.read(_CHUNK)
                if not chunk:
                    break
                size += len(chunk)
                if size > limit:
                    raise _too_big(file, kind)
                out.write(chunk)
        os.replace(part, dest)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    return dest


async def read_upload(file: UploadFile, kind: str = "text") -> bytes:
    """A small upload's bytes, read in chunks up to its cap."""
    limit, got = UPLOAD_LIMITS[kind], bytearray()
    while True:
        chunk = await file.read(_CHUNK)
        if not chunk:
            return bytes(got)
        got += chunk
        if len(got) > limit:
            raise _too_big(file, kind)


#: A Word file is a zip, and a zip can claim to be small and unpack to
#: gigabytes. Checked on the sizes the archive declares -- which Python's
#: zipfile enforces when reading -- before anything is decompressed.
DOCX_MAX_MEMBER = 50 * MB
DOCX_MAX_TOTAL = 200 * MB
DOCX_MAX_MEMBERS = 2000


def check_docx(path: Path) -> None:
    """Refuse (400) a .docx that is not one, or would unpack too large."""
    import zipfile
    try:
        with zipfile.ZipFile(path) as z:
            infos = z.infolist()
    except (zipfile.BadZipFile, OSError, ValueError):
        raise HTTPException(400, f"{path.name} is not a readable Word file.")
    if len(infos) > DOCX_MAX_MEMBERS:
        raise HTTPException(400, f"{path.name} holds {len(infos)} files; a "
                                 f"Word document does not. Refused.")
    names = {i.filename for i in infos}
    if "word/document.xml" not in names:
        raise HTTPException(400, f"{path.name} has no document text in it "
                                 f"(word/document.xml). Is it a Word file?")
    total = sum(i.file_size for i in infos)
    biggest = max((i.file_size for i in infos), default=0)
    if biggest > DOCX_MAX_MEMBER or total > DOCX_MAX_TOTAL:
        raise HTTPException(
            400, f"{path.name} would unpack to {total / MB:.0f} MB "
                 f"(largest part {biggest / MB:.0f} MB). That is not an "
                 f"ordinary Word document; save it as .txt and try again.")

#: Host names this server answers to. Binding to 127.0.0.1 keeps the network
#: out, but not another web page: a site whose DNS name is re-pointed at
#: 127.0.0.1 ("DNS rebinding") becomes, to the browser, the same site as the
#: Studio, and could read projects and scripts with plain GETs. Such a page
#: still carries ITS name in the Host header, so answering only to loopback
#: names closes it. `serve(host=...)` adds a non-loopback bind address, and
#: REALME_ALLOWED_HOSTS (comma-separated) adds others.
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}
_EXTRA_HOSTS: set[str] = set()


def _host_name(header: str) -> str:
    """The name part of a Host header, lower-cased, port removed."""
    h = header.strip().lower()
    if h.startswith("["):                       # [::1]:8000
        return h[:h.find("]") + 1] if "]" in h else h
    return h.rsplit(":", 1)[0] if h.count(":") == 1 else h


def _host_allowed(header: str) -> bool:
    if os.environ.get("COLAB_RELEASE_TAG"):
        # Colab's port proxy is the only way in, it is signed in to your
        # Google account, and its Host header is its own. Rebinding needs a
        # browser that can reach the runtime's loopback; none can.
        return True
    extra = {h.strip().lower() for h in
             os.environ.get("REALME_ALLOWED_HOSTS", "").split(",") if h.strip()}
    return _host_name(header) in (_LOOPBACK_HOSTS | _EXTRA_HOSTS | extra)


@app.middleware("http")
async def _same_site_only(request: Request, call_next):
    """Refuse state-changing requests that come from another website.

    Multipart POSTs are CORS "simple requests", so any page open in the same
    browser could upload a deck, replace the enrolled voice, or start an
    hours-long render on this server without a preflight. Browsers send
    `Origin` (and `Sec-Fetch-Site`) on those; a request from the Studio's own
    page carries a loopback origin or none at all (curl, the CLI, Colab's
    port proxy), and those are allowed through.
    """
    if not _host_allowed(request.headers.get("host", "")):
        return JSONResponse(
            {"detail": "Unknown Host refused. The Studio answers only to "
                       "127.0.0.1 / localhost; set REALME_ALLOWED_HOSTS to add "
                       "a name."}, status_code=421)
    if request.method in ("POST", "PUT", "DELETE", "PATCH"):
        origin = request.headers.get("origin", "")
        site = request.headers.get("sec-fetch-site", "")
        foreign = (origin and not _origin_ok(origin,
                                             request.headers.get("host", ""))) \
            or site == "cross-site"
        if foreign:
            return JSONResponse({"detail": "Cross-site request refused"},
                                status_code=403)
        # Refused before the body is read: Starlette spools a multipart body
        # to disk in full before any route runs, so a per-file cap alone
        # would still let one request fill the disk.
        try:
            length = int(request.headers.get("content-length") or 0)
        except ValueError:
            length = 0
        if length > MAX_REQUEST_BYTES:
            return JSONResponse(
                {"detail": f"That upload is {length / GB:.1f} GB; the Studio "
                           f"accepts at most {MAX_REQUEST_BYTES / GB:.0f} GB "
                           f"in one request."}, status_code=413)
    return await call_next(request)


from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler


@app.exception_handler(RequestValidationError)
async def _bad_request_body(request: Request, exc: RequestValidationError):
    """A body that is not JSON at all is a 400 with a sentence; a field of
    the wrong type stays FastAPI's 422, which names the field."""
    if any(e.get("type") == "json_invalid" for e in exc.errors()):
        return JSONResponse({"detail": "The request body is not valid JSON."},
                            status_code=400)
    return await request_validation_exception_handler(request, exc)


def _netloc(scheme: str, netloc: str) -> tuple[str, int] | None:
    """(host, port) of a scheme://netloc, ports made explicit; None if bad."""
    from urllib.parse import urlsplit
    try:
        u = urlsplit(f"{scheme}://{netloc}")
        port = u.port               # ValueError for 80001 or a non-number
    except ValueError:
        return None
    if not u.hostname or u.username or u.password or u.path or u.query \
            or u.fragment:
        return None
    return u.hostname.lower(), port or {"http": 80, "https": 443}[scheme]


def _origin_ok(origin: str, host_header: str) -> bool:
    """Is `origin` this Studio's own page?

    Parsed and compared whole -- scheme, host and port -- against the Host
    the request was sent to. A prefix test would accept `http://127.0.0.1:
    8000.evil.example` and `http://localhost:80001` as "local".
    """
    from urllib.parse import urlsplit
    try:
        o = urlsplit(origin.strip())
    except ValueError:
        return False
    if o.scheme not in ("http", "https") or o.path not in ("", "/") \
            or o.query or o.fragment:
        return False
    mine = _netloc(o.scheme, o.netloc)
    here = _netloc(o.scheme, host_header.strip())
    return mine is not None and mine == here and _host_allowed(host_header)
runner = JobRunner(state_path=DATA / "jobs.json")
profile = Profile(DATA / "profile")

# TTS price per million characters, for the pre-render estimate.
#: Dollars per million CHARACTERS of narration -- the unit
#: `lec.estimate_cost` divides by. Local engines are absent and cost nothing.
#:
#: gemini-tts is billed per audio token, not per character, so it has to be
#: converted: $9 per million audio tokens at ~32 tokens a second is $1.04 an
#: hour of speech, and this prose runs at about 15 characters a second
#: (measured on the probe's own calibration clips), so a million characters is
#: 18.5 hours and about $19.30. Both halves of the Google price double on
#: 1 January 2027.
PRICE = {"espeak": 0.0, "google_chirp3": 60.0, "elevenlabs": 165.0,
         "chatterbox": 0.0, "gemini-tts": 19.30}


#: What a slide added by a revision carries until narration is written for it.
#: Named rather than typed twice: the revision writes it and the drafting
#: endpoint looks for it, and two spellings of the same sentinel would leave
#: slides that can never be drafted.
NEW_SLIDE_PLACEHOLDER = "(new slide - write the narration)"


def make_writer(name: str, model: str):
    from realme.adapters.script_writer_more import WRITERS, model_for
    if name == "placeholder":
        return PlaceholderScriptWriter()
    # One form field carries the model for every writer. An id from another
    # company's list (the page had not reloaded it yet) is never sent; the
    # writer's own default is used instead.
    if name == "gemini":
        return GeminiScriptWriter(model=model_for("gemini", model) or GEMINI_DEFAULT)
    if name in WRITERS:
        return WRITERS[name](model=model_for(name, model))
    raise HTTPException(400, f"Unknown script writer '{name}'")


from realme.adapters.script_writer import DEFAULT_MODEL as GEMINI_DEFAULT
from realme.pipeline import pending


def build_tts(name: str, **overrides):
    """
    Construct an adapter with everything the profile knows about your voice.

    Delegates to the shared factory, so every caller constructs adapters
    the same way; see realme/adapters/factory.py.
    """
    from realme.adapters.factory import build
    from realme.adapters.base import AdapterUnavailable
    try:
        return build(name, profile=profile, **overrides)
    except AdapterUnavailable as e:
        raise HTTPException(400, str(e))


# One adapter per engine, shared by every preview and render, and one lock
# around synthesis. A fresh `build_tts()` per request would mean a full model
# load for every preview click and, on the PyTorch path, a worker process
# left behind each time; the lock keeps two renders from driving two DLL
# contexts at once.
_TTS_CACHE: dict[str, object] = {}
_TTS_LOCK = threading.Lock()
SYNTH_LOCK = threading.RLock()


class _Serialised:
    """Wraps an adapter so `synthesize` runs one call at a time."""

    def __init__(self, inner):
        self._inner = inner

    def synthesize(self, *a, **kw):
        with SYNTH_LOCK:
            return self._inner.synthesize(*a, **kw)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def make_tts(name: str, **overrides):
    """The adapter for this engine, built once per distinct configuration.

    `overrides` is part of the key: "piper with Amy" and "piper with Ryan"
    are different engines.
    """
    key = name + "|" + repr(sorted(overrides.items()))
    with _TTS_LOCK:
        hit = _TTS_CACHE.get(key)
        if hit is not None:
            return hit
        tts = _Serialised(build_tts(name, **overrides))
        _TTS_CACHE[key] = tts
        return tts


def engine_for(spec: str, *, pace: float | None = None, max_spend=None):
    """The adapter for "engine" or "engine:voice", from the shared cache.

    Every path that speaks goes through here, so the engine warmed up in
    the background is reused. Calling `factory.build` directly would load a
    second copy of a 2 GB model and keep both in memory.
    """
    from realme.pipeline import casting
    name, voice_file = casting.split_spec(spec)
    overrides = casting.voice_overrides(
        name, voice_file, pace if name in casting.PIPER_ENGINES else None)
    # Wrapped per call, not in the cache: the cap counts from now, so it is
    # per render even though the engine underneath is shared; and the saved
    # speed is read now, so a change on the page applies to the next render.
    from realme.pipeline.speak import spend_capped
    tts = casting.with_voice_speed(make_tts(name, **overrides), spec,
                                   profile.data)
    try:
        return spend_capped(tts, max_spend)
    except ValueError as e:
        raise HTTPException(400, str(e))


class VoiceSpeed(BaseModel):
    voice: str = Field(..., description="The voice as the page names it: "
                       "qwen3cpp, gemini-tts, piper:en_US-amy-medium")
    percent: float = Field(0.0, ge=-30, le=30, description="Faster (+) or "
                           "slower (-) than the voice's own speed, in percent. "
                           "0 removes the setting.")


class TranscriptBody(BaseModel):
    text: str = Field(..., description="The exact words read in the voice "
                      "recording")


@app.post("/api/profile/transcript")
def profile_transcript(body: TranscriptBody):
    """The words read in the voice recording, typed or pasted in Twin Setup.

    The local clone and *Measure and match the pace* both need them. Saved
    against the recording that is enrolled now, so a later recording is
    known not to match it.
    """
    text = " ".join(body.text.split())
    if len(text.split()) < 5:
        raise HTTPException(400, "Paste the words you read in the recording "
                                 "-- a few sentences, word for word.")
    if not profile.data.get("voice_reference"):
        raise HTTPException(400, "Choose your voice recording first, then "
                                 "save the words you read in it.")
    profile.set_reference_text(text)
    forget_tts()
    return {"ok": True, "words": len(text.split())}


@app.get("/api/voice-speeds")
def voice_speeds_get():
    """Every saved voice speed, as percent faster (+) or slower (-)."""
    saved = profile.data.get("voice_speeds") or {}
    return {"speeds": {k: round((float(v) - 1.0) * 100, 1)
                       for k, v in saved.items()}}


@app.post("/api/voice-speeds")
def voice_speeds_set(req: VoiceSpeed):
    """Save one voice's speed. It applies to every tab, from the next render
    or preview, and re-records nothing: the stretch is applied to takes
    already on disk."""
    from realme.pipeline import casting
    key = casting.voice_key(req.voice.strip())
    speeds = dict(profile.data.get("voice_speeds") or {})
    if abs(req.percent) < 0.05:
        speeds.pop(key, None)
    else:
        speeds[key] = round(1.0 + req.percent / 100.0, 4)
    profile.data["voice_speeds"] = speeds
    profile.save()
    return voice_speeds_get() | {"voice": key}


def _local_clone_state() -> dict:
    """The free local clone's two settings, for Twin Setup: which model,
    and how many sentences per pass -- each with where its value came from."""
    from realme.adapters.tts_qwen3_dll import (
        DEFAULT_BATCH, DEFAULTED, QUANT_ENV, VOCODER_DEFAULTS, batch_built_at,
        batch_measured, batch_stale, library_path, quant_setting, vocoder_setting)
    from realme.core.env import from_shell
    from realme.engines import cpp as C
    from realme.pipeline.prerecord import BATCH_ENV, MAX_BATCH
    home = Path(C.cpp_home())
    raw = os.environ.get(BATCH_ENV, "").strip()
    try:
        batch = max(1, min(MAX_BATCH, int(raw))) if raw else "auto"
    except ValueError:
        batch = "auto"
    # The local voice lectures use (realme-rev-qwen3-tts): its pass size,
    # measured here or its default, and where that came from.
    rev = {}
    try:
        from realme.engines.rev import NAME as REV, engine_home, batch_default as rev_batch
        from realme.enrollment.choice import local_clone_installed
        from realme.adapters.tts_qwen3_dll import _batch_saved
        if local_clone_installed():
            rh = engine_home()
            got = rev_batch(rh)
            measured_rev = (_batch_saved().get(REV) or (0, ""))[0]
            rev = {"name": REV, "batch": got,
                   "source": "measured" if measured_rev and got == measured_rev else "default"}
    except Exception:
        rev = {}
    return {
        "local_voice": rev,
        "installed": library_path(home).is_file()
                     and (home / "models" / C.F16_FILE).is_file(),
        "quant": quant_setting(),
        "q8_ready": (home / C.Q8_DIR / C.Q8_FILE).is_file(),
        "batch": batch,
        "batches_built": batch_built_at(home),
        "measured": batch_measured(),
        "stale": batch_stale(),
        "defaults": dict(DEFAULT_BATCH),
        "vocoder": vocoder_setting(),
        "shell": [k for k in (QUANT_ENV, BATCH_ENV, *VOCODER_DEFAULTS)
                  if from_shell(k) and k not in DEFAULTED],
    }


@app.get("/api/local-clone")
def local_clone_get():
    return _local_clone_state()


@app.post("/api/local-clone")
async def local_clone_set(patch: dict):
    """Save the local clone's model and sentences per pass, to the same
    settings file `realme key` writes, so the Studio, the launchers and the
    command line agree. Takes effect from the next preview or render."""
    from realme.adapters.tts_qwen3_dll import QUANT_ENV
    from realme.core.env import from_shell, remove_key, set_key
    from realme.pipeline.prerecord import BATCH_ENV
    if any(j["status"] in ("queued", "running") for j in runner.recent(200)):
        # The engine in use is closed and its model changed by this; a
        # render half in one model and half in another is not wanted.
        raise HTTPException(409, "A render is running; change this when it "
                                 "has finished.")
    if "quant" in patch:
        q = str(patch["quant"])
        if q not in ("f16", "q8_0"):
            raise HTTPException(400, "The model is f16 or q8_0.")
        if from_shell(QUANT_ENV):
            raise HTTPException(409, f"{QUANT_ENV} is set outside RealMe; "
                                     f"unset it there to choose here.")
        # Saved either way: with no setting, the 8-bit model is used where
        # it has been made, so "16-bit" has to be said to stick.
        set_key(QUANT_ENV, q)
    if "batch" in patch:
        b = patch["batch"]
        if from_shell(BATCH_ENV):
            raise HTTPException(409, f"{BATCH_ENV} is set outside RealMe; "
                                     f"unset it there to choose here.")
        if b in (None, "", "auto"):
            remove_key(BATCH_ENV)
        else:
            try:
                n = int(b)
            except (TypeError, ValueError):
                raise HTTPException(400, "Sentences per pass: auto, or 1 to 16.")
            if not 1 <= n <= 16:
                raise HTTPException(400, "Sentences per pass: auto, or 1 to 16.")
            set_key(BATCH_ENV, str(n))
    if "vocoder" in patch:
        from realme.adapters.tts_qwen3_dll import DEFAULTED, VOCODER_DEFAULTS, vocoder_defaults
        v = str(patch["vocoder"])
        if v not in ("fast", "original"):
            raise HTTPException(400, "The vocoder is fast or original.")
        busy = [k for k in VOCODER_DEFAULTS if from_shell(k) and k not in DEFAULTED]
        if busy:
            raise HTTPException(409, f"{' and '.join(busy)} set outside RealMe; "
                                     f"unset it there to choose here.")
        for k, on in vocoder_defaults().items():
            if v == "fast":
                remove_key(k)
                if not os.environ.get(k):
                    os.environ[k] = on
                    DEFAULTED.add(k)
            else:
                if k in DEFAULTED:
                    os.environ.pop(k, None)
                    DEFAULTED.discard(k)
                set_key(k, "0")
    # A loaded engine holds the other model; the next one loads this one.
    forget_tts()
    return _local_clone_state()


#: The voice-engine settings "Automatic defaults" takes back to RealMe's own
#: choices for this computer: the model, passes and how they are filled,
#: threads and engine copies, and the engine's switches. Nothing else in the
#: settings file (keys, voices, the Studio's own settings) is touched.
ENGINE_KEYS = ("REALME_QWEN3CPP_QUANT", "REALME_RECORD_BATCH", "REALME_RECORD_STREAM",
               "REALME_RECORD_LOOKAHEAD", "REALME_RECORD_WORKERS", "REALME_ENGINE_THREADS",
               "QWEN3_TTS_THREADS", "QWEN3_TTS_FAST_SNAKE", "QWEN3_TTS_VOCODER_F32",
               "QWEN3_TTS_VOCODER_F32_MAX_FRAMES", "QWEN3_TTS_Q8_BATCH", "QWEN3_TTS_SGEMM",
               "QWEN3_TTS_SGEMM_ALL", "QWEN3_TTS_FFN_F32", "QWEN3_TTS_BROADCAST",
               "QWEN3_TTS_LOW_MEM",
               # realme-rev-qwen3-tts's switches and tests (its own settings
               # are in engines/rev.py and tools/realme-rev-qwen3-tts/local).
               "QWEN3_TTS_CP_STEPS", "QWEN3_TTS_CP_FILL", "QWEN3_TTS_VOCODER_CODEBOOKS",
               "QWEN3_TTS_VOCODER_Q8", "QWEN3_TTS_VOCODER_BACKEND", "QWEN3_TTS_Q8_SHARE",
               "QWEN3_TTS_Q8_VNNI", "QWEN3_TTS_DECODE_ASYNC", "QWEN3_TTS_DECODE_THREADS",
               "QWEN3_TTS_CP_BATCH_ATTN", "QWEN3_TTS_CP_GRAPH_REUSE",
               "QWEN3_TTS_TALK_GRAPH_REUSE", "QWEN3_TTS_STAGE_TIMING", "QWEN3_TTS_CACHE_GROW",
               "QWEN3_TTS_WEIGHT_MMAP", "QWEN3_TTS_TEXT_MMAP", "QWEN3_TTS_ATTN_JOIN",
               "QWEN3_TTS_TOPK_SELECT", "REALME_TAKE_CHECK", "REALME_RUNAWAY_CHECK")
#: The measurement started from Twin Setup, while it runs: its job id.
MEASURING: dict = {"job": None}


def _refuse_while_measuring() -> None:
    """Nothing records while the voice engine is being measured: it would
    slow both and spoil the measurement."""
    jid = MEASURING.get("job")
    job = runner.get(jid) if jid else None
    if job is not None and job.status in ("queued", "running"):
        raise HTTPException(409, "The voice engine is being measured (Twin Setup, "
                                 "Measure this computer); try again when it has finished.")


def reset_engine_settings() -> tuple[list[str], list[str]]:
    """Take ENGINE_KEYS out of the settings file and this process, then set
    RealMe's defaults for the engine as built now. Returns (cleared, kept):
    kept are set outside RealMe (in the shell), which only the user can
    unset."""
    from realme.adapters.tts_qwen3_dll import DEFAULTED, apply_vocoder_defaults
    from realme.core.env import data_home, from_shell, parse, remove_key
    envf = data_home() / ".env"
    saved = parse(envf) if envf.is_file() else {}
    cleared, kept = [], []
    for k in ENGINE_KEYS:
        if from_shell(k) and k not in DEFAULTED:
            kept.append(k)
            continue
        if k in saved:
            cleared.append(k)
        remove_key(k)
        if k in DEFAULTED:
            os.environ.pop(k, None)
            DEFAULTED.discard(k)
    # The defaults of the engine built now (a rebuild may have added a
    # faster vocoder), for this process and the measurement it starts.
    apply_vocoder_defaults()
    forget_tts()
    return cleared, kept


@app.post("/api/local-clone/automatic")
def local_clone_automatic():
    """Automatic defaults, at once: every voice-engine setting back to
    RealMe's own for this computer -- the model, the sentences per pass
    (measured on this engine if it has been, else the default, lowered where
    memory is short), the newest vocoder the engine was built with, threads
    and engine copies by the cores. Measuring is a separate, optional step
    (/api/local-clone/measure)."""
    if any(j["status"] in ("queued", "running") for j in runner.recent(200)):
        raise HTTPException(409, "Something is running; change this when it has finished.")
    from realme.engines import cpp as C
    from realme.enrollment.choice import local_clone_installed
    if not (C.cpp_home() / "build_info.json").is_file() and not local_clone_installed():
        raise HTTPException(409, "The free local clone is not installed here.")
    cleared, kept = reset_engine_settings()
    return {**_local_clone_state(), "cleared": cleared, "kept": kept}


@app.post("/api/local-clone/measure")
def local_clone_measure():
    """Measure how many sentences per pass suit this computer and its engine
    as built (`realme bench-batch`, in its own process, as a job). Optional:
    the defaults are used until it has been done. About 15 minutes."""
    import subprocess
    import sys
    if any(j["status"] in ("queued", "running") for j in runner.recent(200)):
        raise HTTPException(409, "Something is running; start this when it has finished, "
                                 "as it measures how fast this computer records.")
    from realme.adapters.tts_qwen3_dll import batch_built_at
    from realme.engines import cpp as C
    from realme.enrollment.choice import local_clone_installed
    rev_here = local_clone_installed()        # realme-rev-qwen3-tts: what bench-batch measures
    if not rev_here and not (C.cpp_home() / "build_info.json").is_file():
        raise HTTPException(409, "The free local clone is not installed here.")
    if not rev_here and not batch_built_at(C.cpp_home()):
        raise HTTPException(409, "This engine records one sentence at a time; there is "
                                 "nothing to measure. Run _Rebuild_Voice_Engine.bat to "
                                 "record several per pass.")
    forget_tts()

    def work(job):
        job.say("Measuring how many sentences per pass suit this computer "
                "(about 15 minutes; leave the computer alone) ...")
        root = Path(__file__).resolve().parents[2]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["PYTHONIOENCODING"] = "utf-8"
        # No console window of its own on Windows: its lines come here.
        quiet = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if os.name == "nt" else {}
        proc = subprocess.Popen([sys.executable, "-m", "realme", "bench-batch"], cwd=str(root),
                                env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", **quiet)
        sizes = 6
        done = 0
        for line in proc.stdout:
            line = line.rstrip()
            if not line.strip():
                continue
            job.say(line.strip())
            if " per pass:" in line:
                done += 1
                job.progress = min(0.95, 0.1 + 0.85 * (done - 1) / sizes)
            elif "identical" in line or "warming up" in line:
                job.progress = max(job.progress, 0.05)
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError("The measurement did not finish (see the lines above); "
                               "the defaults stay in use.")
        forget_tts()
        st = _local_clone_state()
        lv = st.get("local_voice") or {}
        return {"measured": st.get("measured"),
                "batch_auto": lv.get("batch") or (st.get("measured") or {}).get(st.get("quant"))}

    job = runner.submit("engine", "Measure this computer — voice engine", work)
    MEASURING["job"] = job.id
    return {"job_id": job.id}


#: What the background warm-up is doing. Read by /api/doctor so the page can
#: say "loading" instead of looking idle while 2 GB comes off disk.
WARM: dict = {"engine": "", "state": "idle", "detail": "", "seconds": 0.0}


def warm_engine(name: str = "", *, block: bool = False) -> None:
    """Load the speech engine now, in the background, into the shared cache.

    The model load takes tens of seconds. Doing it now, while the user
    uploads a deck and reads the draft, keeps it off the first click of
    Preview or Render.

    Failure here is not fatal and is deliberately not raised: the engine is
    reported by /api/doctor as usual, and the first render will
    raise the same AdapterUnavailable with the same advice. Set
    REALME_NO_WARMUP=1 to keep the memory until it is asked for.
    """
    import time
    from realme.adapters.tts import DEVELOPED as _DEV
    # REALME_NATIVE_WARMUP_POLICY
    automatic = not name
    name = name or profile.data.get("adapters", {}).get("tts") or _DEV[0]
    if automatic and name in ('qwen3cpp', 'realme-rev-qwen3-tts'):
        # A lecture can select a different build from the global default.
        # Loading one at startup prevents Windows from loading the other pair.
        WARM.update(engine=name, state='idle', detail='loads on first use', seconds=0.0)
        return
    # END_REALME_NATIVE_WARMUP_POLICY
    try:
        _refuse_while_measuring()
    except HTTPException:
        return                  # the measurement has the processor; load later

    def run() -> None:
        WARM.update(engine=name, state="loading", detail="", seconds=0.0)
        t0 = time.perf_counter()
        try:
            tts = engine_for(name)          # the cache every render reads
            with SYNTH_LOCK:
                tts.preflight()
                # The model load is the visible cost; extracting the speaker
                # embedding from the enrolled clip is the larger one -- over
                # a minute on CPU the first time. Both belong before the
                # first click, not after it.
                warm = getattr(tts, "warm_voice", None)
                if callable(warm):
                    WARM.update(state="loading", detail="cloning your voice")
                    warm()
            WARM.update(state="ready", detail="",
                        seconds=round(time.perf_counter() - t0, 1))
        except Exception as e:
            # Drop it from the cache: a half-built adapter must not be handed
            # to a render that would then fail in a less obvious place.
            with _TTS_LOCK:
                _TTS_CACHE.pop(name, None)
            WARM.update(state="failed", detail=f"{type(e).__name__}: {e}"[:300],
                        seconds=round(time.perf_counter() - t0, 1))

    if block:
        run()
    else:
        threading.Thread(target=run, daemon=True, name="warm-engine").start()


def forget_tts() -> None:
    """Drop cached adapters. Called when the enrolled voice changes, so the
    next request builds against the new reference rather than the old one."""
    with _TTS_LOCK:
        old = list(_TTS_CACHE.values())
        _TTS_CACHE.clear()
    WARM.update(state="idle", detail="", seconds=0.0)
    for t in old:
        close = getattr(t, "close", None)
        if callable(close):
            try:
                # Not in the middle of someone's sentence: closing frees the
                # engine the call is using.
                with SYNTH_LOCK:
                    close()
            except Exception:
                pass


import contextlib


@contextlib.contextmanager
def _env_scoped(values: dict[str, str | None]):
    """Set environment variables for a block and restore them after."""
    saved = {k: os.environ.get(k) for k in values}
    try:
        for k, v in values.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


_PID_OK = re.compile(r"[^A-Za-z0-9_\-]")


def project_id_from(filename: str, fallback: str = "lecture") -> str:
    """A project id that is safe as a folder name AND as a URL path segment.

    Every character outside a safe set is replaced, not only spaces:
    `Lecture #3.pdf` would otherwise become a folder the UI asks for as
    `/api/lecture/Lecture_`, which 404s.
    """
    stem = Path(re.split(r"[\\/]", filename or "")[-1]).stem
    pid = _PID_OK.sub("_", stem)
    while "__" in pid:
        pid = pid.replace("__", "_")
    pid = pid.strip("_")[:60]
    return pid or fallback


def safe_filename(name: str | None, fallback: str) -> str:
    """The base name of an upload, with nothing that can leave the folder.

    Starlette passes `filename` through untouched, so `../../x.pdf` would
    otherwise be joined to the project directory as written.
    """
    # Split on both separators: on Linux (Colab) a backslash is not one,
    # and the name would keep it.
    base = re.split(r"[\\/]", name or "")[-1]
    base = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", base).strip(" .")
    return base or fallback


def project_dir(pid: str) -> Path:
    root = PROJECTS.resolve()
    p = (root / pid).resolve()
    # `startswith` would accept a sibling folder that merely shares the
    # prefix (`projects_x`); is_relative_to does not.
    if p == root or not p.is_relative_to(root) or not pid or "/" in pid or "\\" in pid:
        raise HTTPException(400, "Bad project id")
    return p


def new_project(filename: str) -> tuple[str, Path, str]:
    """A folder for a NEW lecture made from `filename`: (id, dir, note).

    Never an existing project's folder, so drafting or importing `Week3.pdf`
    a second time cannot overwrite the approved narration in the `Week3`
    project already there. The deck's name is kept and numbered
    instead (`Week3_2`, `Week3_3`, ...); revising an existing lecture is
    what the Revise path is for. `note` says so when it happens.

    The folder is created here, with no await between the check and the
    mkdir, so two uploads of the same deck cannot both claim one id.
    """
    base = project_id_from(filename)
    pid, n = base, 2
    while True:
        pdir = project_dir(pid)
        try:
            pdir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            if pdir.is_dir() and not any(pdir.iterdir()):
                break                       # an empty folder holds nothing
        pid, n = f"{base[:55]}_{n}", n + 1
    note = "" if pid == base else (
        f"A lecture called '{base}' already exists and was left as it is; "
        f"this one is '{pid}'.")
    return pid, pdir, note


async def _save_into_new(file: UploadFile, dest: Path, kind: str,
                         pdir: Path) -> None:
    """`save_upload` into a folder `new_project` just made; on failure the
    folder is removed again, so a refused upload does not leave a project
    that takes the name."""
    try:
        await save_upload(file, dest, kind)
    except BaseException:
        shutil.rmtree(pdir, ignore_errors=True)
        raise


# ------------------------------------------------------------------ pages

@app.get("/", response_class=HTMLResponse)
def index():
    return (HERE / "ui.html").read_text(encoding="utf-8")


# ------------------------------------------------------------------ status

@app.get("/api/doctor")
def doctor():
    # Re-read .env on every check, not once at import.
    #
    # A key saved after startup then shows up on a page refresh. Reading a
    # small file per status call is cheap.
    _load_env()

    from realme.pipeline import pdfdoc
    needed = ["ffmpeg", "ffprobe"] + ([] if pdfdoc.HAVE_MUPDF else ["pdftoppm"])
    bins = {}
    for b in needed:
        try:
            require(b); bins[b] = True
        except MediaError:
            bins[b] = False

    writers = {}
    from realme.adapters.script_writer_more import WRITERS as _MORE
    for nm, w in (("gemini", GeminiScriptWriter(model=GEMINI_DEFAULT)),
                  *((k, c()) for k, c in _MORE.items()),
                  ("placeholder", PlaceholderScriptWriter())):
        try:
            w.preflight(); writers[nm] = {"ok": True, "detail": "ready"}
        except AdapterUnavailable as e:
            writers[nm] = {"ok": False, "detail": str(e)}

    # Report where the key was looked for: the three paths
    # env.candidates() searches, in order, with what each holds.
    from realme.core.env import candidates, data_home
    key_search = [{"path": str(c), "exists": c.is_file(),
                   "has_gemini": bool(c.is_file() and "GEMINI_API_KEY" in
                                      c.read_text(encoding="utf-8-sig",
                                                  errors="ignore"))}
                  for c in candidates()]

    voices = {}
    have_ref = bool(profile.data.get("voice_reference"))
    # Only the engines this project actually sets up, and asked the cheap
    # question. `preflight()` is not used: for qwen3cpp it IS the 2 GB model
    # load, and for other registry entries it would try to import torch.
    for nm in DEVELOPED:
        cls = TTS_REGISTRY[nm]
        try:
            inst = build_tts(nm)
            inst.check()
            # `clone` is a capability; `cloning_you` is whether it is actually
            # wired to your recording. qwen3cpp preflights fine with no
            # reference at all -- it just synthesises in the default speaker,
            # which is not what "cloning" implies on a status pill.
            voices[nm] = {"ok": True, "detail": "ready",
                          "clone": inst.is_voice_clone,
                          # Gemini's check() already requires an enrolled
                          # voice, so ready there means it is yours.
                          "cloning_you": bool(inst.is_voice_clone and
                                              (have_ref or nm == "gemini-tts")),
                          "placeholder": inst.is_placeholder}
        except Exception as e:
            voices[nm] = {"ok": False, "detail": str(e)[:1200],
                          "clone": getattr(cls, "is_voice_clone", False),
                          "cloning_you": False,
                          "placeholder": getattr(cls, "is_placeholder", False)}
    from realme.core import textio
    return {"binaries": bins, "writers": writers, "voices": voices,
            # Named, not probed. They stay selectable from the command line,
            # but get no status pill: this project neither installs nor
            # tests them.
            "other_engines": NOT_DEVELOPED,
            "engine_warm": dict(WARM),
            "pdf_backend": pdfdoc.backend(),
            # Files this session found written in a legacy Windows encoding
            # and rewrote as utf-8. Reported here because the repair leaves
            # no other trace, and it explains a slow first open of an older
            # project.
            "repaired_encoding": [Path(f).name for f in textio.REPAIRED[-8:]],
            "key_search": key_search,
            "key_write_to": str(data_home() / ".env"),
            "app_dir": str(Path(__file__).resolve().parents[2]),
            "data_dir": str(DATA), "profile_ready": profile.ready_for_cloning}


@app.get("/api/engine/warm")
def engine_warm():
    """Just the warm-up state. Cheap enough to poll while it loads."""
    return dict(WARM)


@app.post("/api/engine/warm")
def engine_warm_start(name: str = Form("")):
    """Load the engine now. Used by the button that appears when it failed."""
    if WARM.get("state") != "loading":
        warm_engine(name)
    return dict(WARM)


@app.get("/api/profile")
def get_profile():
    d = dict(profile.data)
    d["ready_for_cloning"] = profile.ready_for_cloning
    return d


@app.post("/api/profile")
async def set_profile(patch: dict):
    from realme.app.profile import validate_patch
    try:
        validate_patch(patch)
    except ValueError as e:
        raise HTTPException(400, str(e))
    profile.update(patch)
    return get_profile()


@app.get("/api/lexicon")
def lexicon_list():
    """Your pronunciations, and the ones that ship with the project.

    Both, and marked, because only the user's own survive an update; a
    built-in entry may change in the next release. User entries are editable
    here; the built-ins are shown so a term already handled is not added
    again.
    """
    from realme.text.lexicon import (Lexicon, load_user_entries,
                                     user_lexicon_path)
    mine = load_user_entries()
    names = {e.term for e in mine}
    builtin = [e for e in Lexicon().entries.values() if e.term not in names]

    def row(e, own):
        return {"term": e.term,
                "respelling": e.respelling, "ipa": e.ipa,
                "arpabet": e.arpabet, "note": e.note, "mine": own,
                # No pronunciation at all is not an empty entry: it is the
                # "stop spelling this one out" case, which is a real and
                # common reason to add a term.
                "spell_only": not (e.respelling or e.ipa or e.arpabet)}

    p = user_lexicon_path()
    return {"entries": [row(e, True) for e in
                        sorted(mine, key=lambda e: e.term.lower())]
                       + [row(e, False) for e in
                          sorted(builtin, key=lambda e: e.term.lower())],
            "path": str(p), "exists": p.is_file(), "mine": len(mine)}


@app.post("/api/lexicon")
async def lexicon_save(body: dict):
    """Add or replace one of your entries. Same file the CLI writes."""
    from realme.text.lexicon import (Entry, load_user_entries,
                                     save_user_entries, user_lexicon_path)
    term = (body.get("term") or "").strip()
    if not term:
        return JSONResponse({"error": "a term is required"}, status_code=400)
    respelling = (body.get("respelling") or "").strip()
    ipa = (body.get("ipa") or "").strip()
    note = (body.get("note") or "").strip()
    entries = [e for e in load_user_entries() if e.term != term]
    entries.append(Entry(term=term, respelling=respelling, ipa=ipa,
                         note=note, case_sensitive=True))
    save_user_entries(entries)
    # Say what it will do, in the same words the CLI uses, rather than only
    # "saved": a blank pronunciation is a deliberate choice with a different
    # effect, and the page should not make the two look identical.
    did = (f"{term} -> {respelling or ipa}" if (respelling or ipa)
           else f"{term} will no longer be spelled out letter by letter")
    return {"ok": True, "saved": did, "path": str(user_lexicon_path())}


@app.delete("/api/lexicon")
def lexicon_remove_query(term: str = ""):
    """Remove one of yours, named in the query: `?term=and/or`.

    The form the page uses. A term may contain "/" ("and/or", "w/o"), and
    in the path form below a slash -- even sent as %2F -- splits the term
    into two segments, so the path form cannot remove such an entry.
    """
    if not term.strip():
        raise HTTPException(400, "Say which term to remove: ?term=...")
    return lexicon_remove(term)


@app.delete("/api/lexicon/{term}")
def lexicon_remove(term: str):
    """Remove one of yours. A built-in of the same name applies again."""
    from realme.text.lexicon import (load_user_entries, save_user_entries,
                                     Lexicon)
    if not any(e.term == term for e in load_user_entries()):
        raise HTTPException(404, f"'{term}' is not one of your entries.")
    entries = [e for e in load_user_entries() if e.term != term]
    save_user_entries(entries)
    back = any(e.term == term for e in Lexicon().entries.values())
    return {"ok": True,
            "note": (f"{term} removed; the built-in entry applies again"
                     if back else f"{term} removed")}


@app.get("/api/lexicon/check")
def lexicon_check():
    """The same validation `realme lexicon check` runs."""
    from realme.text.lexicon import Lexicon
    lx = Lexicon()
    probs = lx.check()
    return {"problems": probs, "entries": len(lx.entries)}


@app.get("/api/voices/draft")
def draft_voices():
    """
    The piper voices available to iterate against, and which one is chosen.

    Piper is the DRAFT voice: you use it to get the words, the pacing and the
    slide timing right, cheaply, and then render for real with the cloned
    engine. Which voice is more comfortable to work against is taste, so it is
    a switch rather than a decision made for you.
    """
    from realme.adapters.tts import DRAFT_VOICES, draft_voice
    from realme.engines.install import engine_home, piper_home, piper_status
    home = engine_home()
    where = piper_home(home)
    current = draft_voice()
    # `piper_status`, not `piper_voice_file(...).is_file()`.
    #
    # A piper voice is the .onnx AND its .onnx.json. The loader checks the
    # pair, so this endpoint must too, or a voice with a missing .json shows
    # as installed and then fails at load.
    voices = [{"name": n, "label": label,
               "installed": piper_status(home, n)["voice_present"]}
              for n, label in DRAFT_VOICES.items()]
    # Where it looked, always -- not only on failure. "Not installed" is a
    # claim about a folder, and a claim about a folder should name it.
    return {"current": current, "voices": voices,
            "folder": str(where), "folder_exists": Path(where).is_dir(),
            "present": sorted(f.name for f in Path(where).glob("*.onnx"))
                       if Path(where).is_dir() else [],
            "realme_tools": os.environ.get("REALME_TOOLS") or ""}


@app.get("/api/models")
def api_models(writer: str = "gemini"):
    """
    Which Gemini models this key can call -- asked, not remembered.

    A hardcoded list goes out of date whenever Google retires a preview, and
    a retired model fails as a bare HTTP 404 that reads like a broken
    endpoint.

    So the page asks. If the key is missing or the call fails, the configured
    default is offered alone, with the reason.
    """
    from realme.adapters.script_writer import list_models, DEFAULT_MODEL
    from realme.adapters.script_writer_more import WRITERS, models_for
    if writer in WRITERS:
        # Claude and OpenAI: the same idea, asked of that company.
        return models_for(writer)
    try:
        models = [m["id"] for m in list_models()
                  if "generateContent" in m["methods"]]
    except Exception as e:
        return {"current": DEFAULT_MODEL, "models": [DEFAULT_MODEL],
                "detail": str(e)[:200]}
    if DEFAULT_MODEL not in models:
        models.insert(0, DEFAULT_MODEL)
    # Newest first is what people want in a picker, and these ids sort that way
    # in reverse -- 3.8 above 3.6 above 2.5.
    return {"current": DEFAULT_MODEL, "models": sorted(models, reverse=True),
            "detail": ""}


@app.post("/api/profile/audio")
async def upload_profile_audio(kind: str = Form(...), file: UploadFile = File(...)):
    # `gemini_consent` is a THIRD clip, not a replacement for the second.
    # RealMe's consent recording is in the project's own wording and is the
    # artifact an institution would ask for; Google requires its exact
    # sentence and rejects a paraphrase. One recording cannot be both.
    if kind not in ("voice_reference", "consent_recording", "gemini_consent"):
        raise HTTPException(
            400, "kind must be voice_reference, consent_recording or "
                 "gemini_consent")
    tmp = UPLOADS / f"_{kind}_{safe_filename(file.filename, 'audio.wav')}"
    await save_upload(file, tmp, "audio")
    try:
        from realme.core.media import duration_of
        secs = duration_of(tmp)
    except MediaError as e:
        tmp.unlink(missing_ok=True)
        raise HTTPException(400, f"That file isn't decodable audio. {e}")
    dst = profile.store_audio(tmp, kind)
    tmp.unlink(missing_ok=True)
    drift = {}
    if kind == "voice_reference":
        forget_tts()
        # A re-recorded enrolment that is slower than the last one is a change
        # to every future render, because a clone copies the delivery it was
        # cloned from -- and it is invisible unless something measures it at
        # the moment it happens.
        try:
            from realme.enrollment.pace import reference_drift
            text = (profile.reference_text or "").strip()
            if text and profile.transcript_matches_recording:
                drift = reference_drift(dst, text,
                                        profile.data.get("reference_wpm"))
                if drift.get("wpm"):
                    profile.update({"reference_wpm": drift["wpm"]})
            elif text:
                # The transcript describes the PREVIOUS recording. Dividing
                # its word count by this recording's speech time would give
                # a meaningless rate.
                drift = {"note": profile.transcript_note}
        except Exception as e:                     # never block an enrolment
            drift = {"note": "", "error": str(e)[:200]}
    return {"stored": str(dst), "duration_s": round(secs, 2),
            "pace_note": drift.get("note", ""),
            "wpm": drift.get("wpm"),
            "transcript_note": profile.transcript_note,
            "ready_for_cloning": profile.ready_for_cloning}


# ------------------------------------------------------------------ lecture

def _deck_readable(name: str) -> None:
    """Turn away a deck this machine cannot open, before any folder or job.

    PDF always works. A .pptx needs LibreOffice, which is often not
    installed -- so the answer asks for the PDF version.
    """
    why = slides_mod.deck_problem(name)
    if why:
        raise HTTPException(400, why)


@app.post("/api/lecture/script")
async def lecture_script(file: UploadFile = File(...), script_writer: str = Form("gemini"),
                         gemini_model: str = Form(GEMINI_DEFAULT),
                         course_context: str = Form(""), style_notes: str = Form(""),
                         seconds_per_slide: int = Form(0),
                         width: int = Form(1920)):
    if not file.filename:
        raise HTTPException(400, "No deck uploaded")
    _deck_readable(file.filename)
    writer = make_writer(script_writer, gemini_model)
    pid, pdir, note = new_project(file.filename)
    deck = pdir / safe_filename(file.filename, "deck.pdf")
    await _save_into_new(file, deck, "deck", pdir)

    def work(job):
        if note:
            job.say(note)
        manifest, slides = lec.write_script(
            deck, pdir, writer, style=style_notes,
            course_context=course_context, width=width,
            seconds_per_slide=int(seconds_per_slide) or None, log=job.say)
        (pdir / "_work" / "deck_path.txt").write_text(str(deck), encoding="utf-8")
        return {"project_id": pid, "segments": len(manifest.segments),
                "script_source": manifest.script_source,
                "slides": [s.name for s in slides],
                "manifest": json.loads(manifest.model_dump_json())}

    job = runner.submit("script", f"Draft narration — {pid}", work)
    return {"job_id": job.id, "project_id": pid, "note": note}


@app.get("/api/projects")
def list_projects():
    """Projects that have a script, newest first -- what `revise` can act on."""
    out = []
    if PROJECTS.is_dir():
        for d in PROJECTS.iterdir():
            m = d / "_work" / "manifest.json"
            if d.is_dir() and m.is_file():
                out.append((m.stat().st_mtime, d.name))
    return {"projects": [n for _, n in sorted(out, reverse=True)]}


def old_deck_path(pdir: Path) -> Path | None:
    """The deck file this project was last built from, or None.

    `deck_path.txt` is written by every path that ingests a deck. A project
    without it falls back to whatever deck is sitting in the project folder,
    which is where all three ingest paths put it.
    """
    recorded = pdir / "_work" / "deck_path.txt"
    if recorded.is_file():
        p = Path(read_text(recorded).strip())
        if p.is_file():
            return p
        # Recorded but moved: the folder is the next best guess. Say so,
        # so a wrong guess is traceable.
        stray = p.name
        if (pdir / stray).is_file():
            return pdir / stray
    decks = sorted((f for f in pdir.iterdir()
                    if f.is_file()
                    and f.suffix.lower() in {".pdf", ".pptx", ".ppt", ".odp"}),
                   key=lambda f: f.stat().st_mtime, reverse=True)
    return decks[0] if decks else None


@app.post("/api/lecture/{pid}/revise")
async def lecture_revise(pid: str, file: UploadFile = File(...),
                         width: int = Form(1920)):
    """
    A new version of a deck, against the project that already rendered it.

    Reports which slides changed and keeps the narration of the ones that did
    not. Nothing is re-rendered here -- this produces the plan and the updated
    script, and the ordinary render button does the work, re-recording only
    what the ledger cannot serve.
    """
    from realme.pipeline.revise import match_slides
    from realme.pipeline.splice import thumb, difference
    from realme.core.schema import Segment
    pdir = project_dir(pid)
    if not (pdir / "_work" / "manifest.json").is_file():
        raise HTTPException(404, f"No project '{pid}' to revise. Draft or "
                                 f"import a lecture first.")
    if file.filename:
        _deck_readable(file.filename)
    old_m = lec.load_script(pdir)
    old_deck = old_deck_path(pdir)
    if old_deck is None:
        # Refused here rather than in the job, so it is answered on the button
        # press. Without the old deck there is nothing to compare what the
        # slides SAY against, and comparing pictures alone reports a re-themed
        # deck as entirely new.
        raise HTTPException(400,
            f"The deck this project was built from is not on disk any more, so "
            f"there is nothing to compare the new version against. Put it back "
            f"where it was, or draft/import the new deck as a fresh project.")

    # The upload lands in the staging folder, NOT beside the old deck. Two
    # reasons: re-exporting under the same filename is the normal way to revise
    # a deck, and writing the upload to `pdir / file.filename` would destroy
    # the old one before it could be read; and nothing a comparison produces
    # belongs in the project until the Save button says so.
    staged = pending.begin(pdir)
    deck = staged / "deck" / safe_filename(file.filename, "deck.pdf")
    try:
        await save_upload(file, deck, "deck")
    except HTTPException:
        pending.discard(pdir)
        raise

    def work(job):
        work_dir = pdir / "_work"
        old_slides = slides_mod.slide_pages(work_dir / "slides")
        old_texts = slides_mod.deck_texts(old_deck, work_dir / "old_convert")
        job.say(f"old deck: {old_deck.name}, {len(old_texts)} pages")
        if len(old_texts) != len(old_m.segments):
            raise ValueError(
                f"{old_deck.name} has {len(old_texts)} pages but this project "
                f"has {len(old_m.segments)} slides. It is not the deck this "
                f"lecture was written from.")
        words = sum(len(t.split()) for t in old_texts)
        if words < 3 * max(1, len(old_texts)) // 2:
            # Reported prominently. Matching on empty text falls back to
            # pixels, and pixels report a re-themed deck as all new slides.
            job.say(f"  ! only {words} words of text came out of "
                    f"{old_deck.name}. If these are scanned images, slides "
                    f"will be matched on appearance alone and small edits will "
                    f"read as new slides.")
        old_prints = [thumb(p) for p in old_slides]

        # Rasterise the new deck beside the old, not over it: the comparison
        # needs both, and overwriting the old slides first would leave nothing
        # to compare against.
        new_dir = pending.root(pdir) / "slides"
        pdf = slides_mod.deck_to_pdf(deck, work_dir)
        new_slides = slides_mod.rasterize(pdf, new_dir, width_px=width)
        new_texts = [p.text for p in pdfdoc.read_pages(pdf)]
        new_prints = [thumb(p) for p in new_slides]

        def dist(j, i):
            if j >= len(old_prints) or i >= len(new_prints):
                return None
            return difference(old_prints[j], new_prints[i])

        plan = match_slides(old_texts, new_texts, pixel_distance=dist)
        job.say(f"  {plan.summary()}")

        by_old = {s.slide_index + 1: s for s in old_m.segments}
        segs = []
        for c in plan.changes:
            keep = by_old.get(c.old_slide_number) if c.old_slide_number else None
            if keep is not None:
                # Identity decided where the narration goes. Whether the frame
                # is stale is a separate question and does not move notes.
                segs.append(Segment(
                    segment_id=len(segs) + 1, slide_index=c.slide_number - 1,
                    spoken_text=keep.spoken_text, cues=list(keep.cues),
                    prosody=keep.prosody, revision=c.status,
                    # The digest is KEPT either way. Clearing it for a stale
                    # frame would take away the only thing that makes "reuse
                    # it anyway" possible.
                    slide_digest=keep.slide_digest,
                    rerender=c.needs_render))
            else:
                segs.append(Segment(
                    segment_id=len(segs) + 1, slide_index=c.slide_number - 1,
                    spoken_text=NEW_SLIDE_PLACEHOLDER,
                    revision="added", rerender=True))
            job.say(f"  slide {c.slide_number}: {c.status}"
                    + (f" ({c.note})" if c.note else "")
                    + ("" if c.needs_render else " - frame reusable"))
        added = [c.slide_number for c in plan.changes if c.status == "added"]
        for n in plan.removed:
            job.say(f"  old slide {n} is gone; its narration was dropped")
        if plan.removed and added:
            # A slide carrying no text -- a full-page figure -- is matched on
            # appearance, and appearance cannot survive a theme that repaints
            # the background behind it. When that happens the same slide is
            # reported as one removal and one addition. They are not paired
            # automatically, which could move narration onto a slide that
            # only looks similar.
            job.say(f"  note: {len(plan.removed)} slide(s) went and "
                    f"{len(added)} arrived. If any of those are really the "
                    f"same slide -- a figure page usually is, since it has no "
                    f"text to match on -- paste its narration across by hand "
                    f"rather than drafting it again.")

        old_m.segments = segs
        result = {"project_id": pid, "summary": plan.summary(),
                  "to_render": plan.to_render, "to_write": plan.to_write,
                  "removed": plan.removed,
                  "changes": [{"slide": c.slide_number, "status": c.status,
                               "was": c.old_slide_number, "note": c.note,
                               "rerender": c.needs_render}
                              for c in plan.changes]}
        # Staged, not applied. The project still has its old deck, its old
        # slides and its old script until the editor's Save button accepts
        # this -- so a comparison you disagree with costs you nothing.
        pending.finish(pdir, old_m, result)
        job.say("  this is a proposal: nothing in the project has changed yet. "
                "Read it in the editor, then Save to apply it or Discard to "
                "throw it away.")
        result["pending"] = True
        return result

    job = runner.submit("revise", f"Revise — {pid}", work)
    return {"job_id": job.id, "project_id": pid}


@app.post("/api/lecture/script/import")
async def lecture_script_import(file: UploadFile = File(...),
                                notes: UploadFile | None = File(None),
                                width: int = Form(1920)):
    """
    A deck plus narration you already wrote.

    The Studio's route to the importer and its strict 1..N validation, so
    existing notes can be used without the command line and without the
    model rewriting them.
    """
    from realme.pipeline.script_import import (parse_slide_script, to_manifest,
                                               ScriptImportError, looks_like_json,
                                               manifest_from_json)
    if not file.filename:
        raise HTTPException(400, "No deck uploaded")
    if notes is None or not notes.filename:
        raise HTTPException(400, "No notes file uploaded")
    _deck_readable(file.filename)
    # The notes first: a script that does not parse is refused before a
    # project folder is made for it. A RealMe narration file (.json) keeps
    # its cue words, pace and pauses too.
    raw = (await read_upload(notes, "text")).decode("utf-8", "replace")
    as_json = looks_like_json(raw, notes.filename)
    try:
        if as_json:
            from_json, n_json = manifest_from_json(raw, "imported")
            imported = None
        else:
            imported = parse_slide_script(raw)
    except ScriptImportError as e:
        raise HTTPException(400, str(e))

    pid, pdir, note = new_project(file.filename)
    deck = pdir / safe_filename(file.filename, "deck.pdf")
    await _save_into_new(file, deck, "deck", pdir)

    def work(job):
        if note:
            job.say(note)
        # Rasterise, exactly as the drafting path does -- the editor shows
        # slide images beside the narration either way.
        work_dir = pdir / "_work"
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "deck_path.txt").write_text(str(deck), encoding="utf-8")
        pdf = slides_mod.deck_to_pdf(deck, work_dir)
        pages = slides_mod.rasterize(pdf, work_dir / "slides", width_px=width)
        job.say(f"{len(pages)} slides")
        if as_json:
            m = from_json.model_copy(update={"project_id": pid})
            n_sections = n_json
            job.say(f"{n_sections} slide{'s' if n_sections != 1 else ''} (a RealMe narration file)")
        else:
            m = to_manifest(imported, pid)
            n_sections = len(m.segments)
            job.say(f"{n_sections} sections ({imported.marker_format} markers)")
        if n_sections != len(pages):
            # Refused rather than rendered: a script with more or fewer
            # sections than the deck has slides maps onto the wrong pictures,
            # which is easy to miss in the finished video.
            raise ValueError(
                f"The notes have {n_sections} slide sections but the deck "
                f"has {len(pages)} slides. Fix the notes and import again.")
        for w in (imported.warnings if imported else ()):
            job.say(f"  ! {w}")
        lec.save_script(pdir, m)
        return {"project_id": pid, "segments": len(m.segments),
                "slides": len(pages), "script_source": m.script_source}

    job = runner.submit("import", f"Import notes — {pid}", work)
    return {"job_id": job.id, "project_id": pid, "note": note}


@app.get("/api/lecture/{pid}/script")
def get_script(pid: str):
    pdir = project_dir(pid)
    try:
        m = lec.load_script(pdir, pending.script_path(pdir))
    except FileNotFoundError:
        raise HTTPException(404, "No script drafted for this project yet")
    tts = profile.data["adapters"]["tts"]
    waiting = pending.exists(pdir)
    return {"manifest": json.loads(m.model_dump_json()),
            "cost": lec.estimate_cost(m, PRICE.get(tts, 0.0)), "voice": tts,
            # The editor needs to say, above the notes, that these are proposed.
            "pending": waiting,
            # What a render would replace. Re-recording overwrites in place,
            # which is the chosen behaviour -- but it should not be a
            # surprise, so the page can name the file and its date.
            "existing": existing_output(pdir, pid),
            "plan": pending.plan(pdir) if waiting else None}


def existing_output(pdir: Path, pid: str) -> list[dict]:
    """The files a render of this project would overwrite, newest info first."""
    import datetime
    out = []
    for name in (f"{pid}.mp4", f"{pid}_draft.mp4"):
        f = pdir / name
        if f.is_file():
            st = f.stat()
            out.append({
                "name": name,
                "mb": round(st.st_size / 1e6, 1),
                "when": datetime.datetime.fromtimestamp(st.st_mtime)
                        .strftime("%d %b %Y, %H:%M"),
                "draft": name.endswith("_draft.mp4"),
            })
    return out


@app.put("/api/lecture/{pid}/script")
async def put_script(pid: str, body: dict):
    """
    Save the notes -- and, when a revision is waiting, accept it.

    Save is the moment the new deck joins the project. Until it is pressed the
    comparison is a proposal: the project keeps its old deck, its old slides
    and its old script, and Discard puts it back exactly as it was.

    `commit: false` writes into the proposal without accepting it. That is for
    the Studio's own mid-task saves -- drafting notes for the new slides saves
    your hand edits first, and that autosave must not decide, on your behalf,
    that you have approved the revision.
    """
    from realme.core.schema import Manifest
    from pydantic import ValidationError
    pdir = project_dir(pid)
    if not (pdir / "_work" / "manifest.json").is_file():
        raise HTTPException(404, f"No lecture '{pid}' to save into. Draft or "
                                 f"import it first.")
    if not isinstance(body.get("manifest"), dict):
        raise HTTPException(400, "Nothing to save: the request carried no "
                                 "'manifest'.")
    try:
        m = Manifest.model_validate(body["manifest"])
    except ValidationError as e:
        # A slide gap of 90 s, a pace of 0: said, field by field, instead of
        # a 500 and a script that silently did not save.
        raise HTTPException(400, "The notes were not saved: " + "; ".join(
            f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}"
            for err in e.errors()[:6]))
    waiting = pending.exists(pdir)
    commit = bool(body.get("commit", True))
    lec.save_script(pdir, m, pending.script_path(pdir))
    applied = None
    if waiting and commit:
        applied = pending.commit(pdir)
    return {"saved": True, "segments": len(m.segments),
            "applied": applied, "pending": pending.exists(pdir),
            "cost": lec.estimate_cost(m, PRICE.get(profile.data["adapters"]["tts"], 0.0))}


@app.post("/api/lecture/{pid}/discard-revision")
def discard_revision(pid: str):
    """Throw away a proposed revision. The project is left as it was."""
    pdir = project_dir(pid)
    if not pending.discard(pdir):
        raise HTTPException(404, "There is no proposed revision to discard.")
    return {"discarded": True}


@app.post("/api/lecture/{pid}/draft-additions")
async def draft_additions(pid: str, script_writer: str = Form("gemini"),
                          gemini_model: str = Form(GEMINI_DEFAULT),
                          seconds_per_slide: int = Form(0),
                          slides: str = Form(""),
                          course_context: str = Form(""),
                          style_notes: str = Form("")):
    """
    Narration for the slides a revision added, written into the existing deck.

    Only the slides asked for are touched. By default those are the ones the
    revision marked `added`; an edited slide keeps the narration it already
    has, because a slide that gained a bullet or lost a logo usually needs a
    sentence added by hand, not a fresh block that says the same thing in
    different words. `slides` overrides the default with an explicit
    comma-separated list, which is how an edited slide gets rewritten when the
    edit really was substantive.

    The surrounding narration goes to the model verbatim. A slide inserted into
    a finished lecture has to pick up where the previous one stopped; a model
    shown only the new slide writes an opening paragraph for a lecture that is
    already twenty minutes old.
    """
    pdir = project_dir(pid)
    try:
        m = lec.load_script(pdir, pending.script_path(pdir))
    except FileNotFoundError:
        raise HTTPException(404, "No script drafted for this project yet")

    by_number = {s.slide_index + 1: s for s in m.segments}
    if slides.strip():
        try:
            targets = sorted({int(x) for x in slides.replace(",", " ").split()})
        except ValueError:
            raise HTTPException(400, f"Could not read a slide list from {slides!r}")
        unknown = [n for n in targets if n not in by_number]
        if unknown:
            raise HTTPException(400, f"This deck has {len(m.segments)} slides; "
                                     f"asked for {unknown}.")
    else:
        targets = sorted(n for n, seg in by_number.items()
                         if (seg.revision == "added"
                             or seg.spoken_text.strip() == NEW_SLIDE_PLACEHOLDER))
    if not targets:
        raise HTTPException(400, "No slides are waiting for narration. Revise "
                                 "the deck first, or name the slides to write.")

    writer = make_writer(script_writer, gemini_model)
    if not hasattr(writer, "write_additions"):
        raise HTTPException(400, f"The {writer.name} writer cannot write into "
                                 f"an existing lecture.")

    def work(job):
        writer.log = job.say
        slide_pngs = slides_mod.slide_pages(pending.slides_dir(pdir))
        if len(slide_pngs) != len(m.segments):
            # Reported prominently: otherwise narration is written against
            # the wrong pictures, which is easy to miss in the finished video.
            raise ValueError(
                f"{len(slide_pngs)} slide images but {len(m.segments)} "
                f"segments. Re-run the revision before drafting.")
        context = {}
        for n, seg in by_number.items():
            text = (seg.spoken_text or "").strip()
            context[n] = "" if text == NEW_SLIDE_PLACEHOLDER else text
        # A target that already has narration is edited, not written. The
        # instructor has read those words; a slide that gained a bullet wants a
        # sentence added, not the same thing said differently.
        revise = [n for n in targets if context.get(n)]
        fresh = [n for n in targets if n not in revise]
        if fresh:
            job.say(f"writing slide(s) {', '.join(map(str, fresh))} into a "
                    f"{len(m.segments)}-slide lecture")
        if revise:
            job.say(f"revising slide(s) {', '.join(map(str, revise))} in place")
        written = writer.write_additions(
            slide_pngs, targets, context,
            style=style_notes or profile.data.get("style_notes", ""),
            course_context=(course_context
                            or profile.data.get("course_context", "")),
            seconds_per_slide=seconds_per_slide or None,
            revise=revise)

        done = []
        for n in targets:
            got = written.get(n)
            if not got:
                job.say(f"  slide {n}: nothing came back; left as it was")
                continue
            seg = by_number[n]
            seg.spoken_text = got["spoken_text"]
            if got.get("cues"):
                seg.cues = got["cues"]
            # The words changed, so the recorded frame no longer matches what
            # is said over it. This is the one case where new narration must
            # force a re-render regardless of what the picture looks like.
            was = context.get(n, "")
            unchanged = got["spoken_text"].strip() == was.strip()
            if not unchanged:
                # The words changed, so the recorded frame no longer matches
                # what is said over it. New narration forces a re-render
                # whatever the picture looks like.
                seg.rerender = True
            done.append(n)
            job.say(f"  slide {n}: "
                    + ("returned unchanged" if unchanged
                       else f"{len(got['spoken_text'].split())} words"
                            + (f" (was {len(was.split())})" if was else "")))
        if done:
            # Into the proposal if that is what is open. Drafting notes is not
            # approving the revision.
            lec.save_script(pdir, m, pending.script_path(pdir))
        return {"project_id": pid, "written": done,
                "skipped": [n for n in targets if n not in done],
                "manifest": json.loads(m.model_dump_json())}

    job = runner.submit("additions", f"Draft new slides — {pid}", work)
    return {"job_id": job.id, "project_id": pid, "slides": targets}


# ---------------------------------------------------------------- voice

VOICE_DIR = DATA / "voice"


@app.post("/api/voice/analyze")
async def voice_analyze(file: UploadFile = File(...)):
    """Judge a reference recording and render the presets to compare."""
    suffix = Path(safe_filename(file.filename, "in.wav")).suffix or ".wav"
    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = await save_upload(file, VOICE_DIR / ("_upload" + suffix), "audio")
    # The analysis runs ffmpeg several times and beautifies once per preset;
    # under `async def` that would block the event loop and every tab's job
    # polling. Off the loop, into the thread pool.
    from starlette.concurrency import run_in_threadpool
    return await run_in_threadpool(_voice_analyze_sync, tmp)


def _voice_analyze_sync(tmp: Path):
    from realme.enrollment import voice as V
    from dataclasses import asdict
    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    raw = VOICE_DIR / "reference_raw.wav"
    try:
        # normalize container/codec first so analysis sees plain PCM
        from realme.core.media import require, run
        run([require("ffmpeg"), "-y", "-i", str(tmp), "-ac", "1",
             "-c:a", "pcm_s16le", str(raw)], "ingest reference")
    except Exception as e:
        raise HTTPException(400, f"Could not read that audio file. {e}")
    finally:
        tmp.unlink(missing_ok=True)

    report = V.analyze(raw)
    presets_error = ""
    try:
        V.compare(raw, VOICE_DIR)
    except Exception as e:
        # Not swallowed: otherwise the preview endpoint would later say
        # "upload a reference recording first" about one that was uploaded.
        presets_error = f"{type(e).__name__}: {e}"
    return {"report": asdict(report),
            "presets": [{"name": k, "label": v["label"]} for k, v in V.PRESETS.items()],
            "presets_error": presets_error}


@app.post("/api/voice/tune")
def voice_tune(body: dict):
    """
    Render the reference through one custom Treatment so it can be heard.

    Lets the sliders be judged by ear, since a value like "vividness +0.12"
    is not meaningful on its own. One render per change, straight back to
    the browser.
    """
    from realme.enrollment import voice as V
    raw = VOICE_DIR / "reference_raw.wav"
    if not raw.exists():
        raise HTTPException(400, "Upload a reference recording first")
    fields = ("pace", "pitch_semitones", "warmth", "clarity", "smoothing",
              "brightness", "vigor", "vividness")
    try:
        t = V.Treatment(**{k: float(body.get(k, getattr(V.Treatment(), k)))
                           for k in fields}).validated()
    except (ValueError, TypeError) as e:
        raise HTTPException(400, str(e))
    out = VOICE_DIR / "preset_custom.wav"
    V.treat(raw, out, t, repair=bool(body.get("repair", True)))
    from dataclasses import asdict
    report = V.analyze(out)
    return {"preset": "custom", "treatment": asdict(t),
            "duration_s": report.duration_s, "snr_db": report.snr_db,
            "peak_db": report.peak_db}


@app.get("/api/voice/controls")
def voice_controls():
    """The tunable range of each control, so the UI cannot offer a bad value."""
    from realme.enrollment import voice as V
    d = V.Treatment()
    return {
        "defaults": {k: getattr(d, k) for k in
                     ("pace", "pitch_semitones", "warmth", "clarity", "smoothing",
                      "brightness", "vigor", "vividness")},
        "controls": [
            {"key": "warmth", "label": "Warmth", "min": V.WARMTH_MIN,
             "max": V.WARMTH_MAX, "step": 0.01,
             "hint": "Chest and body, low frequencies around 180 Hz"},
            {"key": "clarity", "label": "Clarity", "min": V.CLARITY_MIN,
             "max": V.CLARITY_MAX, "step": 0.01,
             "hint": "Consonant definition around 3.2 kHz. Neutral is 0.50"},
            {"key": "brightness", "label": "Brightness", "min": V.BRIGHTNESS_MIN,
             "max": V.BRIGHTNESS_MAX, "step": 0.01,
             "hint": "Air and openness higher up, around 4.2 kHz"},
            {"key": "vigor", "label": "Energy", "min": V.VIGOR_MIN,
             "max": V.VIGOR_MAX, "step": 0.01,
             "hint": "Presence and level together - forward, not louder"},
            {"key": "vividness", "label": "Vividness", "min": V.VIVIDNESS_MIN,
             "max": V.VIVIDNESS_MAX, "step": 0.01,
             "hint": "Expressive colour in the upper mids, around 2.5 kHz"},
            {"key": "smoothing", "label": "Smoothing", "min": V.SMOOTHING_MIN,
             "max": V.SMOOTHING_MAX, "step": 0.01,
             "hint": "Evens out loud and quiet parts of a sentence"},
            {"key": "pitch_semitones", "label": "Pitch", "min": V.PITCH_MIN,
             "max": V.PITCH_MAX, "step": 0.05, "unit": " st",
             "hint": "Semitones. Pace is held steady when this changes"},
            {"key": "pace", "label": "Pace", "min": V.PACE_MIN,
             "max": V.PACE_MAX, "step": 0.005,
             "hint": "Speaking rate of the reference itself"},
        ],
        "presets": {k: {f: getattr(v, f) for f in
                        ("pace", "pitch_semitones", "warmth", "clarity",
                         "smoothing", "brightness", "vigor", "vividness")}
                    for k, v in V.PRESET_TREATMENTS.items()},
    }


@app.get("/api/voice/preview/{preset}")
def voice_preview(preset: str):
    f = VOICE_DIR / f"preset_{preset}.wav"
    if not f.exists():
        raise HTTPException(404, "Upload a reference recording first")
    return FileResponse(f, media_type="audio/wav")


@app.post("/api/voice/bake")
def voice_bake(body: dict):
    """Install the polished reference as the profile's voice."""
    from realme.enrollment import voice as V
    from dataclasses import asdict
    # `off` by default here too: untouched unless asked. See the CLI's --preset.
    preset = body.get("preset", "off")
    raw = VOICE_DIR / "reference_raw.wav"
    if not raw.exists():
        raise HTTPException(400, "Upload a reference recording first")
    baked = profile.root / "baked_assets" / "voice_reference.wav"
    # Do not bake a stale take over a newer one. Enrolling from the command
    # line refreshes the raw copy, so this should not fire; it guards
    # against silently replacing the enrolled voice with an older recording.
    if baked.exists() and raw.stat().st_mtime < baked.stat().st_mtime - 5:
        raise HTTPException(409,
            f"The recording in {raw.parent.name}\\ is OLDER than the voice you "
            f"are currently using.\n"
            f"Baking would replace your current voice with that older take.\n\n"
            f"  raw upload   {raw}\n"
            f"  in use       {baked}\n\n"
            f"Upload the recording again above if you meant to change it.")
    treatment = body.get("treatment")
    if preset == "custom" and isinstance(treatment, dict):
        try:
            t = V.Treatment(**{k: float(v) for k, v in treatment.items()
                               if k in ("pace", "pitch_semitones", "warmth",
                                        "clarity", "smoothing", "brightness",
                                        "vigor", "vividness")}).validated()
        except (ValueError, TypeError) as e:
            raise HTTPException(400, str(e))
        V.treat(raw, baked, t)
        profile.update({"voice_reference": str(baked), "voice_preset": "custom",
                        "voice_treatment": treatment})
    else:
        if preset not in V.PRESETS:
            raise HTTPException(400, f"Unknown preset {preset!r}. Choose "
                                     f"from: {', '.join(V.PRESETS)}")
        V.beautify(raw, baked, preset)
        profile.update({"voice_reference": str(baked), "voice_preset": preset,
                        "voice_treatment": None})
    return {"baked": str(baked), "preset": preset,
            "report": asdict(V.analyze(baked)),
            "ready_for_cloning": profile.ready_for_cloning}


@app.get("/api/voice/consent")
def voice_consent():
    import datetime
    from realme.enrollment import voice as V
    return {"text": V.consent_text(profile.data.get("display_name", ""),
                                   datetime.date.today().isoformat())}


# ------------------------------------------------------- gemini voice cloning
#
# Separate endpoints from the local enrolment above, on purpose. Enrolling with
# Qwen3 copies a file into the profile directory. Enrolling here uploads the
# instructor's voice to a company and leaves it there for a year. A shared
# endpoint with a flag would make those one action with a parameter.


def _gemini_state() -> dict:
    import os as _os
    from realme.adapters import tts_gemini as G
    g = dict(profile.data.get("gemini_voice") or {})
    return {
        "voice": g.get("voice") or "",
        "model": g.get("model") or G.DEFAULT_MODEL,
        "models": list(G.MODELS),
        "stored": bool(g.get("stored", True)),
        "created_at": g.get("created_at"),
        "expires": ("1 year from last use" if g.get("stored", True)
                    else "7 days"),
        "paid_tier_ack": bool(g.get("paid_tier_ack")
                              or _os.environ.get("REALME_GEMINI_PAID") == "1"),
        "max_chars": g.get("max_chars"),
        "max_chars_effective": (G.MEASURED_MAX_CHARS
                                if g.get("max_chars") is None
                                else (g["max_chars"] or 260)),
        "measured_default": G.MEASURED_MAX_CHARS,
        # The EFFECTIVE correction, not the stored one: the adapter may
        # ignore the stored value, and the badge must show what renders use.
        "rate_match": G.effective_rate_match(g)[0] if g.get("rate_match") else None,
        "rate_match_stored": g.get("rate_match"),
        "rate_match_for": g.get("rate_match_for"),
        "rate_note": G.effective_rate_match(g)[1],
        # Counted, never a balance: the API reports neither a limit nor what
        # is left, and the limit depends on a tier that can change.
        "calls_today": G.calls_today(),
        "calls_exhausted_at": G.exhausted_today(),
        "consent_recording": g.get("consent_recording"),
        "have_consent": bool(g.get("consent_recording")
                             and Path(g["consent_recording"]).is_file()),
        "have_reference": bool(profile.data.get("voice_reference")),
        "have_key": bool(_os.environ.get("GEMINI_API_KEY")),
        "consent_sentence": G.CONSENT_SENTENCE,
        "tier_notice": G.TIER_NOTICE,
    }


@app.get("/api/voice/gemini")
def gemini_status():
    _load_env()
    return _gemini_state()


@app.post("/api/voice/gemini/ack")
def gemini_ack(body: dict):
    g = dict(profile.data.get("gemini_voice") or {})
    g["paid_tier_ack"] = bool(body.get("paid"))
    profile.update({"gemini_voice": g})
    forget_tts()
    return _gemini_state()


@app.post("/api/voice/gemini/chunk")
def gemini_chunk(body: dict):
    """Set characters per call -- only ever a number the probe measured."""
    n = body.get("max_chars")
    g = dict(profile.data.get("gemini_voice") or {})
    try:
        # Empty means "nobody has said": fall back to the engine's measured
        # default. 0 means "use the pipeline's own chunking", which is a
        # different answer and has to survive the round trip.
        g["max_chars"] = None if n in (None, "") else int(n)
    except (TypeError, ValueError):
        raise HTTPException(400, "max_chars must be a number, or empty")
    if g["max_chars"] and not 100 <= g["max_chars"] <= 8000:
        raise HTTPException(400, "max_chars must be between 100 and 8000, "
                                 "or 0 for the pipeline's own chunking")
    profile.update({"gemini_voice": g})
    forget_tts()
    return _gemini_state()


@app.get("/api/voice/gemini/check")
def gemini_check():
    """Compare the two clips before anything is uploaded.

    Google runs a speaker-verification check on the pair and refuses with an
    HTTP 500 some minutes later. Both of the things it refuses for -- clip
    length, and two recordings made on different equipment -- are measurable
    here in a second.
    """
    from realme.adapters import tts_gemini as G
    ref = profile.data.get("voice_reference")
    consent = (profile.data.get("gemini_voice") or {}).get("consent_recording")
    if not ref or not Path(ref).is_file():
        return {"ok": False, "notes": ["No voice reference enrolled yet."],
                "score": None}
    if not consent or not Path(consent).is_file():
        return {"ok": False, "notes": ["No Google consent clip yet."],
                "score": None}
    rep = G.consent_check(Path(ref), Path(consent))
    rep.pop("reference_signature", None)
    rep.pop("consent_signature", None)
    return rep


@app.get("/api/voice/gemini/voices")
def gemini_voices():
    """Every replicated voice in the Google project.

    A voice can exist there without being recorded here, e.g. when an
    enrolment reported failure after Google had created the voice. This
    lists it so it can be adopted or deleted.
    """
    _load_env()
    from realme.adapters import tts_gemini as G
    try:
        rows = G.list_voices()
    except AdapterUnavailable as e:
        return {"voices": [], "detail": str(e)[:300]}
    here = (profile.data.get("gemini_voice") or {}).get("voice")
    return {"voices": [{k: v for k, v in r.items() if k != "raw"} for r in rows],
            "current": here, "detail": ""}


@app.post("/api/voice/gemini/use")
def gemini_use(body: dict):
    """Adopt a voice that already exists at Google. Uploads nothing."""
    _load_env()
    import datetime as _dt
    from realme.adapters import tts_gemini as G
    v = (body.get("voice") or "").strip()
    if not v.startswith(("voice_", "voicekey_")):
        raise HTTPException(400, "A Gemini voice id starts with voice_ or "
                                 "voicekey_.")
    try:
        G.get_voice(v)
    except AdapterUnavailable as e:
        raise HTTPException(400, f"Google does not recognise {v}: {e}")
    g = dict(profile.data.get("gemini_voice") or {})
    g.update({"voice": v, "stored": v.startswith("voice_")})
    g.setdefault("created_at",
                 _dt.datetime.now().isoformat(timespec="seconds"))
    profile.update({"gemini_voice": g})
    forget_tts()
    return _gemini_state()


@app.post("/api/voice/gemini/pace")
def gemini_pace(body: dict | None = None):
    """Measure the clone's actual speaking rate and store the correction.

    Costs one short render -- about a cent -- and is more accurate than
    choosing a number by ear.
    """
    _load_env()
    from realme.enrollment.pace import calibrate_engine
    from realme.core.env import data_home
    g = dict(profile.data.get("gemini_voice") or {})
    if not g.get("voice"):
        raise HTTPException(400, "Create your Gemini voice first (the "
                                 "Create the cloned voice button above).")
    ref = profile.data.get("voice_reference")
    text = (profile.reference_text or "").strip()
    if not ref or not text:
        raise HTTPException(
            400, "Measuring the pace needs your voice recording and the "
                 "words you read in it. Paste them under Your voice "
                 "(\"What you read\") and press Save the text.")
    target = (body or {}).get("target_wpm")
    if not target and not profile.transcript_matches_recording:
        raise HTTPException(400, profile.transcript_note)
    try:
        r = calibrate_engine("gemini-tts", Path(ref), text,
                             target_wpm=float(target) if target else None,
                             workdir=data_home() / "_work", log=lambda *_: None)
    except Exception as e:
        raise HTTPException(400, str(e)[:400])
    g["rate_match"] = r["pace"]
    g["rate_match_for"] = g.get("voice")
    profile.update({"gemini_voice": g})
    if not target:
        profile.note_transcript_pairing()
    forget_tts()
    return {**r, **_gemini_state()}


@app.post("/api/voice/gemini/enroll")
def gemini_enroll(body: dict):
    _load_env()
    import datetime as _dt
    from realme.adapters import tts_gemini as G
    st = _gemini_state()
    if not st["have_key"]:
        raise HTTPException(400, "No Gemini key is set. Close the Studio, "
                                 "double-click 2_Set_Key.bat, then start the "
                                 "Studio again -- it is the same key that "
                                 "drafts the narration.")
    if not st["paid_tier_ack"]:
        raise HTTPException(400, G.TIER_NOTICE)
    ref = profile.data.get("voice_reference")
    if not ref or not Path(ref).is_file():
        raise HTTPException(400, "Record your voice first: Twin Setup -> "
                                 "Your voice, then press Save this voice.")
    consent = profile.data.get("gemini_voice", {}).get("consent_recording")
    if not consent or not Path(consent).is_file():
        raise HTTPException(
            400, "Google needs its own consent clip, in its exact words: "
                 f"\u201c{G.CONSENT_SENTENCE}\u201d")
    rep = G.consent_check(Path(ref), Path(consent))
    if rep["notes"] and not body.get("anyway"):
        raise HTTPException(400, "Not uploaded — these two recordings would "
                                 "very likely be refused:\n\n• "
                                 + "\n\n• ".join(rep["notes"]))
    try:
        made = G.create_voice(Path(ref), Path(consent),
                              store=not bool(body.get("stateless")),
                              display_name=(profile.data.get("display_name")
                                            or "RealMe instructor"))
    except AdapterUnavailable as e:
        raise HTTPException(400, str(e))
    g = G.with_voice(
        profile.data.get("gemini_voice") or {}, made["voice"],
        model=made["model"], stored=made["stored"],
        created_at=_dt.datetime.now().isoformat(timespec="seconds"))
    profile.update({"gemini_voice": g})
    forget_tts()        # so the next render builds the engine with the new id
    return _gemini_state()


@app.post("/api/voice/gemini/forget")
def gemini_forget(body: dict | None = None):
    from realme.adapters import tts_gemini as G
    g = dict(profile.data.get("gemini_voice") or {})
    named = ((body or {}).get("voice") or "").strip()
    voice, note = named or g.get("voice"), ""
    if voice:
        try:
            G.delete_voice(voice)
            note = f"{voice} was deleted from your Google project."
        except AdapterUnavailable as e:
            note = (f"Removed here, but Google would not delete it ({e}). "
                    f"Check aistudio.google.com so it does not sit there.")
    if voice == g.get("voice"):
        profile.update({"gemini_voice": G.with_voice(g, "", created_at=None)})
        forget_tts()
    return {**_gemini_state(), "note": note}


# Request bodies, declared so that /docs and /openapi.json say what each
# route takes. Extra fields are ignored; a wrong TYPE is a 422 naming the
# field, instead of a 500 later on.

class PreviewRequest(BaseModel):
    text: str = Field("", description="A sentence or two of narration")
    tts: str | None = Field(None, description="Engine spec, e.g. qwen3cpp or "
                            "piper:en_US-amy-medium. Default: the profile's.")
    index: int = 0
    pace: float | None = Field(None, gt=0, description="Speed multiplier, 1.0 "
                               "= the engine's own")
    project_id: str | None = Field(None, description="Record the take into "
                                   "this project's ledger, so its render "
                                   "reuses it")


@app.post("/api/preview")
def preview_utterance(req: PreviewRequest):
    """Hear one utterance. Cheap loop for catching a mangled term early."""
    _refuse_while_measuring()
    from realme.pipeline.speak import preview as _preview
    body = req.model_dump()
    if not (body.get("text") or "").strip():
        raise HTTPException(400, "Nothing to hear: the narration is empty.")
    pace = float(body.get("pace") or 1.0)
    # The same engine the render builds -- `engine_for`, so the saved voice
    # speed is applied -- and given the pace the same way the render gives
    # it: as an argument to `synthesize`. On an engine that realises pace
    # itself (qwen3cpp, piper), a preview without it would be heard at 1.0
    # yet filed in the ledger under the slide's pace.
    tts = _PacedSynth(engine_for(body.get("tts")
                                 or profile.data["adapters"]["tts"]), pace)
    try:
        tts.preflight()
    except AdapterUnavailable as e:
        raise HTTPException(400, str(e))
    out = DATA / "previews" / "preview.wav"
    # With a project, the take is recorded into that project's own ledger, so
    # the sentence you just listened to is the one the render uses. Without
    # one it stays a throwaway, which is what the quick check on the Narrate
    # tab wants.
    led = work = None
    pid = (body.get("project_id") or "").strip()
    if pid:
        # A lecture keeps its takes in _work/audio under _work/ledger.json --
        # the same ledger its render reads, so a sentence heard here is not
        # bought again.
        pdir = project_dir(pid)
        if (pdir / "_work" / "manifest.json").is_file():
            from realme.core.ledger import RenderLedger
            led = RenderLedger(pdir / "_work" / "ledger.json")
            work = pdir / "_work" / "audio"
    try:
        return _preview(body.get("text", ""), tts, out,
                        index=max(0, int(body.get("index", 0))),
                        pace=pace, ledger=led, workdir=work)
    except HTTPException:
        raise
    except AdapterUnavailable as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        # Said as a sentence the page can show, not a bare "Internal Server
        # Error" that leaves the button reading "recording..." for ever.
        raise HTTPException(500, f"The line could not be spoken. "
                                 f"{type(e).__name__}: {e}"[:600])


class _PacedSynth:
    """An engine whose `synthesize` is given `pace` unless the caller gives
    one, exactly as `speak_segment` gives it during a render. Everything
    else -- `timing_factor`, `retimes_after`, the voice fingerprint -- is the
    wrapped engine's, so the stretch applied afterwards is the render's too."""

    def __init__(self, inner, pace: float):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_pace", float(pace))

    def synthesize(self, *a, **kw):
        if len(a) < 4:                     # (text, out_wav, voice, pace)
            kw.setdefault("pace", self._pace)
        return self._inner.synthesize(*a, **kw)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __setattr__(self, name, value):
        setattr(self._inner, name, value)


@app.get("/api/preview/audio")
def preview_audio():
    f = DATA / "previews" / "preview.wav"
    if not f.exists():
        raise HTTPException(404, "No preview yet")
    return FileResponse(f, media_type="audio/wav")


@app.post("/api/lecture/{pid}/render")
def render_lecture(pid: str, body: dict | None = None):
    _refuse_while_measuring()
    body = body or {}
    pdir = project_dir(pid)
    if pending.exists(pdir):
        # Refused rather than rendered. Rendering would have to pick one of two
        # decks, and picking silently could render a version the user has not
        # approved.
        raise HTTPException(409,
            "A revised deck is waiting to be accepted. Save the notes to apply "
            "it, or discard the revision, then render.")
    try:
        m = lec.load_script(pdir)
    except FileNotFoundError:
        raise HTTPException(404, f"No lecture '{pid}' with saved notes to "
                                 f"render. Draft or import it first.")
    # The per-render overrides, checked by the same rules as the profile's
    # own values, so a bad one is a 400 now rather than a failed job later.
    from realme.app.profile import validate_patch
    try:
        validate_patch({k: body[k] for k in
                        ("pause_scale", "signalling", "prosody_mode")
                        if body.get(k) is not None}
                       | ({"video": {"layout": body["layout"]}}
                          if body.get("layout") is not None else {}))
    except ValueError as e:
        raise HTTPException(400, str(e))
    # Through the helper, which knows how to find a deck for a project
    # without `deck_path.txt`. Reading the file directly would fail on such
    # a project with a bare FileNotFoundError shown as a 500.
    deck = old_deck_path(pdir)
    if deck is None:
        raise HTTPException(400,
            "No deck found for this project. The slide file it was built from "
            "has moved or been deleted; put it back in the project folder, or "
            "start the lecture again from the deck.")
    v = profile.data["video"]
    # "engine" or "engine:voice", the same spelling the dialogue caster takes,
    # so the lecture page can ask for a specific draft voice.
    from realme.adapters.base import AdapterUnavailable
    spec = body.get("tts") or profile.data["adapters"]["tts"]
    try:
        tts = engine_for(spec, max_spend=body.get("max_spend"))
    except AdapterUnavailable as e:
        raise HTTPException(400, str(e))
    # A draft goes to its own filename. Same deck, same script, one fiftieth
    # of the time -- and nothing about the name would otherwise distinguish it
    # from the finished render it would replace.
    suffix = "" if getattr(tts, "is_voice_clone", False) else "_draft"
    layout = body.get("layout", v["layout"])
    signalling = body.get("signalling", profile.data.get("signalling", "highlight"))
    slide_level = _slide_level(pdir, m.project_id + suffix, tts, layout=layout,
                               signalling=signalling, video=v,
                               fresh=bool(body.get("fresh")))

    def work(job):
        from realme.pipeline import lean
        if slide_level:
            # Kept as its video and text: only changed slides are recorded.
            out = lean.rerecord(
                pdir, deck, tts, name=m.project_id + suffix,
                mode=body.get("prosody_mode",
                              profile.data.get("prosody_mode", "natural")),
                pause_scale=float(body.get(
                    "pause_scale", profile.data.get("pause_scale", 1.0))),
                keep_lean=True, log=job.say,
                progress=lambda p: setattr(job, "progress", p))
            return {"project_id": pid, "report": out["report"],
                    "rerecorded": out.get("rerecorded", []),
                    "files": {k: Path(p).name for k, p in out.items()
                              if k not in ("report", "rerecorded")}}
        out = lec.render(deck, pdir, m, tts, suffix=suffix,
                         layout=body.get("layout", v["layout"]),
                         width=v["width"], height=v["height"], fps=v["fps"],
                         signalling=body.get("signalling",
                                             profile.data.get("signalling", "highlight")),
                         mode=body.get("prosody_mode",
                                       profile.data.get("prosody_mode", "natural")),
                         pause_scale=float(body.get(
                             "pause_scale",
                             profile.data.get("pause_scale", 1.0))),
                         fresh=bool(body.get("fresh")),
                         log=job.say, progress=lambda p: setattr(job, "progress", p))
        if LEAN:
            lean.compact(pdir, log=job.say)
        return {"project_id": pid, "report": out["report"],
                "files": {k: Path(p).name for k, p in out.items() if k != "report"}}

    job = runner.submit("render", f"Render lecture — {pid}", _hold(pid, work))
    return {"job_id": job.id}


# ------------------------------------------- keeping a finished lecture
#
# A finished lecture can be kept as its video and its text files only (the
# narration, every sentence's words and timing, where each slide starts and
# ends), downloaded as one .zip, removed, and brought back from the .zip.
# The work is in pipeline/lean.py; a lecture that is not slimmed down
# renders as usual.

def _slide_level(pdir: Path, name: str, tts, *, layout: str, signalling: str,
                 video: dict, fresh: bool) -> bool:
    """Re-record changed slides only, from the finished video?

    Only for a lecture kept as its video and text (its working audio is
    gone, so an ordinary render would record every sentence again), in the
    same voice and at the same size, frame rate, layout and highlighting --
    otherwise the kept slides could not sit beside new ones, and the whole
    lecture is rendered as usual.
    """
    from realme.pipeline import lean
    if fresh or not lean.is_compact(pdir, name):
        return False
    rec = lean.renders(pdir).get(name) or {}
    return (rec.get("voice") == getattr(tts, "name", tts.__class__.__name__)
            and int(rec.get("width") or 0) == int(video["width"])
            and int(rec.get("height") or 0) == int(video["height"])
            and int(rec.get("fps") or 0) == int(video["fps"])
            and (rec.get("layout") or "slide_only") == layout
            and (rec.get("signalling") or "highlight") == signalling)


#: Lectures with a render being set up or running. Registered before the job
#: is submitted and checked under the same lock as Remove and Keep-only, so
#: neither can act on a lecture in the moment a render of it starts.
_ACTIVE: dict[str, int] = {}
_ACTIVE_LOCK = threading.Lock()


def _project_busy(pid: str) -> bool:
    return _ACTIVE.get(pid, 0) > 0 or any(
        j["status"] in ("queued", "running")
        and j.get("label", "").endswith(f"— {pid}") for j in runner.recent(200))


def _hold(pid: str, work):
    """`work`, with the lecture marked busy from now until it finishes."""
    with _ACTIVE_LOCK:
        _ACTIVE[pid] = _ACTIVE.get(pid, 0) + 1

    def run(job):
        try:
            return work(job)
        finally:
            with _ACTIVE_LOCK:
                _ACTIVE[pid] -= 1
    return run


@app.get("/api/lecture/{pid}/storage")
def lecture_storage(pid: str):
    """What this lecture keeps on disk, and whether it is kept lean."""
    from realme.pipeline import lean
    pdir = _lecture_dir(pid)
    found = lean.renders(pdir)
    total = sum(f.stat().st_size for f in pdir.rglob("*") if f.is_file())
    return {"lean_by_default": LEAN,
            "total_mb": round(total / (1 << 20), 1),
            "working_mb": round(lean.media_bytes(pdir) / (1 << 20), 1),
            "renders": {n: {"compact": lean.is_compact(pdir, n)} for n in found}}


@app.post("/api/lecture/{pid}/compact")
def lecture_compact(pid: str):
    from realme.pipeline import lean
    pdir = _lecture_dir(pid)
    if not lean.renders(pdir):
        raise HTTPException(409, "Render this lecture first: there is no "
                                 "finished video to keep yet.")
    with _ACTIVE_LOCK:
        if _project_busy(pid):
            raise HTTPException(409, "This lecture is rendering. Try again "
                                     "when it has finished.")
        note = lean.compact(pdir, log=lambda *_: None)
    return {**note, **lecture_storage(pid)}


@app.get("/api/lecture/{pid}/bundle")
def lecture_bundle(pid: str):
    """The lecture as one .zip: video, captions, timing, script, deck."""
    import tempfile
    from starlette.background import BackgroundTask
    from realme.pipeline import lean
    pdir = _lecture_dir(pid)
    fd, name = tempfile.mkstemp(dir=UPLOADS, prefix="_bundle_", suffix=".zip")
    os.close(fd)
    tmp = Path(name)
    try:
        lean.bundle(pdir, tmp, deck=old_deck_path(pdir))
    except MediaError as e:
        tmp.unlink(missing_ok=True)
        raise HTTPException(409, str(e))
    return FileResponse(tmp, filename=f"{pid}_lecture.zip",
                        media_type="application/zip",
                        background=BackgroundTask(tmp.unlink, missing_ok=True))


@app.post("/api/lecture/bundle/import")
async def lecture_bundle_import(file: UploadFile = File(...)):
    """A lecture back from its .zip, as a new project."""
    from realme.pipeline import lean
    if not (file.filename or "").lower().endswith(".zip"):
        raise HTTPException(400, "Choose the .zip that RealMe gave you for "
                                 "this lecture.")
    tmp = UPLOADS / f"_import_{safe_filename(file.filename, 'lecture.zip')}"
    await save_upload(file, tmp, "video")
    stem = Path(file.filename).stem
    if stem.endswith("_lecture"):
        stem = stem[:-len("_lecture")]
    pid, pdir, note = new_project(stem + ".pdf")
    try:
        pdir.rmdir()
        info = lean.unbundle(tmp, pdir, log=lambda *_: None)
    except MediaError as e:
        shutil.rmtree(pdir, ignore_errors=True)
        raise HTTPException(400, str(e))
    finally:
        tmp.unlink(missing_ok=True)
    return {"project_id": pid, "renders": info["renders"], "note": note}


@app.delete("/api/projects/{pid}")
def delete_project(pid: str):
    """Remove a lecture -- video, script, deck, everything."""
    pdir = project_dir(pid)
    if not pdir.is_dir():
        raise HTTPException(404, "No such lecture.")
    with _ACTIVE_LOCK:
        if _project_busy(pid):
            raise HTTPException(409, "This lecture is rendering. Try again "
                                     "when it has finished.")
        shutil.rmtree(pdir)
    return {"deleted": pid}


# -------------------------------------------------------------------- tools

# ------------------------------------------------------- inserted videos
#
# A video that plays between two slides -- an animated explainer, a screen
# recording -- remembered by the lecture, so every render puts it back. The
# Tools tab drives these; the work is in pipeline/inserts.py.

def _lecture_dir(pid: str) -> Path:
    pdir = project_dir(pid)
    if not (pdir / "_work" / "manifest.json").is_file():
        raise HTTPException(404, "That lecture has no saved notes.")
    return pdir


def _inserts_view(pid: str, pdir: Path) -> dict:
    from realme.pipeline import inserts as INS
    m = lec.load_script(pdir)
    slides = [{"slide": s.slide_index + 1,
               "opening": " ".join(s.spoken_text.split()[:10])}
              for s in m.segments]
    try:
        items = INS.load(pdir)
    except MediaError as e:
        raise HTTPException(400, str(e))
    a = pdir / "_work" / "assembly.json"
    last = json.loads(read_text(a)) if a.is_file() else None
    return {"project_id": pid, "slides": slides,
            "inserts": [{**i.__dict__,
                         "mb": round((pdir / i.file).stat().st_size / 1e6, 1)
                         if (pdir / i.file).is_file() else None}
                        for i in items],
            # What "apply" would rebuild without rendering: the last render.
            "last_render": last["name"] + ".mp4" if last else None}


@app.get("/api/lecture/{pid}/notes.md")
def lecture_notes_md(pid: str):
    """The narration as a notes file (`[[slide N: Title]]` sections), the
    format *New with my notes* and `realme import-script` read. The script
    as last saved, or the proposed one while a revision waits."""
    from fastapi.responses import Response
    from realme.pipeline.script_import import to_script_text
    pdir = project_dir(pid)
    try:
        m = lec.load_script(pdir, pending.script_path(pdir))
    except FileNotFoundError:
        raise HTTPException(404, "No narration for this lecture yet")
    try:
        titles = slide_titles(pid).get("titles", [])
    except Exception:
        titles = []
    text, _ = to_script_text(m, titles)
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", pid) + "_notes.md"
    return Response(text, media_type="text/markdown; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.get("/api/lecture/{pid}/slide-titles")
def slide_titles(pid: str):
    """Each slide's first line of text, for the editor's slide list.

    Read from the deck's PDF (a .pptx was converted into _work when the
    lecture was made; it is not converted again here) and cached against the
    deck's modification time. Empty when there is nothing to read: the list
    then shows the narration's opening words instead.
    """
    pdir = _lecture_dir(pid)
    deck = old_deck_path(pdir)
    if deck is None:
        return {"titles": []}
    key = f"{deck}|{deck.stat().st_mtime}"
    cache = pdir / "_work" / "slide_titles.json"
    if cache.is_file():
        try:
            data = json.loads(read_text(cache))
            if data.get("key") == key:
                return {"titles": data.get("titles", [])}
        except ValueError:
            pass
    pdf = deck if deck.suffix.lower() == ".pdf" else \
        pdir / "_work" / (deck.stem + ".pdf")
    if not pdf.is_file():
        return {"titles": []}
    titles = []
    for text in slides_mod.extract_pdf_text(pdf):
        line = next((ln.strip() for ln in (text or "").splitlines()
                     if ln.strip()), "")
        titles.append(line[:80])
    (pdir / "_work" / "slide_titles.json").write_text(
        json.dumps({"key": key, "titles": titles}), encoding="utf-8")
    return {"titles": titles}


@app.get("/api/lecture/{pid}/inserts")
def inserts_list(pid: str):
    return _inserts_view(pid, _lecture_dir(pid))


@app.post("/api/lecture/{pid}/inserts")
async def inserts_add(pid: str, files: list[UploadFile] = File(...),
                      after_slide: int = Form(...), title: str = Form("")):
    """Add a video (and, optionally, its .srt) after a slide."""
    from realme.pipeline import inserts as INS
    pdir = _lecture_dir(pid)
    videos = [f for f in files
              if Path(f.filename or "").suffix.lower() in INS.VIDEO_SUFFIXES]
    subs = [f for f in files if Path(f.filename or "").suffix.lower() == ".srt"]
    if len(videos) != 1:
        raise HTTPException(400, "Choose one video (and, if it has one, its "
                                 ".srt captions).")
    if after_slide < 0:
        raise HTTPException(400, "after_slide is 0 (before slide 1) or a slide "
                                 "number.")
    tmp = UPLOADS / f"insert_{pid}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        vid = await save_upload(
            videos[0], tmp / safe_filename(videos[0].filename, "insert.mp4"),
            "video")
        cap = None
        if subs:
            cap = await save_upload(
                subs[0], tmp / safe_filename(subs[0].filename, "insert.srt"),
                "text")
        item = INS.add(pdir, vid, after_slide, title=title, captions=cap)
    except MediaError as e:
        raise HTTPException(400, str(e))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    out = _inserts_view(pid, pdir)
    out["added"] = item.id
    return out


@app.delete("/api/lecture/{pid}/inserts/{iid}")
def inserts_remove(pid: str, iid: str):
    from realme.pipeline import inserts as INS
    pdir = _lecture_dir(pid)
    try:
        INS.remove(pdir, iid)
    except MediaError as e:
        raise HTTPException(404, str(e))
    return _inserts_view(pid, pdir)


@app.post("/api/lecture/{pid}/inserts/apply")
def inserts_apply(pid: str):
    """Re-join the last render with the inserts as they are now. Records and
    draws nothing; a job, because joining an hour of video takes a minute."""
    pdir = _lecture_dir(pid)
    if not (pdir / "_work" / "assembly.json").is_file():
        raise HTTPException(409, "Render this lecture once first; after that "
                                 "inserts are applied without rendering.")

    def work(job):
        from realme.pipeline import lean
        try:
            name = json.loads(read_text(pdir / "_work" / "assembly.json"))["name"]
        except (OSError, ValueError, KeyError):
            name = ""
        compacted = bool(name) and lean.is_compact(pdir, name)
        if compacted:
            lean.restore(pdir, name, log=job.say)
        out = lec.reassemble(pdir, log=job.say)
        if compacted or LEAN:
            lean.compact(pdir, log=job.say)
        return {"project_id": pid, "report": out["report"],
                "files": {k: Path(v).name for k, v in out.items()
                          if k != "report"}}

    job = runner.submit("render", f"Insert videos — {pid}", _hold(pid, work))
    return {"job_id": job.id}


@app.get("/api/tools/splittable")
def splittable():
    """Rendered videos that carry a cut list, so their slide times are known.

    Only these can be split at a slide boundary. A video without its
    `_segments.json` has no known slide times, and this tool does not
    guess them.
    """
    out = []
    if PROJECTS.is_dir():
        for d in sorted(PROJECTS.iterdir()):
            if not d.is_dir():
                continue
            for seg in sorted(d.glob("*_segments.json")):
                video = seg.with_name(seg.name.replace("_segments.json", ".mp4"))
                if not video.is_file():
                    continue
                try:
                    data = json.loads(read_text(seg))
                except (ValueError, OSError):
                    continue
                out.append({
                    "project_id": d.name, "video": video.name,
                    "slides": len(data.get("segments") or []),
                    "total_s": float(data.get("total_s") or 0.0),
                    "mtime": video.stat().st_mtime})
    out.sort(key=lambda r: r["mtime"], reverse=True)
    return {"videos": out}


@app.get("/api/tools/slides/{pid}/{video}")
def split_points(pid: str, video: str):
    """Every slide with its exact start time and the opening of its narration.

    The times come from the sidecar, measured when the video was assembled.
    The words come from the script if it is still there, so each timestamp
    can be recognised by what is said at it.
    """
    from realme.pipeline.split import load_cuts
    pdir = project_dir(pid)
    mp4 = pdir / Path(video).name
    seg = mp4.with_name(mp4.stem + "_segments.json")
    if not seg.is_file():
        raise HTTPException(404, f"No cut list beside {mp4.name}. Only a video "
                                 f"rendered by RealMe can be split by slide.")
    try:
        cuts, meta = load_cuts(seg)
    except ValueError as e:
        raise HTTPException(400, str(e))
    opening = {}
    try:
        m = lec.load_script(pdir)
        for sgm in m.segments:
            words = (sgm.spoken_text or "").split()
            opening[sgm.slide_index + 1] = " ".join(words[:12]) + (
                "…" if len(words) > 12 else "")
    except Exception:
        pass                     # the script is optional; the times are not
    return {"video": mp4.name, "total_s": meta.get("total_s", 0.0),
            "slides": [{**c, "opening": opening.get(c["slide"], "")}
                       for c in cuts]}


@app.post("/api/tools/split")
def split_video(body: dict):
    """Cut a rendered lecture into parts at slide boundaries."""
    from realme.pipeline import split as SPL
    pid = str(body.get("project_id") or "")
    pdir = project_dir(pid)
    mp4 = pdir / Path(str(body.get("video") or "")).name
    if not mp4.is_file():
        raise HTTPException(404, f"No video {mp4.name} in project {pid}.")
    seg = mp4.with_name(mp4.stem + "_segments.json")
    if not seg.is_file():
        raise HTTPException(404, f"No cut list beside {mp4.name}.")
    try:
        starts = [int(x) for x in (body.get("starts") or [])]
        titles = [str(t) for t in (body.get("titles") or [])]
    except (TypeError, ValueError):
        raise HTTPException(400, "starts must be a list of slide numbers.")
    mode = str(body.get("mode") or "auto")
    if mode not in SPL.MODES:
        raise HTTPException(400, f"mode must be one of {', '.join(SPL.MODES)}")
    try:
        cuts, meta = SPL.load_cuts(seg)
        plan = SPL.plan(cuts, starts, titles, stem=mp4.stem,
                        total_s=meta.get("total_s"))
    except ValueError as e:
        raise HTTPException(400, str(e))
    outdir = pdir / f"{mp4.stem}_parts"
    srt = mp4.with_suffix(".srt")
    as_projects = bool(body.get("as_projects", True))
    if as_projects:
        clash = [Path(x.filename).stem for x in plan.parts
                 if (PROJECTS / Path(x.filename).stem).exists()
                 and any((PROJECTS / Path(x.filename).stem).iterdir())]
        if clash:
            raise HTTPException(400,
                f"Splitting this way would create project(s) that already "
                f"exist: {', '.join(clash)}. Rename the parts, or untick "
                f"'make each part a lecture of its own'.")

    def work(job):
        SPL.execute(mp4, plan, outdir, mode=mode, cuts=cuts, render=meta,
                    srt=srt if srt.is_file() else None, log=job.say)
        built = None
        if as_projects:
            built = SPL.to_projects(mp4, plan, outdir, pdir, PROJECTS, log=job.say)
            # The parts folder has served its purpose once every part has a
            # home of its own. Removed only if it is empty, so a part that
            # could not be assembled is never thrown away.
            try:
                outdir.rmdir()
            except OSError:
                pass
        return {"project_id": pid, "outdir": str(outdir),
                "projects": built,
                "summary": plan.summary(), **plan.as_dict()}

    job = runner.submit("split", f"Split — {mp4.stem}", work)
    return {"job_id": job.id, "parts": len(plan.parts)}


@app.get("/api/tools/joinable")
def joinable():
    """Every video in the projects folder, newest first.

    Wider than `splittable` on purpose: joining does not need a cut list, so
    anything RealMe can see is fair game -- a rendered lecture, a part cut from
    one, a draft, or a clip dropped into a project folder by hand.
    """
    out = []
    if PROJECTS.is_dir():
        for d in sorted(PROJECTS.iterdir()):
            if not d.is_dir():
                continue
            for mp4 in sorted(list(d.glob("*.mp4")) + list(d.glob("*_parts/*.mp4"))):
                out.append({"project_id": d.name,
                            "rel": str(mp4.relative_to(d)).replace("\\", "/"),
                            "name": mp4.name,
                            "size_mb": round(mp4.stat().st_size / 1e6, 1),
                            "mtime": mp4.stat().st_mtime})
    out.sort(key=lambda r: r["mtime"], reverse=True)
    return {"videos": out}


def _video_entries(body: dict) -> list[dict]:
    """The `videos` list of a join request, each {project_id, rel}, or 400."""
    entries = body.get("videos") or []
    if not isinstance(entries, list) or not all(
            isinstance(v, dict) and isinstance(v.get("project_id", ""), str)
            and isinstance(v.get("rel", ""), str) for v in entries):
        raise HTTPException(400, "videos must be a list of "
                                 "{project_id, rel} entries.")
    return entries


def _joinable_path(pid: str, rel: str) -> Path:
    """Resolve one entry, refusing anything outside its project folder."""
    pdir = project_dir(pid).resolve()
    target = (pdir / rel).resolve()
    if not str(target).startswith(str(pdir)) or not target.is_file():
        raise HTTPException(404, f"No video {rel} in project {pid}.")
    return target


@app.post("/api/tools/merge/check")
def merge_check(body: dict):
    """Can these be joined without re-encoding, and if not, why not?"""
    from realme.pipeline import merge as MRG
    paths = [_joinable_path(v.get("project_id", ""), v.get("rel", ""))
             for v in _video_entries(body)]
    try:
        plan = MRG.plan(paths)
    except (ValueError, OSError) as e:
        raise HTTPException(400, str(e))
    return {"summary": plan.summary(), "compatible": plan.compatible,
            "reasons": plan.reasons, "total_s": plan.total_s,
            "inputs": [{"name": i.path.name, "seconds": i.seconds,
                        "note": i.note, "captions": bool(i.srt),
                        "cut_list": bool(i.segments)} for i in plan.inputs]}


@app.post("/api/tools/merge")
def merge_videos(body: dict):
    """
    Join videos, and -- when they carry their slides with them -- assemble the
    result into a project of its own.

    A new project folder rather than the first input's. Two sections taken from
    different lectures are a NEW lecture; filing it under whichever one happened
    to be listed first would bury it, and re-rendering it would sit beside a
    deck it has nothing to do with.
    """
    from realme.pipeline import merge as MRG
    from realme.pipeline import assemble as ASM
    entries = _video_entries(body)
    paths = [_joinable_path(v.get("project_id", ""), v.get("rel", ""))
             for v in entries]
    try:
        plan = MRG.plan(paths)
    except (ValueError, OSError) as e:
        raise HTTPException(400, str(e))
    mode = str(body.get("mode") or "auto")
    name = "".join(c if c.isalnum() or c in "-_" else "_"
                   for c in str(body.get("name") or "joined"))[:60] or "joined"
    want_project = bool(body.get("as_project", True))
    # A forced append: put these end to end whatever they are. For an old
    # recording, or something made outside RealMe, there are no notes and no
    # slides to carry and asking for them is only in the way. The result is
    # still a project folder -- it is just a project holding a video, and the
    # report says exactly what it does not have.
    force = bool(body.get("force", False))
    if force:
        want_project = False

    pdir = project_dir(name)
    if pdir.exists() and any(pdir.iterdir()):
        raise HTTPException(400,
            f"A project called '{name}' already exists. Choose another name "
            f"rather than writing over it.")
    out = pdir / f"{name}.mp4"

    sources = [ASM.Source(video=v, project=project_dir(e.get("project_id", "")),
                          slides=ASM.covered_slides(v))
               for v, e in zip(paths, entries)]

    def work(job):
        pdir.mkdir(parents=True, exist_ok=True)
        MRG.execute(plan, out, mode=mode, log=job.say)
        built = None
        if want_project:
            try:
                a = ASM.assemble(sources, pdir, name,
                                 title=str(body.get("title") or ""), log=job.say)
                built = {"project_id": a.project_id, "slides": a.slides,
                         "deck": a.deck, "sources": a.sources}
            except ValueError as e:
                # The video is finished and correct; only the editable part
                # could not be built. Said in full, and not treated as a
                # failure of the join.
                job.say(f"  the video is joined, but no editable lecture was "
                        f"assembled:\n{e}")
                built = {"error": str(e)}
        if force:
            missing = []
            for src, entry in zip(sources, entries):
                lacks = []
                if not src.slides:
                    lacks.append("slide times")
                if not (Path(src.project) / "_work" / "manifest.json").is_file():
                    lacks.append("notes")
                if not Path(src.video).with_suffix(".srt").is_file():
                    lacks.append("captions")
                if lacks:
                    missing.append(f"{Path(src.video).name}: no "
                                   + ", ".join(lacks))
            (pdir / "join_report.json").write_text(json.dumps({
                "name": name, "joined": [str(x) for x in paths],
                "forced": True, "missing": missing,
                "note": "Appended as files. This project has a video and "
                        "whatever captions its inputs carried; it has no "
                        "narration and no slides, so it cannot be edited or "
                        "re-recorded until notes are imported or drafted.",
            }, indent=2), encoding="utf-8")
            for m in missing:
                job.say(f"  {m}")
            job.say("  forced append: a video, and nothing claimed about its "
                    "contents. See join_report.json.")
            built = {"forced": True, "missing": missing}
        return {"out": str(out), "name": out.name, "summary": plan.summary(),
                "project": built, **plan.as_dict()}

    job = runner.submit("merge", f"Join — {name}", work)
    return {"job_id": job.id}


# ------------------------------------------------------------------ narrate

@app.post("/api/narrate")
async def make_narration(text: str = Form(""), tts: str = Form(""),
                         mode: str = Form("natural"), pace: float = Form(1.0),
                         paragraph_gap: float = Form(0.7),
                         captions: bool = Form(True),
                         temperature: float = Form(0.0),
                         spell_acronyms: str = Form(""),
                         file: UploadFile | None = File(None)):
    """
    Read a text file aloud. No deck, no model call, no script to approve.

    The cheapest thing the system does: a quick loop for checking a
    pronunciation, a reference clip, or how a paragraph sounds.
    """
    _refuse_while_measuring()
    from realme.pipeline import narrate as N

    pid = "narr_" + (Path(file.filename).stem if file and file.filename
                     else "pasted")
    pid = "".join(c if c.isalnum() or c == "_" else "_" for c in pid)[:60]
    pdir = project_dir(pid)
    pdir.mkdir(parents=True, exist_ok=True)

    if file is not None and file.filename:
        src = pdir / safe_filename(file.filename, "source.txt")
        is_docx = src.suffix.lower() == ".docx"
        await save_upload(file, src, "document" if is_docx else "text")
        if is_docx:
            try:
                check_docx(src)
            except HTTPException:
                src.unlink(missing_ok=True)
                raise
    elif text.strip():
        src = pdir / "pasted.txt"
        src.write_text(text, encoding="utf-8")
    else:
        raise HTTPException(400, "Nothing to read: paste some text or "
                                 "choose a file.")

    # `tts` may be "engine" or "engine:voice", the same spelling the dialogue
    # caster uses, so "piper:en_US-lessac-medium" works here too.
    from realme.pipeline import casting
    from realme.adapters.base import AdapterUnavailable
    # Blank means the voice the profile reads with (Gemini unless changed).
    tts = tts.strip() or profile.data["adapters"]["tts"]
    engine_name, _voice_file = casting.split_spec(tts)
    # Per-request settings must not leak into os.environ, or a later request
    # asking for the default would inherit them. Temperature is read at
    # construction, so it is
    # set only for the duration of the build; the spelling mode is passed
    # down as an argument.
    try:
        with _env_scoped({"REALME_TTS_TEMPERATURE":
                          str(temperature) if temperature else None}):
            engine = engine_for(tts, pace=pace)
    except AdapterUnavailable as e:
        raise HTTPException(400, str(e))
    acronym_mode = spell_acronyms.strip() or None
    # Piper takes pace as a constructor setting; the cloning engines take it as
    # a retime after synthesis. Do not apply it twice.
    render_pace = 1.0 if engine_name in casting.PIPER_ENGINES else pace
    out = pdir / (src.stem + ".wav")

    def work(job):
        from realme.pipeline.speak import spend_of, took_and_cost
        import time as _t
        t0, spent_before = _t.perf_counter(), spend_of(engine)
        with SYNTH_LOCK:
            r = N.narrate(src, out, engine, mode=mode, pace=render_pace,
                          paragraph_gap_s=paragraph_gap, captions=captions,
                          acronym_mode=acronym_mode, log=job.say)
        files = {"audio": r.wav.name}
        if r.srt:
            files["captions"] = r.srt.name
        return {"project_id": pid,
                "report": {"duration_s": round(r.duration_s, 1),
                           "words": r.words, "paragraphs": r.paragraphs,
                           "utterances": r.utterances,
                           "voice": engine_name,
                           **took_and_cost(_t.perf_counter() - t0, spent_before,
                                           spend_of(engine), engine),
                           "warnings": r.warnings[:12]},
                "files": files}

    job = runner.submit("narrate", f"Narrate — {src.name}", work)
    return {"job_id": job.id, "project_id": pid}


# ------------------------------------------------------------------ dialogue

@app.post("/api/dialogue")
async def make_dialogue(topic: str = Form(...), mode: str = Form("debate"),
                        turns: int = Form(16), script_writer: str = Form("gemini"),
                        gemini_model: str = Form(GEMINI_DEFAULT),
                        instructor_tts: str = Form("espeak"),
                        guest_tts: str = Form("piper"),
                        speed: float = Form(0.0), turn_gap: float = Form(0.0),
                        bookends: bool = Form(True),
                        file: UploadFile | None = File(None)):
    _refuse_while_measuring()
    pid = "dlg_" + "".join(c if c.isalnum() else "_" for c in topic)[:50]
    pdir = project_dir(pid)
    pdir.mkdir(parents=True, exist_ok=True)
    src = None
    if file is not None and file.filename:
        src = pdir / safe_filename(file.filename, "source.txt")
        await save_upload(file, src, "document"
                          if src.suffix.lower() in (".pdf", ".docx") else "text")

    writer = make_writer(script_writer, gemini_model)
    # Through the shared caster, exactly as `realme dialogue` does, so the
    # page honours the same settings as the CLI -- pace, guest voice and so on.
    from realme.pipeline import casting
    from realme.adapters.base import AdapterUnavailable
    try:
        voices = casting.cast(instructor_tts, guest_tts,
                              speed=speed or None, profile=profile)
    except AdapterUnavailable as e:
        raise HTTPException(400, str(e))

    def work(job):
        from realme.pipeline.speak import spend_of, took_and_cost
        import time as _t
        for role, spec in (("instructor", instructor_tts), ("guest", guest_tts)):
            job.say(f"      {role}: {casting.describe(spec, voices[role])}")
        # Both voices, because either side of a conversation can be the
        # hosted one and the bill is the pair.
        t0 = _t.perf_counter()
        before = [spend_of(voices[r]) for r in ("instructor", "guest")]
        out = dlg.build_dialogue(src, topic, pdir, writer, voices,
                                 mode=mode, turns=turns,
                                 turn_gap=turn_gap or dlg.DEFAULT_TURN_GAP_S,
                                 host_name=(profile.data.get("display_name")
                                            or "your host"),
                                 guest_name=casting.speaker_name(
                                     guest_tts, voices["guest"]),
                                 bookends=bookends, log=job.say)
        after = [spend_of(voices[r]) for r in ("instructor", "guest")]
        paid = [(b, a, voices[r]) for b, a, r in
                zip(before, after, ("instructor", "guest")) if b is not None]
        report = dict(out["report"])
        report["voices"] = {"instructor": instructor_tts, "guest": guest_tts}
        report.update(took_and_cost(
            _t.perf_counter() - t0,
            sum(b for b, _, _ in paid) if paid else None,
            sum(a for _, a, _ in paid) if paid else None,
            paid[0][2] if paid else None))
        return {"project_id": pid, "report": report,
                "files": {"audio": Path(out["audio"]).name,
                          "transcript": Path(out["transcript"]).name}}

    job = runner.submit("dialogue", f"{mode.title()} — {topic[:40]}", work)
    return {"job_id": job.id, "project_id": pid}


# ------------------------------------------------------------------ jobs/files

@app.get("/api/jobs")
def list_jobs():
    return runner.recent()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = runner.get(job_id)
    if not job:
        raise HTTPException(404, "No such job")
    return job.public()


@app.get("/api/files/{pid}/{name}")
def get_file(pid: str, name: str):
    # The route pattern keeps `/` out of `name` but not `\`, and on Windows
    # `..\..\.env` is two parent hops -- to the file the API keys live in.
    pdir = project_dir(pid)
    if Path(name).name != name or "\\" in name or name.startswith("."):
        raise HTTPException(404, "No such file")
    f = (pdir / name).resolve()
    if f.parent != pdir or not f.is_file():
        raise HTTPException(404, "No such file")
    return FileResponse(f, filename=f.name)


@app.get("/api/slides/{pid}/{idx}")
def get_slide(pid: str, idx: int):
    from realme.pipeline.slides import slide_pages
    slides = slide_pages(pending.slides_dir(project_dir(pid)))
    if not slides:
        raise HTTPException(404, "No slides rendered")
    # No clamp. `min(idx, len-1)` would serve the last slide for any index
    # past the end, hiding an off-by-one in the manifest behind a plausible
    # picture. An impossible index is an error.
    if not 0 <= idx < len(slides):
        raise HTTPException(
            404, f"Slide {idx} does not exist; this deck has {len(slides)} "
                 f"(0-{len(slides) - 1}). The manifest points outside the deck.")
    return FileResponse(slides[idx])


def is_loopback(host: str) -> bool:
    """Is `host` an address only this computer can reach?"""
    import ipaddress
    h = (host or "").strip().strip("[]").lower()
    if h == "localhost":
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


NETWORK_WARNING = (
    "WARNING: the Studio has no login. Bound to {host}, anyone who can reach "
    "this computer on port {port} can read your lectures, use your cloned "
    "voice and spend on your API keys. Use it only on a network you trust.")


def serve(host="127.0.0.1", port=8000, allow_network: bool = False):
    import sys
    import uvicorn
    if not is_loopback(host):
        # Refused by default: the API is unauthenticated, so a bind address
        # other machines can reach hands it to them. `--host 0.0.0.0` also
        # needs `--allow-network`.
        if not allow_network:
            raise ValueError(
                f"Refusing to listen on {host}: the Studio has no login, and "
                f"that address can be reached from other machines. Use the "
                f"default (127.0.0.1), or add --allow-network if you really "
                f"mean to share it on a network you trust.")
        print(NETWORK_WARNING.format(host=host, port=port), file=sys.stderr)
    if host in ("0.0.0.0", "::"):
        print("Listening on every interface. Other machines must reach it by "
              "a name listed in REALME_ALLOWED_HOSTS, or they are refused.")
    else:
        _EXTRA_HOSTS.add(_host_name(host))
    # Started here rather than at import: the CLI, the tests and `realme
    # migrate` all import this module and none of them want 2 GB loaded.
    if not os.environ.get("REALME_NO_WARMUP"):
        warm_engine()
    uvicorn.run(app, host=host, port=port, log_level="warning")
