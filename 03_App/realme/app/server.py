"""
RealMe Studio -- local web app.

Runs on your own machine or inside a Colab CPU runtime. Serves on 127.0.0.1 by
default and never opens a tunnel: in Colab, reach it through the built-in port
proxy (`google.colab.output.serve_kernel_port_as_window(8000)`) rather than a
share link, which keeps you clear of the "remote proxies" clause in Colab's
terms of service.
"""
from __future__ import annotations
import json, os, re, threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

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

_LOCAL_ORIGINS = ("http://127.0.0.1", "http://localhost", "http://[::1]")


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
    if request.method in ("POST", "PUT", "DELETE", "PATCH"):
        origin = request.headers.get("origin", "")
        site = request.headers.get("sec-fetch-site", "")
        foreign = (origin and not origin.startswith(_LOCAL_ORIGINS)) \
            or site == "cross-site"
        if foreign:
            return JSONResponse({"detail": "Cross-site request refused"},
                                status_code=403)
    return await call_next(request)
runner = JobRunner(state_path=DATA / "jobs.json")
profile = Profile(DATA / "profile")

# TTS price per million characters, for the pre-render estimate.
#: Dollars per million CHARACTERS of narration -- which is the unit
#: `lec.estimate_cost` divides by, and the unit the other entries here were
#: always in. Local engines are absent and cost nothing.
#:
#: gemini-tts is billed per audio token, not per character, so it has to be
#: converted: $9 per million audio tokens at ~32 tokens a second is $1.04 an
#: hour of speech, and this prose runs at about 15 characters a second
#: (measured on the probe's own calibration clips), so a million characters is
#: 18.5 hours and about $19.30. Both halves of the Google price double on
#: 1 January 2027.
#:
#: Written out because the first version of this line said "$1.04" with a
#: comment calling it dollars per hour -- a number that was right about the
#: world and wrong about this dictionary, and would have quoted a 10,000
#: character lecture at one cent.
PRICE = {"espeak": 0.0, "google_chirp3": 60.0, "elevenlabs": 165.0,
         "chatterbox": 0.0, "gemini-tts": 19.30}


#: What a slide added by a revision carries until narration is written for it.
#: Named rather than typed twice: the revision writes it and the drafting
#: endpoint looks for it, and two spellings of the same sentinel would leave
#: slides that can never be drafted.
NEW_SLIDE_PLACEHOLDER = "(new slide - write the narration)"


def make_writer(name: str, model: str):
    if name == "placeholder":
        return PlaceholderScriptWriter()
    if name == "gemini":
        return GeminiScriptWriter(model=model)
    raise HTTPException(400, f"Unknown script writer '{name}'")


from realme.adapters.script_writer import DEFAULT_MODEL as GEMINI_DEFAULT
from realme.pipeline import pending


def build_tts(name: str, **overrides):
    """
    Construct an adapter with everything the profile knows about your voice.

    Delegates to the shared factory. Three call sites used to do this three
    different ways and each got it wrong differently; see
    realme/adapters/factory.py.
    """
    from realme.adapters.factory import build
    from realme.adapters.base import AdapterUnavailable
    try:
        return build(name, profile=profile, **overrides)
    except AdapterUnavailable as e:
        raise HTTPException(400, str(e))


# One adapter per engine, shared by every preview and render, and one lock
# around synthesis. A fresh `build_tts()` per request meant a full model load
# for every preview click and, on the PyTorch path, a worker process left
# behind each time; two renders submitted together drove two DLL contexts
# at once.
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

    `overrides` is part of the key, not decoration: "piper with Amy" and
    "piper with Ryan" are different engines to everyone except the cache.
    """
    key = name + "|" + repr(sorted(overrides.items()))
    with _TTS_LOCK:
        hit = _TTS_CACHE.get(key)
        if hit is not None:
            return hit
        tts = _Serialised(build_tts(name, **overrides))
        _TTS_CACHE[key] = tts
        return tts


def engine_for(spec: str, *, pace: float | None = None):
    """The adapter for "engine" or "engine:voice", from the shared cache.

    Every path that speaks goes through here. Three of them used to call
    `factory.build` directly, so the engine the Studio had already loaded in
    the background was ignored and a second copy of a 2 GB model was loaded
    at the first Render -- both copies then sitting in memory for the life of
    the process.
    """
    from realme.pipeline import casting
    name, voice_file = casting.split_spec(spec)
    overrides = casting.voice_overrides(
        name, voice_file, pace if name in casting.PIPER_ENGINES else None)
    return make_tts(name, **overrides)


#: What the background warm-up is doing. Read by /api/doctor so the page can
#: say "loading" instead of looking idle while 2 GB comes off disk.
WARM: dict = {"engine": "", "state": "idle", "detail": "", "seconds": 0.0}


def warm_engine(name: str = "", *, block: bool = False) -> None:
    """Load the speech engine now, in the background, into the shared cache.

    The model load is tens of seconds and it used to happen at the first
    click of Preview or Render -- so the cost landed exactly when somebody
    was waiting to hear something. Nothing else needs it: uploading a deck
    and drafting a script are a model call and some PDF work. Doing it while
    they read the draft is free wall-clock.

    Failure here is not fatal and is deliberately not raised: the engine is
    reported as it always was by /api/doctor, and the first render will
    raise the same AdapterUnavailable with the same advice. Set
    REALME_NO_WARMUP=1 to keep the memory until it is asked for.
    """
    import time
    from realme.adapters.tts import DEVELOPED as _DEV
    name = name or profile.data.get("adapters", {}).get("tts") or _DEV[0]

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

    Only spaces used to be replaced, so `Lecture #3.pdf` became a folder the
    UI then asked for as `/api/lecture/Lecture_` -- 404, and the editor never
    opened.
    """
    stem = Path(re.split(r"[\\/]", filename or "")[-1]).stem
    pid = _PID_OK.sub("_", stem)
    while "__" in pid:
        pid = pid.replace("__", "_")
    pid = pid.strip("_")[:60]
    return pid or fallback


def safe_filename(name: str | None, fallback: str) -> str:
    """The base name of an upload, with nothing that can leave the folder.

    Starlette passes `filename` through untouched, so `../../x.pdf` was
    joined to the project directory as written.
    """
    # Split on both separators: on Linux (Colab) a backslash is not one,
    # and the name would keep it.
    base = re.split(r"[\\/]", name or "")[-1]
    base = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", base).strip(" .")
    return base or fallback


def project_dir(pid: str) -> Path:
    root = PROJECTS.resolve()
    p = (root / pid).resolve()
    # `startswith` accepted a sibling folder that merely shares the prefix
    # (`projects_x`); is_relative_to does not.
    if p == root or not p.is_relative_to(root) or not pid or "/" in pid or "\\" in pid:
        raise HTTPException(400, "Bad project id")
    return p


# ------------------------------------------------------------------ pages

@app.get("/", response_class=HTMLResponse)
def index():
    return (HERE / "ui.html").read_text(encoding="utf-8")


# ------------------------------------------------------------------ status

@app.get("/api/doctor")
def doctor():
    # Re-read .env on every check, not once at import.
    #
    # Otherwise the obvious thing to do after setting a key -- refresh the page
    # -- cannot possibly work: the process cached the environment at startup,
    # so the browser re-renders the same stale answer and it looks like the key
    # was never saved. Reading a small file per status call costs nothing.
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
    for nm, w in (("gemini", GeminiScriptWriter(model=GEMINI_DEFAULT)),
                  ("placeholder", PlaceholderScriptWriter())):
        try:
            w.preflight(); writers[nm] = {"ok": True, "detail": "ready"}
        except AdapterUnavailable as e:
            writers[nm] = {"ok": False, "detail": str(e)}

    # "No key" is a dead end unless it also says where it looked. These are the
    # three paths env.candidates() searches, in order, with what each holds.
    from realme.core.env import candidates, data_home
    key_search = [{"path": str(c), "exists": c.is_file(),
                   "has_gemini": bool(c.is_file() and "GEMINI_API_KEY" in
                                      c.read_text(encoding="utf-8-sig",
                                                  errors="ignore"))}
                  for c in candidates()]

    voices = {}
    have_ref = bool(profile.data.get("voice_reference"))
    # Only the engines this project actually sets up, and asked the cheap
    # question. This loop used to walk the whole registry calling
    # `preflight()`, which for qwen3cpp IS the 2 GB model load -- so every
    # page refresh loaded the model, threw it away, and the first render
    # then loaded it again. It also tried to import torch on behalf of
    # engines nobody here has installed.
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
                          "cloning_you": bool(inst.is_voice_clone and have_ref),
                          "placeholder": inst.is_placeholder}
        except Exception as e:
            voices[nm] = {"ok": False, "detail": str(e)[:1200],
                          "clone": getattr(cls, "is_voice_clone", False),
                          "cloning_you": False,
                          "placeholder": getattr(cls, "is_placeholder", False)}
    from realme.core import textio
    return {"binaries": bins, "writers": writers, "voices": voices,
            # Named, not probed. They stay selectable from the command line;
            # what would be dishonest is a status pill for an engine this
            # project neither installs nor tests.
            "other_engines": NOT_DEVELOPED,
            "engine_warm": dict(WARM),
            "pdf_backend": pdfdoc.backend(),
            # Files this session found written in the old Windows encoding and
            # rewrote as utf-8. Worth surfacing: it explains why a project made
            # before September 2026 opened after a pause, and it is the only
            # trace of a repair that deliberately leaves no other mark.
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
    profile.update(patch)
    return get_profile()


@app.get("/api/lexicon")
def lexicon_list():
    """Your pronunciations, and the ones that ship with the project.

    Both, and marked, because only yours survive an update -- a list that did
    not distinguish them would invite someone to rely on a built-in entry that
    the next release quietly replaces. Yours are editable here; the built-ins
    are shown so you can see a term is already handled before adding it again.
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


@app.delete("/api/lexicon/{term}")
def lexicon_remove(term: str):
    """Remove one of yours. A built-in of the same name applies again."""
    from realme.text.lexicon import (load_user_entries, save_user_entries,
                                     Lexicon)
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
    # A piper voice is the .onnx AND its .onnx.json, and the two answers
    # disagreed: this endpoint checked the model alone, while the loader
    # checks the pair. A voice with a missing .json therefore showed as
    # installed here and failed at load -- and one whose folder was not where
    # this looked showed as missing with nowhere to look it up.
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
def api_models():
    """
    Which Gemini models this key can call -- asked, not remembered.

    The Studio's model list was three hardcoded ids, one of them labelled
    "free tier" and one of them (`gemini-3-pro-preview`) shut down by Google
    while it sat in the dropdown. A list written into a web page is out of date
    the moment a preview retires, and the failure is a bare HTTP 404 from a URL
    containing the model name, which reads like a broken endpoint.

    So the page asks. If the key is missing or the call fails, the configured
    default is offered alone, with the reason -- a dropdown with one working
    entry beats a dropdown of plausible dead ones.
    """
    from realme.adapters.script_writer import list_models, DEFAULT_MODEL
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
    tmp.write_bytes(await file.read())
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
                # its word count by this recording's speech time is how a
                # 323 words-a-minute "measurement" happened.
                drift = {"note": profile.transcript_note}
        except Exception as e:                     # never block an enrolment
            drift = {"note": "", "error": str(e)[:200]}
    return {"stored": str(dst), "duration_s": round(secs, 2),
            "pace_note": drift.get("note", ""),
            "wpm": drift.get("wpm"),
            "transcript_note": profile.transcript_note,
            "ready_for_cloning": profile.ready_for_cloning}


# ------------------------------------------------------------------ lecture

@app.post("/api/lecture/script")
async def lecture_script(file: UploadFile = File(...), script_writer: str = Form("gemini"),
                         gemini_model: str = Form(GEMINI_DEFAULT),
                         course_context: str = Form(""), style_notes: str = Form(""),
                         seconds_per_slide: int = Form(0),
                         width: int = Form(1920)):
    if not file.filename:
        raise HTTPException(400, "No deck uploaded")
    pid = project_id_from(file.filename)
    pdir = project_dir(pid)
    pdir.mkdir(parents=True, exist_ok=True)
    deck = pdir / safe_filename(file.filename, "deck.pdf")
    deck.write_bytes(await file.read())

    writer = make_writer(script_writer, gemini_model)

    def work(job):
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
    return {"job_id": job.id, "project_id": pid}


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

    `deck_path.txt` is written by every path that ingests a deck. Projects made
    before it existed fall back to whatever deck is sitting in the project
    folder, which is where all three ingest paths put it.
    """
    recorded = pdir / "_work" / "deck_path.txt"
    if recorded.is_file():
        p = Path(read_text(recorded).strip())
        if p.is_file():
            return p
        # Recorded but moved: the folder is the next best guess, and saying
        # nothing here is what made the last failure so hard to see.
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
    # a deck, and writing the upload to `pdir / file.filename` destroyed the old
    # one before it could be read; and nothing a comparison produces belongs in
    # the project until the Save button says so.
    staged = pending.begin(pdir)
    deck = staged / "deck" / safe_filename(file.filename, "deck.pdf")
    deck.write_bytes(await file.read())

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
            # Loud, because this is the condition that used to pass silently.
            # Matching on empty text falls back to pixels, and pixels report a
            # re-themed deck as forty new slides.
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
            # reported as one removal and one addition. Saying so is honest;
            # pairing them automatically would move narration onto a slide that
            # only looks similar, which is the one mistake worth avoiding here.
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

    The importer and its strict 1..N validation existed from the start; the
    lecture page just never offered a way in, so anyone with their own notes
    had to use the command line or let the model rewrite what they had
    already written.
    """
    from realme.pipeline.script_import import (parse_slide_script, to_manifest,
                                               ScriptImportError)
    if not file.filename:
        raise HTTPException(400, "No deck uploaded")
    if notes is None or not notes.filename:
        raise HTTPException(400, "No notes file uploaded")
    pid = project_id_from(file.filename)
    pdir = project_dir(pid)
    pdir.mkdir(parents=True, exist_ok=True)
    deck = pdir / safe_filename(file.filename, "deck.pdf")
    deck.write_bytes(await file.read())

    raw = (await notes.read()).decode("utf-8", "replace")
    try:
        imported = parse_slide_script(raw)
    except ScriptImportError as e:
        raise HTTPException(400, str(e))

    def work(job):
        # Rasterise, exactly as the drafting path does -- the editor shows
        # slide images beside the narration either way.
        work_dir = pdir / "_work"
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "deck_path.txt").write_text(str(deck), encoding="utf-8")
        pdf = slides_mod.deck_to_pdf(deck, work_dir)
        pages = slides_mod.rasterize(pdf, work_dir / "slides", width_px=width)
        job.say(f"{len(pages)} slides")
        m = to_manifest(imported, pid)
        job.say(f"{len(m.segments)} sections ({imported.marker_format} markers)")
        if len(m.segments) != len(pages):
            # Refused rather than rendered: a script with more or fewer
            # sections than the deck has slides maps onto the wrong pictures,
            # and that is the one failure nobody notices until the video is
            # watched.
            raise ValueError(
                f"The notes have {len(m.segments)} slide sections but the deck "
                f"has {len(pages)} slides. Fix the markers and import again.")
        for w in imported.warnings:
            job.say(f"  ! {w}")
        lec.save_script(pdir, m)
        return {"project_id": pid, "segments": len(m.segments),
                "slides": len(pages), "script_source": m.script_source}

    job = runner.submit("import", f"Import notes — {pid}", work)
    return {"job_id": job.id, "project_id": pid}


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
    pdir = project_dir(pid)
    m = Manifest.model_validate(body["manifest"])
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
            # Loud, because the alternative is narration written against the
            # wrong pictures -- the failure nobody notices until they watch it.
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
    data = await file.read()
    suffix = Path(safe_filename(file.filename, "in.wav")).suffix or ".wav"
    # The analysis runs ffmpeg several times and beautifies once per preset;
    # under `async def` that blocked the event loop and froze every tab's
    # job polling for the duration. Off the loop, into the thread pool.
    from starlette.concurrency import run_in_threadpool
    return await run_in_threadpool(_voice_analyze_sync, data, suffix)


def _voice_analyze_sync(data: bytes, suffix: str):
    from realme.enrollment import voice as V
    from dataclasses import asdict
    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    raw = VOICE_DIR / "reference_raw.wav"
    tmp = VOICE_DIR / ("_upload" + suffix)
    tmp.write_bytes(data)
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
        # Swallowing this made the preview endpoint later say "upload a
        # reference recording first" about a recording that had been uploaded.
        presets_error = f"{type(e).__name__}: {e}"
    return {"report": asdict(report),
            "presets": [{"name": k, "label": v["label"]} for k, v in V.PRESETS.items()],
            "presets_error": presets_error}


@app.post("/api/voice/tune")
def voice_tune(body: dict):
    """
    Render the reference through one custom Treatment so it can be heard.

    Sliders are useless without this: nobody can predict what "vividness +0.12"
    does by reading it. One render per change, straight back to the browser.
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
    # line now refreshes the raw copy, so this should never fire -- but if it
    # does, the alternative is silently replacing the voice you are using with
    # one you recorded weeks ago.
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
        t = V.Treatment(**{k: float(v) for k, v in treatment.items()
                           if k in ("pace", "pitch_semitones", "warmth", "clarity",
                                    "smoothing", "brightness", "vigor", "vividness")})
        V.treat(raw, baked, t.validated())
        profile.update({"voice_reference": str(baked), "voice_preset": "custom",
                        "voice_treatment": treatment})
    else:
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
        # The EFFECTIVE correction, not the stored one. The Studio's badge
        # read "pace retimed by 1.25" off the stored field while the adapter
        # was free to ignore it -- a number on screen that no render used.
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

    Exists because a voice can be created there and lost here: the first
    version of the creation parser did not recognise the response shape and
    reported a failure for a voice that had been made. Without this the only
    remedy was paying to make another one.
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

    Costs one short render -- about a cent. The alternative is choosing a
    number by listening, which is how a lecture ends up 5% slow for a term.
    """
    _load_env()
    from realme.enrollment.pace import calibrate_engine
    from realme.core.env import data_home
    g = dict(profile.data.get("gemini_voice") or {})
    if not g.get("voice"):
        raise HTTPException(400, "No Gemini voice enrolled yet.")
    ref = profile.data.get("voice_reference")
    text = (profile.reference_text or "").strip()
    if not ref or not text:
        raise HTTPException(
            400, "Measuring a pace needs the enrolment recording and its "
                 "transcript. Add the transcript in Twin Setup.")
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
        raise HTTPException(400, "No GEMINI_API_KEY. Set it in Twin Setup "
                                 "first -- it is the same key the script "
                                 "writer uses.")
    if not st["paid_tier_ack"]:
        raise HTTPException(400, G.TIER_NOTICE)
    ref = profile.data.get("voice_reference")
    if not ref or not Path(ref).is_file():
        raise HTTPException(400, "Record and save your voice reference first.")
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


@app.post("/api/preview")
def preview_utterance(body: dict):
    """Hear one utterance. Cheap loop for catching a mangled term early."""
    from realme.pipeline.speak import preview as _preview
    tts = engine_for(body.get("tts") or profile.data["adapters"]["tts"])
    try:
        tts.preflight()
    except AdapterUnavailable as e:
        raise HTTPException(400, str(e))
    out = DATA / "previews" / "preview.wav"
    return _preview(body.get("text", ""), tts, out, index=int(body.get("index", 0)))


@app.get("/api/preview/audio")
def preview_audio():
    f = DATA / "previews" / "preview.wav"
    if not f.exists():
        raise HTTPException(404, "No preview yet")
    return FileResponse(f, media_type="audio/wav")


@app.post("/api/lecture/{pid}/render")
def render_lecture(pid: str, body: dict | None = None):
    body = body or {}
    pdir = project_dir(pid)
    if pending.exists(pdir):
        # Refused rather than rendered. Rendering would have to pick one of two
        # decks, and picking silently is how someone ends up with a video of
        # the version they had not agreed to.
        raise HTTPException(409,
            "A revised deck is waiting to be accepted. Save the notes to apply "
            "it, or discard the revision, then render.")
    m = lec.load_script(pdir)
    # Through the helper, which already knows how to find a deck for a
    # project made before `deck_path.txt` existed. Reading the file directly
    # here meant an older project -- exactly the kind somebody reopens to
    # re-record -- died on a bare FileNotFoundError shown as a 500.
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
        tts = engine_for(spec)
    except AdapterUnavailable as e:
        raise HTTPException(400, str(e))
    # A draft goes to its own filename. Same deck, same script, one fiftieth
    # of the time -- and nothing about the name would otherwise distinguish it
    # from the finished render it would replace.
    suffix = "" if getattr(tts, "is_voice_clone", False) else "_draft"

    def work(job):
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
        return {"project_id": pid, "report": out["report"],
                "files": {k: Path(p).name for k, p in out.items() if k != "report"}}

    job = runner.submit("render", f"Render lecture — {pid}", work)
    return {"job_id": job.id}


# -------------------------------------------------------------------- tools

@app.get("/api/tools/splittable")
def splittable():
    """Rendered videos that carry a cut list, so their slide times are known.

    Only these can be split at a slide boundary. A video without its
    `_segments.json` is just a video, and guessing where slide 21 begins is
    exactly what this tool exists not to do.
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
    The words come from the script if it is still there -- they are what makes
    a list of timestamps choosable, since nobody remembers what slide 21 was.
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
    starts = [int(x) for x in (body.get("starts") or [])]
    titles = [str(t) for t in (body.get("titles") or [])]
    mode = str(body.get("mode") or "auto")
    if mode not in SPL.MODES:
        raise HTTPException(400, f"mode must be one of {', '.join(SPL.MODES)}")
    try:
        cuts, meta = SPL.load_cuts(seg)
        plan = SPL.plan(cuts, starts, titles, stem=mp4.stem)
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
             for v in (body.get("videos") or [])]
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
    entries = body.get("videos") or []
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
async def make_narration(text: str = Form(""), tts: str = Form("qwen3cpp"),
                         mode: str = Form("natural"), pace: float = Form(1.0),
                         paragraph_gap: float = Form(0.7),
                         captions: bool = Form(True),
                         temperature: float = Form(0.0),
                         spell_acronyms: str = Form(""),
                         file: UploadFile | None = File(None)):
    """
    Read a text file aloud. No deck, no model call, no script to approve.

    The cheapest thing the system does and the last to reach the Studio, which
    is backwards: it is the step you want when checking a pronunciation, a
    reference clip, or how a paragraph lands -- the loop you run twenty times
    an afternoon, and the one that least deserves a command line.
    """
    from realme.pipeline import narrate as N

    pid = "narr_" + (Path(file.filename).stem if file and file.filename
                     else "pasted")
    pid = "".join(c if c.isalnum() or c == "_" else "_" for c in pid)[:60]
    pdir = project_dir(pid)
    pdir.mkdir(parents=True, exist_ok=True)

    if file is not None and file.filename:
        src = pdir / safe_filename(file.filename, "source.txt")
        src.write_bytes(await file.read())
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
    engine_name, _voice_file = casting.split_spec(tts)
    # Per-request settings used to be written into os.environ and never
    # cleared, so a later request asking for the default inherited whatever
    # the previous one set. Temperature is read at construction, so it is
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
    pid = "dlg_" + "".join(c if c.isalnum() else "_" for c in topic)[:50]
    pdir = project_dir(pid)
    pdir.mkdir(parents=True, exist_ok=True)
    src = None
    if file is not None and file.filename:
        src = pdir / safe_filename(file.filename, "source.txt")
        src.write_bytes(await file.read())

    writer = make_writer(script_writer, gemini_model)
    # Through the shared caster, exactly as `realme dialogue` does. Building
    # the two adapters here by hand is why the web page ignored every dialogue
    # setting the CLI had -- pace, guest voice, the lot.
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
    # No clamp. `min(idx, len-1)` quietly served the last slide for any index
    # past the end, which is exactly how a one-off in the manifest looked like
    # "the last slide is duplicated" instead of like an error -- the editor
    # showed a plausible picture for an impossible index, and the real fault
    # stayed hidden a level up.
    if not 0 <= idx < len(slides):
        raise HTTPException(
            404, f"Slide {idx} does not exist; this deck has {len(slides)} "
                 f"(0-{len(slides) - 1}). The manifest points outside the deck.")
    return FileResponse(slides[idx])


def serve(host="127.0.0.1", port=8000):
    import uvicorn
    # Started here rather than at import: the CLI, the tests and `realme
    # migrate` all import this module and none of them want 2 GB loaded.
    if not os.environ.get("REALME_NO_WARMUP"):
        warm_engine()
    uvicorn.run(app, host=host, port=port, log_level="warning")
