#!/usr/bin/env python3
"""
The server and CLI as another program sees them.

  1. The server answers only to loopback host names.
  2. `--json` gives a program one object to parse.
  3. A spending cap stops a paid render before it goes past it.
  4. The routes declare what they take.

Each section has a control: the same input under the permissive setting (a
host on the list, no cap), so a check that passes because nothing was
exercised shows up as a failed control.
"""
from __future__ import annotations
import contextlib, io, json, math, os, struct, sys, tempfile, wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

_ISO = tempfile.mkdtemp(prefix="t_agent_")
os.environ["REALME_HOME"] = str(Path(_ISO) / "home")
os.environ["REALME_TOOLS"] = str(Path(_ISO) / "tools")
os.environ["HOME"] = _ISO
os.environ["USERPROFILE"] = _ISO
os.environ["REALME_NO_WARMUP"] = "1"
for k in ("REALME_MAX_SPEND_USD", "REALME_ALLOWED_HOSTS",
          "COLAB_RELEASE_TAG"):
    os.environ.pop(k, None)

from t_util import check, FAILED, summary          # noqa: E402

WORK = Path(_ISO) / "work"
WORK.mkdir(parents=True, exist_ok=True)
def tone(path: Path, seconds: float, rate=24000):
    n = int(rate * seconds)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", int(6000 * math.sin(
            2 * math.pi * 180 * i / rate))) for i in range(n)))


class PaidStub:
    """Charges a cent a call, like a hosted engine that reports its spend."""
    name = "paidstub"
    is_voice_clone = False
    is_placeholder = False
    speaks_languages = ()
    phoneme_syntax = None
    retimes_after = False

    def __init__(self, per_call=0.01):
        self.per_call, self.calls, self._spent = per_call, [], 0.0

    @property
    def spend_usd(self):
        return self._spent

    def preflight(self):
        pass

    def voice_fingerprint(self):
        return "paidstub:v1"

    def timing_factor(self, pace=1.0):
        return 1.0

    def synthesize(self, text, out_wav, voice=None, pace=1.0, **kw):
        self.calls.append(text)
        self._spent += self.per_call
        tone(Path(out_wav), 0.4)
        return Path(out_wav)


class FreeStub(PaidStub):
    """A local engine: no `spend_usd` at all."""
    spend_usd = None


def section_host():
    print("\n1d. the server answers only to loopback names")
    from fastapi.testclient import TestClient
    import realme.app.server as S
    tc = TestClient(S.app, base_url="http://127.0.0.1")
    for host, want in (("127.0.0.1", 200), ("localhost:8000", 200),
                       ("[::1]:8000", 200), ("evil.example", 421),
                       ("127.0.0.1.evil.example", 421),
                       ("localhost.evil.example:8000", 421)):
        r = tc.get("/", headers={"host": host})
        check(r.status_code == want, f"GET / with Host {host} -> {want}",
              str(r.status_code))
    r = TestClient(S.app, base_url="http://evil.example").post(
        "/api/lecture/x/render", json={})
    check(r.status_code == 421, "and a write from a rebinding page is refused "
                                "before any route runs", str(r.status_code))
    os.environ["REALME_ALLOWED_HOSTS"] = "studio.lan, other.lan"
    try:
        r = TestClient(S.app, base_url="http://studio.lan:8000").get("/")
        check(r.status_code == 200, "REALME_ALLOWED_HOSTS adds a name",
              str(r.status_code))
    finally:
        os.environ.pop("REALME_ALLOWED_HOSTS")
    check(S._host_name("LocalHost:8000") == "localhost"
          and S._host_name("[::1]:8000") == "[::1]"
          and S._host_name("127.0.0.1") == "127.0.0.1",
          "Host headers are parsed with and without ports")

def section_spend():
    print("\n3. a spending cap stops a paid render before it goes past it")
    from realme.pipeline import speak
    from realme.adapters.base import SpendCapReached, QuotaExceeded

    paid = PaidStub(0.01)
    capped = speak.spend_capped(paid, 0.025)
    out = WORK / "s.wav"
    stopped = None
    for i in range(10):
        try:
            capped.synthesize(f"line {i}", out)
        except SpendCapReached as e:
            stopped = e
            break
    check(stopped is not None and len(paid.calls) == 3,
          "three one-cent calls are made under a 2.5 cent cap, the fourth is "
          "refused", f"calls={len(paid.calls)}")
    check(isinstance(stopped, QuotaExceeded),
          "and it is a kind of quota, so every caller that stops cleanly at "
          "a quota stops cleanly here")
    check(stopped is not None and "--max-spend" in str(stopped),
          "the message says how to continue")

    later = speak.spend_capped(paid, 0.025)
    later.synthesize("again", out)
    check(len(paid.calls) == 4,
          "the cap counts from when it was applied, so it is per render")

    free = FreeStub()
    fcap = speak.spend_capped(free, 0.0)
    for i in range(5):
        fcap.synthesize(f"x{i}", out)
    check(len(free.calls) == 5, "a local engine is never stopped")

    check(speak.spend_capped(paid) is paid,
          "with no cap set, the engine is returned unchanged")
    os.environ["REALME_MAX_SPEND_USD"] = "1.5"
    try:
        w = speak.spend_capped(PaidStub())
        check(isinstance(w, speak.SpendCapped) and w.cap_usd == 1.5,
              "REALME_MAX_SPEND_USD applies when no cap is passed")
        os.environ["REALME_MAX_SPEND_USD"] = "two dollars"
        try:
            speak.spend_capped(PaidStub())
            check(False, "a cap that is not a number is refused, not ignored")
        except ValueError:
            check(True, "a cap that is not a number is refused, not ignored")
    finally:
        os.environ.pop("REALME_MAX_SPEND_USD")

    w = speak.spend_capped(PaidStub(), 1)
    w.rate_match = 0.9
    check(w._inner.rate_match == 0.9 and w.voice_fingerprint() == "paidstub:v1",
          "the wrapper passes attributes through both ways")

    print("\n3b. a narration that hits the cap resumes without paying twice")
    from realme.pipeline import narrate as N
    src = WORK / "talk.txt"
    src.write_text("One short sentence. Two short sentence. Three short "
                   "sentence.\n\nFour short sentence. Five short sentence.",
                   encoding="utf-8")
    eng = PaidStub(0.01)
    try:
        N.narrate(src, WORK / "talk.wav", speak.spend_capped(eng, 0.015),
                  mode="sentence", captions=False, log=lambda *a: None)
        hit = False
    except SpendCapReached:
        hit = True
    first = len(eng.calls)
    check(hit and first == 2, "it stops after the calls the cap allowed",
          f"calls={first}")
    N.narrate(src, WORK / "talk.wav", speak.spend_capped(eng, 5.0),
              mode="sentence", captions=False, log=lambda *a: None)
    total = len(eng.calls)
    # Control: a fresh engine with no ledger history pays for every line.
    fresh = PaidStub(0.01)
    N.narrate(src, WORK / "talk_fresh.wav", fresh, mode="sentence",
              captions=False, log=lambda *a: None)
    check(total == len(fresh.calls),
          "re-run with a higher cap: only the lines not yet recorded are paid "
          "for", f"{first} + {total - first} vs {len(fresh.calls)} from scratch")

    print("\n3c. the CLI and the Studio apply it where the engine is chosen")
    import argparse
    import realme.adapters.factory as F
    import realme.cli as C
    real_build = F.build
    F.build = lambda name, **kw: PaidStub()
    try:
        t = C._tts("x", argparse.Namespace(max_spend=0.5))
        check(isinstance(t, speak.SpendCapped) and t.cap_usd == 0.5,
              "--max-spend wraps the CLI's engine")
        t = C._tts("x", argparse.Namespace())
        check(not isinstance(t, speak.SpendCapped),
              "control: without it, the CLI's engine is bare")
    finally:
        F.build = real_build
    import realme.app.server as S
    S._TTS_CACHE.clear()
    real = S.build_tts
    S.build_tts = lambda name, **kw: PaidStub()
    try:
        a = S.engine_for("paidstub", max_spend=2)
        b = S.engine_for("paidstub")
        check(isinstance(a, speak.SpendCapped) and a._inner is b,
              "the Studio wraps per request, over the one cached engine")
    finally:
        S.build_tts = real
        S._TTS_CACHE.clear()
    from realme.cli import _main  # noqa: F401  -- the flag exists on the parsers
    for cmd in ("lecture", "narrate"):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.suppress(SystemExit):
            C._main([cmd, "--help"])
        check("--max-spend" in buf.getvalue(), f"`realme {cmd}` takes --max-spend")


def run_json(argv) -> tuple[int, dict, str]:
    import realme.cli as C
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = C.main(argv)
    return code, json.loads(out.getvalue()), err.getvalue()


def section_json():
    print("\n2. --json: one object on stdout, prose on stderr")
    code, j, err = run_json(["voice", "speed", "--voice", "gemini-tts",
                             "--percent", "-3", "--json"])
    check(code == 0 and j["ok"] and j["command"] == "voice speed",
          "a successful command reports ok", str(j)[:120])
    check(j.get("report", {}).get("speeds_percent", {}).get("gemini-tts") == -3.0,
          "and its result as data", str(j.get("report")))
    check("Saved" in err, "the readable lines went to stderr")

    code, j, _ = run_json(["narrate", str(WORK / "nope.txt"), "--json"])
    check(code == 1 and j["ok"] is False and "nope.txt" in j.get("error", ""),
          "a failure is still one object, with the error", str(j))

    # Control: without --json the same command prints prose, not JSON.
    import realme.cli as C
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        C.main(["voice", "speed"])
    try:
        json.loads(buf.getvalue())
        is_json = True
    except ValueError:
        is_json = False
    check(not is_json, "control: without --json the output is for a person")


def section_models():
    print("\n5. the routes declare what they take")
    import realme.app.server as S
    spec = S.app.openapi()

    def props(path):
        op = spec["paths"][path]["post"]
        ref = op["requestBody"]["content"]["application/json"]["schema"]
        ref = ref.get("$ref") or next(x["$ref"] for x in ref.get("anyOf", [])
                                      if "$ref" in x)
        return set(spec["components"]["schemas"][ref.split("/")[-1]]
                   ["properties"])
    check({"project_id", "pace", "text"} <= props("/api/preview"),
          "preview: project_id, pace, text")
    check({"voice", "percent"} <= props("/api/voice-speeds"),
          "voice speeds: voice, percent")
    from fastapi.testclient import TestClient
    c = TestClient(S.app, base_url="http://127.0.0.1")
    r = c.post("/api/voice-speeds", json={"voice": "x", "percent": "fast"})
    check(r.status_code == 422 and "percent" in r.text,
          "a wrong type is refused and the field named", r.text[:100])
    r = c.post("/api/preview", json={"pace": -1})
    check(r.status_code == 422 and "pace" in r.text,
          "an impossible pace is refused and the field named", r.text[:100])


class PacedStub(FreeStub):
    """A local engine that realises pace itself (as qwen3cpp and piper do),
    and records the pace it was asked for."""
    name = "pacedstub"

    def __init__(self):
        super().__init__()
        self.paces = []

    def synthesize(self, text, out_wav, voice=None, pace=1.0, **kw):
        self.paces.append(round(float(pace), 3))
        Path(out_wav).parent.mkdir(parents=True, exist_ok=True)  # as engines do
        return super().synthesize(text, out_wav, voice=voice, pace=pace, **kw)


def _client():
    from fastapi.testclient import TestClient
    import realme.app.server as S
    return S, TestClient(S.app, base_url="http://127.0.0.1",
                         raise_server_exceptions=False)


def _wait(S, job_id, secs=30.0):
    import time
    t0 = time.time()
    while time.time() - t0 < secs:
        j = S.runner.get(job_id)
        if j is None or j.status in ("done", "error"):
            return j
        time.sleep(0.05)
    return S.runner.get(job_id)


def section_review_fixes():
    """Studio routes and page: project ids, uploads, origins, lexicon, profile."""
    S, c = _client()
    ui = (HERE / "realme/app/ui.html").read_text(encoding="utf-8")
    src = (HERE / "realme/app/server.py").read_text(encoding="utf-8")

    print("\nR1. a new lecture never overwrites a project of the same name")
    old = S.PROJECTS / "Week3"
    (old / "_work").mkdir(parents=True, exist_ok=True)
    (old / "_work" / "manifest.json").write_text('{"keep": 1}', encoding="utf-8")
    (old / "Week3.pdf").write_bytes(b"old deck")
    notes = b"[[slide 1]]\nHello there.\n"
    r = c.post("/api/lecture/script/import",
               files={"file": ("Week3.pdf", b"new deck", "application/pdf"),
                      "notes": ("n.txt", notes, "text/plain")})
    pid = r.json().get("project_id")
    check(r.status_code == 200 and pid == "Week3_2",
          "importing a deck named like an existing lecture gets a fresh id",
          f"{r.status_code} {r.text[:120]}")
    check("already exists" in (r.json().get("note") or ""),
          "and says so")
    if r.status_code == 200:
        _wait(S, r.json()["job_id"])
    check((old / "_work" / "manifest.json").read_text(encoding="utf-8")
          == '{"keep": 1}' and (old / "Week3.pdf").read_bytes() == b"old deck",
          "the existing project's notes and deck are untouched")
    r = c.post("/api/lecture/script",
               data={"script_writer": "placeholder"},
               files={"file": ("Week3.pdf", b"newer", "application/pdf")})
    check(r.status_code == 200 and r.json().get("project_id") == "Week3_3",
          "drafting it again numbers on", r.text[:120])
    if r.status_code == 200:
        _wait(S, r.json()["job_id"])
    r = c.post("/api/lecture/script/import",
               files={"file": ("Fresh deck.pdf", b"x", "application/pdf"),
                      "notes": ("n.txt", notes, "text/plain")})
    check(r.status_code == 200 and r.json().get("project_id") == "Fresh_deck"
          and not r.json().get("note"),
          "control: a new name keeps its own id", r.text[:120])
    if r.status_code == 200:
        _wait(S, r.json()["job_id"])

    print("\nR2. a previewed line is spoken at the render's pace and speed")
    real = S.build_tts
    stub = PacedStub()
    S.build_tts = lambda name, **kw: stub
    S._TTS_CACHE.clear()
    saved = dict(S.profile.data.get("voice_speeds") or {})
    try:
        S.profile.data["voice_speeds"] = {"pacedstub": 1.25}
        r = c.post("/api/preview", json={"text": "One short sentence here.",
                                         "tts": "pacedstub", "pace": 0.9})
        check(r.status_code == 200 and stub.paces == [0.9],
              "the engine is asked for the slide's pace, as a render asks",
              f"{r.status_code} paces={stub.paces} {r.text[:100]}")
        d = r.json().get("duration_s") if r.status_code == 200 else None
        check(d is not None and abs(d - 0.4 / 1.25) < 0.03,
              "and the saved voice speed is applied after it",
              f"duration {d}, want {0.4 / 1.25:.2f}")
        from realme.pipeline import speak
        rec = PacedStub()
        speak.speak_segment("One short sentence here.", rec, WORK / "seg",
                            1, _Ledger(), pace=0.9, mode="stable")
        check(rec.paces == stub.paces,
              "control: speak_segment gives the engine the same pace",
              f"render {rec.paces} vs preview {stub.paces}")
        r = c.post("/api/preview", json={"text": "  ", "tts": "pacedstub"})
        check(r.status_code == 400, "an empty line is a 400, not a 500",
              str(r.status_code))
    finally:
        S.build_tts = real
        S._TTS_CACHE.clear()
        S.profile.data["voice_speeds"] = saved
    fn = ui[ui.index("async function hear("):]
    fn = fn[:fn.index("\n}\n")]
    check("pace: segPace(i)" in fn and "rvoice" in fn,
          "the editor's Hear button sends the slide's pace and the render's voice")

    print("\nR3. the Studio refuses a network address without --allow-network")
    check(S.is_loopback("127.0.0.1") and S.is_loopback("::1")
          and S.is_loopback("[::1]") and S.is_loopback("localhost")
          and S.is_loopback("127.0.0.2"), "loopback names are recognised")
    check(not S.is_loopback("0.0.0.0") and not S.is_loopback("192.168.1.5")
          and not S.is_loopback("studio.lan"), "network addresses are not")
    import uvicorn
    real_run, ran = uvicorn.run, []
    uvicorn.run = lambda *a, **kw: ran.append(kw.get("host"))
    try:
        try:
            S.serve("0.0.0.0", 8810)
            refused = False
        except ValueError as e:
            refused = "--allow-network" in str(e)
        check(refused and not ran, "serve('0.0.0.0') is refused and does not "
                                   "start", str(ran))
        err = io.StringIO()
        with contextlib.redirect_stderr(err), \
                contextlib.redirect_stdout(io.StringIO()):
            S.serve("192.168.1.5", 8810, allow_network=True)
        check(ran == ["192.168.1.5"] and "no login" in err.getvalue(),
              "with allow_network it starts, with a warning", str(ran))
        check("192.168.1.5" in S._EXTRA_HOSTS,
              "and the Host allowlist still learns the bind address")
        S._EXTRA_HOSTS.discard("192.168.1.5")
        import realme.cli as C
        real_serve, called = S.serve, []
        S.serve = lambda **kw: called.append(kw)
        try:
            with contextlib.redirect_stderr(io.StringIO()) as e2, \
                    contextlib.redirect_stdout(io.StringIO()):
                code = C.main(["studio", "--host", "0.0.0.0"])
            check(code == 2 and not called,
                  "`realme studio --host 0.0.0.0` exits 2 without serving",
                  f"{code} {called}")
            with contextlib.redirect_stdout(io.StringIO()):
                C.main(["studio", "--host", "0.0.0.0", "--allow-network"])
                C.main(["studio"])
            check([k.get("allow_network") for k in called] == [True, False]
                  and called[-1].get("host") == "127.0.0.1",
                  "--allow-network passes through; the default is loopback",
                  str(called))
        finally:
            S.serve = real_serve
    finally:
        uvicorn.run = real_run

    print("\nR4. uploads are streamed to disk and capped")
    check("await file.read()" not in src and ".read())" not in src.replace(
        "notes.read", ""), "no route reads a whole upload into memory")
    stub = FreeStub()
    S.build_tts = lambda name, **kw: stub
    S._TTS_CACHE.clear()
    lim = dict(S.UPLOAD_LIMITS)
    try:
        S.UPLOAD_LIMITS["text"] = 1000
        r = c.post("/api/narrate", data={"tts": "freestub"},
                   files={"file": ("big.txt", b"word " * 400, "text/plain")})
        check(r.status_code == 413
              and "big.txt" in r.text and "Nothing was saved" in r.text,
              "a file over its cap is a 413 that names it", r.text[:160])
        check(not list(S.PROJECTS.glob("narr_big/big.txt*")),
              "and nothing of it is left on disk")
        r = c.post("/api/narrate", data={"tts": "freestub"},
                   files={"file": ("small.txt", b"Two words.", "text/plain")})
        check(r.status_code == 200, "control: a file under the cap is taken",
              r.text[:160])
        if r.status_code == 200:
            _wait(S, r.json()["job_id"])
    finally:
        S.UPLOAD_LIMITS.clear()
        S.UPLOAD_LIMITS.update(lim)
    real_max = S.MAX_REQUEST_BYTES
    S.MAX_REQUEST_BYTES = 500
    try:
        r = c.post("/api/lexicon", content=b"x" * 2000,
                   headers={"content-type": "application/json"})
        check(r.status_code == 413, "a request larger than any upload is "
                                    "refused before it is read",
              str(r.status_code))
    finally:
        S.MAX_REQUEST_BYTES = real_max

    print("\nR5. a .docx that would unpack too large is refused")
    import zipfile
    bomb = WORK / "bomb.docx"
    real_member = S.DOCX_MAX_MEMBER
    S.DOCX_MAX_MEMBER = 1 << 20
    try:
        with zipfile.ZipFile(bomb, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("word/document.xml", b"\0" * (3 << 20))
        check(bomb.stat().st_size < 100_000, "(the test file is small on disk)",
              str(bomb.stat().st_size))
        r = c.post("/api/narrate", data={"tts": "freestub"},
                   files={"file": ("bomb.docx", bomb.read_bytes(),
                                   "application/octet-stream")})
        check(r.status_code == 400 and "unpack" in r.text,
              "a zip bomb named .docx is a 400", r.text[:160])
        check(not (S.PROJECTS / "narr_bomb" / "bomb.docx").exists(),
              "and is not kept")
        r = c.post("/api/narrate", data={"tts": "freestub"},
                   files={"file": ("junk.docx", b"not a zip", "x/y")})
        check(r.status_code == 400 and "Word" in r.text,
              "a .docx that is not a zip is a 400", r.text[:120])
        ok = WORK / "ok.docx"
        with zipfile.ZipFile(ok, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("word/document.xml",
                       "<w:p><w:t>A real paragraph.</w:t></w:p>")
        r = c.post("/api/narrate", data={"tts": "freestub"},
                   files={"file": ("ok.docx", ok.read_bytes(), "x/y")})
        check(r.status_code == 200, "control: an ordinary .docx is read",
              r.text[:160])
        if r.status_code == 200:
            _wait(S, r.json()["job_id"])
    finally:
        S.DOCX_MAX_MEMBER = real_member
        S.build_tts = real
        S._TTS_CACHE.clear()

    print("\nR6. Origin is compared whole, not by prefix")
    for origin, host in (("http://127.0.0.1:8000.evil.example", "127.0.0.1:8000"),
                         ("http://localhost:80001", "localhost:8000"),
                         ("http://localhost.evil.example", "localhost"),
                         ("http://localhost:8001", "localhost:8000"),
                         # userinfo; split so the GitHub packager's email
                         # check does not read the literal as an address
                         ("https://127.0.0.1" + "@" + "evil.example", "127.0.0.1"),
                         ("null", "127.0.0.1")):
        r = c.post("/api/lexicon", json={"term": ""},
                   headers={"origin": origin, "host": host})
        check(r.status_code == 403, f"Origin {origin} is refused",
              str(r.status_code))
    for origin, host in (("http://127.0.0.1", "127.0.0.1"),
                         ("http://localhost:8000", "localhost:8000"),
                         ("http://[::1]:8000", "[::1]:8000"),
                         ("http://127.0.0.1:80", "127.0.0.1")):
        r = c.post("/api/lexicon", json={"term": ""},
                   headers={"origin": origin, "host": host})
        check(r.status_code == 400, f"control: Origin {origin} reaches the "
                                    f"route", str(r.status_code))

    print("\nR7. ordinary bad input is a 400/404 with a sentence, not a 500")
    pdir = S.PROJECTS / "lec_bad"
    (pdir / "_work").mkdir(parents=True, exist_ok=True)
    from realme.pipeline.lecture import save_script
    from realme.core.schema import Manifest, Segment
    save_script(pdir, Manifest(project_id="lec_bad", title="T",
                               script_source="s",
                               segments=[Segment(segment_id=1, slide_index=0,
                                                 spoken_text="words")]))
    good = json.loads((pdir / "_work" / "manifest.json").read_text("utf-8"))
    bad_m = json.loads(json.dumps(good))
    bad_m["segments"][0]["prosody"] = {"pace": 0, "lead_in_s": 90}
    for name, (meth, url, kw, want) in {
        "render of a project that does not exist":
            ("post", "/api/lecture/nope/render", {"json": {}}, 404),
        "saving notes into a project that does not exist":
            ("put", "/api/lecture/nope/script", {"json": {"manifest": good}}, 404),
        "saving with no manifest":
            ("put", "/api/lecture/lec_bad/script", {"json": {}}, 400),
        "saving a slide gap of 90 s":
            ("put", "/api/lecture/lec_bad/script", {"json": {"manifest": bad_m}}, 400),
        "a body that is not JSON":
            ("post", "/api/lecture/lec_bad/render",
             {"content": b"{bad", "headers": {"content-type": "application/json"}}, 400),
        "a join list of the wrong shape":
            ("post", "/api/tools/merge/check", {"json": {"videos": ["x"]}}, 400),
        "a render with a pause scale of 'lots'":
            ("post", "/api/lecture/lec_bad/render", {"json": {"pause_scale": "lots"}}, 400),
    }.items():
        r = getattr(c, meth)(url, **kw)
        detail = r.json().get("detail") if r.headers.get(
            "content-type", "").startswith("application/json") else None
        check(r.status_code == want and isinstance(detail, str)
              and bool(detail.strip()),
              f"{name} -> {want}", f"{r.status_code} {r.text[:120]}")
    check(json.loads((pdir / "_work" / "manifest.json").read_text("utf-8"))
          == good, "and the refused save changed nothing")

    print("\nR8. a lexicon term with a / in it can be removed")
    c.post("/api/lexicon", json={"term": "and/or", "respelling": "and or"})
    from realme.text.lexicon import load_user_entries
    check(any(e.term == "and/or" for e in load_user_entries()),
          "(added)")
    r = c.delete("/api/lexicon/and%2For")
    check(r.status_code == 404, "control: the path form cannot reach it",
          str(r.status_code))
    r = c.delete("/api/lexicon", params={"term": "and/or"})
    check(r.status_code == 200 and not any(e.term == "and/or"
                                           for e in load_user_entries()),
          "DELETE /api/lexicon?term=and/or removes it", r.text[:120])
    check("/api/lexicon?term=" in ui, "and the page uses that form")

    print("\nR9. profile video settings are validated")
    before = c.get("/api/profile").json()["video"]
    for patch in ({"video": {"width": "abc"}}, {"video": {"width": 1921}},
                  {"video": {"fps": 0}}, {"video": {"height": 99999}},
                  {"video": {"layout": "cinema"}}, {"pause_scale": 3},
                  {"video": {"fps": True}}, {"signalling": "flash"}):
        r = c.post("/api/profile", json=patch)
        check(r.status_code == 400 and "Not saved" in r.text,
              f"{json.dumps(patch)} is refused", f"{r.status_code} {r.text[:100]}")
    check(c.get("/api/profile").json()["video"] == before,
          "and nothing was written")
    r = c.post("/api/profile", json={"video": {"width": 1280, "height": 720,
                                               "fps": 25}})
    check(r.status_code == 200 and r.json()["video"]["width"] == 1280,
          "control: a sensible size is saved", r.text[:100])
    c.post("/api/profile", json={"video": before})
    check("const res = await fetch('/api/profile'" in ui
          and "if(!res.ok)" in ui[ui.index("async function saveProfile"):],
          "the page shows a refused save instead of 'Saved.'")

    print("\nR10. the Podcast tab has its own script-writer control")
    check('id="dwriter"' in ui and 'id="dgmodel"' in ui,
          "a writer and a model picker on the Podcast tab")
    fn = ui[ui.index("async function makeDialogue("):]
    fn = fn[:fn.index("\n}\n")]
    check("$('dwriter').value" in fn and "$('dgmodel').value" in fn
          and "$('writer')" not in fn,
          "and the podcast sends those, not the Lecture tab's")

    print("\nR11. accessibility basics")
    check('id="pills" role="status" aria-live="polite"' in ui,
          "status pills are a live region")
    check('id="job-log" role="log" aria-live="polite"' in ui,
          "the job log is a live log")
    check('aria-label="Hide the activity log"' in ui
          and "aria-label=\"Remove ${esc(v.rel)} from the join\"" in ui,
          "icon-only buttons are named")
    check("button:focus-visible" in ui, "a visible focus ring for buttons")
    zs = ui[ui.index("function zoomSlide("):]
    zs = zs[:zs.index("\n}\n")]
    check("'Escape'" in zs and "role', 'dialog'" in zs and ".focus()" in zs,
          "the enlarged slide is a dialog that Escape closes")
    check('aria-label="Narration for slide' in ui,
          "each narration box is labelled")

    print("\nR12. Read aloud defaults to the profile's own voice")
    check("Your voice (profile default" in ui and "$('ntts').value = '';" in ui,
          "the voice picker starts on 'your voice', which sends blank")
    seen = []
    stub = FreeStub()
    S.build_tts = lambda name, **kw: (seen.append(name), stub)[1]
    S._TTS_CACHE.clear()
    try:
        r = c.post("/api/narrate", data={"tts": "", "text": "Hello there."})
        check(r.status_code == 200 and seen
              == [S.profile.data["adapters"]["tts"]],
              "and the server reads blank as the profile's voice",
              f"{r.status_code} {seen}")
        if r.status_code == 200:
            _wait(S, r.json()["job_id"])
    finally:
        S.build_tts = real
        S._TTS_CACHE.clear()


class _Ledger:
    """The smallest ledger speak_segment accepts: remembers nothing."""
    def get(self, key):
        return None

    def put(self, key, path, dur):
        pass


def section_pdf_default():
    """PDF is the deck format; a .pptx needs a converter, asked up front."""
    from realme.pipeline import slides as SL
    from realme.core.media import MediaError
    S, c = _client()
    def chk(name, cond, detail=""):
        return check(bool(cond), name, detail)
    print("\nPDF is the default deck format")
    chk("a PDF always reads", SL.deck_problem("Week 4.pdf") is None)
    real = SL.converters
    SL.converters = lambda: []
    try:
        why = SL.deck_problem("Week 4.pptx") or ""
        chk("a .pptx with no converter is refused with the PDF route",
              "PDF" in why and "Save As" in why, why[:120])
        before = set(p.name for p in S.PROJECTS.iterdir()) if S.PROJECTS.is_dir() else set()
        r = c.post("/api/lecture/script",
                   files={"file": ("NoConv.pptx", b"PK\x03\x04 not really", "application/octet-stream")},
                   data={"script_writer": "placeholder"})
        after = set(p.name for p in S.PROJECTS.iterdir()) if S.PROJECTS.is_dir() else set()
        chk("the Studio answers 400 at the upload", r.status_code == 400,
              f"{r.status_code} {r.text[:120]}")
        chk("and says to save as PDF", "PDF" in r.text, r.text[:120])
        chk("and makes no project folder for it", before == after,
              str(after - before))
        pptx = WORK / "nc.pptx"
        pptx.write_bytes(b"x")
        try:
            SL.deck_to_pdf(pptx, WORK / "nc_work")
            raised = ""
        except MediaError as e:
            raised = str(e)
        chk("deck_to_pdf refuses the same way", "Save As" in raised, raised[:120])
    finally:
        SL.converters = real
    chk("other formats are turned away too",
          "PDF" in (SL.deck_problem("notes.key") or ""))

    print("\nhidden slides do not shift the speaker notes")
    from pptx import Presentation
    prs = Presentation()
    for i, note in enumerate(["first", "hidden one", "third"]):
        sl = prs.slides.add_slide(prs.slide_layouts[5])
        sl.shapes.title.text = f"Slide {i + 1}"
        sl.notes_slide.notes_text_frame.text = note
        if i == 1:
            sl._element.set("show", "0")
    deck = WORK / "hidden.pptx"
    prs.save(str(deck))
    notes = SL.extract_pptx_notes(deck)
    chk("the hidden slide's notes are left out", notes == ["first", "third"],
          str(notes))
    chk("and so is its text",
          SL.extract_pptx_text(deck) == ["Slide 1", "Slide 3"],
          str(SL.extract_pptx_text(deck)))


def section_voice_choice():
    """Voice cloning is optional: a new installation is asked once."""
    import tempfile
    from realme.app.profile import Profile
    from realme.enrollment import choice as CH
    def chk(name, cond, detail=""):
        return check(bool(cond), name, detail)
    print("\nvoice cloning is a choice, asked once")
    said = []
    def fresh():
        return Profile(Path(tempfile.mkdtemp(prefix="vc_", dir=_ISO)) / "profile")
    pr = fresh()
    got = CH.choose_voice(pr, if_new=True, write=said.append, interactive=False)
    chk("with no one to ask, nothing is changed", got["choice"] is None
        and pr.data["adapters"]["tts"] == "gemini-tts", str(got))
    pr = fresh()
    got = CH.choose_voice(pr, if_new=True, read=lambda _: "3", write=said.append,
                          interactive=True)
    chk("'no' means the draft voices", got["tts"] == "piper"
        and pr.data["voice_choice"] == "none", str(got))
    chk("and the saved profile says so",
        Profile(pr.root).data["adapters"]["tts"] == "piper")
    pr = fresh(); said.clear()
    real = CH.local_clone_installed
    CH.local_clone_installed = lambda: False
    try:
        got = CH.choose_voice(pr, read=lambda _: "2", write=said.append,
                              interactive=True)
    finally:
        CH.local_clone_installed = real
    chk("'locally' without the clone installed waits on the draft voices",
        got["tts"] == "piper" and got["choice"] == "local", str(got))
    chk("and says how to build it, or where a ready one comes from",
        any("3_Get_Voice_Engine.bat" in m for m in said)
        and any("Xinhua Yu" in m for m in said), " | ".join(said)[:160])
    pr = fresh()
    got = CH.choose_voice(pr, read=lambda _: "", write=said.append, interactive=True)
    chk("Enter means Gemini", got["tts"] == "gemini-tts" and got["choice"] == "gemini")
    again = CH.choose_voice(Profile(pr.root), if_new=True, write=said.append,
                            read=lambda _: (_ for _ in ()).throw(AssertionError("asked")),
                            interactive=True)
    chk("running the installer again does not ask again", again["asked"] is False)
    pr = fresh()
    pr.data["adapters"]["tts"] = "qwen3cpp"
    pr.data["voice_reference"] = "x.wav"
    pr.save()
    got = CH.choose_voice(Profile(pr.root), if_new=True, write=said.append,
                          read=lambda _: (_ for _ in ()).throw(AssertionError("asked")),
                          interactive=True)
    chk("an existing local-clone user is left on the local clone",
        got["choice"] == "local" and Profile(pr.root).data["adapters"]["tts"] == "qwen3cpp",
        str(got))
    code, j, _ = run_json(["voice", "choose", "--choice", "none", "--json"])
    chk("realme voice choose --choice none works from the command line",
        code == 0 and j.get("report", {}).get("tts") == "piper", str(j)[:160])
    run_json(["voice", "choose", "--choice", "gemini", "--json"])
    inst = (HERE.parent / "00_Windows" / "1_Install.bat").read_text(encoding="latin-1")
    chk("1_Install.bat asks on a new installation", "voice choose --if-new" in inst)
    ui = (HERE / "realme/app/ui.html").read_text(encoding="utf-8")
    chk("the Studio does not call the draft voice a fault",
        "lectures: draft voice" in ui)
    chk("and says how to get the local clone", "3_Get_Voice_Engine.bat" in ui
        and "Ask Xinhua Yu" not in ui)


class BrokenStub(FreeStub):
    """An engine that fails the way a real crash does: a Python error."""
    name = "brokenstub"

    def synthesize(self, text, out_wav, voice=None, pace=1.0, **kw):
        raise AttributeError("'NoneType' object has no attribute 'synthesize_wav'")


def section_plain_language():
    """What the page says to someone who is not a programmer."""
    S, c = _client()
    ui = (HERE / "realme/app/ui.html").read_text(encoding="utf-8")
    start = (HERE.parent / "START_HERE.md").read_text(encoding="utf-8")

    def fn_body(name):
        f = ui[ui.index(name):]
        return f[:f.index("\n}\n")]

    print("\nP1. voices are named as a lecturer names them; values unchanged")
    check("'gemini-tts': 'Gemini'" in ui and "'qwen3cpp': 'free local clone'" in ui,
          "engine ids map to 'Gemini' and 'free local clone'")
    eo = fn_body("function engineOption(")
    check('value="${esc(k)}"' in eo and "Draft voice — no cloning" in eo,
          "menu options keep the engine id as their value, with a friendly label")
    check("`Draft voice — ${dv.label}`" in ui and "NOT your voice`" not in ui,
          "the Recording menu says 'Draft voice — Amy', not a raw engine id")
    check("(placeholder)" not in eo and "' — unavailable'" not in ui,
          "the draft voice is not called a 'placeholder' and nothing just says 'unavailable'")
    check("needs a Gemini key (2_Set_Key.bat)" in ui
          and "not created yet (Twin Setup)" in ui,
          "an unavailable voice says what to do about it")

    print("\nP2. errors are plain sentences, with the raw text folded away")
    pe = fn_body("function plainError(")
    check("No GEMINI_API_KEY" in pe and "2_Set_Key.bat" in pe
          and "No replicated voice yet" in pe and "could not reach ${who}" in pe
          and "'Anthropic (Claude)'" in pe and "'Google'" in pe,
          "the common failures are re-said with the next step")
    check("Something went wrong inside RealMe" in pe and "AttributeError" in pe,
          "a Python error is said to be a program fault, not the user's")
    check("Technical details" in fn_body("function errorHtml("),
          "the raw message stays available behind 'Technical details'")
    tl = fn_body("function tidyLogLine(")
    check("Traceback" in tl and "j.log.map(tidyLogLine)" in ui,
          "tracebacks in the Activity log are replaced by one line")
    check("showError('start-draft'" in fn_body("async function draftScript(")
          and "showError('editor'" not in fn_body("async function draftScript(")
          and "showError('start-import'" in fn_body("async function importScript("),
          "a failed draft or import is reported on its own card, not in an empty editor")
    check("jsonOf(r)" in fn_body("async function hear(")
          and "jsonOf(r)" in fn_body("async function hearTuned("),
          "Hear first line cannot be left reading 'recording…' by a crash page")

    print("\nP3. a voice that cannot speak yet is not the one Render uses")
    check("picked.disabled" in ui and "RVOICE_WAITING" in fn_body("function costLine("),
          "the Recording menu falls back to a draft voice and says why")
    check('<option value="1" selected>as measured (1.0)</option>' in ui,
          "'Between sentences' starts at 1.0, not at the first option")

    print("\nP4. the draft-voice choice reads as a choice, not a fault")
    gp = fn_body("function gemPaint(")
    check("optional — no key" in gp and "'pill bad'" in gp,
          "the Gemini panel is 'optional' for a draft-voice user, red only for a Gemini user")
    check('id="gem-role"' in ui and 'id="gem-next"' in ui,
          "it says whether it is the lecture voice, and what is missing to create it")
    check('id="envsum"' in ui and 'id="env-tech"' in ui
          and ui.index('id="env-tech"') < ui.index('id="doctor"'),
          "Environment shows a plain summary, the raw listing folded below it")
    check("cloning adapters" not in ui and "voice adapter you explicitly" not in ui,
          "no 'adapter' jargon in Twin Setup")

    print("\nP5. the server's own messages")
    real = S.build_tts
    S.build_tts = lambda name, **kw: BrokenStub()
    S._TTS_CACHE.clear()
    try:
        r = c.post("/api/preview", json={"text": "One line.", "tts": "brokenstub"})
        try:
            body = r.json()
        except ValueError:
            body = {}
        check(r.status_code == 500 and "could not be spoken" in str(body.get("detail", "")),
              "a crash in Hear first line answers with a sentence, as JSON",
              f"{r.status_code} {r.text[:120]}")
    finally:
        S.build_tts = real
        S._TTS_CACHE.clear()
    # The route reloads .env files, and a developer's machine may have a real
    # key in one of them (03_App\.env, say). Hold the reload off so "no key"
    # really means no key wherever the suite runs.
    had = os.environ.pop("GEMINI_API_KEY", None)
    real_load = S._load_env
    S._load_env = lambda *a, **k: None
    try:
        if not S._gemini_state()["have_key"]:
            r = c.post("/api/voice/gemini/enroll", json={})
            check(r.status_code == 400 and "2_Set_Key.bat" in r.text
                  and "Set it in Twin Setup" not in r.text,
                  "creating a Gemini voice with no key points to 2_Set_Key.bat",
                  r.text[:120])
    finally:
        S._load_env = real_load
        if had is not None:
            os.environ["GEMINI_API_KEY"] = had

    print("\nP6. START_HERE says what the screens say")
    check("your voice: Gemini" in start and "your voice: gemini-tts" not in start,
          "the pill is quoted as the page shows it")
    check("Save this voice" in start and "Save this voice" in ui,
          "the guide names the button that stores the voice recording")
    check("Record\nevery sentence again" in start.replace("*", "")
          or "Record every sentence again" in start.replace("*", "").replace("\n", " "),
          "and the re-record tick by its label")
    check("Render draft" in start and "Recording" in start,
          "and the Recording menu and the Render draft button")


def section_transcript_box():
    """Twin Setup takes the words read in the voice recording."""
    S, c = _client()
    def chk(name, cond, detail=""):
        return check(bool(cond), name, detail)
    print("\nthe words read in the recording can be typed in the Studio")
    keep = S.profile.data.get("voice_reference")
    S.profile.data["voice_reference"] = None
    r = c.post("/api/profile/transcript", json={"text": "one two three four five six"})
    chk("without a recording it says to choose one first",
        r.status_code == 400 and "recording first" in r.text, r.text[:120])
    S.profile.data["voice_reference"] = "x.wav"
    r = c.post("/api/profile/transcript", json={"text": "too short"})
    chk("a scrap is refused in plain words", r.status_code == 400, r.text[:120])
    words = "Today we talk about confounding and how to control it in practice"
    r = c.post("/api/profile/transcript", json={"text": words})
    chk("the text is saved", r.status_code == 200 and S.profile.reference_text == words,
        r.text[:120])
    S.profile.data["voice_reference"] = keep
    ui = (HERE / "realme/app/ui.html").read_text(encoding="utf-8")
    chk("Twin Setup has the box and its button",
        'id="vtext"' in ui and "saveTranscript()" in ui)
    src = (HERE / "realme/app/server.py").read_text(encoding="utf-8")
    chk("no message still says the Studio has no box for it",
        "no box" not in src and "no box" not in ui)

    print("\nthe finished video appears below the editor")
    chk("the video panel comes after the slides and the render bar",
        0 < ui.find('id="renderbar"') < ui.find('id="lec-out"'))
    import re as _re
    _rb = _re.search(r"\.renderbar\{[^}]*\}", ui)
    chk("the render bar follows the slides instead of floating over them",
        _rb is not None and "sticky" not in _rb.group(0), _rb.group(0)[:80] if _rb else "")
    print("\nthe Activity corner says what is happening, in plain words")
    import re as _re2
    blk = ui[ui.find("const STAGES = ["):ui.find("function stageOf")]
    rules = [(m.group(1), m.group(2)) for m in
             _re2.finditer(r"\[/(.+?)/, (?:m)?\(?\)?=>(?:'|`)(.+?)(?:'|`)\]", blk)]
    def stage(log):
        for line in reversed(log):
            for rx, say in rules:
                mm = _re2.search(rx, line)
                if mm:
                    return _re2.sub(r"\$\{m\[(\d)\]\}", lambda g: mm.group(int(g.group(1))), say)
        return "Starting"
    for log, want in [
            (["Preflight", "Rasterizing deck", "Drafting narration via gemini"], "Drafting the narration"),
            (["about 30 engine calls for this deck", "slide 3 of 12 (120 words) ..."], "Recording slide 3 of 12"),
            (["slide 12 of 12 (90 words) ...", "Assembling master"], "Creating the video"),
            (["slide 12 of 12 (90 words) ...", "Creating 12 slide videos, 4 at a time",
              "  slide video 5 of 12 (slide 7)"], "Creating the slide videos — 5 of 12"),
            (["Assembling master", "Verifying"], "Final check of the video"),
            (["[3/4] Synthesizing turns", "      turn 5 [Amy]: 7.2s"], "Recording the conversation — turn 5"),
            (["  [2/3] 4 utterance(s)"], "Reading paragraph 2 of 3")]:
        chk(f"'{log[-1][:28]}' shows as '{want}'", stage(log) == want, stage(log))
    chk("the corner button carries the stage and the panel repeats it",
        'id="act-label"' in ui and 'id="job-stage"' in ui and "showStage(title, stageOf(j.log)" in ui)
    chk("and the page does not jump back to the top for it",
        "$('lec-out').scrollIntoView({behavior:'smooth', block:'start'})" not in ui)


def main() -> int:
    import shutil
    if not shutil.which("ffmpeg"):
        print("ffmpeg not on PATH; skipping")
        return 0
    section_host()
    section_spend()
    section_json()
    section_models()
    section_review_fixes()
    section_pdf_default()
    section_voice_choice()
    section_plain_language()
    section_transcript_box()
    return summary()


if __name__ == "__main__":
    raise SystemExit(main())
