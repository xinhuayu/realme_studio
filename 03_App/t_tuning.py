#!/usr/bin/env python3
"""
Voice and timing controls: a saved speed per voice, and slide gaps.

What is promised, and checked here:

  * A voice's saved speed changes how fast it speaks by that factor, for a
    local engine and for one that re-times after synthesis (Gemini) alike.
  * Changing it records nothing again: every take already on disk is reused
    and only re-stretched. The control is "no saved speed", which must leave
    the engine object, its timing and its cache exactly as they were.
  * The Studio saves a speed per voice, applies it wherever that voice is
    built, and refuses one outside +/-30%.
  * The lecture tab carries the controls above Render, and Render sends the
    sentence-pause setting.
"""
from __future__ import annotations
import math, os, struct, sys, tempfile, wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

_ISO = tempfile.mkdtemp(prefix="t_tuning_")
os.environ["REALME_HOME"] = str(Path(_ISO) / "home")
os.environ["REALME_TOOLS"] = str(Path(_ISO) / "tools")
os.environ["HOME"] = _ISO
os.environ["USERPROFILE"] = _ISO
os.environ["REALME_NO_WARMUP"] = "1"

from t_util import check, summary          # noqa: E402

WORK = Path(_ISO) / "work"
WORK.mkdir(parents=True, exist_ok=True)

TEXT = ("The risk ratio compares two groups. It is not the odds ratio. "
        "When the outcome is rare, the two are close. When it is common, "
        "they part company, and the odds ratio overstates the effect.")


def tone(path: Path, seconds: float, rate=24000):
    n = int(rate * seconds)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", int(6000 * math.sin(
            2 * math.pi * 180 * i / rate))) for i in range(n)))


class LocalStub:
    """Speaks at a fixed rate; applies `pace` itself, like piper."""
    name = "localstub"
    is_voice_clone = False
    is_placeholder = False
    speaks_languages = ()
    phoneme_syntax = None
    retimes_after = False

    def __init__(self):
        self.calls, self.paces = [], []

    def preflight(self):
        pass

    def voice_fingerprint(self):
        return "localstub:v1"

    def timing_factor(self, pace=1.0):
        return 1.0

    def synthesize(self, text, out_wav, voice=None, pace=1.0, **kw):
        self.calls.append(text)
        self.paces.append(pace)
        tone(Path(out_wav), len(text.split()) * 0.3 / (pace or 1.0))
        return Path(out_wav)


class RetimeStub(LocalStub):
    """Like Gemini: ignores pace, and asks the caller to stretch afterwards
    by pace times a measured correction."""
    name = "retimestub"
    retimes_after = True
    rate_match = 1.1

    def voice_fingerprint(self):
        return "retimestub:v1"

    def timing_factor(self, pace=1.0):
        return float(pace) * self.rate_match

    def synthesize(self, text, out_wav, voice=None, pace=1.0, **kw):
        self.calls.append(text)
        self.paces.append(pace)
        tone(Path(out_wav), len(text.split()) * 0.3)
        return Path(out_wav)


def spoken(tts, tag):
    """Total seconds of speech for TEXT, through the real speak path."""
    from realme.core.ledger import RenderLedger
    from realme.pipeline.speak import speak_segment
    work = WORK / tag
    (work / "audio").mkdir(parents=True, exist_ok=True)
    utts = speak_segment(TEXT, tts, work / "audio", 1,
                         RenderLedger(work / "ledger.json"), mode="sentence")
    return sum(u.duration_s or 0 for u in utts)


def section_engine():
    from realme.pipeline.speak import SpeedTrimmed, speed_trimmed

    print("\na saved speed changes the speed, and records nothing again")
    for Stub, label in ((LocalStub, "a local engine"),
                        (RetimeStub, "an engine that re-times afterwards")):
        eng = Stub()
        base = spoken(eng, label[:7] + "_1")
        made = len(eng.calls)
        # Same ledger folder: the second pass must find every take.
        from realme.core.ledger import RenderLedger
        from realme.pipeline.speak import speak_segment
        work = WORK / (label[:7] + "_1")
        for factor in (1.15, 0.93):
            utts = speak_segment(TEXT, speed_trimmed(eng, factor),
                                 work / "audio", 1,
                                 RenderLedger(work / "ledger.json"),
                                 mode="sentence")
            got = sum(u.duration_s or 0 for u in utts)
            check(abs(got - base / factor) / (base / factor) < 0.03,
                  f"{label} at x{factor} takes 1/{factor} of the time",
                  f"{got:.2f}s vs {base / factor:.2f}s expected")
        check(len(eng.calls) == made,
              f"{label}: no utterance was recorded again",
              f"{len(eng.calls) - made} extra calls")
        speed_trimmed(eng, 1.15).synthesize("two words", WORK / "p.wav",
                                            pace=1.0)
        check(eng.paces[-1] == 1.0,
              f"{label}: the engine is asked for the pace it always was")

    eng = LocalStub()
    check(speed_trimmed(eng, 1.0) is eng and speed_trimmed(eng, None) is eng,
          "control: with no saved speed the engine is returned untouched")
    w = SpeedTrimmed(RetimeStub(), 0.97)
    check(abs(w.timing_factor(1.0) - 1.1 * 0.97) < 1e-9,
          "a measured correction and a saved speed multiply")
    check(w.voice_fingerprint() == "retimestub:v1",
          "the voice's fingerprint is unchanged, so paid takes stay valid")
    w.rate_match = 1.2
    check(w._inner.rate_match == 1.2, "attributes pass through to the engine")


def section_casting():
    print("\nthe speed is filed under the voice, however it is asked for")
    from realme.pipeline import casting as C
    from realme.pipeline.speak import SpeedTrimmed

    class Piperish(LocalStub):
        model = "en_US-amy-medium"
    check(C.voice_key("piper", Piperish()) == "piper:en_US-amy-medium",
          "a bare `piper` is filed under the voice it loaded")
    check(C.voice_key("piper:en_US-ryan-medium") == "piper:en_US-ryan-medium"
          and C.voice_key("gemini-tts") == "gemini-tts",
          "a named voice and a cloning engine keep their names")
    data = {"voice_speeds": {"piper:en_US-amy-medium": 1.15,
                             "piper:en_US-ryan-medium": 0.93,
                             "gemini-tts": 0.97, "qwen3cpp": 9.0}}
    check(C.saved_voice_speed("gemini-tts", data) == 0.97,
          "a saved speed is read back")
    check(C.saved_voice_speed("qwen3cpp", data) == 1.0,
          "one outside +/-30% reads as no setting, not as a clamp")

    class _Prof:
        pass
    prof = _Prof()
    prof.data = data
    voices = C.cast("qwen3cpp", "piper:en_US-amy-medium", speed=1.0,
                    profile=prof, registers=False,
                    build=lambda name, **kw: LocalStub())
    g = voices["guest"]
    check(isinstance(g, SpeedTrimmed) and g.speed_factor == 1.15,
          "a podcast's interlocutor is cast at its saved speed")
    check(not isinstance(voices["instructor"], SpeedTrimmed),
          "and a voice with no saved speed is cast as it was")


def section_studio():
    print("\nthe Studio saves a speed per voice and uses it")
    from fastapi.testclient import TestClient
    import realme.app.server as S
    from realme.pipeline.speak import SpeedTrimmed
    c = TestClient(S.app, base_url="http://127.0.0.1")
    r = c.post("/api/voice-speeds", json={"voice": "piper:en_US-amy-medium",
                                          "percent": 15})
    check(r.status_code == 200 and r.json()["speeds"].get(
        "piper:en_US-amy-medium") == 15.0, "Amy +15% is saved", r.text[:120])
    c.post("/api/voice-speeds", json={"voice": "piper:en_US-ryan-medium",
                                      "percent": -7})
    c.post("/api/voice-speeds", json={"voice": "gemini-tts", "percent": -3})
    got = c.get("/api/voice-speeds").json()["speeds"]
    check(got == {"piper:en_US-amy-medium": 15.0,
                  "piper:en_US-ryan-medium": -7.0, "gemini-tts": -3.0},
          "each voice keeps its own", str(got))
    from realme.app.profile import Profile
    on_disk = Profile(S.DATA / "profile").data.get("voice_speeds")
    check(on_disk.get("gemini-tts") == 0.97, "and it is in the saved profile",
          str(on_disk))

    S._TTS_CACHE.clear()
    real = S.build_tts
    S.build_tts = lambda name, **kw: LocalStub()
    try:
        a = S.engine_for("piper:en_US-amy-medium")
        q = S.engine_for("qwen3cpp")
        check(isinstance(a, SpeedTrimmed) and abs(a.speed_factor - 1.15) < 1e-9,
              "the engine the Studio builds for Amy speaks 15% faster")
        check(not isinstance(q, SpeedTrimmed),
              "control: a voice with no saved speed is built as before")
    finally:
        S.build_tts = real
        S._TTS_CACHE.clear()

    print("\na sentence heard on the Lecture tab is kept for that lecture")
    # The page says "kept for the render", so a sentence previewed on the
    # Lecture tab goes into the lecture's own ledger.
    pdir = S.PROJECTS / "lec_keep"
    (pdir / "_work").mkdir(parents=True, exist_ok=True)
    (pdir / "_work" / "manifest.json").write_text("{}", encoding="utf-8")
    S._TTS_CACHE.clear()
    stub = LocalStub()
    S.build_tts = lambda name, **kw: stub
    try:
        body = {"text": "The odds ratio was two point three.", "tts": "qwen3cpp",
                "project_id": "lec_keep"}
        r1 = c.post("/api/preview", json=body)
        r2 = c.post("/api/preview", json=body)
    finally:
        S.build_tts = real
        S._TTS_CACHE.clear()
    check(r1.status_code == 200 and (pdir / "_work" / "ledger.json").is_file(),
          "the take is written into the lecture's ledger", r1.text[:120])
    check(r2.json().get("reused") is True and len(stub.calls) == 1,
          "and hearing it again does not record it again",
          f"{len(stub.calls)} calls")

    r = c.post("/api/voice-speeds", json={"voice": "gemini-tts", "percent": 0})
    check("gemini-tts" not in r.json()["speeds"], "0% removes the setting")
    r = c.post("/api/voice-speeds", json={"voice": "gemini-tts", "percent": 45})
    check(r.status_code == 422, "more than 30% is refused", str(r.status_code))

    print("\nthe lecture tab carries the controls, beside Render")
    ui = (HERE / "realme" / "app" / "ui.html").read_text(encoding="utf-8")

    def block(open_tag_id):
        i = ui.find(f'id="{open_tag_id}"')
        tag = ui.rfind("<", 0, i)
        name = ui[tag + 1:ui.find(" ", tag)]
        depth, j = 0, tag
        while True:
            o, c = ui.find(f"<{name}", j + 1), ui.find(f"</{name}>", j + 1)
            if o != -1 and o < c:
                depth, j = depth + 1, o
            elif depth:
                depth, j = depth - 1, c
            else:
                return ui[tag:c]
    tune, bar = block("tune"), block("renderbar")
    check(all(f'id="{k}"' in tune for k in ("tspeed", "tpause", "tlead", "tafter",
                                            "rfresh")),
          "speed, sentence pause, both slide gaps and re-record sit together "
          "under Options")
    check(all(f'id="{k}"' in bar for k in ("rvoice", "btn-save", "btn-render", "cost")),
          "and the voice, Save, Render and the cost sit in the bar that stays "
          "on screen")
    check(ui.find('id="tune"') < ui.find('id="renderbar"'),
          "with Options just above it")
    print("\nthe page shows what the moment needs")
    import re as _re
    cards = _re.findall(r'class="startcard" data-start="(\w+)"', ui)
    check(sorted(cards) == ["draft", "import", "open", "revise"]
          and all(f'id="start-{c}"' in ui for c in cards),
          "four ways to start, each card with its own form")
    load = ui[ui.find("async function loadEditor"):]
    load = load[:load.find("\nfunction ")]
    check("enterEditor(" in load and "buildSlideNav(" in load,
          "opening a lecture shows just that lecture, with a slide list")
    check('id="seg${i}"' in ui and 'id="slidenav"' in ui,
          "and every slide can be jumped to")
    watch = ui[ui.find("function watch("):]
    check("activity-on" in watch[:watch.find("\n}\n")],
          "the activity log appears when something starts running")
    check("function tidyHints" in ui and "#p-help" not in
          ui[ui.find("function tidyHints"):ui.find("function jumpBars")],
          "long explanations fold behind 'More about this', except in Help")

    body = ui[ui.find("async function renderLecture"):]
    body = body[:body.find("\n}\n")]
    check("pause_scale" in body, "Render sends the sentence-pause setting")


def section_cli_and_docs():
    print("\nthe command line and the documents say the same thing")
    import contextlib, io, json
    import realme.cli as C

    def run(*argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = C.main(list(argv) + ["--json"])
        return code, json.loads(out.getvalue())
    code, j = run("voice", "speed", "--voice", "piper:en_US-ryan-medium",
                  "--percent", "-7")
    check(code == 0 and j["report"]["speeds_percent"].get(
        "piper:en_US-ryan-medium") == -7.0,
        "`realme voice speed --percent -7` saves Ryan at -7%", str(j)[:120])
    from realme.core.env import data_home
    from realme.app.profile import Profile
    check(Profile(data_home() / "profile").data["voice_speeds"].get(
        "piper:en_US-ryan-medium") == 0.93,
        "into the same profile the Studio reads")
    code, j = run("voice", "speed", "--voice", "piper:en_US-ryan-medium",
                  "--percent", "40")
    check(code == 1 and "between" in j.get("error", ""),
          "beyond 30% is refused", str(j)[:120])
    code, j = run("voice", "speed", "--voice", "piper:en_US-ryan-medium",
                  "--percent", "0")
    check(code == 0 and "piper:en_US-ryan-medium" not in
          j["report"]["speeds_percent"], "0 clears it")

    root = HERE.parent
    start = (root / "START_HERE.md").read_text(encoding="utf-8")
    ui = (HERE / "realme" / "app" / "ui.html").read_text(encoding="utf-8")
    check("realme voice speed" in start and "Voice speed" in start
          and "### Speed: voice speed and pace" in start,
          "START_HERE explains voice speed against pace, with the command")
    check("Two kinds of speed" in ui and "function speedNote" in ui,
          "the Help says it too, and each voice menu shows a saved speed")


def section_lecture():
    print("\na lecture re-rendered at a new speed is at the new speed")
    # The slide videos are cached under a key of their own. The saved speed
    # is deliberately not in the voice's fingerprint (so a paid take survives
    # it), so it has to be in that key -- or a re-render after changing the
    # speed re-serves every slide at the old one and calls it a cache hit.
    try:
        import pymupdf
    except ImportError:
        print("  --   pymupdf not installed; skipped")
        return
    from realme.adapters.script_writer import PlaceholderScriptWriter
    from realme.core.media import duration_of
    from realme.pipeline import lecture as lec
    from realme.pipeline.speak import speed_trimmed
    deck = WORK / "deck.pdf"
    doc = pymupdf.open()
    for i in range(2):
        pg = doc.new_page(width=960, height=540)
        pg.insert_text((72, 100), f"Slide {i + 1}: risk and odds.", fontsize=24)
        pg.insert_text((72, 150), "The odds ratio was 2.3 in the cohort.",
                       fontsize=16)
    doc.save(str(deck))
    proj = WORK / "lecture"
    m, _ = lec.write_script(deck, proj, PlaceholderScriptWriter(),
                            log=lambda *_: None)
    for seg in m.segments:                 # short gaps, so speech dominates
        seg.prosody.lead_in_s = seg.prosody.pause_after_s = 0.3
    eng = LocalStub()
    quiet = dict(check_audio=False, log=lambda *_: None)
    a = lec.render(deck, proj, m, eng, **quiet)
    base, made = duration_of(a["master"]), len(eng.calls)
    b = lec.render(deck, proj, m, eng, **quiet)
    check(abs(duration_of(b["master"]) - base) < 0.05 and len(eng.calls) == made,
          "control: the same speed twice is a full cache hit")
    c = lec.render(deck, proj, m, speed_trimmed(eng, 1.25), **quiet)
    fast = duration_of(c["master"])
    check(fast < base - 0.5,
          f"at +25% the video is shorter ({base:.2f}s -> {fast:.2f}s)")
    check(len(eng.calls) == made,
          f"without recording anything again ({len(eng.calls) - made} calls)")
    d = lec.render(deck, proj, m, eng, **quiet)
    check(abs(duration_of(d["master"]) - base) < 0.05,
          "and back at its own speed, the first render's slides are reused")

    print("\na slide whose audio changed is never served from an old video")
    # A slide's audio is made from what the text layer turns its narration
    # into, so the video key must cover that spoken text, not only the
    # WRITTEN narration. Otherwise a lexicon entry, a change to how money or
    # markup is read, or a new acronym rule re-records the sentence but
    # reuses the old video: the wrong words, and a duration the script no
    # longer has ("FAIL: duration_matches_script").
    from realme.text.lexicon import Entry, save_user_entries
    save_user_entries([Entry(term="cohort", respelling=
                             "cohort, that is the group we followed for years")])
    try:
        made = len(eng.calls)
        e = lec.render(deck, proj, m, eng, **quiet)
        rep = e["report"]
        check(len(eng.calls) > made,
              "a changed reading records the affected sentences again",
              f"{len(eng.calls) - made} calls")
        check(rep["checks"]["duration_matches_script"],
              "and the video carries the new audio: its length matches the "
              "script", f"drift {rep.get('drift_s')}s")
        check(duration_of(e["master"]) > base + 1.0,
              "so the lecture is longer by what the new words take",
              f"{base:.2f}s -> {duration_of(e['master']):.2f}s")
    finally:
        save_user_entries([])


def main() -> int:
    import shutil
    if not shutil.which("ffmpeg"):
        print("ffmpeg not on PATH; skipping")
        return 0
    section_engine()
    section_casting()
    section_studio()
    section_cli_and_docs()
    section_lecture()
    return summary()


if __name__ == "__main__":
    raise SystemExit(main())
