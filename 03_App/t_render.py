#!/usr/bin/env python3
"""
The render path, exercised with a fake engine that records what it was asked.

Every check here is one that passed with the code broken: the pace edit that
re-served old audio, the dialogue guest played at the instructor's sample
rate, the pause-only segment that crashed Piper, the renumbered segment that
re-rendered everything after it. None of the other tests build a ledger or a
TTS, so none could have caught them.

Needs ffmpeg. Everything else is the real pipeline.
"""
from __future__ import annotations
import json, math, os, shutil, struct, sys, tempfile, wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

#: Where this file lives. Every path below is relative to it, never to the
#: working directory: these suites are run from the project root as often as
#: from 03_App, and `Path("realme/...")` silently means a different thing in
#: each. `verify_tree.py` already anchored on __file__; the t_*.py files did
#: not, so t_render died with FileNotFoundError when run from the root.
HERE = Path(__file__).resolve().parent
os.environ.setdefault("REALME_HOME", tempfile.mkdtemp(prefix="realme_t_render_"))

FAILS = 0


def check(cond, what):
    global FAILS
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        FAILS += 1


def tone(path: Path, seconds: float, rate: int, freq: float = 440.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(seconds * rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(b"".join(
            struct.pack("<h", int(12000 * math.sin(2 * math.pi * freq * i / rate)))
            for i in range(n)))


class FakeTTS:
    """Writes a half-second tone and remembers every request."""
    name = "fake"
    is_voice_clone = False
    is_placeholder = False
    speaks_languages = ()
    phoneme_syntax = None

    def __init__(self, rate=24000, voice="v1"):
        self.rate, self.voice, self.calls = rate, voice, []

    def preflight(self): pass

    def voice_fingerprint(self): return f"fake:{self.voice}"

    def synthesize(self, text, out_wav, voice=None, pace=1.0, **kw):
        self.calls.append({"text": text, "pace": pace, "voice": voice})
        tone(Path(out_wav), 0.5 / pace, self.rate)
        return Path(out_wav)


def main():
    if not shutil.which("ffmpeg"):
        print("ffmpeg not on PATH; skipping"); return 0
    from realme.core.ledger import RenderLedger
    from realme.pipeline.speak import speak_segment, join_utterances
    from realme.pipeline import compose
    from realme.core.media import duration_of

    tmp = Path(tempfile.mkdtemp(prefix="t_render_"))
    print("cache keys")
    tts = FakeTTS()
    ledger = RenderLedger(tmp / "ledger.json")
    u1 = speak_segment("The odds ratio was two.", tts, tmp / "a", 1, ledger)
    n1 = len(tts.calls)
    speak_segment("The odds ratio was two.", tts, tmp / "a", 1, ledger)
    check(len(tts.calls) == n1, "identical text, voice and pace is a cache hit")
    speak_segment("The odds ratio was two.", tts, tmp / "a", 1, ledger, pace=1.2)
    check(len(tts.calls) == n1 + 1 and tts.calls[-1]["pace"] == 1.2,
          "a pace change is a cache miss and reaches the engine")
    n2 = len(tts.calls)
    speak_segment("The odds ratio was two.", tts, tmp / "a", 7, ledger)
    check(len(tts.calls) == n2, "the same words under a different segment id hit the cache")
    tts2 = FakeTTS(voice="v2")
    speak_segment("The odds ratio was two.", tts2, tmp / "a", 1, ledger)
    check(len(tts2.calls) == 1, "a different voice fingerprint is a miss")
    check(all("seg_" not in Path(u.wav).name for u in u1 if u.wav),
          "utterance files are named by key, not by position")
    k = RenderLedger.key("utt", "x|1.000|en-US", "fake:v1")
    check(":" in k and k.split(":")[0] == "utt" and RenderLedger.stem(k) == k.replace(":", "_"),
          "key carries no position and has a filename stem")

    print("pause-only text")
    try:
        utts = speak_segment("[[pause:500]]", tts, tmp / "b", 2, ledger)
        check(utts == [], "a marker-only segment produces no utterance and no engine call")
        out = join_utterances(utts, tmp / "b" / "seg.wav", tail_s=1.0)
        check(0.9 < duration_of(out) < 1.2, "and joins to silence of the tail length")
    except Exception as e:
        check(False, f"marker-only segment raised {type(e).__name__}: {e}")
    utts = speak_segment("Hello [[pause:700]] world", tts, tmp / "b", 3, ledger)
    check(len(utts) == 2 and utts[0].gap_ms >= 700, "a pause between words lands as a gap")

    print("what the log box shows")
    # One line per slide, not per utterance. Sixty lines for a twelve-slide
    # lecture was accurate and unreadable; nothing at all for minutes looked
    # like a hang. The totals come back with the utterances so the caller can
    # say it once.
    seen = []
    n = len(tts.calls)
    u = speak_segment("One sentence here. And a second one.", tts,
                      tmp / "c", 9, ledger, log=seen.append)
    check(seen == [], f"speak_segment says nothing per utterance (got {seen})")
    check(getattr(u, "cost", None) is not None, "but reports what it cost")
    check(u.cost["spoken"] == len(tts.calls) - n and u.cost["reused"] == 0,
          f"a first pass counts them all as spoken ({u.cost})")
    u2 = speak_segment("One sentence here. And a second one.", tts,
                       tmp / "c", 9, ledger, log=seen.append)
    check(u2.cost["reused"] == len(u2) and u2.cost["spoken"] == 0,
          f"a second pass counts them all as reused ({u2.cost})")
    check(u2.cost["synth_s"] == 0.0, "and charges no synthesis time for them")

    print("how much silence goes between units")
    # Measured from one instructor's enrolled clip: 0.336 s median pause,
    # 1.192 s longest, 18 breath-length gaps in 62 s. The old lift turned
    # that into 442 ms after EVERY sentence against a 185 ms floor -- his own
    # mid-sentence breathing applied at sentence boundaries, on top of the
    # trailing breath the engine already renders per utterance.
    from realme.pipeline.prosody import gap_ms
    HIS = dict(reference_pause_seconds=0.336,
               reference_longest_pause_seconds=1.192,
               breath_pause_candidates=18)
    sentence = gap_ms("The odds ratio was 2.3.", mode="natural", **HIS)
    clause = gap_ms("the odds ratio was 2.3,", mode="natural", **HIS)
    check(sentence == 238, f"a sentence gap is 238 ms, not the old 442 (got {sentence})")
    check(clause == 151, f"a clause gap is 151 ms, not the old 202 (got {clause})")
    floor_s = gap_ms("The odds ratio was 2.3.", mode="natural")
    check(floor_s < sentence, "the speaker's rhythm still lifts it above the floor")
    check(sentence < 442, "but is no longer the dominant term")
    check(gap_ms("Anything.", mode="stable", **HIS) == 0,
          "stable mode inserts nothing at all")
    # A longer pause in the reference must still mean a longer gap, or the
    # measurement has stopped meaning anything.
    slower = gap_ms("The odds ratio was 2.3.", mode="natural",
                    reference_pause_seconds=0.6,
                    reference_longest_pause_seconds=1.8,
                    breath_pause_candidates=10)
    check(slower > sentence, f"a slower speaker still gets longer gaps ({slower} ms)")
    # And the scale knob reaches the number, within its documented clamp.
    low = gap_ms("The odds ratio was 2.3.", mode="natural", pause_scale=0.85, **HIS)
    high = gap_ms("The odds ratio was 2.3.", mode="natural", pause_scale=1.25, **HIS)
    check(low < sentence < high, f"pause_scale moves it ({low} / {sentence} / {high})")
    check(gap_ms("A.", mode="natural", pause_scale=0.1, **HIS) == low,
          "and is clamped rather than letting a typo silence the gaps")

    print("the scale reaches the renderer")
    # The wrapper's own docstring records this exact bug: `mode` was added to
    # `render` and to the CLI and not to `build_lecture`, so every
    # `realme lecture` died on a TypeError before rendering a frame.
    import inspect as _i
    from realme.pipeline.lecture import render as _r, build_lecture as _bl
    from realme.pipeline.speak import speak_segment as _ss
    for fn in (_r, _bl, _ss):
        check("pause_scale" in _i.signature(fn).parameters,
              f"{fn.__name__} takes pause_scale")
    from realme.app.profile import DEFAULTS as _PD
    check(_PD.get("pause_scale") == 1.0, "a profile carries the default")
    gaps = []
    import realme.pipeline.prosody as _P
    real = _P.gap_ms
    _P.gap_ms = lambda *a, **k: gaps.append(k.get("pause_scale")) or real(*a, **k)
    try:
        speak_segment("One. Two.", tts, tmp / "ps", 5, ledger, pause_scale=0.9)
    finally:
        _P.gap_ms = real
    check(gaps and all(g == 0.9 for g in gaps),
          f"and speak_segment hands it to gap_ms ({gaps})")

    print("dialogue master")
    a24, a22 = tmp / "m" / "a24.wav", tmp / "m" / "a22.wav"
    tone(a24, 1.0, 24000); tone(a22, 1.0, 22050)
    mp3 = compose.audio_master([a24, a22], tmp / "m" / "master.mp3")
    d = duration_of(mp3)
    check(1.9 < d < 2.2, f"24 kHz + 22.05 kHz turns concatenate to ~2.0 s (got {d:.2f})")
    mp3b = compose.audio_master([a22, a24, a22], tmp / "m" / "master2.mp3")
    d = duration_of(mp3b)
    check(2.9 < d < 3.2, f"and in the other order (got {d:.2f})")

    print("script validation")
    from realme.adapters import script_writer as SW
    w = SW.GeminiScriptWriter(api_key="x", model="m")
    seen = []
    w.log = seen.append
    replies = iter([
        {"title": "T", "segments": [{"slide_number": 1, "spoken_text": "one"}]},
        {"title": "T", "segments": [{"slide_number": 1, "spoken_text": "one"}]},
    ])
    w._call = lambda parts, schema, system=None: next(replies)
    png = tmp / "s.png"
    from PIL import Image
    Image.new("RGB", (64, 36), "white").save(png)
    try:
        w.write_lecture([png, png], ["", ""], "", "")
        check(False, "one segment for two slides was accepted")
    except RuntimeError as e:
        check("twice" in str(e), "a persistent count mismatch is refused, not sorted into place")
    check(any("asking once more" in m for m in seen), "the retry is announced through the writer's log hook")
    replies = iter([
        {"title": "T", "segments": [{"slide_number": 1, "spoken_text": "one"},
                                    {"slide_number": 2, "spoken_text": "  "}]},
    ])
    try:
        w.write_lecture([png, png], ["", ""], "", "")
        check(False, "an empty segment was accepted")
    except RuntimeError as e:
        check("no narration" in str(e), "an empty spoken_text is refused")
    replies = iter([
        {"title": "T", "segments": [{"slide_number": 1, "spoken_text": "one", "pause_after_s": None},
                                    {"slide_number": 2, "spoken_text": "two"}]},
    ])
    m = w.write_lecture([png, png], ["", ""], "", "")
    check(len(m.segments) == 2 and m.segments[0].prosody.pause_after_s == 3.0,
          "a null pause_after_s becomes the default rather than a TypeError")
    check(SW.slug_of("Is BMI a good measure?") == "is_bmi_a_good_measure",
          "a question-shaped topic becomes a legal filename")
    part = SW._image_part(png)
    check(part["inline_data"]["mime_type"] == "image/jpeg", "slides go to the model as JPEG")

    print("which model gets asked")
    check(SW.DEFAULT_MODEL == "gemini-3.8-flash",
          f"the default is 3.8-flash (got {SW.DEFAULT_MODEL})")
    check("preview" not in SW.DEFAULT_MODEL,
          "a preview retires on someone else's schedule, so it is never the default")
    # The profile does not store a model. It used to, which gave the id two
    # homes and let the saved one rot: a profile was still holding
    # `gemini-3-flash-preview` months after that preview was switched off,
    # influencing nothing and contradicting the menu.
    from realme.app.profile import Profile, DEFAULTS
    check("gemini_model" not in DEFAULTS["adapters"],
          "a new profile stores no script model")
    for stored in ("gemini-3-flash-preview", "gemini-3.6-flash", "gemini-3.8-pro"):
        d = tmp / ("prof_" + stored.replace(".", "_"))
        d.mkdir(parents=True, exist_ok=True)
        (d / "profile.json").write_text(json.dumps(
            {"adapters": {"script_writer": "gemini", "gemini_model": stored,
                          "tts": "qwen3cpp", "guest_tts": "piper:en_US-amy-medium",
                          "draft_tts": "piper"}}), encoding="utf-8")
        ad = Profile(d).data["adapters"]
        check("gemini_model" not in ad,
              f"an old profile holding {stored!r} has it dropped")
        check(json.loads((d / "profile.json").read_text(encoding="utf-8"))
              ["adapters"].get("gemini_model") is None,
              "and the file on disk is rewritten without it")
        check(ad["tts"] == "qwen3cpp",
              "while the rest of the profile is untouched")

    print("roman numerals")
    from realme.text.acronyms import expand
    check(expand("A Type II error.")[0] == "A Type 2 error.", "Type II -> Type 2")
    check(expand("an IV line")[0] == "an I V line", "a bare IV is still spelled")

    print("server helpers")
    os.environ["REALME_HOME"] = str(tmp / "home")
    from realme.app import server as S
    check(S.project_id_from("Lecture #3 (final).pdf") == "Lecture_3_final", "project id is URL- and folder-safe")
    check(S.project_id_from("..\\..\\x.pdf") == "x", "and cannot climb out of the projects folder")
    check(S.safe_filename("../../evil.pdf", "deck.pdf") == "evil.pdf", "upload filename loses its path")
    check(S.safe_filename("..\\..\\.env", "deck.pdf") == "env", "and its backslashes and leading dots")
    (S.PROJECTS / "p1").mkdir(parents=True, exist_ok=True)
    (S.PROJECTS / "p1" / "ok.txt").write_text("x", encoding="utf-8")
    (S.DATA / ".env").write_text("SECRET=1", encoding="utf-8")
    from fastapi import HTTPException
    for bad in ("..\\..\\.env", "../.env", ".env", "..%5C.env"):
        try:
            S.get_file("p1", bad)
            check(False, f"{bad!r} was served")
        except HTTPException as e:
            check(e.status_code == 404, f"{bad!r} is refused")
    r = S.get_file("p1", "ok.txt")
    check(str(getattr(r, "path", "")).endswith("ok.txt"), "a plain name inside the project is served")
    for bad in ("../x", "..\\x", "../projects_x", ""):
        try:
            S.project_dir(bad)
            check(False, f"project id {bad!r} accepted")
        except HTTPException:
            check(True, f"project id {bad!r} refused")
    S.build_tts = lambda name: FakeTTS(voice=name)
    S._TTS_CACHE.clear()
    a = S.make_tts("fake"); b = S.make_tts("fake")
    check(a is b, "the same engine is built once and shared")
    S.forget_tts()
    check(S.make_tts("fake") is not a, "and rebuilt after the voice changes")

    # The gaps are applied when utterances are JOINED, not when they are
    # synthesised, so they have to be in the video key or a re-render with a
    # different pause_scale is a cache hit and the knob does nothing. This is
    # the third time a key has been missing something that changes the output.
    print("changing the gaps re-renders the video, not the speech")
    try:
        import pymupdf
    except ImportError:
        print("  --   pymupdf not installed; skipped")
    else:
        from realme.adapters.script_writer import PlaceholderScriptWriter
        from realme.pipeline import lecture as lec
        deck = tmp / "gapdeck.pdf"
        doc = pymupdf.open()
        for i in range(2):
            pg = doc.new_page(width=960, height=540)
            pg.insert_text((72, 100), f"Slide {i+1}: Confounding, bias, and error.",
                            fontsize=24)
            pg.insert_text((72, 150), "The odds ratio was 2.3. We adjusted for age.",
                            fontsize=16)
        doc.save(str(deck))
        proj = tmp / "gapproj"
        gm, _ = lec.write_script(deck, proj, PlaceholderScriptWriter(),
                                 log=lambda *_: None)
        gt = FakeTTS()
        a = lec.render(deck, proj, gm, gt, pause_scale=1.0, log=lambda *_: None)
        wide = duration_of(a["master"])
        spoke = len(gt.calls)
        b = lec.render(deck, proj, gm, gt, pause_scale=1.0, log=lambda *_: None)
        check(abs(duration_of(b["master"]) - wide) < 0.05 and len(gt.calls) == spoke,
              "the same scale twice is a full cache hit")
        c = lec.render(deck, proj, gm, gt, pause_scale=0.85, log=lambda *_: None)
        tight = duration_of(c["master"])
        check(tight < wide - 0.1,
              f"a lower scale gives a shorter video ({wide:.2f}s -> {tight:.2f}s)")
        check(len(gt.calls) == spoke,
              f"without re-synthesising a single utterance "
              f"({len(gt.calls) - spoke} extra calls)")

    print("re-recording an existing project")
    # There was no way back into a project: PROJECT was set only by drafting,
    # importing or revising, so after a page reload the only route to Render
    # was re-drafting -- a model call that REPLACES the script. The render
    # endpoint could already do it from the saved manifest.
    import inspect as _i2
    from realme.pipeline.lecture import render as _r2, build_lecture as _bl2
    from realme.pipeline.speak import speak_segment as _ss2
    for fn, arg in ((_r2, "fresh"), (_bl2, "fresh"), (_ss2, "reuse")):
        check(arg in _i2.signature(fn).parameters,
              f"{fn.__name__} takes {arg}")
    ui = (HERE / "realme/app/ui.html").read_text(encoding="utf-8")
    check("openProject" in ui and 'id="opproj"' in ui,
          "the page offers a project to reopen")
    check('id="rfresh"' in ui and "fresh: $('rfresh').checked" in ui,
          "and a tick that reaches the render call")
    # Reuse is the default, and turning it off really does speak again.
    t2 = FakeTTS()
    l2 = RenderLedger(tmp / "fresh.json")
    speak_segment("Say this once.", t2, tmp / "f", 1, l2)
    n = len(t2.calls)
    speak_segment("Say this once.", t2, tmp / "f", 1, l2)
    check(len(t2.calls) == n, "the same text is reused by default")
    speak_segment("Say this once.", t2, tmp / "f", 1, l2, reuse=False)
    check(len(t2.calls) == n + 1, "reuse=False speaks it again")
    speak_segment("Say this once.", t2, tmp / "f", 1, l2)
    check(len(t2.calls) == n + 1,
          "and the ledger was still written, so the next render is cheap again")
    # The overwrite notice has to come from the file on disk, not a guess.
    from realme.app import server as S2
    pd = S2.PROJECTS / "reopen_demo"
    pd.mkdir(parents=True, exist_ok=True)
    check(S2.existing_output(pd, "reopen_demo") == [],
          "a project with no render reports nothing to replace")
    (pd / "reopen_demo.mp4").write_bytes(b"x" * 1_200_000)
    got = S2.existing_output(pd, "reopen_demo")
    check(len(got) == 1 and got[0]["name"] == "reopen_demo.mp4"
          and got[0]["mb"] == 1.2 and got[0]["draft"] is False,
          f"and names the file a render would overwrite ({got})")
    (pd / "reopen_demo_draft.mp4").write_bytes(b"x" * 100_000)
    got = S2.existing_output(pd, "reopen_demo")
    check(len(got) == 2 and got[1]["draft"] is True,
          "including a draft take, marked as one")

    # An older project is exactly what somebody reopens, and the oldest ones
    # have no `deck_path.txt`. `old_deck_path` has always known how to cope;
    # the render endpoint read the file directly and died on a bare
    # FileNotFoundError, surfaced as a 500.
    old_proj = S2.PROJECTS / "no_pointer"
    (old_proj / "_work").mkdir(parents=True, exist_ok=True)
    (old_proj / "slides.pdf").write_bytes(b"%PDF-1.4 ")
    found = S2.old_deck_path(old_proj)
    check(found is not None and found.name == "slides.pdf",
          "a project with no deck_path.txt still finds its deck")
    from fastapi.testclient import TestClient
    from realme.pipeline.lecture import save_script
    from realme.core.schema import Manifest as _M, Segment as _S, Prosody as _Pr
    bare = S2.PROJECTS / "no_deck"
    (bare / "_work").mkdir(parents=True, exist_ok=True)
    save_script(bare, _M(project_id="no_deck", title="T", script_source="s",
                         segments=[_S(segment_id=1, slide_index=0,
                                      spoken_text="words", prosody=_Pr())]))
    resp = TestClient(S2.app).post("/api/lecture/no_deck/render", json={})
    check(resp.status_code == 400 and "deck" in resp.text.lower(),
          f"and a project whose deck is gone says so, not 500 "
          f"({resp.status_code})")

    print()
    print("all good" if not FAILS else f"{FAILS} FAILED")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
