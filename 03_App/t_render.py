#!/usr/bin/env python3
"""
The render path, exercised with a fake engine that records what it was asked.

Guards against: a pace edit re-serving old audio, the dialogue guest played
at the main voice's sample rate, a pause-only segment crashing Piper, and a
renumbered segment re-rendering everything after it. Each needs a real
ledger and a TTS to show.

Needs ffmpeg. Everything else is the real pipeline.
"""
from __future__ import annotations
import json, math, os, shutil, struct, sys, tempfile, wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

#: Where this file lives. Every path below is relative to it, never to the
#: working directory: these suites are run from the project root as often as
#: from 03_App, and `Path("realme/...")` means a different thing in each.
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

    # A pace correction belongs to the voice, not to the words, and for an
    # engine that realises pace with ffmpeg it can be applied after the fact.
    # The guarantee: changing it rebuilds the stretched copy and buys nothing.
    # The factor is kept out of `voice_fingerprint`, so re-measuring a
    # correction does not invalidate every utterance in the project.
    class Retimes(FakeTTS):
        retimes_after = True

        def __init__(self, factor=1.0, **kw):
            super().__init__(**kw)
            self.factor = factor

        def timing_factor(self, pace=1.0):
            return pace * self.factor

    slow = Retimes(0.8, voice="v3")
    led2 = RenderLedger(tmp / "ledger2.json")
    a = speak_segment("The odds ratio was two.", slow, tmp / "b", 1, led2)
    bought = len(slow.calls)
    check(bought == 1, f"the take is bought once ({bought})")
    check(abs(a[0].duration_s - 0.5 / 0.8) < 0.05,
          f"and the duration the pipeline reports is the stretched one "
          f"({a[0].duration_s:.3f}s)")

    faster = Retimes(1.1, voice="v3")
    faster.calls = slow.calls                    # the same meter
    b = speak_segment("The odds ratio was two.", faster, tmp / "b", 1, led2)
    check(len(slow.calls) == bought,
          "a different correction buys nothing from the engine")
    check(Path(b[0].wav) != Path(a[0].wav),
          "it is a different file, so no audio at the old speed is served")
    check(b[0].duration_s < a[0].duration_s - 0.05,
          f"and it is genuinely faster ({b[0].duration_s:.3f}s vs "
          f"{a[0].duration_s:.3f}s)")
    again = speak_segment("The odds ratio was two.", faster, tmp / "b", 1, led2)
    check(Path(again[0].wav) == Path(b[0].wav) and len(slow.calls) == bought,
          "and repeating it is free twice over")

    k = RenderLedger.key("utt", "x|1.000|en-US", "fake:v1")
    check(":" in k and k.split(":")[0] == "utt" and RenderLedger.stem(k) == k.replace(":", "_"),
          "key carries no position and has a filename stem")

    print("stopping at a quota loses nothing")
    # A hosted engine's daily allowance runs out mid-render. The ledger is
    # written after every take, so the requests already spent are recorded
    # and the same command resumes from there.
    from realme.adapters.base import QuotaExceeded
    # Every sentence distinct: repeats would be cache hits on the first take,
    # and the segment would finish without reaching the quota.
    long_text = " ".join(
        f"Point number {n} concerns the design of the study and what it can "
        f"support." for n in range(1, 25))

    class RunsOut(FakeTTS):
        def __init__(self, allow, **kw):
            super().__init__(**kw)
            self.allow = allow

        def synthesize(self, text, out_wav, voice=None, pace=1.0, **kw):
            if len(self.calls) >= self.allow:
                raise QuotaExceeded("no requests left", retry_after_s=42704.0,
                                    limit=100, period="day")
            return super().synthesize(text, out_wav, voice=voice, pace=pace)

    led3 = RenderLedger(tmp / "ledger3.json")
    stingy = RunsOut(2, voice="v4")
    try:
        speak_segment(long_text, stingy, tmp / "c", 1, led3)
        stopped = False
    except QuotaExceeded:
        stopped = True
    spent = len(stingy.calls)
    check(stopped, "the render stops rather than carrying on silently")
    check(spent == 2, f"after spending exactly what was allowed ({spent})")
    kept = sum(1 for k in led3.state if k.startswith("utt:"))
    check(kept == spent,
          f"and every take it paid for is in the ledger ({kept} of {spent})")

    plenty = RunsOut(99, voice="v4")
    utts = speak_segment(long_text, plenty, tmp / "c", 1, led3)
    check(len(plenty.calls) + spent > spent,
          "re-running continues instead of starting over")
    check(len(plenty.calls) == len(utts) - spent,
          f"buying only what was missing ({len(plenty.calls)} more for "
          f"{len(utts)} utterances, {spent} already paid for)")
    check(all(Path(u.wav).is_file() for u in utts if u.wav),
          "and the finished segment is whole")

    print("a deck that is not there says where it looked")
    # `realme lecture deck.pdf` run from 00_Windows instead of the project
    # root must name the directory it looked in, not only the file, and not
    # as a pymupdf FileNotFoundError deep in a traceback.
    from realme.core.media import MediaError
    from realme.cli import main as cli_main
    from realme.pipeline import slides as slides_mod
    proj = tmp / "proj"
    (proj / "00_Windows").mkdir(parents=True, exist_ok=True)
    (proj / "Guide.pdf").write_bytes(b"%PDF-1.4\n")
    was = Path.cwd()
    try:
        os.chdir(proj / "00_Windows")
        try:
            slides_mod.require_deck("Guide.pdf")
            said = ""
        except MediaError as e:
            said = str(e)
        check(bool(said), "a missing deck is refused before any work")
        check("00_Windows" in said,
              "and the message names the directory it looked in")
        check(str(proj / "Guide.pdf") in said,
              "and the file of that name one level up, which is the mistake")
        check("from " + str(proj) in said,
              "with where to run it from")
        check(Path(slides_mod.require_deck(proj / "Guide.pdf")).is_file(),
              "while a deck that is there comes straight back")
        rc = cli_main(["lecture", "Guide.pdf", "--script", "imported",
                       "-o", "out", "--tts", "piper"])
        check(rc == 1, f"and the CLI exits 1 rather than raising ({rc})")
    finally:
        os.chdir(was)

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

    print("a long narration joins in batches, and sounds the same")
    import math as _m, struct as _st, wave as _wv
    from realme.pipeline.speak import (JOIN_BATCH, _join_one, _join_batched,
                                       Utterance as _U)

    long_dir = tmp / "many"
    long_dir.mkdir(parents=True, exist_ok=True)
    many = []
    for i in range(JOIN_BATCH * 3 + 7):          # several batches, uneven last
        w = long_dir / f"utt_{i:03d}.wav"
        with _wv.open(str(w), "wb") as _f:
            _f.setnchannels(1); _f.setsampwidth(2); _f.setframerate(24000)
            _f.writeframes(b"".join(
                _st.pack("<h", int(8000 * _m.sin(2 * _m.pi * (200 + i) * n / 24000)))
                for n in range(int(24000 * (0.2 + (i % 3) * 0.05)))))
        many.append(_U(index=i, text=f"s{i}", engine_text=f"s{i}", wav=w,
                       gap_ms=120 + (i % 4) * 40))

    def _samples(path):
        with _wv.open(str(path), "rb") as f:
            return f.getparams(), f.readframes(f.getnframes())

    one = _samples(_join_one(many, long_dir / "one.wav", None, 1.0, 0.6))
    lots = _samples(_join_batched(many, long_dir / "lots.wav", None, 1.0, 0.6))
    check(one[0] == lots[0],
          f"the batches come back in the same format ({one[0].framerate} Hz, "
          f"{one[0].nchannels} ch)")
    # Sample for sample: every gap is still the gap that utterance earned,
    # and the batches are stitched by copy rather than re-encoded.
    check(one[1] == lots[1],
          f"and identical sample for sample, not merely the same length "
          f"({len(one[1])} vs {len(lots[1])} bytes)")
    wrapped = _samples(join_utterances(many, long_dir / "wrap.wav", None, 1.0, 0.6))
    check(wrapped[1] == lots[1], "which is the path a long list takes")
    few = many[:5]
    check(_samples(join_utterances(few, long_dir / "few.wav", None, 1.0, 0.6))[1]
          == _samples(_join_one(few, long_dir / "few1.wav", None, 1.0, 0.6))[1],
          "while a short list is joined exactly as before")

    # The reason it is batched at all. Windows refuses a command line past
    # about 32,000 characters, and `narrate` joins a whole document at once.
    def _one_call_len(n):
        d = (r"C:\Users\someone\RealMeStudio\_work\anim_a_long_project_name"
             r"\narration_work\audio")
        args = ["ffmpeg", "-y", "-f", "lavfi", "-t", "0.600", "-i",
                "anullsrc=r=24000:cl=mono"]
        fl = ["[0:a]anull[lead]"]
        for i in range(n):
            args += ["-i", d + "\\utt_%020x.wav" % i]
            fl.append(f"[{i+1}:a]apad=pad_dur=0.18[a{i}]")
        fc = ";".join(fl) + ";[lead]" + "".join(f"[a{i}]" for i in range(n)) \
            + f";concat=n={n+1}:v=0:a=1[out]"
        args += ["-filter_complex", fc, "-map", "[out]", "-c:a", "pcm_s16le",
                 "out.wav"]
        return sum(len(a) + 1 for a in args)

    check(_one_call_len(250) > 32000,
          f"250 utterances in one call would exceed the Windows limit "
          f"({_one_call_len(250):,d} chars)")
    check(_one_call_len(JOIN_BATCH) < 8000,
          f"while a batch of {JOIN_BATCH} stays far inside it "
          f"({_one_call_len(JOIN_BATCH):,d} chars)")

    print("a moved project does not pay for its audio twice")
    import shutil as _sh
    from realme.core.ledger import RenderLedger as _RL
    was_dir = tmp / "before" / "narration_work"
    (was_dir / "audio").mkdir(parents=True, exist_ok=True)
    take = was_dir / "audio" / "utt_deadbeef.wav"
    take.write_bytes(b"RIFF----WAVEfmt ")
    _RL(was_dir / "ledger.json").put("utt:deadbeef", take, 1.25)
    moved = tmp / "after" / "narration_work"
    moved.parent.mkdir(parents=True, exist_ok=True)
    _sh.move(str(was_dir), str(moved))
    hit = _RL(moved / "ledger.json").get("utt:deadbeef")
    check(bool(hit), "a renamed project still finds its recorded sentences")
    check(bool(hit) and Path(hit["path"]).is_file() and hit["duration"] == 1.25,
          "at their new path, with the measured duration kept")
    check(_RL(moved / "ledger.json").get("utt:nosuchkey") is None,
          "and a key that was never recorded is still a miss")

    print("what the log box shows")
    # One line per slide, not per utterance: enough to show progress without
    # flooding the log. The totals come back with the utterances so the
    # caller can say it once.
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
    # A typical enrolled clip: 0.336 s median pause, 1.192 s longest, 18
    # breath-length gaps in 62 s. Mid-sentence breathing must not be applied
    # in full at sentence boundaries, on top of the trailing breath the
    # engine already renders per utterance.
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
    # Every keyword `render` and the CLI pass must also be accepted by
    # `build_lecture`, or `realme lecture` fails on a TypeError before
    # rendering a frame.
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
    # The profile does not store a model, so the id has one home. A model
    # an older profile still holds is ignored rather than contradicting
    # the menu.
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

    print("words written in capitals")
    from realme.text.acronyms import expand as _expand, READ_AS_WORDS, SAY_AS
    said, spelled = _expand("A AND B, but NOT C, and neither D NOR E.")
    check("AND" in said and "NOT" in said and "NOR" in said,
          "connectives in capitals are read as words, not spelled")
    check(spelled == [], f"and none of them is reported as spelled ({spelled})")
    said, _ = _expand("A XOR B")
    check("ex or" in said, f"XOR is said 'ex or' ({said})")
    # The ambiguous ones stay spelled: each is both a word and something
    # people really do spell out.
    said, spelled = _expand("the US cohort, OR the IT system")
    check(set(spelled) >= {"US", "OR", "IT"},
          f"while OR, IT and US are still spelled ({spelled})")
    # The one that would be a clinical error.
    said, spelled = _expand("children with ALL")
    check("ALL" in spelled,
          "ALL is spelled, because in this field it is a leukaemia")
    check("ALL" not in READ_AS_WORDS and "OR" not in READ_AS_WORDS,
          "and neither is in the table")
    # A decision somebody recorded still beats the table.
    class _Lx:
        def get(self, term):
            return {"say": "and so on"} if term == "AND" else None
    said, _ = _expand("A AND B", lexicon=_Lx())
    check("AND" in said, "a lexicon entry is left for the lexicon to apply")
    check("XNOR" in SAY_AS, "XNOR has its own spoken form too")

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
    # different pause_scale is a cache hit and the knob does nothing.
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

        # What it took, in the report, so the Studio can show it beside what
        # it produced rather than only in the scrolling log.
        rep = a["report"]
        check(rep.get("render_s", 0) > 0,
              f"the report says how long the render took ({rep.get('render_s')}s)")
        check(rep.get("recording_s") is not None
              and rep["recording_s"] <= rep["render_s"],
              f"and how much of that was the engine "
              f"({rep.get('recording_s')}s of {rep.get('render_s')}s)")
        check(rep.get("takes_spoken") == spoke,
              f"with the takes it bought ({rep.get('takes_spoken')} vs {spoke})")
        # Against the first render's TOTAL, not its purchases: this deck
        # repeats a line across both slides, so the first render already
        # reused some of its own takes.
        utterances = rep["takes_spoken"] + rep["takes_reused"]
        check(b["report"]["takes_spoken"] == 0
              and b["report"]["takes_reused"] == utterances,
              f"and a cached re-render reports every take as reused "
              f"({b['report'].get('takes_spoken')} spoken, "
              f"{b['report'].get('takes_reused')} of {utterances} reused)")

    print("re-recording an existing project")
    # An existing project can be rendered again after a page reload, from
    # its saved manifest, without re-drafting (a model call that REPLACES
    # the script).
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

    # An older project may have no `deck_path.txt`. The render endpoint goes
    # through `old_deck_path`, which copes, rather than reading the file and
    # failing with a 500.
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
    resp = TestClient(S2.app, base_url="http://127.0.0.1").post("/api/lecture/no_deck/render", json={})
    check(resp.status_code == 400 and "deck" in resp.text.lower(),
          f"and a project whose deck is gone says so, not 500 "
          f"({resp.status_code})")

    print()
    print("all good" if not FAILS else f"{FAILS} FAILED")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
