#!/usr/bin/env python3
"""
Recording a lecture's sentences with several engines at once.

A stand-in engine plays the local clone: safe to copy, a fixed time per word,
and it writes down every sentence it is asked for and which process asked.
Checked, against what would go wrong:

  * the planning step asks for exactly the sentences the render loop would
    record -- controls, pauses, a French block, a masked phrase, a runaway
    sentence and all -- so after it the loop records nothing itself;
  * two engines really run, and every sentence is recorded once, not twice;
  * the lecture comes out the same as with one engine: same sentences, same
    captions, same length;
  * an unchanged render records nothing; an edited slide records only its
    sentences; "record every sentence again" records each exactly once;
  * an engine that cannot be copied, too little memory, or extra engines
    that fail to start all fall back to one engine and still finish.
"""
from __future__ import annotations
import json, math, os, struct, sys, tempfile, time, wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# The engine copies run in fresh processes that import this file again; they
# must find the same folder (and the same record of calls), not make another.
_ISO = os.environ.get("T_PRERECORD_ISO") or tempfile.mkdtemp(prefix="t_prerecord_")
os.environ["T_PRERECORD_ISO"] = _ISO
os.environ["REALME_HOME"] = str(Path(_ISO) / "home")
os.environ["REALME_TOOLS"] = str(Path(_ISO) / "tools")
os.environ["HOME"] = _ISO
os.environ["USERPROFILE"] = _ISO
os.environ["REALME_NO_WARMUP"] = "1"

from t_util import check, summary          # noqa: E402

WORK = Path(_ISO) / "work"
WORK.mkdir(parents=True, exist_ok=True)
CALLS = WORK / "calls.jsonl"


def tone(path: Path, seconds: float, rate=24000):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(rate * seconds)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", int(5000 * math.sin(
            2 * math.pi * 170 * i / rate))) for i in range(n)))


class Stub:
    """Copyable, like the local clone; slow enough per word to overlap."""
    name = "stub-local"
    is_voice_clone = True
    is_placeholder = False
    phoneme_syntax = None
    retimes_after = False
    parallel_safe = True
    speaks_languages = ("fr",)

    def __init__(self, fail_in_copies: bool = False):
        self.threads = 6
        self.parent = os.getpid()
        self.fail_in_copies = fail_in_copies

    def preflight(self):
        if self.fail_in_copies and os.getpid() != self.parent:
            raise RuntimeError("this copy cannot start")

    def voice_fingerprint(self):
        return "stub-local:v1"

    def timing_factor(self, pace=1.0):
        return 1.0

    def synthesize(self, text, out_wav, voice=None, pace=1.0, language=None):
        time.sleep(0.03 * len(text.split()))
        tone(Path(out_wav), max(0.5, len(text.split()) * 0.2))
        with open(CALLS, "a", encoding="utf-8") as f:
            f.write(json.dumps({"pid": os.getpid(), "text": text,
                                "lang": language}) + "\n")
        return Path(out_wav)


class BatchStub(Stub):
    """Plays a local clone built with the batch entry point."""
    def __init__(self, fail_batches: bool = False):
        super().__init__()
        self.fail_batches = fail_batches

    def batch_built(self):
        return True

    def synthesize_batch(self, items, voice=None):
        if self.fail_batches:
            raise RuntimeError("this pass failed")
        langs = {lang for _, _, _, lang in items}
        time.sleep(0.03 * max(len(t.split()) for t, _, _, _ in items))
        for t, o, p, lang in items:
            tone(Path(o), max(0.5, len(t.split()) * 0.2))
        with open(CALLS, "a", encoding="utf-8") as f:
            for t, o, p, lang in items:
                f.write(json.dumps({"pid": os.getpid(), "text": t, "lang": lang,
                                    "batch": len(items),
                                    "mixed": len(langs) > 1}) + "\n")
        return [Path(o) for _, o, _, _ in items]


class NetStub(Stub):
    """Plays Gemini: requests wait on the network, several may be in flight
    from threads sharing one adapter, a take can come back hurried, Google
    can ask to slow down, and the day's allowance can run out."""
    parallel_safe = False
    network_bound = True
    concurrent_requests = 3
    max_chars_per_call = None

    def __init__(self, *, rush_word=None, slow_after=None, quota_after=None,
                 minute_at=None):
        super().__init__()
        self.minute_at = minute_at
        import threading
        self._tl = threading.local()
        self._mu = threading.Lock()
        self.in_flight = self.peak = self.n_calls = 0
        self.rate_waits = 0
        self.rush_word, self.slow_after, self.quota_after = (
            rush_word, slow_after, quota_after)
        self.rushed_texts = []

    def synthesize(self, text, out_wav, voice=None, pace=1.0, language=None):
        from realme.adapters.base import QuotaExceeded
        with self._mu:
            self.n_calls += 1
            n = self.n_calls
            if self.quota_after and n > self.quota_after:
                raise QuotaExceeded("no requests left today", period="day")
            if self.minute_at and n == self.minute_at:
                raise QuotaExceeded("too many this minute", period="minute",
                                    retry_after_s=20)
            if self.slow_after and n == self.slow_after:
                self.rate_waits += 1
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
        try:
            time.sleep(0.25)                       # waiting for Google
            tone(Path(out_wav), max(0.5, len(text.split()) * 0.2))
            with open(CALLS, "a", encoding="utf-8") as f:
                f.write(json.dumps({"pid": os.getpid(), "text": text}) + "\n")
            hurried = (self.rush_word and self.rush_word in text
                       and len(text.split()) > 6)
            self._tl.rush = 1.2 if hurried else None
            if hurried:
                self.rushed_texts.append(text)
        finally:
            with self._mu:
                self.in_flight -= 1
        return Path(out_wav)

    def last_rush(self):
        r, self._tl.rush = getattr(self._tl, "rush", None), None
        return r


class NoCopy(Stub):
    parallel_safe = False


class Refuse(Stub):
    """Fails the test if the render loop itself asks for a sentence."""
    def synthesize(self, *a, **kw):
        raise AssertionError(f"the loop recorded {a[0]!r} itself")


def calls() -> list[dict]:
    if not CALLS.is_file():
        return []
    return [json.loads(l) for l in CALLS.read_text("utf-8").splitlines() if l]


def main() -> int:
    import shutil
    if not shutil.which("ffmpeg"):
        print("ffmpeg not on PATH; skipping")
        return 0
    try:
        import pymupdf
    except ImportError:
        print("pymupdf not installed; skipping")
        return 0
    from realme.adapters.script_writer import PlaceholderScriptWriter
    from realme.core.ledger import RenderLedger
    from realme.core.media import duration_of
    from realme.pipeline import lecture as lec, prerecord as PR
    from realme.pipeline.speak import speak_segment
    # Free memory is the test's to decide, not this machine's: the stand-in
    # engine needs none, and a small test machine must not turn the
    # parallel path off.
    import realme.parbench as _PB
    _PB.available_memory_gb = lambda: 60.0

    deck = WORK / "deck.pdf"
    doc = pymupdf.open()
    for i in range(4):
        pg = doc.new_page(width=960, height=540)
        pg.insert_text((72, 100), f"Slide {i + 1}: selection bias {i + 1}.",
                       fontsize=24)
    doc.save(str(deck))
    long_one = ("A sentence that runs on, " * 30).strip() + " and finally ends."
    texts = [
        "Selection bias enters when who is studied depends on the outcome. "
        "[[pause:600]] That is the whole idea, said once.",
        "[[slow]]Read this slowly.[[/slow]] Then at the usual pace. "
        "[[mask]]a word nobody hears[[/mask]] And on.",
        "In French: [[fr-FR]]Le biais de sélection.[[/fr-FR]] Back in English.",
        long_one,
    ]

    def project(name):
        proj = Path(os.environ["REALME_HOME"]) / "projects" / name
        m, _ = lec.write_script(deck, proj, PlaceholderScriptWriter(),
                                log=lambda *_: None)
        for seg, t in zip(m.segments, texts):
            seg.spoken_text = t
            seg.prosody.lead_in_s = seg.prosody.pause_after_s = 0.3
        lec.save_script(proj, m)
        return proj, m

    quiet = dict(check_audio=False, log=lambda *_: None, width=640, height=360,
                 fps=30, signalling="off")

    print("the plan matches the render loop")
    proj, m = project("plan")
    work = proj / "_work"
    ledger = RenderLedger(work / "ledger.json")
    takes = PR.plan(m, Stub(), ledger, work / "audio")
    check(len(takes) >= 6 and all(t.text.strip() for t in takes),
          "the sentences to record are found", len(takes))
    check(not any("nobody hears" in t.text for t in takes),
          "a masked phrase is never sent to the engine")
    check(any(t.language == "fr" for t in takes),
          "a French block goes to the engine as French")
    os.environ[PR.ENV] = "2"
    CALLS.unlink(missing_ok=True)
    pre = PR.run(m, Stub(), work / "audio", ledger, log=lambda *_: None)
    check(pre["spoken"] == len(takes) and pre["engines"] == 2,
          "two engines record them all", pre)
    refuse = Refuse()
    for seg in m.segments:
        speak_segment(seg.spoken_text, refuse, work / "audio", seg.segment_id,
                      ledger, pace=seg.prosody.pace, prerecorded=pre["keys"])
    check(True, "after that the render loop records nothing itself")
    pids = {c["pid"] for c in calls()}
    texts_said = [c["text"] for c in calls()]
    check(len(pids) == 2, "two processes really did the recording", pids)
    check(len(texts_said) == len(set(texts_said)) == len(takes),
          "each sentence recorded once, none twice",
          (len(texts_said), len(set(texts_said)), len(takes)))

    print("\nthe same lecture as with one engine")
    os.environ[PR.ENV] = "1"
    one, m1 = project("one_engine")
    a = lec.render(deck, one, m1, Stub(), **quiet)
    os.environ[PR.ENV] = "2"
    two, m2 = project("two_engines")
    CALLS.unlink(missing_ok=True)
    b = lec.render(deck, two, m2, Stub(), **quiet)
    check(a["report"]["verdict"].startswith("PASS")
          and b["report"]["verdict"].startswith("PASS"),
          "both pass the checks", (a["report"]["verdict"], b["report"]["verdict"]))
    check(b["report"].get("recording_engines") == 2
          and "recording_engines" not in a["report"],
          "the report says when two engines recorded")
    check(b["report"]["takes_spoken"] == a["report"]["takes_spoken"]
          and b["report"]["takes_reused"] == 0,
          "and counts every sentence as spoken, as one engine does",
          (a["report"]["takes_spoken"], b["report"]["takes_spoken"],
           b["report"]["takes_reused"]))
    da, db = duration_of(a["master"]), duration_of(b["master"])
    check(abs(da - db) < 0.05, "the videos are the same length", (da, db))
    sa = Path(a["srt"]).read_text("utf-8")
    sb = Path(b["srt"]).read_text("utf-8")
    check(sa == sb, "and have the same captions, to the millisecond")

    print("\nre-rendering")
    CALLS.unlink(missing_ok=True)
    c = lec.render(deck, two, lec.load_script(two), Stub(), **quiet)
    check(not calls() and c["report"]["takes_reused"] > 0,
          "an unchanged lecture records nothing", len(calls()))
    m2 = lec.load_script(two)
    m2.segments[0].spoken_text = "A new first slide. It has two sentences now."
    lec.save_script(two, m2)
    CALLS.unlink(missing_ok=True)
    lec.render(deck, two, m2, Stub(), **quiet)
    said = [c_["text"] for c_ in calls()]
    check(said and all("first slide" in t or "two sentences" in t for t in said),
          "an edited slide records only its own sentences", said)
    CALLS.unlink(missing_ok=True)
    f = lec.render(deck, two, lec.load_script(two), Stub(),
                   **{**quiet, "fresh": True})
    said = [c_["text"] for c_ in calls()]
    check(len(said) == len(set(said)) == f["report"]["takes_spoken"]
          and f["report"]["takes_reused"] == 0,
          "record every sentence again records each exactly once",
          (len(said), len(set(said)), f["report"]["takes_spoken"]))

    print("\nseveral sentences per pass")
    from realme.adapters.tts_qwen3_dll import BATCH_FILE
    from realme.core.env import data_home
    os.environ.pop(PR.BATCH_ENV, None)
    (data_home() / BATCH_FILE).unlink(missing_ok=True)
    check(PR.batch_for(BatchStub(), 20) == 8,
          "unmeasured, the 16-bit model's default: 8 per pass")
    check(PR.batch_for(BatchStub(), 5) == 5, "never more than there are sentences")
    data_home().mkdir(parents=True, exist_ok=True)
    (data_home() / BATCH_FILE).write_text('{"batch": 3}', encoding="utf-8")
    check(PR.batch_for(BatchStub(), 20) == 3, "measured, what was measured")
    os.environ["REALME_QWEN3CPP_QUANT"] = "q8_0"
    check(PR.batch_for(BatchStub(), 20) == 4,
          "but not for the other model, which is measured on its own (its "
          "default is 4)")
    (data_home() / BATCH_FILE).write_text(
        '{"f16": {"batch": 2}, "q8_0": {"batch": 4}}', encoding="utf-8")
    check(PR.batch_for(BatchStub(), 20) == 4, "the 8-bit model's own measurement")
    os.environ.pop("REALME_QWEN3CPP_QUANT")
    check(PR.batch_for(BatchStub(), 20) == 2, "and the 16-bit model's, kept beside it")
    os.environ["REALME_QWEN3CPP_QUANT"] = "q8_0"
    os.environ.pop("REALME_QWEN3CPP_QUANT")
    (data_home() / BATCH_FILE).write_text('{"batch": 3}', encoding="utf-8")
    check(PR.batch_for(Stub(), 20) == 1 and PR.batch_for(NetStub(), 20) == 1,
          "never for an engine without batches, nor for a hosted voice")
    os.environ[PR.BATCH_ENV] = "2"
    check(PR.batch_for(BatchStub(), 20) == 2 and PR.batch_for(BatchStub(), 1) == 1,
          "REALME_RECORD_BATCH overrides it; one sentence is one at a time")
    os.environ[PR.BATCH_ENV] = "3"
    os.environ.pop(PR.ENV, None)         # unset: engines as the machine allows
    import realme.core.cpu as _cpu
    _cores = _cpu.physical_cores
    _cpu.physical_cores = lambda: 6      # one engine's worth of cores
    bproj, bm = project("batched")
    bwork = bproj / "_work"
    bledger = RenderLedger(bwork / "ledger.json")
    btakes = PR.plan(bm, BatchStub(), bledger, bwork / "audio")
    CALLS.unlink(missing_ok=True)
    pre = PR.run(bm, BatchStub(), bwork / "audio", bledger, log=lambda *_: None)
    said = calls()
    check(pre["spoken"] == len(btakes) and pre["engines"] == 1
          and pre["batch"] == 3, "one engine records them all, three per pass",
          {k: v for k, v in pre.items() if k != "keys"})
    check(max(c["batch"] for c in said) == 3 and len({c["pid"] for c in said}) == 1,
          "in passes of up to three, in one process",
          [c["batch"] for c in said])
    check(len(said) == len({c["text"] for c in said}) == len(btakes),
          "each sentence recorded once, none twice")
    for seg in bm.segments:
        speak_segment(seg.spoken_text, Refuse(), bwork / "audio", seg.segment_id,
                      bledger, pace=seg.prosody.pace, prerecorded=pre["keys"])
    check(True, "and the render loop records nothing itself")
    fproj, fm = project("batch_fails")
    fwork = fproj / "_work"
    fledger = RenderLedger(fwork / "ledger.json")
    ftakes = PR.plan(fm, Stub(), fledger, fwork / "audio")
    CALLS.unlink(missing_ok=True)
    pre = PR.run(fm, BatchStub(fail_batches=True), fwork / "audio", fledger,
                 log=lambda *_: None)
    check(pre["spoken"] == len(ftakes) and len(calls()) == len(ftakes)
          and all("batch" not in c for c in calls()),
          "a pass that fails is recorded one sentence at a time instead", pre["spoken"])
    rproj, rm = project("batched_render")
    r = lec.render(deck, rproj, rm, BatchStub(), **quiet)
    check(r["report"]["verdict"].startswith("PASS")
          and r["report"].get("recording_batch") == 3,
          "a lecture renders, and its report says three per pass",
          r["report"].get("recording_batch"))
    check(isinstance(r["report"].get("recording_wall_s"), float)
          and 0 < r["report"]["recording_wall_s"] <= r["report"]["render_s"],
          "and how long the recording itself took, within the render's time",
          (r["report"].get("recording_wall_s"), r["report"].get("render_s")))
    os.environ["REALME_EARLY_VIDEOS"] = "1"
    try:
        eproj, em = project("early_videos")
        r = lec.render(deck, eproj, em, BatchStub(), **quiet)
    finally:
        os.environ.pop("REALME_EARLY_VIDEOS", None)
    check(r["report"]["verdict"].startswith("PASS")
          and r["report"].get("videos_during_recording") is True
          and isinstance(r["report"].get("videos_after_audio_s"), float),
          "with a local engine, slide videos made during recording give the "
          "same passing lecture, and the report times what was left after",
          {k: r["report"].get(k) for k in ("verdict", "videos_during_recording",
                                            "videos_after_audio_s")})
    os.environ.pop(PR.BATCH_ENV, None)
    (data_home() / BATCH_FILE).unlink(missing_ok=True)
    _cpu.physical_cores = _cores

    print("\nslide videos made while recording")
    import types
    from realme.pipeline import lecture as LC
    real_cpu = os.cpu_count
    try:
        os.cpu_count = lambda: 12
        local = types.SimpleNamespace(network=False, n=1,
                                      base=types.SimpleNamespace(threads=6))
        check(LC.early_video_plan(local) == (2, 3),
              "a local engine on 6 of 12 threads: two videos alongside, 3 threads each",
              LC.early_video_plan(local))
        local.base.threads = 8
        check(LC.early_video_plan(local) == (1, 4), "fewer spare threads, one video")
        os.cpu_count = lambda: 6
        local.base.threads = 6
        check(LC.early_video_plan(local) is None,
              "no spare threads: the videos wait for the audio, as before")
        os.environ[LC.EARLY_ENV] = "1"
        check(LC.early_video_plan(local) == (1, 1), "unless asked for")
        os.environ[LC.EARLY_ENV] = "0"
        net = types.SimpleNamespace(network=True, n=2, base=None)
        check(LC.early_video_plan(net) is None and LC.early_video_plan(None) is None,
              "REALME_EARLY_VIDEOS=0 turns it off for every voice")
        os.environ.pop(LC.EARLY_ENV)
        check(LC.early_video_plan(net) == (LC.video_workers(), None),
              "a hosted voice keeps making them as before")
    finally:
        os.cpu_count = real_cpu
        os.environ.pop(LC.EARLY_ENV, None)
    import inspect
    from realme.pipeline import compose as CP
    src = inspect.getsource(CP.render_segment)
    check(src.count('["-threads", str(threads)] if threads else []') == 2,
          "and their encoder keeps to the threads it is given")

    print("\nthe local clone can be copied")
    import pickle, threading
    from realme.adapters.tts_qwen3_dll import Qwen3DllTTS
    q = Qwen3DllTTS(home=Path(_ISO) / "cpp", temperature=0.8)
    q._lib, q._handle = threading.Lock(), threading.Lock()   # stand-ins: not copyable
    q2 = pickle.loads(pickle.dumps(q))
    check(q2.voice_fingerprint() == q.voice_fingerprint()
          and q2._lib is None and q2._handle is None and q2.parallel_safe,
          "with its settings, and without the loaded library or model")

    print("\na hosted voice: several requests in flight")
    os.environ.pop(PR.ENV, None)
    os.environ[PR.ENV] = "1"
    solo, ms = project("net_one")
    a = lec.render(deck, solo, ms, NetStub(), **quiet)
    os.environ.pop(PR.ENV, None)
    net, mn = project("net_three")
    logs = []
    eng = NetStub()
    CALLS.unlink(missing_ok=True)
    t0 = time.perf_counter()
    b = lec.render(deck, net, mn, eng, **{**quiet, "log": logs.append})
    took = time.perf_counter() - t0
    said = [c_["text"] for c_ in calls()]
    check(eng.peak >= 2 and b["report"].get("recording_engines") == 3,
          "requests go out several at once, from one adapter",
          (eng.peak, b["report"].get("recording_engines")))
    check(len(said) == len(set(said)), "each sentence asked for once")
    check(any("begun while recording" in l for l in logs),
          "slide videos are made while later slides are still recording")
    check(Path(a["srt"]).read_text("utf-8") == Path(b["srt"]).read_text("utf-8")
          and abs(duration_of(a["master"]) - duration_of(b["master"])) < 0.05,
          "and the lecture is the same as one request at a time")

    print("\na hosted voice: Google's limits and the hurried-take check")
    slow, msl = project("net_slow")
    logs = []
    eng = NetStub(slow_after=2)
    r = lec.render(deck, slow, msl, eng, **{**quiet, "log": logs.append})
    check(r["report"]["verdict"].startswith("PASS")
          and any("asked to slow down; now 2" in l for l in logs),
          "asked to slow down, it sends fewer at once and still finishes",
          [l for l in logs if "slow" in l])
    hur, mh = project("net_rush")
    logs = []
    eng = NetStub(rush_word="Selection")
    r = lec.render(deck, hur, mh, eng, **{**quiet, "log": logs.append})
    check(eng.rushed_texts and any("came back" in l and "fast" in l for l in logs)
          and r["report"]["verdict"].startswith("PASS"),
          "a hurried take is still caught and recorded again in shorter pieces",
          [l for l in logs if "fast" in l][:2])
    out_of, mq = project("net_quota")
    eng = NetStub(quota_after=3)
    CALLS.unlink(missing_ok=True)
    from realme.adapters.base import QuotaExceeded
    try:
        lec.render(deck, out_of, mq, eng, **quiet)
        stopped = False
    except QuotaExceeded:
        stopped = True
    ledger = RenderLedger(out_of / "_work" / "ledger.json")
    kept = sum(1 for k in ledger.state if k.startswith("utt"))
    check(stopped and kept >= 1,
          "the day's allowance running out stops the render, keeping what "
          "was recorded", (stopped, kept))
    check(kept == len(calls()),
          "every sentence Google answered is in the ledger, including those "
          "still in flight when the render stopped (paid for once)",
          (kept, len(calls())))
    time.sleep(0.6)
    calls_then = eng.n_calls
    time.sleep(0.6)
    check(eng.n_calls == calls_then, "and nothing keeps asking Google after it")

    mproj, mm = project("net_minute")
    logs = []
    eng = NetStub(minute_at=2)
    r = lec.render(deck, mproj, mm, eng, **{**quiet, "log": logs.append})
    check(r["report"]["verdict"].startswith("PASS")
          and any("per-minute limit; now 2" in l for l in logs)
          and r["report"].get("recording_engines"),
          "a per-minute limit sends fewer at once and carries on, rather "
          "than giving the rest up", [l for l in logs if "minute" in l])

    print("\nREALME_RECORD_WORKERS=1 turns it all off")
    os.environ[PR.ENV] = "1"
    os.environ[PR.BATCH_ENV] = "3"
    oproj, om = project("all_off")
    CALLS.unlink(missing_ok=True)
    r = lec.render(deck, oproj, om, BatchStub(), **quiet)
    check("recording_engines" not in r["report"] and "recording_batch" not in r["report"]
          and all("batch" not in c for c in calls()),
          "no copies, no passes: one sentence at a time, as before")
    os.environ.pop(PR.BATCH_ENV, None)

    print("\nfalling back to one engine")
    check(PR.workers_for(NoCopy(), 10) == 1, "an engine that is not safe to copy")
    import realme.parbench as PB
    real = PB.available_memory_gb
    PB.available_memory_gb = lambda: 1.0
    try:
        check(PR.workers_for(Stub(), 10) == 1, "too little free memory")
    finally:
        PB.available_memory_gb = real
    os.environ[PR.ENV] = "1"
    check(PR.workers_for(Stub(), 10) == 1, "REALME_RECORD_WORKERS=1 turns it off")

    class Threaded(Stub):
        def applies_threads(self):
            return True
    os.environ.pop(PR.ENV, None)
    from realme.core import cpu as CPU
    real_pc, real_mem = CPU.physical_cores, PB.available_memory_gb
    PB.available_memory_gb = lambda: 60.0
    try:
        for cores, want in ((6, 1), (12, 2), (16, 2), (24, 4), (64, 4)):
            CPU.physical_cores = lambda c=cores: c
            check(PR.workers_for(Threaded(), 10) == want,
                  f"{cores} cores: {want} engine(s) of 6 threads",
                  PR.workers_for(Threaded(), 10))
        CPU.physical_cores = lambda: 24
        PB.available_memory_gb = lambda: 9.0
        check(PR.workers_for(Threaded(), 10) == 2,
              "and no more than the free memory has room for",
              PR.workers_for(Threaded(), 10))
    finally:
        CPU.physical_cores, PB.available_memory_gb = real_pc, real_mem
    os.environ[PR.ENV] = "2"
    check(PR.workers_for(Threaded(), 10) == 2, "unless two are asked for")
    os.environ[PR.ENV] = "2"
    bad, mb = project("copies_fail")
    r = lec.render(deck, bad, mb, Stub(fail_in_copies=True), **quiet)
    check(r["report"]["verdict"].startswith("PASS")
          and r["report"].get("recording_engines", 1) == 1,
          "extra engines that cannot start leave the render to one, finished",
          (r["report"]["verdict"], r["report"].get("recording_engines")))
    return summary()


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        import shutil
        shutil.rmtree(_ISO, ignore_errors=True)
