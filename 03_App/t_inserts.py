#!/usr/bin/env python3
"""
A video inserted between a lecture's slides, remembered by the lecture.

The test plays the part of another video tool: a short clip at a different size and
frame rate, with no sound and a caption file beside it, inserted after slide
2 of a three-slide lecture. Then it checks what a viewer and the rest of
RealMe would notice if it were wrong:

  * the clip is in the video, in its place, at the lecture's size, with sound;
  * the captions, chapters and cut list all move by its length, and its own
    caption is carried in;
  * the duration check knows about it, so the video's contents and the
    script agree;
  * splitting at a slide keeps it with the part it follows;
  * taking it out or putting it in again re-joins the last render without
    recording a word;
  * and there is no Animate tab; `realme setup` moves the retired animation
    code of an older install into _Archive intact.
"""
from __future__ import annotations
import contextlib, io, json, math, os, struct, subprocess, sys, tempfile, wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

_ISO = tempfile.mkdtemp(prefix="t_inserts_")
os.environ["REALME_HOME"] = str(Path(_ISO) / "home")
os.environ["REALME_TOOLS"] = str(Path(_ISO) / "tools")
os.environ["HOME"] = _ISO
os.environ["USERPROFILE"] = _ISO
os.environ["REALME_NO_WARMUP"] = "1"

from t_util import check, summary          # noqa: E402

WORK = Path(_ISO) / "work"
WORK.mkdir(parents=True, exist_ok=True)
CLIP_S = 3.0


def tone(path: Path, seconds: float, rate=24000):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(rate * seconds)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", int(6000 * math.sin(
            2 * math.pi * 180 * i / rate))) for i in range(n)))


class Stub:
    name = "stub"
    is_voice_clone = True
    is_placeholder = False
    speaks_languages = ()
    phoneme_syntax = None
    retimes_after = False

    def __init__(self):
        self.calls = []

    def preflight(self):
        pass

    def voice_fingerprint(self):
        return "stub:v1"

    def timing_factor(self, pace=1.0):
        return 1.0

    def synthesize(self, text, out_wav, voice=None, pace=1.0, **kw):
        self.calls.append(text)
        tone(Path(out_wav), max(0.6, len(text.split()) * 0.25))
        return Path(out_wav)


def make_clip(path: Path) -> Path:
    """What another tool might hand over: another size, another rate, silent."""
    from realme.core.media import require
    subprocess.run([require("ffmpeg"), "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", f"testsrc=size=640x480:rate=24:duration={CLIP_S}",
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    path.with_suffix(".srt").write_text(
        "1\n00:00:00,500 --> 00:00:02,000\nAn animated caption.\n",
        encoding="utf-8")
    return path


def streams(path: Path) -> dict:
    from realme.core.media import probe
    out = {}
    for s in probe(path)["streams"]:
        out.setdefault(s["codec_type"], s)
    return out


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
    from realme.core.media import duration_of
    from realme.pipeline import inserts as INS, lecture as lec, split as SPL

    deck = WORK / "deck.pdf"
    doc = pymupdf.open()
    for i in range(3):
        pg = doc.new_page(width=960, height=540)
        pg.insert_text((72, 100), f"Slide {i + 1}: confounding, part {i + 1}.",
                       fontsize=24)
    doc.save(str(deck))
    proj = Path(os.environ["REALME_HOME"]) / "projects" / "lec_ins"
    m, _ = lec.write_script(deck, proj, PlaceholderScriptWriter(),
                            log=lambda *_: None)
    for seg in m.segments:
        seg.prosody.lead_in_s = seg.prosody.pause_after_s = 0.4
    lec.save_script(proj, m)
    eng = Stub()
    quiet = dict(check_audio=False, log=lambda *_: None, width=640, height=360,
                 fps=30)

    print("\nwithout an insert (the control)")
    a = lec.render(deck, proj, m, eng, **quiet)
    base = duration_of(a["master"])
    name = Path(a["master"]).stem
    made = len(eng.calls)
    check(a["report"]["verdict"].startswith("PASS"), "the lecture renders",
          a["report"]["verdict"])

    print("\nan animated clip after slide 2")
    clip = make_clip(WORK / "confounding.mp4")
    item = INS.add(proj, clip, 2, title="Confounding, animated")
    check((proj / item.file).is_file() and bool(item.captions),
          "the clip and its captions are copied into the lecture", str(item))
    b = lec.render(deck, proj, lec.load_script(proj), eng, **quiet)
    rep = b["report"]
    got = duration_of(b["master"])
    check(rep["verdict"].startswith("PASS")
          and rep["checks"]["duration_matches_script"],
          "the duration check counts it, and passes", rep["verdict"])
    check(abs(got - (base + CLIP_S)) < 0.25,
          f"the video is longer by the clip ({base:.2f}s -> {got:.2f}s)")
    check(len(eng.calls) == made, "and nothing was recorded again",
          f"{len(eng.calls) - made} calls")
    st = streams(b["master"])
    check(st["video"]["width"] == 640 and st["video"]["height"] == 360
          and "audio" in st,
          "the clip was fitted to the lecture's frame, with a sound track",
          str({k: (v.get('width'), v.get('height')) for k, v in st.items()}))

    seg = json.loads((proj / f"{name}_segments.json").read_text())
    cuts = {c["slide"]: c for c in seg["segments"]}
    span = (seg.get("inserts") or [{}])[0]
    check(abs(span.get("start_s", -1) - cuts[2]["end_s"]) < 0.01
          and abs(cuts[3]["start_s"] - span.get("end_s", -1)) < 0.01,
          "the cut list puts it between slide 2 and slide 3",
          f"{cuts[2]['end_s']} | {span} | {cuts[3]['start_s']}")
    check(abs(seg["total_s"] - got) < 0.5, "and its total is the video's",
          f"{seg['total_s']} vs {got:.2f}")

    srt = (proj / f"{name}.srt").read_text(encoding="utf-8")
    check("An animated caption." in srt, "its own caption is carried in")
    from realme.pipeline.captions import _secs
    stamp = next(l for l in srt.splitlines()
                 if "-->" in l and srt.splitlines()[srt.splitlines().index(l) + 1]
                 == "An animated caption.")
    check(abs(_secs(stamp.split("-->")[0]) - (cuts[2]["end_s"] + 0.5)) < 0.05,
          "at its own moment inside the clip", stamp)
    after = [l for l in srt.splitlines() if "-->" in l]
    last = _secs(after[-1].split("-->")[0])
    check(last > cuts[3]["start_s"] - 0.01,
          "and slide 3's captions come after it", f"{last} vs {cuts[3]['start_s']}")
    chapters = (proj / f"{name}_chapters.txt").read_text(encoding="utf-8")
    check("Confounding, animated" in chapters
          and chapters.index("Confounding, animated") < chapters.index("Slide 3"),
          "it is a chapter of its own, before slide 3", chapters)

    print("\nsplitting keeps it with the part it follows")
    cuts_l, meta = SPL.load_cuts(proj / f"{name}_segments.json")
    plan = SPL.plan(cuts_l, [3], stem="x", total_s=meta["total_s"])
    p1, p2 = plan.parts
    check(abs(p1.end_s - cuts[3]["start_s"]) < 0.01 and p2.start_s == p1.end_s,
          "part 1 runs to where slide 3 starts, so the clip is in it",
          f"{p1.end_s} vs {cuts[3]['start_s']}")
    no_total = SPL.plan(cuts_l, [], stem="x")
    check(abs(SPL.plan(cuts_l, [], stem="x", total_s=meta["total_s"]).parts[0].end_s
              - meta["total_s"]) < 0.01 and no_total.parts[0].end_s > 0,
          "and one part covers the whole video")

    print("\ntaking it out and putting it back re-joins, it does not render")
    INS.remove(proj, item.id)
    c = lec.reassemble(proj, log=lambda *_: None)
    check(abs(duration_of(c["master"]) - base) < 0.25
          and c["report"]["verdict"].startswith("PASS"),
          "removed: the video is back to the lecture alone",
          f"{duration_of(c['master']):.2f}s vs {base:.2f}s")
    INS.add(proj, clip, 0, title="Opening animation")
    d = lec.reassemble(proj, log=lambda *_: None)
    check(abs(duration_of(d["master"]) - (base + CLIP_S)) < 0.25
          and d["report"]["checks"]["duration_matches_script"],
          "before slide 1: it opens the lecture", d["report"]["verdict"])
    check(len(eng.calls) == made, "and neither recorded anything",
          f"{len(eng.calls) - made} calls")
    seg = json.loads((proj / f"{name}_segments.json").read_text())
    check(seg["inserts"][0]["start_s"] == 0.0
          and abs(seg["segments"][0]["start_s"] - CLIP_S) < 0.1,
          "and slide 1 now starts after it")
    said = []
    INS.add(proj, clip, 9, title="Too far")
    lec.reassemble(proj, log=said.append)
    check(any("plays at the end" in s for s in said),
          "a slide number past the end plays at the end, and says so",
          " | ".join(said)[:120])
    for i in INS.load(proj):
        INS.remove(proj, i.id)

    print("\nthe Studio and the command line")
    from fastapi.testclient import TestClient
    import realme.app.server as S
    tc = TestClient(S.app, base_url="http://127.0.0.1")
    r = tc.get("/api/lecture/lec_ins/inserts")
    check(r.status_code == 200 and len(r.json()["slides"]) == 3
          and r.json()["last_render"] == f"{name}.mp4",
          "the Tools tab is told the slides and the last render", r.text[:120])
    with open(clip, "rb") as v, open(clip.with_suffix(".srt"), "rb") as cap:
        r = tc.post("/api/lecture/lec_ins/inserts",
                    files=[("files", ("anim.mp4", v.read(), "video/mp4")),
                           ("files", ("anim.srt", cap.read(), "text/plain"))],
                    data={"after_slide": "1", "title": "From the page"})
    j = r.json()
    check(r.status_code == 200 and len(j["inserts"]) == 1
          and bool(j["inserts"][0]["captions"]),
          "a video and its .srt are added in one go", r.text[:150])
    r = tc.post("/api/lecture/lec_ins/inserts",
                files=[("files", ("notes.txt", b"x", "text/plain"))],
                data={"after_slide": "1"})
    check(r.status_code == 400, "something that is not a video is refused")
    r = tc.post("/api/lecture/lec_ins/inserts/apply")
    job = r.json().get("job_id")
    import time
    for _ in range(300):
        st = tc.get(f"/api/jobs/{job}").json()
        if st["status"] in ("done", "error"):
            break
        time.sleep(0.2)
    check(st["status"] == "done"
          and st["result"]["report"]["verdict"].startswith("PASS"),
          "Put them into the last render now: done, and verified",
          str(st.get("error") or st.get("result", {}).get("report", {})
              .get("verdict")))
    iid = j["inserts"][0]["id"]
    r = tc.delete(f"/api/lecture/lec_ins/inserts/{iid}")
    check(r.status_code == 200 and r.json()["inserts"] == [],
          "and one is taken out again")

    import realme.cli as C
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        code = C.main(["insert", str(proj), str(clip), "--after", "2",
                       "--title", "From the CLI", "--json"])
    jj = json.loads(out.getvalue())
    check(code == 0 and jj["report"]["inserts"][0]["title"] == "From the CLI",
          "`realme insert LECTURE VIDEO --after 2` adds one", str(jj)[:120])

    print("\nthe Animate tab is gone, and its code archived rather than lost")
    ui = (HERE / "realme" / "app" / "ui.html").read_text(encoding="utf-8")
    check('id="p-animate"' not in ui and "/api/animate" not in ui,
          "the tab and its calls are out of the page")
    check(not any(p.startswith("/api/animate") for p in S.app.openapi()["paths"]),
          "and its routes out of the server")
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        try:
            C._main(["animate", "stills", "x.py"])
            gone = False
        except SystemExit as e:
            gone = e.code == 2
    check(gone, "`realme animate` is no longer a command")
    from realme.setup_check import archive_retired, RETIRED_CODE
    root = WORK / "tree"
    for rel in RETIRED_CODE["animate_2026-10"]:
        f = root / rel
        if Path(rel).suffix:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("# old", encoding="utf-8")
        else:
            f.mkdir(parents=True, exist_ok=True)
            (f / "scenes.py").write_text("# old", encoding="utf-8")
    moved = archive_retired(root)
    check(len(moved) == 4 and not (root / "03_App/realme/pipeline/animate.py").exists()
          and (root / "_Archive/animate_2026-10/03_App/realme/pipeline/animate.py")
          .read_text() == "# old"
          and (root / "_Archive/animate_2026-10/03_App/examples/guide_v3/scenes.py")
          .is_file(),
          "`realme setup` moves the old engine into _Archive, intact",
          str(moved))
    check(archive_retired(root) == [], "and a second run moves nothing")

    regressions(deck, proj, clip, eng, quiet, name, base)
    parallel_videos()
    return summary()


def parallel_videos():
    """Slide videos are made several at a time, after all the audio."""
    from PIL import Image
    from realme.core.ledger import RenderLedger
    from realme.core.media import MediaError, duration_of
    from realme.pipeline import lecture as lec
    print("\nslide videos are made several at a time, in slide order")
    d = WORK / "par"
    d.mkdir(parents=True, exist_ok=True)
    jobs_src = []
    for i in range(5):
        png = d / f"s{i}.png"
        Image.new("RGB", (320, 180), (40 * i, 90, 160)).save(png)
        wav = d / f"a{i}.wav"
        tone(wav, 0.6 + 0.2 * i)
        jobs_src.append((png, wav))

    def jobs(tag):
        return [{"at": i, "vk": f"{tag}{i}", "slide": i + 1,
                 "args": (png, wav, d / f"{tag}{i}.mp4"),
                 "kwargs": dict(width=320, height=180, fps=10,
                                layout="slide_only", cues=[])}
                for i, (png, wav) in enumerate(jobs_src)]
    results = {}
    for n in (1, 3):
        led = RenderLedger(d / f"ledger{n}.json")
        vids = [None] * 5
        said = []
        lec._make_slide_videos(jobs(f"w{n}_"), vids, led, log=said.append,
                               workers=n)
        results[n] = [round(duration_of(v), 1) for v in vids]
        check(all(v is not None and Path(v).is_file() for v in vids)
              and all(Path(v).name == f"w{n}_{i}.mp4" for i, v in enumerate(vids)),
              f"{n} at a time: every video lands at its own slide",
              str([Path(v).name if v else None for v in vids]))
        check(all(led.get(f"w{n}_{i}") for i in range(5)),
              f"{n} at a time: each one is in the ledger, so the next render reuses it")
    check(results[1] == results[3],
          "one at a time and several at a time make the same videos",
          f"{results[1]} vs {results[3]}")
    bad = jobs("bad_")
    bad[2]["args"] = (jobs_src[2][0], d / "missing.wav", d / "bad_2.mp4")
    vids = [None] * 5
    led = RenderLedger(d / "ledger_bad.json")
    try:
        lec._make_slide_videos(bad, vids, led, log=lambda *_: None, workers=3)
        raised = False
    except Exception:
        raised = True
    check(raised and not led.get("bad_2"),
          "a slide that cannot be made stops the render, and is not recorded as made")
    old = os.environ.get("REALME_VIDEO_WORKERS")
    os.environ["REALME_VIDEO_WORKERS"] = "1"
    try:
        check(lec.video_workers() == 1, "REALME_VIDEO_WORKERS=1 makes them one at a time")
    finally:
        if old is None:
            os.environ.pop("REALME_VIDEO_WORKERS", None)
        else:
            os.environ["REALME_VIDEO_WORKERS"] = old
    check(1 <= lec.video_workers() <= 4, "by default at most four at once",
          str(lec.video_workers()))


def regressions(deck, proj, clip, eng, quiet, name, base):
    """Edge cases, each pinned by the smallest check that shows it."""
    import shutil
    from realme.core.ledger import RenderLedger
    from realme.core.media import (MediaError, concat_line, duration_of,
                                   write_concat_list)
    from realme.pipeline import (compose, inserts as INS, lecture as lec,
                                 split as SPL)
    from realme.pipeline.captions import _ts

    print("\nan apostrophe in a folder name does not break a join")
    odd = WORK / "O'Brien's lectures"
    odd.mkdir(parents=True, exist_ok=True)
    # Built from the resolved path, so it holds on Windows too ("C:/x/...").
    _want = Path("/x/O'Brien/a.mp4").resolve().as_posix().replace("'", "'\\''")
    check(concat_line("/x/O'Brien/a.mp4").strip() == f"file '{_want}'"
          and "O'\\''Brien" in concat_line("/x/O'Brien/a.mp4"),
          "a quote is written as '\\'' in a concat list",
          concat_line("/x/O'Brien/a.mp4"))
    a1, a2 = odd / "a.mp4", odd / "b.mp4"
    shutil.copy(clip, a1); shutil.copy(clip, a2)
    try:
        joined = compose.concat([a1, a2], odd / "joined.mp4")
        ok = abs(duration_of(joined) - 2 * CLIP_S) < 0.3
    except MediaError as e:
        ok = False
        print("   ", str(e)[:200])
    check(ok, "two videos in a folder called O'Brien's join")
    hand_written = [p.name for p in (HERE / "realme" / "pipeline").glob("*.py")
                    if "f\"file '" in p.read_text(encoding="utf-8")
                    or "f'file '" in p.read_text(encoding="utf-8")]
    check(not hand_written,
          "every concat list in the pipeline is written by the one helper",
          str(hand_written))

    print("\ncaption times never print 1000 milliseconds")
    check(_ts(59.9996) == "00:01:00,000" and _ts(1.9999, ".") == "00:00:02.000"
          and SPL._stamp(3599.9997) == "01:00:00,000"
          and _ts(12.3456) == "00:00:12,346",
          "a time that rounds up carries into the seconds",
          f"{_ts(59.9996)} {_ts(1.9999, '.')} {SPL._stamp(3599.9997)}")

    print("\nthe ledger is written whole or not at all")
    led_path = WORK / "ledger_t" / "ledger.json"
    led = RenderLedger(led_path)
    led.put("k:1", led_path, 1.0)
    import realme.core.ledger as LM
    real_replace = LM.os.replace

    def boom(*_a, **_k):
        raise OSError("disk full")
    LM.os.replace = boom
    try:
        led.put("k:2", led_path, 2.0)
        failed = False
    except OSError:
        failed = True
    finally:
        LM.os.replace = real_replace
    kept = json.loads(led_path.read_text(encoding="utf-8"))
    check(failed and set(kept) == {"k:1"}
          and not list(led_path.parent.glob("*.tmp")),
          "a failed write leaves the previous ledger intact and no temp file",
          str(kept))

    print("\nan insert before slide 1 stays in the first part when splitting")
    opening = INS.add(proj, clip, 0, title="Opening")
    lec.reassemble(proj, log=lambda *_: None)
    cuts_l, meta = SPL.load_cuts(proj / f"{name}_segments.json")
    parts = SPL.plan(cuts_l, [2], stem="x", total_s=meta["total_s"]).parts
    check(parts[0].start_s == 0.0
          and abs(parts[0].end_s - next(c for c in cuts_l
                                        if c["slide"] == 2)["start_s"]) < 0.01,
          "part 1 starts at 0s, where the opening video plays",
          f"{parts[0].start_s} (slide 1 at {cuts_l[0]['start_s']})")
    INS.remove(proj, opening.id)

    print("\na missing inserted video stops the render before it starts")
    gone = INS.add(proj, clip, 1, title="Will go missing")
    (proj / gone.file).rename(proj / (gone.file + ".away"))

    class Watch(Stub):
        started = False

        def preflight(self):
            Watch.started = True
    try:
        lec.render(deck, proj, lec.load_script(proj), Watch(), **quiet)
        msg = ""
    except MediaError as e:
        msg = str(e)
    check("missing" in msg and Path(gone.file).name in msg
          and not Watch.started,
          "it names the file, before the engine is even started", msg[:150])
    (proj / (gone.file + ".away")).rename(proj / gone.file)
    INS.remove(proj, gone.id)

    print("\nthe slide video is made again when its cues or frame rate change")
    m = lec.load_script(proj)
    m.segments[0].cues = ["confounding"]
    lec.save_script(proj, m)
    lec.render(deck, proj, lec.load_script(proj), eng, **quiet)
    rec = lambda: json.loads((proj / "_work" / "assembly.json")
                             .read_text(encoding="utf-8"))
    first = rec()["seg_videos"]
    m = lec.load_script(proj)
    m.segments[0].cues = ["part"]
    lec.save_script(proj, m)
    lec.render(deck, proj, lec.load_script(proj), eng, **quiet)
    second = rec()["seg_videos"]
    cue_report = json.loads((proj / f"{name}_cues.json").read_text())
    check(first[0] != second[0] and first[1:] == second[1:],
          "a different cue word is a different video for that slide only",
          f"{Path(first[0]).name} vs {Path(second[0]).name}; {str(cue_report)[:80]}")
    lec.render(deck, proj, lec.load_script(proj), eng, **{**quiet, "fps": 25})
    third = rec()["seg_videos"]
    rate = streams(Path(third[1]))["video"]["r_frame_rate"]
    check(not set(third) & set(second) and rate == "25/1",
          "a new frame rate re-makes every slide video", rate)
    lec.render(deck, proj, lec.load_script(proj), eng, **quiet)

    print("\napplying inserts after a draft re-joins the finished render")

    class Draft(Stub):
        name = "draft"
        is_voice_clone = False

        def voice_fingerprint(self):
            return "draft:v1"

        def synthesize(self, text, out_wav, voice=None, pace=1.0, **kw):
            tone(Path(out_wav), max(0.4, len(text.split()) * 0.1))
            return Path(out_wav)
    real = duration_of(proj / f"{name}.mp4")
    dr = lec.render(deck, proj, lec.load_script(proj), Draft(),
                    suffix="_draft", **quiet)
    drafted = duration_of(dr["master"])
    INS.add(proj, clip, 1, title="After the draft")
    out = lec.reassemble(proj, log=lambda *_: None)
    check(Path(out["master"]).name == f"{name}.mp4"
          and abs(duration_of(out["master"]) - (real + CLIP_S)) < 0.25
          and out["report"]["checks"]["duration_matches_script"],
          "the finished render gets the insert, at its own timings",
          f"{Path(out['master']).name} {duration_of(out['master']):.2f}s, "
          f"real {real:.2f}s, draft {drafted:.2f}s")
    od = lec.reassemble(proj, log=lambda *_: None, name=f"{name}_draft")
    check(Path(od["master"]).name == f"{name}_draft.mp4"
          and abs(duration_of(od["master"]) - (drafted + CLIP_S)) < 0.25
          and od["report"]["checks"]["duration_matches_script"],
          "and the draft is still there to re-join by name",
          f"{duration_of(od['master']):.2f}s vs {drafted + CLIP_S:.2f}s")
    from fastapi.testclient import TestClient
    import realme.app.server as S
    r = TestClient(S.app, base_url="http://127.0.0.1").get(
        "/api/lecture/lec_ins/inserts")
    check(r.status_code == 200 and r.json()["last_render"] == f"{name}.mp4",
          "the Tools tab names the finished render, not the draft", r.text[:120])

    print("\nre-joining after the script was edited says to render again")
    m = lec.load_script(proj)
    keep = m.model_copy(deep=True)
    m.segments[1].spoken_text += " A sentence added after the render."
    m.segments = m.segments[:2]
    lec.save_script(proj, m)
    try:
        lec.reassemble(proj, log=lambda *_: None)
        msg = ""
    except MediaError as e:
        msg = str(e)
    except Exception as e:                    # anything else is a crash
        msg = f"CRASH {type(e).__name__}: {e}"
    check("Render the lecture again" in msg and "removed" in msg
          and "edited" in msg,
          "a stale render is refused in words, not a crash", msg[:160])
    lec.save_script(proj, keep)
    ok = lec.reassemble(proj, log=lambda *_: None)
    check(ok["report"]["checks"]["duration_matches_script"],
          "and the script put back re-joins again (a save alone is no edit)",
          ok["report"]["verdict"])
    for i in INS.load(proj):
        INS.remove(proj, i.id)


if __name__ == "__main__":
    raise SystemExit(main())
