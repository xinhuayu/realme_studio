#!/usr/bin/env python3
"""
A lecture kept as its video and text, and edited slide by slide.

Renders a three-slide lecture with a stand-in voice, then checks what the
person would notice if it went wrong:

  * after compacting, the working audio and slide videos are gone and the
    video, captions, cut list and script remain;
  * an unchanged lecture is not recorded again;
  * editing one slide records that slide only -- every sentence on it, and
    nothing from the others -- and the result passes the same checks as a
    full render, with the other slides' sound and pictures kept;
  * a video inserted between slides survives all of it;
  * the .zip holds what is needed to bring the lecture back, and a lecture
    brought back from it can be edited the same way;
  * a .zip that tries to write outside its folder, or is not RealMe's, is
    refused.
"""
from __future__ import annotations
import json, math, os, struct, subprocess, sys, tempfile, wave, zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

_ISO = tempfile.mkdtemp(prefix="t_lean_")
os.environ["REALME_HOME"] = str(Path(_ISO) / "home")
os.environ["REALME_TOOLS"] = str(Path(_ISO) / "tools")
os.environ["HOME"] = _ISO
os.environ["USERPROFILE"] = _ISO
os.environ["REALME_NO_WARMUP"] = "1"

from t_util import check, summary          # noqa: E402

WORK = Path(_ISO) / "work"
WORK.mkdir(parents=True, exist_ok=True)


def tone(path: Path, seconds: float, hz: float, rate=24000):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(rate * seconds)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", int(6000 * math.sin(
            2 * math.pi * hz * i / rate))) for i in range(n)))


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
        tone(Path(out_wav), max(0.6, len(text.split()) * 0.25), 180)
        return Path(out_wav)


class Other(Stub):
    name = "another-voice"


def media_in(work: Path) -> list[str]:
    return sorted(f.name for f in work.rglob("*")
                  if f.is_file() and f.suffix in (".wav", ".mp4"))


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
    from realme.core.media import MediaError, duration_of, require
    from realme.pipeline import inserts as INS, lean, lecture as lec

    deck = WORK / "deck.pdf"
    doc = pymupdf.open()
    for i in range(3):
        pg = doc.new_page(width=960, height=540)
        pg.insert_text((72, 100), f"Slide {i + 1}: effect modification {i + 1}.",
                       fontsize=24)
    doc.save(str(deck))
    proj = Path(os.environ["REALME_HOME"]) / "projects" / "lec_lean"
    m, _ = lec.write_script(deck, proj, PlaceholderScriptWriter(),
                            log=lambda *_: None)
    for seg in m.segments:
        seg.prosody.lead_in_s = seg.prosody.pause_after_s = 0.4
    lec.save_script(proj, m)
    shutil.copy(deck, proj / "deck.pdf")
    eng = Stub()
    quiet = dict(check_audio=False, log=lambda *_: None, width=640, height=360,
                 fps=30)
    a = lec.render(proj / "deck.pdf", proj, m, eng, **quiet)
    name = Path(a["master"]).stem
    check(a["report"]["verdict"].startswith("PASS"), "the lecture renders",
          a["report"]["verdict"])

    clip = WORK / "clip.mp4"
    subprocess.run([require("ffmpeg"), "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", "testsrc=size=320x240:rate=24:duration=2",
                    "-pix_fmt", "yuv420p", str(clip)], check=True)
    INS.add(proj, clip, 1, title="An animation")
    lec.reassemble(proj, log=lambda *_: None)
    before = duration_of(proj / f"{name}.mp4")

    print("\ncompacting")
    work = proj / "_work"
    big = lean.media_bytes(proj)
    note = lean.compact(proj, log=lambda *_: None)
    check(note["freed_bytes"] == big and big > 0 and not media_in(work),
          "the working audio and slide videos are gone", media_in(work)[:4])
    kept = [f"{name}.mp4", f"{name}.srt", f"{name}_segments.json",
            "inserts.json"]
    check(all((proj / k).is_file() for k in kept)
          and (work / "manifest.json").is_file()
          and (work / lec.ASSEMBLY_DIR / f"{name}.json").is_file(),
          "the video, captions, cut list, script and record remain")
    rec = json.loads((work / lec.ASSEMBLY_DIR / f"{name}.json").read_text("utf-8"))
    cues = [c for s in rec["manifest"]["segments"] for c in s["spoken_cues"]]
    check(len(cues) >= 3 and all(len(c) == 3 and c[2] for c in cues),
          "each utterance's words and timing are kept as text", cues[:2])
    check(lean.is_compact(proj, name), "and the lecture knows it is compact")

    print("\nnothing changed")
    eng.calls.clear()
    r = lean.rerecord(proj, proj / "deck.pdf", eng, name=name, check_audio=False,
                      log=lambda *_: None)
    check(r["rerecorded"] == [] and not eng.calls,
          "an unchanged lecture is not recorded again", eng.calls)

    print("\none slide edited")
    m = lec.load_script(proj)
    m.segments[1].spoken_text = ("This slide was rewritten. It says something "
                                 "new about effect modification.")
    lec.save_script(proj, m)
    old_cuts = json.loads((proj / f"{name}_segments.json").read_text("utf-8"))
    eng.calls.clear()
    r = lean.rerecord(proj, proj / "deck.pdf", eng, name=name, check_audio=False,
                      log=lambda *_: None)
    said = " ".join(eng.calls)
    check(r["rerecorded"] == [2], "only the edited slide is re-recorded",
          r["rerecorded"])
    check("rewritten" in said and "something new" in said
          and all("Slide 1" not in c and "Slide 3" not in c for c in eng.calls),
          "every sentence of it, and nothing from the other slides", eng.calls)
    check(r["report"]["verdict"].startswith("PASS"),
          "the joined video passes the duration and caption checks",
          r["report"]["verdict"])
    new_cuts = json.loads((proj / f"{name}_segments.json").read_text("utf-8"))
    d1o = old_cuts["segments"][0]["end_s"] - old_cuts["segments"][0]["start_s"]
    d1n = new_cuts["segments"][0]["end_s"] - new_cuts["segments"][0]["start_s"]
    check(abs(d1o - d1n) < 0.01, "a kept slide keeps its length", (d1o, d1n))
    check(len(new_cuts.get("inserts", [])) == 1,
          "the inserted video is still between the slides")
    total = duration_of(proj / f"{name}.mp4")
    check(abs(total - new_cuts["total_s"]) < 0.35,
          "and the video is as long as its cut list says",
          (total, new_cuts["total_s"], before))
    srt = (proj / f"{name}.srt").read_text("utf-8")
    check("rewritten" in srt, "the captions carry the new words")
    check(not media_in(work), "and the working media are let go again",
          media_in(work)[:4])

    print("\nanother voice")
    m.segments[2].spoken_text = "Slide three, said again."
    lec.save_script(proj, m)
    try:
        lean.rerecord(proj, proj / "deck.pdf", Other(), name=name,
                      check_audio=False, log=lambda *_: None)
        mixed = True
    except MediaError as e:
        mixed = "mix two voices" not in str(e)
    check(not mixed, "re-recording one slide in a different voice is refused")

    print("\nthe .zip")
    z = lean.bundle(proj, WORK / "lecture.zip", deck=proj / "deck.pdf")
    with zipfile.ZipFile(z) as zz:
        names = set(zz.namelist())
    check({f"{name}.mp4", f"{name}.srt", f"{name}_segments.json",
           "narration.txt", "record/script.json", f"record/{name}.json",
           "deck/deck.pdf", "inserts/inserts.json", "inserts/clip.mp4",
           "README.txt"} <= names, "it holds the video, text, deck and insert",
          sorted(names))
    check(not any(n.endswith(".wav") for n in names),
          "and no working audio")
    back = proj.parent / "lec_back"
    info = lean.unbundle(z, back, log=lambda *_: None)
    check(info["renders"] == [name] and (back / "deck.pdf").is_file(),
          "a lecture comes back from its .zip", info)
    eng.calls.clear()
    r = lean.rerecord(back, back / "deck.pdf", eng, name=name,
                      check_audio=False, log=lambda *_: None)
    check(r["rerecorded"] == [3] and r["report"]["verdict"].startswith("PASS"),
          "and is edited slide by slide like any other",
          (r["rerecorded"], r["report"]["verdict"]))

    print("\nbad .zip files")
    evil = WORK / "evil.zip"
    with zipfile.ZipFile(evil, "w") as zz:
        with zipfile.ZipFile(z) as src:
            for n in src.namelist():
                zz.writestr(n, src.read(n))
        zz.writestr("../escaped.txt", "x")
        zz.writestr("record/../../escaped2.txt", "x")
        zz.writestr("inserts/inserts.json", json.dumps({"version": 1, "inserts": [
            {"id": "x", "after_slide": 1, "file": "../../../secret.mp4"}]}))
    lean.unbundle(evil, proj.parent / "lec_evil", log=lambda *_: None)
    ins = INS.load(proj.parent / "lec_evil")
    check(all(i.file.startswith("inserts/") and ".." not in i.file for i in ins),
          "an insert list cannot point outside the lecture's own folder",
          [i.file for i in ins])
    check(not (proj.parent / "escaped.txt").exists()
          and not list(Path(_ISO).rglob("escaped*")),
          "a member that climbs out of the folder is not written")
    junk = WORK / "junk.zip"
    with zipfile.ZipFile(junk, "w") as zz:
        zz.writestr("hello.txt", "hi")
    try:
        lean.unbundle(junk, proj.parent / "lec_junk", log=lambda *_: None)
        refused = False
    except MediaError:
        refused = not (proj.parent / "lec_junk").exists()
    check(refused, "a .zip that is not RealMe's is refused and leaves nothing")

    studio_routes(deck)
    return summary()


def studio_routes(deck: Path) -> None:
    """The same, through the Studio: what its buttons call."""
    import io, time
    from fastapi.testclient import TestClient
    from realme.adapters.script_writer import PlaceholderScriptWriter
    from realme.app import server as S
    from realme.pipeline import lecture as lec
    print("\nthrough the Studio, kept lean")
    S.LEAN = True
    eng = Stub()
    S.engine_for = lambda spec, **kw: eng
    S.profile.data["video"].update({"width": 640, "height": 360, "fps": 30})
    proj = S.PROJECTS / "studio_lean"
    import shutil
    proj.mkdir(parents=True, exist_ok=True)
    shutil.copy(deck, proj / "deck.pdf")
    m, _ = lec.write_script(proj / "deck.pdf", proj, PlaceholderScriptWriter(),
                            log=lambda *_: None)
    lec.save_script(proj, m)
    c = TestClient(S.app, base_url="http://127.0.0.1")

    def run(body):
        j = c.post("/api/lecture/studio_lean/render", json=body).json()
        for _ in range(600):
            st = c.get(f"/api/jobs/{j['job_id']}").json()
            if st["status"] in ("done", "error"):
                return st
            time.sleep(0.2)
        return st

    st = run({"tts": "stub"})
    check(st["status"] == "done", "a render through the Studio finishes",
          st.get("error"))
    store = c.get("/api/lecture/studio_lean/storage").json()
    check(store["working_mb"] == 0 and all(r["compact"] for r in store["renders"].values()),
          "and is kept as its video and text files", store)
    m = lec.load_script(proj)
    m.segments[0].spoken_text = "A new opening for the first slide."
    lec.save_script(proj, m)
    eng.calls.clear()
    st = run({"tts": "stub"})
    check(st["status"] == "done" and st["result"].get("rerecorded") == [1]
          and all("first slide" in t or "opening" in t for t in eng.calls),
          "Render after an edit records the edited slide only",
          (st.get("error"), st.get("result", {}).get("rerecorded"), eng.calls))
    r = c.get("/api/lecture/studio_lean/bundle")
    check(r.status_code == 200 and r.content[:2] == b"PK",
          "Download everything gives a .zip", r.status_code)
    zbytes = r.content
    r = c.delete("/api/projects/studio_lean")
    check(r.status_code == 200 and not proj.exists(),
          "Remove this lecture removes it")
    r = c.post("/api/lecture/bundle/import",
               files={"file": ("studio_lean_lecture.zip", zbytes, "application/zip")})
    j = r.json()
    check(r.status_code == 200 and j.get("project_id") == "studio_lean"
          and "studio_lean" in c.get("/api/projects").json()["projects"],
          "and the .zip brings it back", j)
    r = c.post("/api/lecture/bundle/import",
               files={"file": ("x.zip", b"#EXTM3U\n/etc/passwd\n", "application/zip")})
    check(r.status_code == 400, "a text file posing as a .zip is refused")
    m = lec.load_script(S.PROJECTS / "studio_lean")
    m.segments[2].spoken_text = "The last slide, said differently."
    lec.save_script(S.PROJECTS / "studio_lean", m)
    st = run({"tts": "stub"})
    check(st["status"] == "done" and st["result"].get("rerecorded") == [3],
          "a lecture brought back is edited slide by slide in the Studio too",
          (st.get("error"), st.get("result", {}).get("rerecorded")))


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        import shutil
        shutil.rmtree(_ISO, ignore_errors=True)
