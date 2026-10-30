#!/usr/bin/env python3
"""
`realme gemini-parallel`: does it find how many requests at once a Google
project takes, without calling Google?

A stand-in plays a project that serves two requests at once and makes a
third wait (and says so, as Google's 429 does). The measurement should find
2, stop when today's allowance runs out, and save an answer that the Gemini
adapter then uses.
"""
from __future__ import annotations
import json, math, os, struct, sys, tempfile, threading, time, wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

_ISO = tempfile.mkdtemp(prefix="t_gemparallel_")
os.environ["REALME_HOME"] = str(Path(_ISO) / "home")
os.environ["HOME"] = _ISO
os.environ["USERPROFILE"] = _ISO

from t_util import check, summary          # noqa: E402


def tone(path: Path, seconds: float, rate=24000):
    n = int(rate * seconds)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", int(4000 * math.sin(
            2 * math.pi * 150 * i / rate))) for i in range(n)))


class Project:
    """Serves `room` requests at once; the next waits and is told to."""
    def __init__(self, room=2, quota_after=None):
        self.room, self.quota_after = room, quota_after
        self.mu = threading.Lock()
        self.busy = self.calls = self.rate_waits = 0
        self.spend_usd = 0.0

    def synthesize(self, text, out, pace=1.0):
        from realme.adapters.base import QuotaExceeded
        with self.mu:
            self.calls += 1
            if self.quota_after and self.calls > self.quota_after:
                raise QuotaExceeded("limit: 6 requests per day", period="day")
            crowded = self.busy >= self.room
            if crowded:
                self.rate_waits += 1
            self.busy += 1
        try:
            time.sleep(0.25 + (0.5 if crowded else 0.0))
            tone(Path(out), 1.0)
            with self.mu:
                self.spend_usd += 0.001
        finally:
            with self.mu:
                self.busy -= 1
        return Path(out)


def main() -> int:
    from realme import gemparallel as GP
    out = Path(_ISO) / "out"

    print("finding the project's limit")
    est = GP.estimate([1, 2, 3, 4])
    check(est["requests"] == 4 + 4 + 6 + 8 and est["usd"] > 0,
          "the cost and the requests are said before anything is sent", est)
    p = Project(room=2)
    res = GP.measure(p, [1, 2, 3, 4], outdir=out, log=lambda *_: None)
    rows = {r["level"]: r for r in res["levels"]}
    check(set(rows) == {1, 2, 3, 4} and all(r["ok"] == r["requests"]
                                             for r in rows.values()),
          "every level is measured", list(rows))
    check(rows[3]["slow_down_asked"] > 0 and rows[2]["slow_down_asked"] == 0,
          "being asked to slow down is noticed",
          {k: r["slow_down_asked"] for k, r in rows.items()})
    v = GP.verdict(res["levels"])
    check(v["safe"] == 2, "the project's limit is found: 2 at a time", v)
    check(len({f.name for f in out.glob("level*_*.wav")}) == est["requests"],
          "the audio is kept to listen to")

    print("\ntoday's allowance")
    q = Project(room=4, quota_after=6)
    res = GP.measure(q, [1, 2, 3, 4], outdir=Path(_ISO) / "q", log=lambda *_: None)
    check(res["stopped"] and q.calls <= 6 + 4,
          "it stops asking once Google says the day's allowance is gone",
          (res["stopped"], q.calls))
    v = GP.verdict(res["levels"])
    check(v["safe"] >= 1, "and still saves what it learned", v)

    print("\na failed measurement is not saved")
    row = lambda lv, ok, req, pm, w=0: {"level": lv, "ok": ok, "requests": req,
                                        "per_minute": pm, "slow_down_asked": w}
    v = GP.verdict([row(1, 3, 4, 10), row(2, 4, 4, 30)])
    check(v["safe"] == 1 and not v["measured"],
          "one at a time failing measures nothing", v)
    v = GP.verdict([row(1, 4, 4, 10), row(2, 3, 4, 30), row(3, 6, 6, 40)])
    check(v["safe"] == 1 and v["measured"],
          "a level that failed stops the comparison there", v)
    v = GP.verdict([row(1, 4, 4, 10), row(2, 4, 4, 20), row(3, 6, 6, 30)])
    check(v["safe"] == 3 and v["measured"], "clean levels are compared in order", v)
    import inspect
    src = inspect.getsource(GP.main_cli)
    check('res["stopped"] or not v.get("measured")' in src,
          "and a run the day's allowance cut short is not saved")

    print("\nthe answer is a ceiling, not a setting")
    from realme.adapters.tts_gemini import GEMINI_PARALLEL_FILE, GeminiTTS
    from realme.core.env import data_home
    f = data_home() / GEMINI_PARALLEL_FILE
    g = GeminiTTS(voice="voice_x", api_key="k")
    check(g.concurrent_requests == 2, "unmeasured, two at a time")
    data_home().mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"safe": 4}), encoding="utf-8")
    check(g.concurrent_requests == 2,
          "a file from the first probe (4) no longer raises it")
    GP.save(measured=4, when="t", model="m")
    check(g.concurrent_requests == 2 and json.loads(f.read_text())
          .get("measured") == 4, "measuring 4 keeps two at a time")
    GP.save(measured=1)
    check(g.concurrent_requests == 1, "measuring 1 lowers it to one")
    GP.save(measured=4)
    logs = []
    check(GP.choose(3, log=logs.append) == 0 and g.concurrent_requests == 3,
          "--use 3 raises it", logs)
    check(GP.choose(6, log=logs.append) == 1 and g.concurrent_requests == 3,
          "--use above the measured limit is refused", logs[-1:])
    GP.save(measured=2)
    check(g.concurrent_requests == 2 and json.loads(f.read_text())["use"] == 2,
          "a lower new measurement lowers the choice too")
    f.unlink()
    check(GP.choose(5, log=logs.append) == 0 and g.concurrent_requests == 5,
          "unmeasured, the user's choice stands")
    f.write_text("[1, 2]", encoding="utf-8")
    check(g.concurrent_requests == 2, "a damaged file means two")
    f.write_text(json.dumps({"use": 3}), encoding="utf-8")
    from realme.pipeline import prerecord as PR
    os.environ.pop(PR.ENV, None)
    check(PR.workers_for(g, 50) == 3, "and the recorder sends what was chosen")
    os.environ[PR.ENV] = "1"
    check(PR.workers_for(g, 50) == 1, "unless told one at a time")
    os.environ.pop(PR.ENV, None)
    return summary()


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        import shutil
        shutil.rmtree(_ISO, ignore_errors=True)
