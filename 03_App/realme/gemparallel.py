"""
`realme gemini-parallel` -- how many requests at once your Google project takes.

A Gemini recording is mostly waiting for Google, so a lecture records faster
with several sentences in flight (`pipeline/prerecord.py`). How many is a
property of the Google PROJECT, not of RealMe or this computer: Google sets a
per-minute allowance per project and tier, does not publish the figure for
its voice models, and shows it only in AI Studio. So this measures it.

It sends a few short sentences at 1, 2, 3 and 4 at a time (by default), in
your own cloned voice, and for each level reports how long the batch took,
sentences per minute, and whether Google asked to slow down. The highest
level that was clearly faster than the one below it, without Google asking
to slow down, is saved as this project's ceiling.

Lectures still keep 2 at a time by default. The probe is a short burst; a
lecture at the measured pace would keep it up for many minutes, which the
probe does not test. `realme gemini-parallel --use N` chooses another number,
up to the measured ceiling once there is one.

It costs money and uses part of today's allowance: about 22 short requests,
a few cents. Both are printed before anything is sent, and it asks first.
The audio is kept, to listen to.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

#: Short, varied, ordinary narration. Each request gets a different one so
#: nothing can be served from anywhere but a fresh synthesis.
SENTENCES = [
    "Confounding is a property of the question, not of the data.",
    "A risk ratio compares two groups on the same scale.",
    "Selection bias enters when who is studied depends on the outcome.",
    "Random error shrinks with sample size; bias does not.",
    "An odds ratio approximates a risk ratio when the outcome is rare.",
    "Effect modification is a finding, not a nuisance.",
    "Missing data cost precision, and sometimes validity.",
    "A directed acyclic graph states what we assume, in pictures.",
    "Matching balances the measured covariates and nothing else.",
    "Incidence counts new cases; prevalence counts all of them.",
    "A confidence interval is a statement about the procedure.",
    "Regression to the mean looks like an effect and is not one.",
]

#: How much faster a level must be than the last good one to count.
GAIN = 1.15


def per_level_default(level: int) -> int:
    return max(4, 2 * level)


def estimate(levels, per_level=None, model: str = "") -> dict:
    from realme.adapters.tts_gemini import PRICE_PER_MTOK, TOKENS_PER_SECOND
    n = sum(per_level or per_level_default(l) for l in levels)
    words = sum(len(s.split()) for s in SENTENCES) / len(SENTENCES)
    audio_s = n * words / 2.5
    usd = audio_s * TOKENS_PER_SECOND / 1e6 * PRICE_PER_MTOK.get(model, 9.0)
    return {"requests": n, "audio_s": round(audio_s), "usd": round(usd, 3)}


def measure(tts, levels, per_level=None, *, outdir: Path, log=print) -> dict:
    """Send the batches; never more after Google says today's allowance is
    gone. Returns the measurements, level by level."""
    from realme.adapters.base import QuotaExceeded
    from realme.core.media import duration_of
    from realme.text.lexicon import Lexicon
    from realme.text.prepare import prepare_for_speech
    outdir.mkdir(parents=True, exist_ok=True)
    lx = Lexicon()
    rows, k, stopped = [], 0, ""
    for level in levels:
        n = per_level or per_level_default(level)
        texts = []
        for _ in range(n):
            base = SENTENCES[k % len(SENTENCES)]
            texts.append((k, prepare_for_speech(
                base if k < len(SENTENCES) else f"{base} Once more.",
                "plain", lx).engine_text))
            k += 1
        waits0 = int(getattr(tts, "rate_waits", 0) or 0)
        spend0 = float(getattr(tts, "spend_usd", 0.0) or 0.0)
        lat, audio, errors = [], [], []

        def one(item):
            i, text = item
            out = outdir / f"level{level}_{i:02d}.wav"
            t0 = time.perf_counter()
            tts.synthesize(text, out)
            return time.perf_counter() - t0, duration_of(out)

        log(f"\n  {level} at a time: {n} requests ...")
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=level) as pool:
            futures = [pool.submit(one, t) for t in texts]
            for f in futures:
                try:
                    a, d = f.result()
                    lat.append(a)
                    audio.append(d)
                except QuotaExceeded as e:
                    stopped = str(e).splitlines()[0]
                    for g in futures:
                        g.cancel()
                    errors.append(stopped)
                except Exception as e:
                    errors.append(f"{e.__class__.__name__}: {e}"[:160])
        wall = time.perf_counter() - t0
        waits = int(getattr(tts, "rate_waits", 0) or 0) - waits0
        row = {"level": level, "requests": n, "ok": len(lat),
               "wall_s": round(wall, 1),
               "per_minute": round(60.0 * len(lat) / wall, 1) if wall else 0.0,
               "mean_latency_s": round(sum(lat) / len(lat), 1) if lat else None,
               "audio_s": round(sum(audio), 1), "slow_down_asked": waits,
               "errors": errors,
               "usd": round(float(getattr(tts, "spend_usd", 0.0) or 0.0)
                            - spend0, 4)}
        rows.append(row)
        log(f"    {row['ok']} of {n} in {wall:.0f}s: {row['per_minute']} a "
            f"minute, {row['mean_latency_s']}s each"
            + (f"; Google asked to slow down {waits} time(s)" if waits else "")
            + (f"; {len(errors)} failed" if errors else ""))
        if stopped:
            log(f"    stopped: {stopped}")
            break
    return {"levels": rows, "stopped": stopped}


def verdict(rows: list[dict]) -> dict:
    """The highest level clearly faster than the one below it, with every
    request answered and no request to slow down."""
    clean = lambda r: r["requests"] and r["ok"] == r["requests"]
    if not rows or rows[0]["level"] != 1 or not clean(rows[0]):
        return {"safe": 1, "measured": False,
                "say": "One at a time did not finish cleanly, so nothing is "
                       "compared and nothing is saved."}
    best = rows[0]
    # In order, stopping at the first level that failed, was asked to slow
    # down, or was not clearly faster: each is compared with the one below.
    for r in rows[1:]:
        if not clean(r) or r["slow_down_asked"]:
            break
        if r["per_minute"] >= best["per_minute"] * GAIN:
            best = r
        else:
            break
    gain = best["per_minute"] / rows[0]["per_minute"] if rows[0]["per_minute"] else 1.0
    say = (f"{best['level']} at a time is the most this project takes with a "
           f"clear gain: {best['per_minute']} sentences a minute, {gain:.1f} "
           f"times one at a time." if best["level"] > 1 else
           "Sending more than one at a time did not help this project; "
           "lectures record one sentence at a time.")
    return {"safe": best["level"], "measured": True, "speedup": round(gain, 2),
            "say": say}


def main_cli(levels, per_level, yes: bool, outdir: Path, log=print) -> int:
    from realme.adapters.factory import build
    from realme.adapters.tts_gemini import calls_today, gemini_parallel_use
    tts = build("gemini-tts")
    tts.preflight()
    est = estimate(levels, per_level, getattr(tts, "model", ""))
    log(f"  voice: {getattr(tts, 'voice', '?')} on {getattr(tts, 'model', '?')}")
    log(f"  levels: {', '.join(map(str, levels))} at a time; "
        f"{est['requests']} requests, about {est['audio_s']}s of speech, "
        f"about ${est['usd']:.2f}")
    log(f"  requests already made today from this computer: {calls_today()}; "
        f"these use {est['requests']} more of today's allowance")
    if not yes and input("  Continue? [y/N] ").strip().lower() not in ("y", "yes"):
        return 0
    res = measure(tts, levels, per_level, outdir=outdir, log=log)
    v = verdict(res["levels"])
    log("\n  " + v["say"])
    stamp = time.strftime("%Y%m%d_%H%M%S")
    record = {"when": stamp, "model": getattr(tts, "model", ""),
              "safe": v["safe"], **res, "verdict": v}
    (outdir / f"gemini_parallel_{stamp}.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8")
    if res["stopped"] or not v.get("measured"):
        # A run cut short by the day's allowance, or one where even one at a
        # time failed, measured nothing about the project; saving it would
        # lower every later lecture to one request at a time.
        log("  Not saved: the measurement did not finish cleanly"
            + (f" ({res['stopped']})" if res["stopped"] else "")
            + ". Try again later.")
    elif res["levels"]:
        saved = save(measured=v["safe"], when=stamp, model=record["model"])
        use = gemini_parallel_use()
        log(f"  Saved: this project took up to {v['safe']} at a time. Lectures "
            f"in your Gemini voice record {use} at a time"
            + (" (the default)" if "use" not in saved else " (your choice)")
            + (f"; `realme gemini-parallel --use {v['safe']}` would use all "
               f"of it." if v["safe"] > use else "."))
    log(f"\n  results and audio: {outdir}")
    return 0


def save(**fields) -> dict:
    """Merge `fields` into the saved file and return what it now holds. A new
    measurement keeps the user's choice, lowered to fit under it."""
    from realme.adapters.tts_gemini import (GEMINI_PARALLEL_FILE,
                                            gemini_parallel_saved)
    from realme.core.env import data_home
    saved = gemini_parallel_saved()
    saved.pop("safe", None)
    saved.update({k: v for k, v in fields.items() if v is not None})
    if "use" in saved and "measured" in saved:
        saved["use"] = min(int(saved["use"]), int(saved["measured"]))
    data_home().mkdir(parents=True, exist_ok=True)
    (data_home() / GEMINI_PARALLEL_FILE).write_text(
        json.dumps(saved, indent=2), encoding="utf-8")
    return saved


def choose(n: int, log=print) -> int:
    """`--use N`: record N at a time from now on, sending nothing."""
    from realme.adapters.tts_gemini import (GEMINI_PARALLEL_MAX,
                                            gemini_parallel_saved,
                                            gemini_parallel_use)
    if not 1 <= n <= GEMINI_PARALLEL_MAX:
        log(f"  --use takes 1 to {GEMINI_PARALLEL_MAX}.")
        return 1
    measured = gemini_parallel_saved().get("measured")
    if measured and n > int(measured):
        log(f"  This project was measured to take {measured} at a time; "
            f"{n} is more than that. Measure again first "
            f"(`realme gemini-parallel --levels 1,2,...,{n}`).")
        return 1
    save(use=n)
    log(f"  Lectures in your Gemini voice now record {gemini_parallel_use()} "
        f"at a time (REALME_RECORD_WORKERS overrides it).")
    return 0
