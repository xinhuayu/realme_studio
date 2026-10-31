"""
`realme bench-batch` -- does recording several sentences in one pass help on
this computer, and how many per pass?

The local clone's two models spend their time reading their weights from
memory, once per step. A pass of several sentences reads them once for all
of them (`engines/qwen3cpp_batch`). How much that saves depends on the
computer -- its memory, its cores -- so it is measured here, with your own
voice, and the answer is saved for lectures to use.

Two checks, in this order:

  identical   two sentences with sampling switched off, recorded one at a
              time and then together: the sound must match sample for
              sample, or nothing is saved.
  faster      the same sentences one at a time, then 2, 4, 8, 12 and 16 (by
              default) per pass, as seconds of work per second of speech.
              Each size records whole passes: 16 sentences, or 12 at 12 per
              pass.

The fastest pass size is saved -- or a smaller one within NEAR of it,
since a smaller pass needs less memory -- provided it is clearly (GAIN)
faster than one at a time. It is saved (`qwen3_batch.json` in the data folder). REALME_RECORD_BATCH
overrides it; 1 switches it off. Needs an engine built with the batch entry
point: run _Rebuild_Voice_Engine.bat first.
"""
from __future__ import annotations

import json
import time
import wave
from pathlib import Path

#: How much faster than one at a time a pass must be to be used at all.
GAIN = 1.15
#: A smaller pass within this of the fastest is taken instead of it. Each
#: size is compared with the fastest, not with the size below it, so a
#: larger pass that is still clearly faster is not missed.
NEAR = 0.06
DEFAULT_SIZES = (1, 2, 4, 8, 12, 16)
#: Opt-in: GGML's tinyBLAS for every product it takes. A pass and one
#: sentence still match while the engine's own 8-bit kernel takes a pass's
#: products; without it they differ, and the identity check turns passes off.
ALL_ENV = "QWEN3_TTS_SGEMM_ALL"


def _same(a: Path, b: Path) -> bool:
    with wave.open(str(a)) as fa, wave.open(str(b)) as fb:
        return fa.readframes(fa.getnframes()) == fb.readframes(fb.getnframes())


def identical(tts, texts, outdir: Path, log=print) -> bool:
    """Sampling off: one at a time vs together, sample for sample."""
    keep = tts.temperature
    tts.temperature = 0.0
    try:
        single = []
        for i, t in enumerate(texts):
            single.append(tts.synthesize(t, outdir / f"same_single_{i}.wav"))
        together = tts.synthesize_batch(
            [(t, outdir / f"same_batch_{i}.wav", 1.0, None)
             for i, t in enumerate(texts)])
    finally:
        tts.temperature = keep
    return all(_same(a, b) for a, b in zip(single, together))


def measure(tts, sizes, texts, outdir: Path, log=print) -> list[dict]:
    from realme.core.media import duration_of
    rows = []
    for b in sizes:
        # Whole passes only, so no size is measured on a short last pass.
        use = texts[:max(b, len(texts) // b * b)]
        log(f"\n  {b} per pass: {len(use)} sentences ...")
        t0 = time.perf_counter()
        paths = []
        for i in range(0, len(use), b):
            chunk = use[i:i + b]
            outs = [outdir / f"b{b}_{i + j:02d}.wav" for j in range(len(chunk))]
            if b == 1:
                paths.append(tts.synthesize(chunk[0], outs[0]))
            else:
                paths += tts.synthesize_batch(
                    [(t, o, 1.0, None) for t, o in zip(chunk, outs)])
        wall = time.perf_counter() - t0
        audio = sum(duration_of(p) for p in paths)
        rows.append({"batch": b, "wall_s": round(wall, 1),
                     "audio_s": round(audio, 1),
                     "work_per_speech_s": round(wall / audio, 2) if audio else None})
        log(f"    {wall:.0f}s for {audio:.0f}s of speech: "
            f"{wall / audio:.2f} seconds of work per second" if audio else
            "    no audio")
    return rows


def verdict(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("work_per_speech_s")]
    if not ok or ok[0]["batch"] != 1:
        return {"batch": 1, "say": "One at a time did not measure; nothing "
                                   "to compare."}
    one = ok[0]["work_per_speech_s"]
    # Only sizes clearly faster than one at a time are candidates; of those,
    # the smallest within NEAR of the fastest.
    good = [r for r in ok[1:] if r["work_per_speech_s"] * GAIN <= one]
    if good:
        fastest = min(r["work_per_speech_s"] for r in good)
        best = min((r for r in good
                    if r["work_per_speech_s"] <= fastest * (1 + NEAR)),
                   key=lambda r: r["batch"])
    else:
        best = ok[0]
    gain = one / best["work_per_speech_s"]
    say = (f"{best['batch']} sentences per pass recorded {gain:.2f} times as "
           f"fast as one at a time." if best["batch"] > 1 else
           "Several per pass did not help clearly on this computer; lectures "
           "record one sentence at a time.")
    return {"batch": best["batch"], "speedup": round(gain, 2), "say": say}


def main_cli(sizes, sentences: int, outdir: Path, reference: Path | None,
             log=print, quant: str | None = None, save: bool = True) -> int:
    import os
    from realme.adapters.factory import build
    from realme.adapters.tts_qwen3_dll import (BATCH_FILE, BATCH_FN, QUANT_ENV,
                                               quant_setting)
    if quant:
        # Measured for the model lectures will use: the 8-bit model reads
        # less, so a pass may pay off differently.
        os.environ[QUANT_ENV] = quant
    log(f"  model: {quant_setting()}")
    all_on = os.environ.get(ALL_ENV, "").strip() == "1"
    if all_on:
        log(f"  {ALL_ENV}=1: tinyBLAS for every product. The sound differs "
            f"slightly from the default; listen before keeping it.")
    from realme.bench import DEFAULT_SCRIPT
    from realme.core.env import data_home
    outdir.mkdir(parents=True, exist_ok=True)
    from realme.pipeline.prerecord import base_engine
    tts = base_engine(build("qwen3cpp", reference_wav=reference))
    tts.preflight()
    if getattr(getattr(tts, "_lib", None), BATCH_FN, None) is None:
        log("  This voice engine was built before the batch entry point. Run "
            "_Rebuild_Voice_Engine.bat, then this again.")
        return 1
    low = os.environ.get("QWEN3_TTS_LOW_MEM", "")
    if not tts.batch_built() or (low and low != "0"):
        log("  Batches are not used with this engine (a GPU build, or "
            "low-memory mode); it records one sentence at a time. Nothing to "
            "measure.")
        return 1
    from realme.parbench import available_memory_gb
    from realme.pipeline.prerecord import (GB_PER_BATCH_SENTENCE,
                                           GB_PER_ENGINE, GB_SPARE)
    free = available_memory_gb()
    if free is not None:
        fit = [b for b in sizes
               if b == 1 or GB_PER_ENGINE + b * GB_PER_BATCH_SENTENCE
               + GB_SPARE <= free]
        if fit != list(sizes):
            log(f"  {free:.0f} GB free: measuring {fit} only")
        sizes = fit
    n_texts = max(2, sentences, max(sizes))
    texts = (list(DEFAULT_SCRIPT) * (n_texts // len(DEFAULT_SCRIPT) + 1))[:n_texts]
    log("  warming up ...")
    tts.synthesize(texts[0], outdir / "warmup.wav")
    log("  checking the sound is identical (sampling off) ...")
    from realme.adapters.base import AdapterUnavailable
    try:
        same = identical(tts, texts[:2], outdir, log=log)
    except AdapterUnavailable as e:
        log(f"  The engine would not record a batch ({e}); batches stay off.")
        return 1
    log("    identical, sample for sample" if same else
        f"    DIFFERENT -- batches stay off. Remove {ALL_ENV} and measure "
        f"again." if all_on else
        "    DIFFERENT -- batches stay off; send the results file")
    rows = measure(tts, sizes, texts, outdir, log=log) if same else []
    v = verdict(rows) if same else {"batch": 1, "say": "Not identical; off."}
    log("\n  " + v["say"])
    stamp = time.strftime("%Y%m%d_%H%M%S")
    record = {"when": stamp, "identical": same, "sizes": list(sizes),
              "quant": quant_setting(), "sgemm_all": all_on,
              "rows": rows, "verdict": v,
              "threads": getattr(tts, "threads", None)}
    rf = outdir / f"batch_{stamp}.json"
    rf.write_text(json.dumps(record, indent=2), encoding="utf-8")
    if not save:
        # A comparison (two models side by side, say): the results file and
        # the audio, and the pass size lectures use left as it was.
        log("  Not saved as this computer's pass size (--no-save).")
        log(f"\n  results: {rf}")
        return 0
    data_home().mkdir(parents=True, exist_ok=True)
    # One entry per model, so measuring the 8-bit model does not replace
    # the 16-bit model's answer.
    f = data_home() / BATCH_FILE
    try:
        saved = json.loads(f.read_text(encoding="utf-8"))
        if "batch" in saved:                       # older flat, single-model format
            saved = {saved.get("quant") or "f16": saved}
    except (OSError, ValueError):
        saved = {}
    saved[quant_setting()] = {"batch": v["batch"], "when": stamp,
                              "speedup": v.get("speedup")}
    f.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    log(f"  Saved: lectures in your local voice record {v['batch']} "
        f"sentence{'s' if v['batch'] != 1 else ''} per pass "
        f"(REALME_RECORD_BATCH overrides it).")
    log(f"\n  results: {rf}")
    return 0
