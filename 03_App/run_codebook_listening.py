"""Fewer code predictor steps: what they save and what they sound like.

Each frame of speech is 16 codes; the code predictor makes 15 of them, one
step after another, each a finer correction to the sound. Fewer steps save
predictor time and lose the finest detail. This renders the same sentences
several ways, saves every take as a WAV for listening, and reports the time
each took. It changes nothing in Studio or the engine.

Two stages, in the order to listen to them:

  vocoder  The codes made as usual (all 15 steps), the vocoder told to hear
           only the first 13, 11 or 9 codebooks (QWEN3_TTS_VOCODER_CODEBOOKS).
           The best that 12, 10 or 8 steps could sound: the talker and the
           predictor are not touched, so this has no time saving, and the
           same sentence lasts exactly as long. If 9 already sounds bad
           here, 8 steps will not do.
  steps    The predictor really stops after 12, 10 or 8 steps
           (QWEN3_TTS_CP_STEPS), with the talker given each skipped
           codebook's mean (fill "mean") or nothing (fill "zero") in their
           place. Saves time; the talker hears something it was not trained
           on, so listen for drift, odd pacing and sentences that run on.

The vocoder stage also renders the 8-bit vocoder (QWEN3_TTS_VOCODER_Q8=8,
and =7 with 7-bit input, the one faster on a processor without AVX-VNNI):
the same codes, so its "sound kept" figure is meaningful, and its time is
the vocoder's own saving.

    python run_codebook_listening.py                    (both stages)
    python run_codebook_listening.py --stage vocoder    (the first, ~2 minutes)
    python run_codebook_listening.py --repeats 2        (timings worth comparing)
    python run_codebook_listening.py --plan-only

Results: results/codebooks_<time>/listen/<sentence>_<arm>.wav side by side,
REPORT.md with the times, each sentence's length against the full take, and
for the vocoder stage how far the sound moved (signal-to-difference in dB;
the same codes, so the takes line up sample for sample). Greedy decoding by
default, so the arms differ only by the switch.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_cpu_combinations as RC          # noqa: E402

#: name -> (stage, environment). "full" is the reference every arm is compared with.
ARMS = {
    "full": ("both", {}),
    "voc13": ("vocoder", {"QWEN3_TTS_VOCODER_CODEBOOKS": "13"}),
    "voc11": ("vocoder", {"QWEN3_TTS_VOCODER_CODEBOOKS": "11"}),
    "voc9": ("vocoder", {"QWEN3_TTS_VOCODER_CODEBOOKS": "9"}),
    # The 8-bit vocoder (same codes, so compared sample by sample too).
    "vocq8": ("vocoder", {"QWEN3_TTS_VOCODER_Q8": "8"}),
    "vocq7": ("vocoder", {"QWEN3_TTS_VOCODER_Q8": "7"}),
    "steps12_mean": ("steps", {"QWEN3_TTS_CP_STEPS": "12", "QWEN3_TTS_CP_FILL": "mean"}),
    "steps10_mean": ("steps", {"QWEN3_TTS_CP_STEPS": "10", "QWEN3_TTS_CP_FILL": "mean"}),
    "steps8_mean": ("steps", {"QWEN3_TTS_CP_STEPS": "8", "QWEN3_TTS_CP_FILL": "mean"}),
    "steps12_zero": ("steps", {"QWEN3_TTS_CP_STEPS": "12", "QWEN3_TTS_CP_FILL": "zero"}),
    "steps10_zero": ("steps", {"QWEN3_TTS_CP_STEPS": "10", "QWEN3_TTS_CP_FILL": "zero"}),
    "steps8_zero": ("steps", {"QWEN3_TTS_CP_STEPS": "8", "QWEN3_TTS_CP_FILL": "zero"}),
}
#: Sentences that end by themselves under greedy decoding (the NUC sweep's).
SENTENCES = [
    "The first slide introduces the main question.",
    "We compare the observed results with the expected values and explain why the difference matters.",
    "Please look at the horizontal axis.",
    "The next example illustrates how the same method applies to a different population.",
    "These estimates depend on the assumptions stated earlier in the lecture.",
    "The evidence supports a cautious interpretation because several alternative explanations remain possible.",
]


def arms_for(stage: str) -> list[str]:
    return [n for n, (s, _) in ARMS.items() if s == "both" or stage in ("all", s)]


def plan(names: list[str], repeats: int) -> list[tuple[int, str]]:
    """(repeat, arm): the reference first in each repeat; the order of the
    rest reversed on alternate repeats, so no arm always runs warmest."""
    out = []
    rest = [n for n in names if n != "full"]
    for r in range(repeats):
        order = rest if r % 2 == 0 else rest[::-1]
        out += [(r, "full")] + [(r, n) for n in order]
    return out


def read_wav(path: Path) -> list[float]:
    import array
    import wave
    with wave.open(str(path)) as w:
        data = array.array("h")
        data.frombytes(w.readframes(w.getnframes()))
    return [x / 32768.0 for x in data]


def snr_db(reference: list[float], other: list[float]) -> float | None:
    """Signal-to-difference in dB, for takes of the same length (the
    vocoder stage); None when they differ in length."""
    if len(reference) != len(other) or not reference:
        return None
    signal = sum(x * x for x in reference)
    noise = sum((x - y) ** 2 for x, y in zip(reference, other))
    if noise == 0:
        return math.inf
    return 10 * math.log10(signal / noise) if signal > 0 else None


def run_arm(name: str, folder: Path, a, base: dict) -> dict:
    cmd = [sys.executable, str(HERE / "check_lecture_cpu.py"), "--home", str(a.home),
           "--embedding", str(a.embedding), "--out", str(folder), "--threads", str(a.threads),
           "--slots", str(a.slots), "--sentences", str(len(a.texts)), "--mode", "stream",
           "--frames", str(a.frames), "--corpus-file", str(a.corpus_file), "--save-audio",
           "--temperature", str(a.temperature)]
    if a.warmup:
        cmd.append("--warmup-pass")
    if a.library_dir:
        cmd += ["--library-dir", str(a.library_dir)]
    if a.model_dir:
        cmd += ["--model-dir", str(a.model_dir)]
    env = dict(base, QWEN3_TTS_STAGE_TIMING="1", **ARMS[name][1])
    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(HERE), env=env, capture_output=True, text=True)
    (folder.parent / (folder.name + ".log")).write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
    row = dict(arm=name, folder=str(folder), returncode=proc.returncode,
               wall_seconds=time.perf_counter() - started)
    result = folder / "result.json"
    if proc.returncode == 0 and result.is_file():
        data = json.loads(result.read_text())
        row.update(seconds=data["synthesis_seconds"], audio_seconds=data["audio_seconds"],
                   samples={str(o["corpus_index"]): o["samples"] for o in data["outputs"]},
                   stages=RC.parse_stages(proc.stderr))
    else:
        row["error"] = (proc.stderr or proc.stdout)[-2000:]
    return row


def analyse(rows: list[dict], out: Path, texts: list[str], frames: int) -> dict:
    ok = [r for r in rows if r.get("seconds")]
    names = list(dict.fromkeys(r["arm"] for r in rows))
    med = {n: statistics.median(r["seconds"] for r in ok if r["arm"] == n)
           for n in names if any(r["arm"] == n for r in ok)}
    pred = {n: statistics.median(r["stages"].get("predictor_compute", 0) for r in ok if r["arm"] == n)
            for n in med}
    voc = {n: statistics.median(r["stages"].get("vocoder", 0) for r in ok if r["arm"] == n)
           for n in med}
    first = {}
    for r in ok:
        first.setdefault(r["arm"], r)
    ref = first.get("full")
    per = {}
    listen = out / "listen"
    listen.mkdir(exist_ok=True)
    for n, r in first.items():
        for i in range(len(texts)):
            src = Path(r["folder"]) / f"{i:04d}.wav"
            if src.is_file():
                shutil.copy2(src, listen / f"{i + 1:02d}_{n}.wav")
        if not ref or n == "full":
            continue
        lengths, snrs, ran_on = [], [], []
        for i in range(len(texts)):
            a, b = ref["samples"].get(str(i)), r["samples"].get(str(i))
            if a and b:
                lengths.append(b / a)
                if b >= frames * 1920 * 0.95 or b / a > 2:
                    ran_on.append(i + 1)
            if ARMS[n][0] == "vocoder":
                fa, fb = Path(ref["folder"]) / f"{i:04d}.wav", Path(r["folder"]) / f"{i:04d}.wav"
                if fa.is_file() and fb.is_file():
                    s = snr_db(read_wav(fa), read_wav(fb))
                    if s is not None:
                        snrs.append(s)
        per[n] = dict(length_ratio=statistics.median(lengths) if lengths else None,
                      snr_db=statistics.median(snrs) if snrs else None, ran_on=ran_on)
    return dict(medians=med, predictor=pred, vocoder=voc, per_arm=per)


def report(summary: dict) -> str:
    res = summary["result"]
    med, pred, per = res["medians"], res["predictor"], res["per_arm"]
    voc = res.get("vocoder", {})
    full = med.get("full")
    lines = [f"# Fewer code predictor steps: {summary['host']}", "",
             f"{len(summary['texts'])} sentences, {summary['slots']} per pass, {summary['threads']} threads, "
             f"{summary['repeats']} repeat(s), temperature {summary['temperature']}.", "",
             "| Arm | What changes | Seconds (median) | vs full | Predictor s | Vocoder s | Length vs full | Sound kept (dB) |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for n in med:
        env = ", ".join(f"{k.replace('QWEN3_TTS_', '')}={v}" for k, v in ARMS[n][1].items()) or "nothing (reference)"
        p = per.get(n, {})
        ratio = f"{med[n] / full:.3f}" if full else ""
        length = f"{p['length_ratio']:.2f}" if p.get("length_ratio") else ""
        snr = ("identical" if p.get("snr_db") == math.inf else f"{p['snr_db']:.1f}") if p.get("snr_db") is not None else ""
        lines.append(f"| {n} | {env} | {med[n]:.1f} | {ratio} | {pred.get(n, 0):.1f} | {voc.get(n, 0):.1f} | {length} | {snr} |")
    ran = {n: p["ran_on"] for n, p in per.items() if p.get("ran_on")}
    lines += ["", "Ran on past their text (sentence numbers): " +
              ("; ".join(f"{n}: {v}" for n, v in ran.items()) if ran else "none."),
              "", "## How to read this", "",
              "- Listen in `listen/`: each sentence's takes are side by side (`03_full.wav`, `03_voc11.wav`, ...).",
              "  Start with the vocoder stage: it is the best each step count can sound.",
              "- \"Sound kept\" (vocoder stage only) is the full take's energy over the difference: 30 dB and",
              "  more is hard to hear; below about 15 dB expect audible loss. It says nothing about whether",
              "  the loss is pleasant; the ears decide.",
              "- \"Length vs full\" near 1 means the pacing held. Much longer means the talker drifted.",
              "- A time saving from one repeat is not a measurement; use --repeats 2 or more before deciding.",
              "- Nothing here changes Studio. To use a setting for a test lecture, put the arm's lines",
              "  (e.g. QWEN3_TTS_CP_STEPS=12) in RealMeStudio/.env, and take them out again after."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["vocoder", "steps", "all"], default="all")
    p.add_argument("--home", type=Path, help="engine home (default: tools\\realme-rev-qwen3-tts)")
    p.add_argument("--library-dir", type=Path)
    p.add_argument("--model-dir", type=Path)
    p.add_argument("--embedding", type=Path, help="voice embedding .f32 (default: the newest)")
    p.add_argument("--threads", type=int, default=RC.default_threads())
    p.add_argument("--slots", type=int, help="sentences per pass (default: Studio's, else 8)")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--frames", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--corpus-file", type=Path, help="JSON list of sentences (default: six lecture sentences)")
    p.add_argument("--no-warmup", dest="warmup", action="store_false")
    p.add_argument("--out", type=Path)
    p.add_argument("--plan-only", action="store_true")
    a = p.parse_args(argv)
    a.texts = json.loads(a.corpus_file.read_text(encoding="utf-8")) if a.corpus_file else SENTENCES
    a.slots = a.slots or int(RC.studio_setting("REALME_RECORD_BATCH") or 0) or 8
    a.slots = min(a.slots, len(a.texts))
    runs = plan(arms_for(a.stage), a.repeats)
    if a.plan_only:
        print(json.dumps(dict(threads=a.threads, slots=a.slots, sentences=len(a.texts),
                              runs=[(r, n, ARMS[n][1]) for r, n in runs]), indent=2))
        return 0
    from realme.engines.rev import cpu_settings, find_models
    from realme.engines.rev_build import library_directory
    from realme.engines.rev import engine_home
    a.home = a.home or engine_home()
    if not (a.home / "local" / "engine_active.json").is_file():
        raise ValueError(f"Build RealMe's voice first (_Rebuild_Voice_Engine.bat), or pass --home; not built: {a.home}")
    a.home = a.home.resolve()
    if "onedrive" in [x.lower() for x in a.home.parts]:
        raise ValueError("Use the local engine folder, not a OneDrive copy")
    a.library_dir = a.library_dir or library_directory(a.home)
    a.model_dir = a.model_dir or find_models(a.home)
    dll = next((a.library_dir / n for n in ("qwen3tts.dll", "libqwen3tts.so") if (a.library_dir / n).is_file()), None)
    if dll is None:
        raise FileNotFoundError(f"No engine library in {a.library_dir}")
    blob = dll.read_bytes()
    if b"QWEN3_TTS_CP_STEPS" not in blob or b"QWEN3_TTS_VOCODER_CODEBOOKS" not in blob:
        raise RuntimeError(f"Rebuild the voice engine first: {dll} predates these switches")
    if a.embedding is None:
        folder = a.home.parent / "qwen3cpp" / "embeddings"
        found = sorted((q for q in folder.glob("*.f32") if q.stat().st_size == 4096), key=lambda q: q.stat().st_mtime)
        if not found:
            raise FileNotFoundError("No voice embedding found; pass --embedding")
        a.embedding = found[-1]
    out = a.out or a.home / "results" / ("codebooks_" + time.strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=False)
    a.corpus_file = out / "corpus.json"
    a.corpus_file.write_text(json.dumps(a.texts, indent=2), encoding="utf-8")
    base = {k: v for k, v in os.environ.items() if not k.startswith("QWEN3_TTS_")}
    base.update({k: v for k, v in cpu_settings(a.threads).items() if v is not None})
    base["QWEN3_TTS_CP_BATCH_ATTN"] = "1"      # the fast path, with kept graphs
    summary = dict(host=platform.node(), threads=a.threads, slots=a.slots, repeats=a.repeats,
                   temperature=a.temperature, texts=a.texts, dll_sha256=RC.sha(dll),
                   started=time.strftime("%Y-%m-%d %H:%M:%S"), runs=[])
    for i, (r, name) in enumerate(runs, 1):
        row = run_arm(name, out / f"{i:02d}_r{r}_{name}", a, base)
        row["repeat"] = r
        summary["runs"].append(row)
        print(f"[{i}/{len(runs)}] {name}: " + (f"{row['seconds']:.1f} s" if row.get("seconds") else "FAILED"),
              flush=True)
        RC.dump(out / "summary.json", summary)
    summary["result"] = analyse(summary["runs"], out, a.texts, a.frames)
    summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    RC.dump(out / "summary.json", summary)
    text = report(summary)
    (out / "REPORT.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"Results: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
