"""
`realme bench-parallel` -- would several local voice engines at once be faster?

The local clone (qwen3cpp) records one sentence at a time with the model held
in memory. Adding threads to that one engine was measured and changed
nothing beyond the noise (see the note on `threads` in
adapters/tts_qwen3_dll.py), which points at memory speed rather than cores as
the limit. Running two or three engines side by side, each on its own
sentences, might still finish a lecture sooner -- or might only share the
same memory speed among more copies of the model. That is a question about a
particular machine, so this measures it there instead of guessing.

What it does, for each layout (workers x threads each):

  * starts that many separate engine processes, each loading its own copy of
    the model, and waits until all are loaded -- loading is reported, never
    counted as recording time;
  * hands the same sentences out to whichever worker is free, and times the
    whole batch from the first sentence to the last;
  * does all of it more than once, because the engine samples without a
    seed and one run is one draw.

It reports, per layout: the memory each worker took, the time for the batch,
and the speed relative to one worker. It recommends parallel recording only
when the best layout is faster in EVERY run than one worker is in ANY run --
a difference smaller than the noise is not a difference.

Nothing in rendering changes because of this. The audio is kept so the
layouts can be listened to as well as timed.
"""
from __future__ import annotations

import json
import os
import platform
import queue
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

#: Memory one qwen3cpp worker is assumed to need before it has been measured:
#: the 2.2 GB of weights plus working space. A layout that would not fit in
#: the memory free now is skipped rather than allowed to push the machine
#: into swapping, which would measure the disk.
GB_PER_WORKER_GUESS = 4.0

#: Times real time the local clone is assumed to take, for the estimate
#: printed before anything starts. Measured values replace it in the report.
RTF_GUESS = 4.0


# ---------------------------------------------------------------- machine

def available_memory_gb() -> float | None:
    """Memory free right now, in GB, or None when it cannot be read."""
    return _memory_gb(total=False)


def total_memory_gb() -> float | None:
    """The machine's memory, in GB, or None when it cannot be read."""
    return _memory_gb(total=True)


def _memory_gb(total: bool) -> float | None:
    try:
        if os.name == "nt":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            ms = MS()
            ms.dwLength = ctypes.sizeof(MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                return (ms.ullTotalPhys if total else ms.ullAvailPhys) / 1e9
            return None
        want = "MemTotal:" if total else "MemAvailable:"
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith(want):
                return int(line.split()[1]) * 1024 / 1e9
    except Exception:
        return None
    return None


def default_layouts(cores: int, physical: int | None = None) -> list[tuple[int, int]]:
    """One engine at one thread per core (what the Studio uses), the same
    cores shared by two and three engines, two engines with a full share
    each, and -- on a machine with room -- as many full engines as the cores
    allow. `cores` is logical processors; `physical` the cores themselves,
    assumed half the logical count when not given."""
    per_core = physical or max(2, cores // 2)
    one = max(2, min(6, per_core))
    out = []
    full = min(4, per_core // one)
    share = lambda k: max(2, min(6, per_core // k))
    for lay in ((1, one), (2, share(2)), (2, one), (3, share(3)), (full, one)):
        if lay[0] >= 1 and lay not in out:
            out.append(lay)
    return out


def parse_layouts(text: str) -> list[tuple[int, int]]:
    """`1x6,2x4,3x3` -> [(1, 6), (2, 4), (3, 3)]."""
    out = []
    for part in (text or "").replace(" ", "").split(","):
        if not part:
            continue
        try:
            w, t = part.lower().split("x")
            w, t = int(w), int(t)
        except ValueError:
            raise ValueError(f"{part!r} is not a layout: write workers x "
                             f"threads, like 2x4.")
        if not (1 <= w <= 8 and 1 <= t <= 64):
            raise ValueError(f"{part}: between 1 and 8 workers, 1 and 64 threads.")
        out.append((w, t))
    if not out:
        raise ValueError("No layouts given.")
    if out[0][0] != 1:
        raise ValueError("The first layout is the comparison and must be one "
                         "worker, like 1x6.")
    return out


# ------------------------------------------------------------------ worker

class _SelfTest:
    """A stand-in engine, for testing this module without the model: it
    takes a fixed time per word, sleeping, so it scales perfectly."""
    name = "selftest"
    phoneme_syntax = None

    def __init__(self, threads: int = 1):
        self.threads = threads

    def preflight(self):
        time.sleep(0.2)

    def synthesize(self, text, out_wav, pace=1.0, **kw):
        import math, struct, wave
        words = max(1, len(text.split()))
        time.sleep(0.02 * words)
        rate, secs = 16000, 0.25 * words
        with wave.open(str(out_wav), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
            w.writeframes(b"".join(struct.pack("<h", int(3000 * math.sin(
                2 * math.pi * 200 * i / rate))) for i in range(int(rate * secs))))
        return Path(out_wav)


def _worker(spec: str, reference: str | None, threads: int, outdir: str,
            tag: str, jobs, results, go) -> None:
    """One engine process: load, say so, wait for the start, then record
    whatever sentences are handed to it until told to stop."""
    wid = os.getpid()
    try:
        if spec == "selftest":
            tts = _SelfTest(threads)
        else:
            from realme.bench import _make
            tts = _make(spec, Path(reference) if reference else None, threads)
        t0 = time.perf_counter()
        tts.preflight()
        results.put(("ready", wid, time.perf_counter() - t0, None))
    except Exception as e:
        results.put(("failed", wid, 0.0, f"{e.__class__.__name__}: {e}"))
        return
    from realme.text.lexicon import Lexicon
    from realme.text.prepare import prepare_for_speech
    lx = Lexicon()
    syntax = getattr(tts, "phoneme_syntax", None) or "none"
    go.wait()
    while True:
        item = jobs.get()
        if item is None:
            return
        i, text, pace = item
        out = Path(outdir) / f"{tag}_{i:02d}.wav"
        try:
            spoken = prepare_for_speech(text, syntax, lx).engine_text
            t0 = time.perf_counter()
            tts.synthesize(spoken, out, pace=pace)
            wall = time.perf_counter() - t0
            from realme.core.media import duration_of
            results.put(("done", i, wall, duration_of(out)))
        except Exception as e:
            results.put(("error", i, 0.0, f"{e.__class__.__name__}: {e}"))


# --------------------------------------------------------------------- run

@dataclass
class Run:
    workers: int
    threads: int
    repeat: int
    wall_s: float = 0.0
    audio_s: float = 0.0
    load_s: float = 0.0
    gb_per_worker: float | None = None
    failed: int = 0
    note: str = ""


@dataclass
class Layout:
    workers: int
    threads: int
    runs: list[Run] = field(default_factory=list)
    skipped: str = ""

    @property
    def label(self) -> str:
        return f"{self.workers} x {self.threads}"

    def walls(self) -> list[float]:
        return [r.wall_s for r in self.runs if r.wall_s and not r.failed]


def run_layout(spec: str, script: list[tuple[str, float]], workers: int,
               threads: int, outdir: Path, repeat: int,
               reference: Path | None, log=print,
               load_timeout_s: float = 900.0) -> Run:
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    jobs, results, go = ctx.Queue(), ctx.Queue(), ctx.Event()
    tag = f"{workers}x{threads}_run{repeat}"
    outdir.mkdir(parents=True, exist_ok=True)
    run = Run(workers, threads, repeat)
    free_before = available_memory_gb()
    procs = [ctx.Process(target=_worker,
                         args=(spec, str(reference) if reference else None,
                               threads, str(outdir), tag, jobs, results, go),
                         daemon=True)
             for _ in range(workers)]
    for p in procs:
        p.start()
    try:
        ready, loads = 0, []
        deadline = time.time() + load_timeout_s
        while ready < workers:
            left = deadline - time.time()
            if left <= 0:
                run.note = "the engines did not finish loading in time"
                run.failed = len(script)
                return run
            try:
                kind, _wid, secs, msg = results.get(timeout=left)
            except queue.Empty:
                continue
            if kind == "failed":
                run.note = f"an engine could not start: {msg}"
                run.failed = len(script)
                return run
            ready += 1
            loads.append(secs)
        run.load_s = round(max(loads), 1)
        free_after = available_memory_gb()
        if free_before is not None and free_after is not None:
            run.gb_per_worker = round(max(0.0, free_before - free_after) / workers, 2)
        for i, (text, pace) in enumerate(script):
            jobs.put((i, text, pace))
        for _ in procs:
            jobs.put(None)
        t0 = time.perf_counter()
        go.set()
        got = 0
        while got < len(script):
            try:
                kind, i, wall, extra = results.get(timeout=3600)
            except queue.Empty:
                run.note = "no sentence finished for an hour"
                run.failed += len(script) - got
                run.wall_s = 0.0
                return run
            got += 1
            if kind == "done":
                run.audio_s += float(extra or 0.0)
            else:
                run.failed += 1
                log(f"    sentence {i + 1}: {extra}")
        run.wall_s = round(time.perf_counter() - t0, 1)
        run.audio_s = round(run.audio_s, 1)
        return run
    finally:
        for p in procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()


def estimate_s(script: list[tuple[str, float]], layouts, repeat: int) -> float:
    words = sum(len(t.split()) for t, _ in script)
    audio = words / 2.5                     # about 150 words a minute
    total = 0.0
    for w, _t in layouts:
        total += repeat * (20.0 + audio * RTF_GUESS / max(1, min(w, 2) * 0.8))
    return total


def verdict(layouts: list[Layout]) -> dict:
    """Is any layout faster than one worker by more than the noise?"""
    base = layouts[0]
    bw = base.walls()
    if not bw:
        return {"worth_it": False, "best": None,
                "say": "One worker did not complete a run, so there is "
                       "nothing to compare against."}
    best, best_speed = None, 1.0
    for lay in layouts[1:]:
        w = lay.walls()
        if not w:
            continue
        speed = statistics.median(bw) / statistics.median(w)
        if speed > best_speed:
            best, best_speed = lay, speed
    if best is None:
        return {"worth_it": False, "best": None,
                "say": f"Nothing was faster than one worker "
                       f"({base.label} threads). Parallel recording would "
                       f"not help on this machine."}
    clear = max(best.walls()) < min(bw)
    worth = clear and best_speed >= 1.2
    say = (f"{best.label} finished the same sentences {best_speed:.2f} times "
           f"as fast as one worker")
    if worth:
        say += (", in every run. Parallel recording is worth building for "
                f"this machine, with {best.workers} workers of "
                f"{best.threads} threads.")
    elif not clear:
        say += (", but the runs overlap: the difference is inside the "
                "run-to-run noise. Not worth building on this evidence; "
                "--repeat 3 would settle it.")
    else:
        say += (". Consistent, but under 1.2 times -- too small to be worth "
                "the extra memory and heat.")
    return {"worth_it": worth, "best": best.label if best else None,
            "speedup": round(best_speed, 2), "say": say}


def table(layouts: list[Layout]) -> str:
    base_w = layouts[0].walls()
    base = statistics.median(base_w) if base_w else None
    rows = [f"  {'workers x threads':<18}{'memory each':>12}{'load':>8}"
            f"{'batch time (runs)':>26}{'speed':>8}"]
    for lay in layouts:
        if lay.skipped:
            rows.append(f"  {lay.label:<18}  skipped: {lay.skipped}")
            continue
        w = lay.walls()
        mem = [r.gb_per_worker for r in lay.runs if r.gb_per_worker is not None]
        loads = [r.load_s for r in lay.runs if r.load_s]
        if not w:
            why = next((r.note for r in lay.runs if r.note), "failed")
            rows.append(f"  {lay.label:<18}  did not finish: {why}")
            continue
        med = statistics.median(w)
        runs = " / ".join(f"{x:.0f}" for x in w)
        rows.append(
            f"  {lay.label:<18}"
            f"{(f'{max(mem):.1f} GB' if mem else '?'):>12}"
            f"{(f'{max(loads):.0f}s' if loads else '?'):>8}"
            f"{f'{med:.0f}s ({runs})':>26}"
            f"{(f'{base / med:.2f}x' if base else '?'):>8}")
    return "\n".join(rows)


def can_start(spec: str, reference: Path | None) -> str:
    """Why this engine cannot run here, in a sentence, or "" if it can.

    Asked before the time estimate, in seconds and without loading the
    model, so a computer that cannot run the engine at all says so at once
    instead of after the first long load.
    """
    if spec == "selftest":
        return ""
    try:
        from realme.bench import _make
        tts = _make(spec, reference)
        check = getattr(tts, "check", None) or tts.preflight
        check()
        return ""
    except Exception as e:
        return f"{e.__class__.__name__}: {e}"


def main(spec: str, script: list[tuple[str, float]], layouts, *, repeat: int,
         outdir: Path, reference: Path | None, log=print) -> dict:
    cores = os.cpu_count() or 4
    free = available_memory_gb()
    log(f"  machine: {platform.processor() or platform.machine()}, {cores} "
        f"logical cores, "
        + (f"{free:.0f} GB memory free" if free is not None else "free memory unknown"))
    log(f"  engine: {spec}; {len(script)} sentences; each layout run {repeat} times")
    results: list[Layout] = []
    gb_each = GB_PER_WORKER_GUESS if spec != "selftest" else 0.0
    for w, t in layouts:
        lay = Layout(w, t)
        results.append(lay)
        need = w * gb_each
        if free is not None and need > 0.8 * free:
            lay.skipped = (f"needs about {need:.0f} GB and {free:.0f} GB is "
                           f"free")
            log(f"\n  {lay.label}: skipped -- {lay.skipped}")
            continue
        for r in range(1, repeat + 1):
            log(f"\n  {lay.label} threads, run {r} of {repeat}: loading "
                f"{w} engine{'s' if w > 1 else ''} ...")
            run = run_layout(spec, script, w, t, outdir, r, reference, log=log)
            lay.runs.append(run)
            if run.wall_s:
                log(f"    {run.wall_s:.0f}s for {run.audio_s:.0f}s of speech "
                    f"(loaded in {run.load_s:.0f}s"
                    + (f", about {run.gb_per_worker:.1f} GB each"
                       if run.gb_per_worker else "") + ")")
            else:
                log(f"    did not finish: {run.note}")
                if lay is results[0] and "could not start" in run.note:
                    # One engine cannot start, so neither can two: stop
                    # here rather than fail the same way for every layout.
                    log("\n  The engine does not start on this computer, so "
                        "there is nothing to compare. Nothing was changed.")
                    return {"engine": spec, "layouts": [], "verdict": {
                        "worth_it": False, "best": None, "say": run.note}}
                break
            if run.gb_per_worker:
                gb_each = max(gb_each, run.gb_per_worker * 1.15)
    v = verdict(results)
    log("\n" + table(results))
    log("\n  " + v["say"])
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = {"when": stamp, "engine": spec, "cores": cores,
           "memory_free_gb": free, "sentences": len(script), "repeat": repeat,
           "layouts": [{**{k: v_ for k, v_ in asdict(l).items() if k != "runs"},
                        "runs": [asdict(r) for r in l.runs]} for l in results],
           "verdict": v}
    f = outdir / f"parallel_{stamp}.json"
    f.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"\n  results: {f}\n  audio:   {outdir}")
    return out
