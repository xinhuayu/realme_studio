"""Combinations of the CPU engine's switches, measured together, then confirmed.

Testing one switch at a time can miss a combination that helps: a switch that
does nothing at one width may help once another removes a different
bottleneck. This runs a full two-level factorial design over a few switches
(every combination, each repeated, in a shuffled order that is reversed for
the second repeat), checks that every combination gives the same audio,
estimates each switch's effect and each pair's interaction from all the runs,
and then confirms the fastest combination against the current setting in
alternating runs. Nothing in Studio or the engine is changed: the result is a
report and the lines to put in RealMeStudio/.env for a final lecture render.

    python run_cpu_combinations.py                 (this computer's defaults)
    python run_cpu_combinations.py --slots 4,8     (a 16 GB computer)
    python run_cpu_combinations.py --plan-only     (show the cases, run none)
    python run_cpu_combinations.py --home <tools>\\qwen3cpp   (the free clone; default: RealMe's own voice once built)
    python run_cpu_combinations.py --splits 6+0,6+2,5+1,5+2,4+2,6+6
                         (decoding beside the pass with other thread splits)
    python run_cpu_combinations.py --splits 6+0,6+gpu,4+0,4+gpu
                         (the vocoder on the GPU: an engine built with REALME_BUILD_VULKAN=1)

Every case is a fresh process running check_lecture_cpu.py (the same probe
harness as the NUC sweep), streaming the sentences with one untimed warm-up
pass. Greedy decoding, so every combination must give identical samples.

Switches (two levels each):
    cp_batch_attn   QWEN3_TTS_CP_BATCH_ATTN: the code predictor's attention
                    for all sentences in one set of operations (its graphs
                    then kept frame to frame unless QWEN3_TTS_CP_GRAPH_REUSE=0)
    decode_async    QWEN3_TTS_DECODE_ASYNC: finished sentences decoded beside
                    the pass, on the vocoder's own backend (QWEN3_TTS_DECODE_THREADS)
    attn_join       the talker's ordered attention gather (runtime option 4096)
    slots           sentences per pass, the two values of --slots

Thread splits (--splits): instead of the factorial, the fastest switches found
so far (cp_batch_attn and attn_join on, Studio's width) with the threads divided
differently between the transformers and the vocoder: "T+0" decodes in the
loop on T threads; "T+V" decodes beside the pass, the transformers on T threads
and the vocoder on V (6+6 on a six-core, twelve-thread processor puts the
vocoder on the hyperthreads). Compared with the first split, then confirmed.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ATTN_JOIN_BIT = 4096
FACTORS = ("cp_batch_attn", "decode_async", "attn_join", "slots")
STAGE_LINE = re.compile(r"\[realme\]\s+(\w+)\s+([0-9.]+) s\s+([0-9.]+)%")


def sha(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# ----------------------------------------------------------------- the plan
def cases(slots: tuple[int, int]) -> list[dict]:
    """Every combination of the four switches (16 cases)."""
    out = []
    for cp, dec, join, s in itertools.product((0, 1), (0, 1), (0, 1), slots):
        out.append(dict(cp_batch_attn=cp, decode_async=dec, attn_join=join, slots=s))
    return out


def quick_cases(width: int) -> list[dict]:
    """The two new switches, off and on, at one width (4 cases)."""
    return [dict(cp_batch_attn=cp, decode_async=dec, attn_join=0, slots=width)
            for cp, dec in itertools.product((0, 1), (0, 1))]


def split_cases(splits: str, width: int) -> list[dict]:
    """--splits "6+0,5+1": transformer threads + vocoder threads (0: in the loop)."""
    out = []
    for item in splits.split(","):
        t, v = item.strip().split("+")
        t = int(t)
        if v.lower() == "gpu":
            # The vocoder on the GPU beside the pass (an engine built with
            # REALME_BUILD_VULKAN=1): its samples differ slightly from the
            # CPU's, so it is compared with other GPU cases only.
            out.append(dict(cp_batch_attn=1, decode_async=1, attn_join=1, slots=width,
                            threads=t, decode_threads=0, vocoder="gpu"))
            continue
        v = int(v)
        assert 1 <= t <= 64 and 0 <= v <= 64, item
        out.append(dict(cp_batch_attn=1, decode_async=int(v > 0), attn_join=1, slots=width,
                        threads=t, decode_threads=v))
    return out


def quick_verdict(rows: list[dict]) -> tuple[bool, str]:
    failed = [key(r["case"]) for r in rows if r.get("error") or not r.get("seconds")]
    changed = [key(r["case"]) for r in rows if r.get("seconds") and not r.get("exact")]
    if failed:
        return False, "QUICK TEST FAILED: these cases did not finish: " + ", ".join(failed) + \
            ". Their .log files beside the result folders say why."
    if changed:
        return False, "QUICK TEST FAILED: audio differed in " + ", ".join(changed) + \
            "; keep the switches off and send the results folder for review."
    return True, ("QUICK TEST PASS: the new engine loads and records, and with each new switch "
                  "off or on every sentence's audio is identical.")


def key(case: dict) -> str:
    k = (f"cp{case['cp_batch_attn']}_dec{case['decode_async']}"
         f"_join{case['attn_join']}_s{case['slots']}")
    if "threads" in case:
        k += f"_t{case['threads']}+" + ("gpu" if case.get("vocoder") == "gpu" else str(case["decode_threads"]))
    return k


def schedule(all_cases: list[dict], repeats: int, seed: int) -> list[tuple[int, dict]]:
    """(repeat, case) in run order: shuffled once, reversed every other repeat,
    so slow drift (heat, background work) does not favour one combination."""
    order = list(all_cases)
    random.Random(seed).shuffle(order)
    runs = []
    for r in range(repeats):
        runs += [(r, c) for c in (order if r % 2 == 0 else list(reversed(order)))]
    return runs


def environment(case: dict, base: dict, decode_threads: int) -> dict:
    env = dict(base)
    if "threads" in case:
        env["QWEN3_TTS_THREADS"] = str(case["threads"])
        decode_threads = case["decode_threads"] or decode_threads
    if case.get("vocoder") == "gpu":
        env["QWEN3_TTS_VOCODER_BACKEND"] = "gpu"
    else:
        env.pop("QWEN3_TTS_VOCODER_BACKEND", None)
    env["QWEN3_TTS_CP_BATCH_ATTN"] = str(case["cp_batch_attn"])
    env["QWEN3_TTS_DECODE_ASYNC"] = str(case["decode_async"])
    if case["decode_async"]:
        env["QWEN3_TTS_DECODE_THREADS"] = str(decode_threads)
    else:
        env.pop("QWEN3_TTS_DECODE_THREADS", None)
    env["QWEN3_TTS_STAGE_TIMING"] = "1"
    return env


# ------------------------------------------------------------- the analysis
def effects(rows: list[dict], slots: tuple[int, int]) -> dict:
    """Main effects and two-way interactions on log time, from every exact
    run (a balanced 2^4 design: each estimate uses all runs). A ratio below 1
    means faster with the switch on (for slots: with the larger width)."""
    level = lambda r, f: (r["case"][f] == slots[1]) if f == "slots" else bool(r["case"][f])
    good = [r for r in rows if r.get("exact") and r.get("seconds")]
    if not good:
        return {}
    logs = {id(r): math.log(r["seconds"]) for r in good}
    def contrast(sign):
        on = [logs[id(r)] for r in good if sign(r) > 0]
        off = [logs[id(r)] for r in good if sign(r) < 0]
        if not on or not off:
            return None
        return math.exp(statistics.mean(on) - statistics.mean(off))
    out = {"main": {}, "interaction": {}}
    for f in FACTORS:
        out["main"][f] = contrast(lambda r, f=f: 1 if level(r, f) else -1)
    for a, b in itertools.combinations(FACTORS, 2):
        # Time with the two at the same level over time with them at
        # different levels: 1, no interaction; below 1, the pair on together
        # is faster than their two effects alone predict; above 1, slower.
        out["interaction"][f"{a} x {b}"] = contrast(
            lambda r, a=a, b=b: 1 if level(r, a) == level(r, b) else -1)
    return out


def medians(rows: list[dict]) -> dict:
    by = {}
    for r in rows:
        if r.get("exact") and r.get("seconds"):
            by.setdefault(key(r["case"]), []).append(r["seconds"])
    return {k: statistics.median(v) for k, v in by.items()}


def best_case(rows: list[dict], all_cases: list[dict]) -> dict | None:
    m = medians(rows)
    ranked = sorted((m[key(c)], key(c), c) for c in all_cases if key(c) in m)
    return ranked[0][2] if ranked else None


def exactness(rows: list[dict]) -> None:
    """Mark each run exact when every sentence's samples equal those most
    runs gave (greedy decoding: no switch may change a sample)."""
    from collections import Counter
    # Cases with the vocoder on a GPU are compared among themselves.
    groups = {}
    for r in rows:
        groups.setdefault((r.get("case") or {}).get("vocoder", "cpu"), []).append(r)
    for group in groups.values():
        seen = Counter(json.dumps(r["hashes"], sort_keys=True) for r in group if r.get("hashes"))
        reference = json.loads(seen.most_common(1)[0][0]) if seen else None
        for r in group:
            r["exact"] = bool(r.get("hashes")) and r["hashes"] == reference


def confirmation(pairs: list[dict]) -> dict:
    """Best against the current setting, run alternately: each pair's ratio,
    and whether every pair favoured the best combination by at least 3%."""
    ratios = [p["best"] / p["current"] for p in pairs if p.get("best") and p.get("current")]
    return dict(ratios=ratios,
                median_ratio=statistics.median(ratios) if ratios else None,
                confirmed=bool(ratios) and all(x < 0.97 for x in ratios))


# ---------------------------------------------------------------- the runs
def parse_stages(text: str) -> dict:
    return {m.group(1): float(m.group(2)) for m in STAGE_LINE.finditer(text)}


def run_case(case: dict, folder: Path, args, base_env: dict) -> dict:
    cmd = [sys.executable, str(HERE / "check_lecture_cpu.py"), "--home", str(args.home),
           "--embedding", str(args.embedding), "--out", str(folder),
           "--threads", str(case.get("threads", args.threads)), "--slots", str(case["slots"]),
           "--sentences", str(args.sentences), "--mode", "stream", "--warmup-pass",
           "--frames", str(getattr(args, "frames", 4096))]
    if getattr(args, "corpus_file", None):
        cmd += ["--corpus-file", str(args.corpus_file)]
    if args.library_dir:
        cmd += ["--library-dir", str(args.library_dir)]
    if args.model_dir:
        cmd += ["--model-dir", str(args.model_dir)]
    if case["attn_join"]:
        cmd += ["--cpu-mask", str(ATTN_JOIN_BIT), "--cpu-enabled", str(ATTN_JOIN_BIT)]
    env = environment(case, base_env, args.decode_threads)
    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(HERE), env=env, capture_output=True, text=True)
    (folder.parent / (folder.name + ".log")).write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
    row = dict(case=case, folder=str(folder), wall_seconds=time.perf_counter() - started,
               returncode=proc.returncode)
    result = folder / "result.json"
    if proc.returncode == 0 and result.is_file():
        data = json.loads(result.read_text())
        row.update(seconds=data["synthesis_seconds"], rtf=data["real_time_factor"],
                   audio_seconds=data["audio_seconds"],
                   peak_private_gib=data["memory"].get("PeakPagefileUsage", 0) / 2**30,
                   hashes={str(o["corpus_index"]): [o["samples"], o["sha256"]] for o in data["outputs"]},
                   stages=parse_stages(proc.stderr))
    else:
        row["error"] = (proc.stderr or proc.stdout)[-2000:]
    return row


def studio_setting(name: str) -> str:
    """A setting as Studio sees it: the environment, else its .env files
    (RealMeStudio/.env holds the NUC's REALME_RECORD_BATCH=16)."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        sys.path.insert(0, str(HERE))
        from realme.core.env import candidates, parse
        for path in candidates():
            value = parse(path).get(name)
            if value:
                return value
    except Exception:
        pass
    return ""


def default_slots() -> tuple[int, int]:
    try:
        sys.path.insert(0, str(HERE))
        from realme.parbench import total_memory_gb
        total = total_memory_gb() or 16
    except Exception:
        total = 16
    return (8, 16) if total >= 32 else (4, 8)


def default_threads() -> int:
    try:
        from realme.core.cpu import physical_cores
        return max(1, min(6, physical_cores()))
    except Exception:
        return max(1, min(6, (os.cpu_count() or 2) // 2))


def report_text(summary: dict) -> str:
    lines = [f"# CPU switch combinations: {summary['host']}", "",
             f"{summary['threads']} threads; slots {summary['slots'][0]} and {summary['slots'][1]}; "
             f"{summary['sentences']} sentences per case; {summary['repeats']} repeats; greedy.", ""]
    limit = summary.get("frames")
    if limit:
        capped = sorted({k for r in summary["runs"] for k, (n, _) in (r.get("hashes") or {}).items()
                         if n >= (limit - 1) * 1920})
        if capped:
            lines.append(f"Sentences that ran to the {limit}-frame limit (never ended): corpus "
                         + ", ".join(capped) + ". They run alone at the end and dominate the times.")
    bad = [r for r in summary["runs"] if not r.get("exact")]
    lines.append(f"Runs: {len(summary['runs'])}; identical audio in all but {len(bad)}.")
    if bad:
        lines.append("Runs whose audio differed or failed (excluded): "
                     + ", ".join(key(r["case"]) + (" (failed)" if r.get("error") else "") for r in bad))
    lines += ["", "## Median seconds per combination", "", "| Combination | Median s | RTF |", "|---|---:|---:|"]
    m = summary["medians"]
    audio = next((r["audio_seconds"] for r in summary["runs"] if r.get("audio_seconds")), None)
    for k in sorted(m, key=m.get):
        lines.append(f"| {k} | {m[k]:.2f} | {m[k] / audio:.3f} |" if audio else f"| {k} | {m[k]:.2f} | |")
    eff = summary.get("effects") or {}
    if eff:
        lines += ["", "## Effects (time ratio; below 1 is faster)", ""]
        for f, v in eff["main"].items():
            lines.append(f"- {f}: {v:.3f}" if v else f"- {f}: n/a")
        lines += ["", "Two-way interactions (1 is none; below 1 the pair helps more together):", ""]
        for f, v in eff["interaction"].items():
            lines.append(f"- {f}: {v:.3f}" if v else f"- {f}: n/a")
    if summary.get("best"):
        lines += ["", f"## Fastest combination: {key(summary['best'])}"]
    conf = summary.get("confirmation")
    if conf:
        lines += ["", f"Confirmation against the current setting ({key(summary['current'])}): ratios "
                  + ", ".join(f"{x:.3f}" for x in conf["ratios"])
                  + (" -- confirmed (every pair at least 3% faster)." if conf["confirmed"]
                     else " -- not confirmed; keep the current setting.")]
    if summary.get("env_lines"):
        lines += ["", "## For a final lecture render", "",
                  "Put these lines in RealMeStudio/.env, restart Studio, record the same lecture again",
                  "(fresh), and compare its recording time with the earlier one. Remove them to go back.",
                  "", "```"] + summary["env_lines"] + ["```"]
    return "\n".join(lines) + "\n"


def env_lines(case: dict, decode_threads: int, rev: bool = False) -> list[str]:
    if "threads" in case:
        decode_threads = case["decode_threads"] or decode_threads
    lines = [f"QWEN3_TTS_CP_BATCH_ATTN={case['cp_batch_attn']}",
             f"QWEN3_TTS_DECODE_ASYNC={case['decode_async']}",
             f"REALME_RECORD_BATCH={case['slots']}"]
    if case["decode_async"]:
        lines.append(f"QWEN3_TTS_DECODE_THREADS={decode_threads}")
    if case["attn_join"]:
        lines.append("QWEN3_TTS_ATTN_JOIN=1")
        if rev:
            lines.append("# RealMe rev: its profile's option mask overrides this line; set "
                         '"enabled": 4096 in local/active.json instead')
    if case.get("vocoder") == "gpu":
        lines.append("QWEN3_TTS_VOCODER_BACKEND=gpu")
    if "threads" in case:
        lines.append(f"# threads: {case['threads']} ("
                     + ('"threads" in local/active.json' if rev else "QWEN3_TTS_THREADS") + ")")
    return lines


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--home", type=Path, help="engine home (default: realme-rev-qwen3-tts once built, else qwen3cpp)")
    p.add_argument("--library-dir", type=Path)
    p.add_argument("--model-dir", type=Path, help="folder with the Q8 model pair (default: the engine's own)")
    p.add_argument("--embedding", type=Path, help="voice embedding .f32 (default: the newest in the home)")
    p.add_argument("--threads", type=int, default=default_threads())
    p.add_argument("--slots", default=None, help="two widths, e.g. 8,16 (default by memory)")
    p.add_argument("--sentences", type=int, default=16)
    p.add_argument("--long-corpus", action="store_true",
                   help="the harness's 12-entry corpus with four long lecture sentences; under greedy "
                        "decoding one of them never ends and runs to the frame limit alone")
    p.add_argument("--frames", type=int, default=1024,
                   help="per-sentence frame limit (1024 = 82 s); a sentence that reaches it is reported")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--decode-threads", type=int, default=2)
    p.add_argument("--current-slots", type=int,
                   help="sentences per pass Studio uses now (default: REALME_RECORD_BATCH from Studio's .env, else 8)")
    p.add_argument("--confirm", type=int, default=3, help="best-vs-current pairs")
    p.add_argument("--seed", type=int, default=20261101)
    p.add_argument("--out", type=Path)
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--splits", help='thread splits instead of the factorial, e.g. "6+0,6+2,5+1,6+6" '
                   "(transformer threads + vocoder threads beside the pass; +0 decodes in the loop)")
    p.add_argument("--quick", action="store_true",
                   help="a short check that a new engine works: the two new switches off and on "
                        "(4 cases) at Studio's width, 4 sentences, once, identical audio required")
    a = p.parse_args(argv)
    if a.quick:
        a.repeats, a.confirm = 1, 0
        if "--sentences" not in (argv if argv is not None else sys.argv):
            a.sentences = 4
    slots = tuple(int(x) for x in a.slots.split(",")) if a.slots else default_slots()
    assert len(slots) == 2 and 0 < slots[0] < slots[1] <= 16, "--slots takes two widths, smaller first"
    # Studio's setting today: every switch off, at its sentences per pass
    # (--current-slots; else REALME_RECORD_BATCH; else 8 where 8 is tested).
    asked = a.current_slots or int(studio_setting("REALME_RECORD_BATCH") or 0)
    width = asked if asked in slots else (8 if 8 in slots else slots[0])
    if a.quick:
        width = asked if 0 < asked <= 16 else width
    current = dict(cp_batch_attn=0, decode_async=0, attn_join=0, slots=width)
    all_cases = quick_cases(width) if a.quick else cases(slots)
    if a.splits:
        all_cases = split_cases(a.splits, width)
        current = all_cases[0]
    runs = schedule(all_cases, a.repeats, a.seed)
    if a.plan_only:
        print(json.dumps(dict(slots=slots, threads=a.threads, cases=[key(c) for c in all_cases],
                              runs=[(r, key(c)) for r, c in runs], current=key(current)), indent=2))
        return 0
    sys.path.insert(0, str(HERE))
    from realme.engines.cpp import cpp_home
    from realme.engines.rev import cpu_settings
    # RealMe's own voice (tools/realme-rev-qwen3-tts) once Studio has built
    # it; otherwise the free local clone (qwen3cpp).
    from realme.engines.rev import engine_home
    built = engine_home() if (engine_home() / "local" / "engine_active.json").is_file() else None
    a.home = (a.home or built or cpp_home()).resolve()
    if "onedrive" in [x.lower() for x in a.home.parts]:
        raise ValueError("Use the local engine folder, not a OneDrive copy")
    rev = (a.home / "local" / "engine_active.json").is_file() or (a.home / "local" / "active.json").is_file()
    if rev:
        # RealMe rev (tools/realme-rev-qwen3-tts): its selected DLL folder, the
        # converted model pair beside it, the voice embeddings of qwen3cpp.
        from realme.engines.rev_build import library_directory
        from realme.engines.rev import find_models
        a.library_dir = a.library_dir or library_directory(a.home)
        a.model_dir = a.model_dir or find_models(a.home)
    lib = (a.library_dir or a.home / "build")
    # The switches are read from the environment, so an engine without them
    # would run every case the same; its library must name them.
    dll = next((lib / n for n in ("qwen3tts.dll", "libqwen3tts.so", "libqwen3tts.dylib") if (lib / n).is_file()), None)
    if dll is None:
        raise FileNotFoundError(f"No engine library in {lib}")
    blob = dll.read_bytes()
    if b"QWEN3_TTS_CP_BATCH_ATTN" not in blob or b"QWEN3_TTS_DECODE_ASYNC" not in blob:
        raise RuntimeError(f"Rebuild the voice engine first: {dll} predates these switches")
    if a.embedding is None:
        # The voice's own cache first, then the previous engine's.
        folders = [a.home / "embeddings"] + ([a.home.parent / "qwen3cpp" / "embeddings"] if rev else [])
        found = sorted((q for f in folders for q in f.glob("*.f32")), key=lambda q: q.stat().st_mtime)
        found = [q for q in found if q.stat().st_size == 4096]
        if not found:
            raise FileNotFoundError("No voice embedding yet: preview a sentence in your voice in the Studio once, or pass --embedding")
        a.embedding = found[-1]
    out = a.out or a.home / "results" / ("combinations_" + time.strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=False)
    # The NUC sweep's short sentences (eight, cycled): every one ends by
    # itself under greedy decoding, so a case measures passes, not one
    # sentence running on alone to the frame limit.
    a.corpus_file = None
    if not a.long_corpus:
        from check_cache_growth import TEXTS
        a.corpus_file = out / "corpus.json"
        a.corpus_file.write_text(json.dumps(TEXTS[:8], indent=2), encoding="utf-8")
    base = {k: v for k, v in os.environ.items() if not k.startswith("QWEN3_TTS_")}
    base.update({k: v for k, v in cpu_settings(a.threads).items() if v is not None})
    summary = dict(status="running", host=platform.node(), threads=a.threads, slots=list(slots), frames=a.frames,
                   sentences=a.sentences, repeats=a.repeats, decode_threads=a.decode_threads,
                   dll_sha256=sha(dll),
                   embedding_sha256=sha(a.embedding), runner_sha256=sha(__file__),
                   current=current, runs=[], started=time.strftime("%Y-%m-%d %H:%M:%S"))
    dump(out / "summary.json", summary)
    total = len(runs) + 2 * a.confirm
    for i, (r, case) in enumerate(runs, 1):
        folder = out / f"{i:03d}_r{r}_{key(case)}"
        row = run_case(case, folder, a, base)
        row["repeat"] = r
        summary["runs"].append(row)
        exactness(summary["runs"])
        print(f"[{i}/{total}] {key(case)}: "
              + (f"{row['seconds']:.1f} s" if row.get("seconds") else "FAILED"), flush=True)
        dump(out / "summary.json", summary)
    summary["medians"] = medians(summary["runs"])
    summary["effects"] = {} if a.splits else effects(summary["runs"], slots)
    best = best_case(summary["runs"], all_cases)
    summary["best"] = best
    pairs = []
    if best and key(best) != key(current) and a.confirm > 0:
        for j in range(a.confirm):
            pair = {}
            for which, case in (("current", current), ("best", best)) if j % 2 == 0 else \
                    (("best", best), ("current", current)):
                folder = out / f"confirm{j}_{which}_{key(case)}"
                row = run_case(case, folder, a, base)
                row.update(phase="confirm", which=which)
                summary["runs"].append(row)
                exactness(summary["runs"])
                pair[which] = row.get("seconds") if row.get("exact") else None
                print(f"[confirm {j + 1}/{a.confirm}] {which} {key(case)}: "
                      + (f"{row['seconds']:.1f} s" if row.get("seconds") else "FAILED"), flush=True)
            pairs.append(pair)
        summary["confirmation"] = confirmation(pairs)
        if summary["confirmation"]["confirmed"]:
            summary["env_lines"] = env_lines(best, a.decode_threads, rev)
    summary["status"] = "done"
    summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    text = report_text(summary)
    ok = True
    if a.quick:
        ok, verdict = quick_verdict(summary["runs"])
        summary["quick_verdict"] = verdict
        text += "\n## Quick test\n\n" + verdict + \
            "\nTimings from four sentences run once are not a speed result.\n"
    dump(out / "summary.json", summary)
    (out / "REPORT.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"Results: {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
