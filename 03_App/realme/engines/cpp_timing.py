"""
`realme engine-timing` -- where the local clone's time goes, measured.

The local clone (qwen3cpp) turns text into speech in three stages, every
twelfth of a second of audio:

  main model     one step of the 28-layer model ("talker")
  small model    15 small steps that fill in the rest of that slice
                 ("code predictor"), each with a calculation plan built and
                 allocated from scratch -- identical plans, every slice
  vocoder        at the end, the codes are turned into sound

The engine's authors put a detailed stopwatch into the code and left it
switched off. This builds a second copy of the engine with it switched on,
in its own folder (tools/qwen3cpp_timing) -- the engine the Studio uses is
not touched -- and records a few sentences with it.

Upstream stores a thread setting but does not pass it on, so its
calculations run on GGML's default of 4 threads. The copy, like the working
engine, reads QWEN3_TTS_THREADS and applies it, so the measurement shows
how the time changes with the thread count.

Nothing here changes how lectures are recorded. It reports, and keeps the
raw stopwatch output beside the summary.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

#: The thread patch is the one the working engine gets (engines/cpp.py);
#: the timing copy applies it the same way, so the two cannot drift apart.
from realme.engines.cpp import (THREADS_ANCHOR as _THREADS_ANCHOR,  # noqa: E402
                                THREADS_INCLUDE_ANCHOR as _INCLUDE_ANCHOR,
                                apply_broadcast_patch, apply_ffn_patch,
                                apply_quant_patch, apply_snake_patch,
                                apply_path_patch, apply_thread_patch,
                                apply_vocoder_patch)
_GGML_DIR_LINE = 'set(GGML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ggml")'

#: The default thread counts to compare. 4 is GGML's default.
DEFAULT_THREADS = (4, 6, 8, 12)


def timing_home(home: Path | None = None) -> Path:
    from realme.engines.cpp import cpp_home
    home = Path(home or cpp_home())
    return home.parent / (home.name + "_timing")


def library(out: Path) -> Path:
    from realme.adapters.tts_qwen3_dll import library_path
    return library_path(out)


# ------------------------------------------------------------------- build

def patch_sources(out: Path, ggml_dir: Path) -> None:
    """The timing copy's own edits: GGML from the working engine's folder,
    and the thread count applied. Each anchor must be found exactly once."""
    cm = out / "CMakeLists.txt"
    text = cm.read_text(encoding="utf-8")
    if text.count(_GGML_DIR_LINE) != 1:
        raise RuntimeError("CMakeLists.txt no longer sets GGML_DIR where the "
                           "timing build expects it; the vendored engine has "
                           "changed. Nothing was built.")
    cm.write_text(text.replace(
        _GGML_DIR_LINE, f'set(GGML_DIR "{ggml_dir.as_posix()}")'), encoding="utf-8")
    apply_thread_patch(out)
    apply_ffn_patch(out)
    apply_broadcast_patch(out)
    apply_snake_patch(out)
    apply_quant_patch(out)
    apply_vocoder_patch(out)
    apply_path_patch(out)
    apply_profile_patch(out)


#: A stopwatch for the vocoder, which the engine's own timing reports only as
#: one number. With QWEN3_TTS_OP_PROFILE=1 every calculation in the vocoder
#: is run and timed on its own, and the totals by kind (convolution,
#: transposed convolution, matrix multiply, ...) are printed after each
#: sentence. Timing copy only: running each calculation separately is itself
#: slower, so the shares are what matter, not the total.
PROFILE_INCLUDE_ANCHOR = '#include <numeric>\n\n#define QWEN3_TTS_DEC_MAX_NODES 32768\n'
PROFILE_INCLUDE = """#include <numeric>
#include <chrono>
#include <map>
#include <string>
#include <vector>

#define QWEN3_TTS_DEC_MAX_NODES 32768

// RealMe op profile (QWEN3_TTS_OP_PROFILE=1, timing copy only)
namespace {
std::map<std::string, std::pair<double, int>> g_realme_prof;
std::chrono::high_resolution_clock::time_point g_realme_mark;
bool realme_prof_cb(struct ggml_tensor * t, bool ask, void *) {
    auto now = std::chrono::high_resolution_clock::now();
    if (ask) { g_realme_mark = now; return true; }
    auto & e = g_realme_prof[ggml_op_desc(t)];
    e.first += std::chrono::duration<double, std::milli>(now - g_realme_mark).count();
    e.second += 1;
    return true;
}
void realme_prof_print() {
    if (g_realme_prof.empty()) return;
    std::vector<std::pair<double, std::string>> rows;
    double tot = 0;
    for (auto & kv : g_realme_prof) { rows.push_back({kv.second.first, kv.first}); tot += kv.second.first; }
    std::sort(rows.rbegin(), rows.rend());
    fprintf(stderr, "=== Vocoder op profile (%.1f ms) ===\\n", tot);
    for (auto & r : rows) {
        fprintf(stderr, "  op %-22s %9.1f ms %5.1f%%  x%d\\n", r.second.c_str(), r.first,
                tot > 0 ? 100.0 * r.first / tot : 0.0, g_realme_prof[r.second].second);
    }
    g_realme_prof.clear();
}
}
"""
PROFILE_SCHED_ANCHOR = ("    state_.sched = ggml_backend_sched_new(backends.data(), nullptr, "
                        "(int)backends.size(), QWEN3_TTS_DEC_MAX_NODES, false, true);\n")
PROFILE_SCHED = PROFILE_SCHED_ANCHOR + """    if (state_.sched && std::getenv("QWEN3_TTS_OP_PROFILE")) {
        ggml_backend_sched_set_eval_callback(state_.sched, realme_prof_cb, nullptr);
    }
"""
PROFILE_PRINT_ANCHOR = """    int64_t n_samples = audio_tensor->ne[0];
    samples.resize(n_samples);"""
PROFILE_PRINT = """    realme_prof_print();
""" + PROFILE_PRINT_ANCHOR


def apply_profile_patch(tree: Path) -> bool:
    """The vocoder stopwatch, into the timing copy's decoder. Idempotent;
    raises when an anchor is not found exactly once."""
    f = Path(tree) / "src" / "audio_tokenizer_decoder.cpp"
    text = f.read_text(encoding="utf-8")
    if "realme_prof_cb" in text:
        return False
    if (text.count(PROFILE_INCLUDE_ANCHOR) != 1
            or text.count(PROFILE_SCHED_ANCHOR) != 1
            or text.count(PROFILE_PRINT_ANCHOR) != 1):
        raise RuntimeError("audio_tokenizer_decoder.cpp has changed upstream; "
                           "the vocoder stopwatch does not fit.")
    text = (text.replace(PROFILE_INCLUDE_ANCHOR, PROFILE_INCLUDE)
                .replace(PROFILE_SCHED_ANCHOR, PROFILE_SCHED)
                .replace(PROFILE_PRINT_ANCHOR, PROFILE_PRINT))
    f.write_text(text, encoding="utf-8")
    return True


def patched(out: Path) -> bool | None:
    """Does this timing copy carry every change measured here? None when its
    source is not there to look at."""
    try:
        tc = (out / "src" / "tts_transformer.cpp").read_text(encoding="utf-8")
        dc = (out / "src" / "audio_tokenizer_decoder.cpp").read_text(encoding="utf-8")
        ca = (out / "src" / "qwen3tts_c_api.cpp").read_text(encoding="utf-8")
    except OSError:
        return None
    from realme.engines.cpp import (BCAST_MARK, FFN_MARK, QUANT_MARK, SNAKE_MARK,
                                    VOC_MARK)
    return (FFN_MARK in tc and "ggml_is_quantized" in tc and BCAST_MARK in dc
            and SNAKE_MARK in dc and VOC_MARK in dc and "realme_prof_cb" in dc
            and QUANT_MARK in ca)


def _settings(home: Path) -> dict:
    """What the working engine was built for, so the copy matches it."""
    try:
        info = json.loads((home / "build_info.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        info = {}
    return {"cpu": info.get("cpu") or "avx2", "gpu": info.get("gpu") or "off"}


NO_COMPILER = """There is no C++ compiler on this computer, so the stopwatch copy of
  the engine cannot be built here. Two ways on:

  1. On the computer that built the engine (it has the compiler), run
         _Engine_Timing.bat --build-only
     then copy its folder  tools\\qwen3cpp_timing  into this computer's
     RealMe\\tools  folder, beside qwen3cpp, and run _Engine_Timing.bat here
     again. It is built for the same processors as your engine ({cpu}).

  2. Or install a compiler here -- Visual Studio Build Tools, "Desktop
     development with C++" -- and run _Engine_Timing.bat again.

  ({detail})"""


def build(home: Path | None = None, log=print, force: bool = False) -> Path:
    """Build the timing copy of the engine. Returns its shared library.

    Two routes, the same result. Where the engine was built on this computer,
    its staged source and its GGML build are reused (minutes). Where it came
    in a migration package -- binaries and weights only -- the copy is built
    from the source vendored inside RealMe, GGML included, for the same
    processor target the engine was built for.
    """
    from realme.engines import cpp
    home = Path(home or cpp.cpp_home())
    out = timing_home(home)
    lib = library(out)
    if lib.is_file() and not force:
        # A copy brought from another computer may come without its source;
        # it is used as it is. One whose source predates the changes measured
        # here is built again.
        if patched(out) is not False:
            log(f"  timing engine already built: {lib}")
            return lib
        log("  the timing engine predates the speed changes; building it again")
        force = True
    settings = _settings(home)
    kind, detail = cpp.find_compiler()
    if kind == "none":
        raise RuntimeError(NO_COMPILER.format(cpu=settings["cpu"], detail=detail))
    cmake = cpp.find_cmake(log)

    reuse = (home / "CMakeLists.txt").is_file() and (home / "ggml" / "build").is_dir()
    # The source always comes from the copy inside RealMe, untouched, so
    # this RealMe's changes are the ones applied: the working engine's
    # staged source carries whatever the RealMe that last built it applied.
    # Only its GGML build is reused.
    vendored = cpp.source_dir()
    src_root = vendored if (vendored / "CMakeLists.txt").is_file() else home
    if not (src_root / "CMakeLists.txt").is_file():
        raise RuntimeError(f"The engine's source is not in {src_root}; this "
                           f"RealMe installation is incomplete. Run 0_Update.bat.")
    if force and out.exists():
        shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    log(f"  copying the engine's source into {out}"
        + (" (from the copy inside RealMe)" if src_root == vendored else ""))
    for name in ("CMakeLists.txt", "src", "tests", "main.cpp", "qwen3_tts.cpp",
                 "qwen3_tts.h"):
        src = src_root / name
        if src.is_dir():
            shutil.copytree(src, out / name, dirs_exist_ok=True)
        elif src.is_file():
            shutil.copy2(src, out / name)

    env, generator = None, []
    if os.name == "nt":
        generator = ["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={cpp.find_ninja(log)}"]
        if kind == "msvc":
            env = cpp.msvc_env(detail, log)
    elif kind == "msvc":
        env = cpp.msvc_env(detail, log)
        generator = ["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={cpp.find_ninja(log)}"]
    j = str(max(1, os.cpu_count() or 2))

    if reuse:
        ggml = home / "ggml"
    else:
        ggml = cpp.fetch_ggml(out, log)
        try:
            if cpp.apply_sgemm_patch(ggml):
                shutil.rmtree(ggml / "build", ignore_errors=True)
            sgemm = [cpp.SGEMM_FLAG]
        except RuntimeError:
            sgemm = ["-DGGML_LLAMAFILE=OFF"]
        cpp.normalise_mtimes(out, log)
        # A build folder that came along with the source was configured
        # somewhere else; CMake refuses it. Start it over here.
        cache = ggml / "build" / "CMakeCache.txt"
        if cache.is_file() and str(ggml.resolve()).replace("\\", "/").lower() \
                not in cache.read_text(encoding="utf-8", errors="ignore") \
                .replace("\\", "/").lower():
            shutil.rmtree(ggml / "build", ignore_errors=True)
        log(f"  building GGML for {settings['cpu']} processors, CPU only "
            f"(a few minutes)")
        cpp._run([cmake, "-S", str(ggml), "-B", str(ggml / "build"), *generator,
                  "-DCMAKE_BUILD_TYPE=Release", "-DGGML_METAL=OFF",
                  *cpp.CPU_TARGETS.get(settings["cpu"], cpp.CPU_TARGETS["avx2"]),
                  *sgemm, "-DGGML_BUILD_TESTS=OFF", "-DGGML_BUILD_EXAMPLES=OFF"],
                 "ggml configure", log, env=env)
        cpp._run([cmake, "--build", str(ggml / "build"), "--config", "Release",
                  "-j", j], "ggml build", log, env=env)
    patch_sources(out, ggml)

    extra = ["-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=ON"] if os.name == "nt" else []
    log("  configuring the timing engine (stopwatch on, threads applied)")
    cpp._run([cmake, "-S", str(out), "-B", str(out / "build"), *generator,
              "-DCMAKE_BUILD_TYPE=Release", "-DQWEN3_TTS_TIMING=ON", *extra],
             "timing configure", log, env=env)
    log("  building it")
    cpp._run([cmake, "--build", str(out / "build"), "--config", "Release",
              "--target", "qwen3tts_shared", "-j", j],
             "timing build", log, env=env)
    if os.name == "nt":
        dest = library(out).parent
        for d in (ggml / "build").rglob("*.dll"):
            shutil.copy2(d, dest / d.name)
    lib = library(out)
    if not lib.is_file():
        raise RuntimeError(f"The build reported success but {lib} is not there.")
    (out / "build_info.json").write_text(json.dumps(
        {**settings, "gpu": "off", "timing": True, "threads_applied": True},
        indent=2), encoding="utf-8")
    log(f"  built: {lib}")
    return lib


# ----------------------------------------------------------------- parsing

_NUM = r"([0-9]+(?:\.[0-9]+)?)"


def parse(stderr_text: str) -> list[dict]:
    """One record per sentence from the engine's own stopwatch output.

    Each sentence prints a detailed generation block, then the overall
    "Timing:" block. Sections repeat the same labels ("Graph build"), so the
    section a line sits in is tracked.
    """
    out: list[dict] = []
    cur: dict | None = None
    section = ""
    for raw in stderr_text.splitlines():
        line = raw.strip()
        m = re.match(r"=== Detailed Generation Timing \((\d+) frames\) ===", line)
        if m:
            cur = {"frames": int(m.group(1))}
            out.append(cur)
            section = ""
            continue
        if cur is None:
            continue
        if line.startswith("Prefill:"):
            section = "prefill"; continue
        if line.startswith("Talker forward_step"):
            section = "talker"; continue
        if line.startswith("Code predictor"):
            section = "codepred"; continue
        if line.startswith("Timing:"):
            section = "overall"; continue
        m = re.match(r"([A-Za-z/ ()0-9]+?):\s+" + _NUM + r"\s*(ms|s)\b", line)
        if not m:
            continue
        label = m.group(1).strip().lower()
        value = float(m.group(2)) * (1000.0 if m.group(3) == "s" else 1.0)
        key = {
            "embed lookups": "embed_ms", "other/overhead": "other_ms",
            "total generate": "generate_ms",
        }.get(label)
        if key:
            cur[key] = value
            continue
        if section in ("prefill", "talker", "codepred"):
            name = {"build graph": "build_graph", "forward total": "forward",
                    "total": "total", "graph build": "graph_build",
                    "graph alloc": "graph_alloc", "compute": "compute",
                    "data i/o": "data", "init/kv/embed": "init",
                    "prefill (2tok)": "prefill", "steps (14)": "steps",
                    "coreml total": "coreml"}.get(label)
            if name:
                cur[f"{section}_{name}_ms"] = value
        elif section == "overall":
            name = {"tokenization": "tokenize_ms",
                    "speaker encode": "encode_ms",
                    "code generation": "codegen_ms",
                    "vocoder decode": "vocoder_ms", "total": "total_ms",
                    "audio duration": "audio_ms"}.get(label)
            if name:
                cur[name] = value
    return [r for r in out if "total_ms" in r and r.get("audio_ms")]


def shares(records: list[dict]) -> dict:
    """Where the time went, as fractions of the whole, summed over sentences."""
    tot = sum(r["total_ms"] for r in records) or 1.0
    g = lambda k: sum(r.get(k, 0.0) for r in records)
    replan = (g("talker_graph_build_ms") + g("talker_graph_alloc_ms")
              + g("codepred_graph_build_ms") + g("codepred_graph_alloc_ms")
              + g("prefill_graph_build_ms") + g("prefill_graph_alloc_ms")
              + g("prefill_build_graph_ms"))
    parts = {
        "vocoder": g("vocoder_ms"),
        "main_compute": g("talker_compute_ms"),
        "main_replan": g("talker_graph_build_ms") + g("talker_graph_alloc_ms"),
        "small_compute": g("codepred_compute_ms"),
        "small_replan": g("codepred_graph_build_ms") + g("codepred_graph_alloc_ms"),
        "small_data": g("codepred_data_ms") + g("codepred_init_ms"),
        "prefill": g("prefill_build_graph_ms") + g("prefill_forward_ms"),
    }
    parts["other"] = max(0.0, tot - sum(parts.values()))
    return {"total_ms": tot, "audio_ms": g("audio_ms"),
            "rtf": tot / (g("audio_ms") or 1.0), "replan_share": replan / tot,
            **{k: v / tot for k, v in parts.items()}}


# --------------------------------------------------------------- measuring

def _worker(spec: dict) -> int:
    """One engine process at one thread count: load, warm up, record, report.
    The engine's stopwatch goes to stderr; the parent keeps that file."""
    os.environ["QWEN3_TTS_THREADS"] = str(spec["threads"])
    for k, v in (spec.get("env") or {}).items():
        os.environ[k] = str(v)
    from realme.adapters import tts_qwen3_dll as D
    lib = Path(spec["library"])
    D.library_path = lambda home: lib        # the timing copy, this process only
    from realme.adapters.factory import build as make
    ref = Path(spec["reference"]) if spec.get("reference") else None
    tts = make("qwen3cpp", reference_wav=ref, threads=spec["threads"])
    tts.preflight()
    outdir = Path(spec["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)
    from realme.text.lexicon import Lexicon
    from realme.text.prepare import prepare_for_speech
    lx = Lexicon()
    walls = []
    tag = spec.get("tag") or f"t{spec['threads']}"
    if spec.get("identity"):
        # One sentence with sampling off, so two builds can be compared
        # sample for sample: the same numbers must give the same sound.
        keep = tts.temperature
        tts.temperature = 0.0
        print("=== identity ===", file=sys.stderr, flush=True)
        tts.synthesize(prepare_for_speech(spec["identity"], "plain", lx)
                       .engine_text, outdir / f"identity_{tag}.wav")
        tts.temperature = keep
    for i, text in enumerate([spec["sentences"][0]] + spec["sentences"]):
        spoken = prepare_for_speech(text, "plain", lx).engine_text
        print(f"=== sentence {i} ===", file=sys.stderr, flush=True)
        t0 = time.perf_counter()
        tts.synthesize(spoken, outdir / f"{tag}_{i:02d}.wav")
        walls.append(time.perf_counter() - t0)
    print(json.dumps({"walls": walls}))
    return 0


def measure(threads=DEFAULT_THREADS, sentences: int = 3, *, outdir: Path,
            reference: Path | None, library_path: Path, log=print) -> dict:
    from realme.bench import DEFAULT_SCRIPT
    texts = (list(DEFAULT_SCRIPT) * 3)[:max(1, sentences)]
    outdir.mkdir(parents=True, exist_ok=True)
    results = {}
    for n in threads:
        log(f"\n  {n} threads: loading the timing engine and recording "
            f"{len(texts)} sentences (plus one to warm up) ...")
        spec = {"threads": n, "library": str(library_path),
                "reference": str(reference) if reference else "",
                "sentences": texts, "outdir": str(outdir)}
        err = outdir / f"stopwatch_{n}threads.txt"
        with open(err, "w", encoding="utf-8") as fe:
            proc = subprocess.run(
                [sys.executable, "-m", "realme.engines.cpp_timing", "--worker",
                 json.dumps(spec)],
                stdout=subprocess.PIPE, stderr=fe, text=True,
                encoding="utf-8", errors="replace")
        raw = err.read_text(encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            tail = "\n".join(raw.strip().splitlines()[-6:])
            log(f"    did not finish:\n{tail}")
            results[n] = {"error": tail}
            continue
        recs = parse(raw)[1:]                    # the first is the warm-up
        applied = f"CPU threads: {n}" in raw
        if not recs:
            log("    no stopwatch output found -- was the timing build used?")
            results[n] = {"error": "no stopwatch output", "applied": applied}
            continue
        s = shares(recs)
        results[n] = {"applied": applied, "sentences": recs, **s}
        log(f"    {s['total_ms'] / 1000:.0f}s for {s['audio_ms'] / 1000:.0f}s of "
            f"speech: {s['rtf']:.2f} times real time"
            + ("" if applied else "  (WARNING: thread count not confirmed)"))
    return results


def report(results: dict) -> tuple[str, dict]:
    ok = {n: r for n, r in results.items() if "rtf" in r}
    if not ok:
        return "No thread count produced a measurement.", {}
    base_n = 4 if 4 in ok else min(ok)
    b = ok[base_n]
    pct = lambda x: f"{100 * x:4.0f}%"
    lines = [f"Where the time goes ({base_n} threads):",
             f"  turning codes into sound (vocoder)      {pct(b['vocoder'])}",
             f"  main model, once per slice:  computing  {pct(b['main_compute'])}",
             f"                               re-planning{pct(b['main_replan'])}",
             f"  small model, 15 per slice:   computing  {pct(b['small_compute'])}",
             f"                               re-planning{pct(b['small_replan'])}",
             f"                               moving data{pct(b['small_data'])}",
             f"  reading the sentence in (prefill)       {pct(b['prefill'])}",
             f"  everything else                         {pct(b['other'])}",
             "",
             f"Rebuilding the same calculation plans: {pct(b['replan_share']).strip()} "
             f"of the time.", "", "By thread count:"]
    for n in sorted(ok):
        r = ok[n]
        lines.append(f"  {n:>2} threads: {r['rtf']:.2f} times real time"
                     + (f"  ({b['rtf'] / r['rtf']:.2f}x vs {base_n})"
                        if n != base_n else "  (today)"))
    best_n = min(ok, key=lambda n: ok[n]["rtf"])
    gain = b["rtf"] / ok[best_n]["rtf"]
    verdict = {"replan_share": round(b["replan_share"], 3),
               "vocoder_share": round(b["vocoder"], 3),
               "best_threads": best_n, "thread_gain": round(gain, 2)}
    lines.append("")
    if b["replan_share"] >= 0.15:
        lines.append(f"Reusing the plans could save up to about "
                     f"{100 * b['replan_share']:.0f}% -- worth building.")
    elif b["replan_share"] >= 0.05:
        lines.append(f"Reusing the plans could save up to about "
                     f"{100 * b['replan_share']:.0f}% -- modest.")
    else:
        lines.append("Re-planning is a small share; reusing the plans would "
                     "not be worth the change.")
    if gain >= 1.10:
        lines.append(f"{best_n} threads were {gain:.2f} times as fast as 4. "
                     f"Applying the thread count in the real engine is worth "
                     f"doing (one small change, then rebuild).")
    else:
        lines.append("More threads did not help much even when applied; the "
                     "time is in steps too small to share out.")
    lines.append("Measured once per setting on a few sentences: treat "
                 "differences under about 10% as noise.")
    return "\n".join(lines), verdict


# --------------------------------------------------------------- comparing

#: The run-time switches for the changes, all off ("before") or all on.
CHANGES = {"QWEN3_TTS_FFN_F32": "ffn_down converted once (both models)",
           "QWEN3_TTS_BROADCAST": "no per-channel copies (vocoder)",
           "QWEN3_TTS_FAST_SNAKE": "the vocoder's snake in one pass ("
                                   "'exact': library sine; '1': RealMe's)",
           "QWEN3_TTS_VOCODER_F32": "the vocoder's convolutions in 32 bits, "
                                    "by matrix multiply (tinyBLAS)"}
IDENTITY_TEXT = ("Confounding is a property of the question, not of the data, "
                 "and no amount of data can answer a question that was never "
                 "asked.")


def parse_ops(stderr_text: str) -> dict:
    """The vocoder's time by kind of calculation, summed over the measured
    sentences (the identity sentence and the warm-up are left out)."""
    text = stderr_text.split("=== sentence 1 ===", 1)[-1]
    ops: dict = {}
    for m in re.finditer(r"^\s*op (\S+)\s+" + _NUM + r" ms", text, re.M):
        ops[m.group(1)] = ops.get(m.group(1), 0.0) + float(m.group(2))
    return ops


def same_audio(a: Path, b: Path) -> bool | None:
    """Sample-for-sample equal? None when either file is missing."""
    import wave
    try:
        with wave.open(str(a)) as fa, wave.open(str(b)) as fb:
            return (fa.getparams()[:3] == fb.getparams()[:3]
                    and fa.readframes(fa.getnframes()) == fb.readframes(fb.getnframes()))
    except (OSError, EOFError, wave.Error):
        return None


#: What each run switches on. "before" and "after" must sound identical;
#: the two one-pass snakes are measured against "after".
#: Every variant states every switch, so a setting saved for lectures
#: (`realme key`, Twin Setup) cannot leak into the one it is compared with.
_BASE = {"REALME_QWEN3CPP_QUANT": "f16", "QWEN3_TTS_VOCODER_F32": "0"}
VARIANTS = (
    ("before", {**_BASE, "QWEN3_TTS_FFN_F32": "0", "QWEN3_TTS_BROADCAST": "0",
                "QWEN3_TTS_FAST_SNAKE": "0"}),
    ("after", {**_BASE, "QWEN3_TTS_FFN_F32": "1", "QWEN3_TTS_BROADCAST": "1",
               "QWEN3_TTS_FAST_SNAKE": "0"}),
    ("snake", {**_BASE, "QWEN3_TTS_FFN_F32": "1", "QWEN3_TTS_BROADCAST": "1",
               "QWEN3_TTS_FAST_SNAKE": "exact"}),
    ("fast", {**_BASE, "QWEN3_TTS_FFN_F32": "1", "QWEN3_TTS_BROADCAST": "1",
              "QWEN3_TTS_FAST_SNAKE": "1"}),
    ("voc", {**_BASE, "QWEN3_TTS_FFN_F32": "1", "QWEN3_TTS_BROADCAST": "1",
             "QWEN3_TTS_FAST_SNAKE": "1", "QWEN3_TTS_VOCODER_F32": "1"}),
    ("q8", {**_BASE, "QWEN3_TTS_FFN_F32": "1", "QWEN3_TTS_BROADCAST": "1",
            "QWEN3_TTS_FAST_SNAKE": "1", "REALME_QWEN3CPP_QUANT": "q8_0"}),
)
#: Runs when none are named: everything but the 8-bit model and the 32-bit
#: vocoder, which are asked for by name (--variants fast,q8 / fast,voc).
DEFAULT_VARIANTS = ("before", "after", "snake", "fast")


def pick_variants(names) -> tuple:
    known = dict(VARIANTS)
    names = [n.strip() for n in (names or DEFAULT_VARIANTS) if n.strip()]
    bad = [n for n in names if n not in known]
    if bad:
        raise ValueError(f"unknown: {', '.join(bad)}; known: {', '.join(known)}")
    return tuple((n, known[n]) for n in names)


def audio_difference(a: Path, b: Path) -> dict | None:
    """How far apart two recordings of the same codes are: the largest
    difference in 16-bit steps, the share of samples that differ, and the
    difference's level below the speech in dB. None if either is missing."""
    import math
    import wave
    from array import array
    try:
        with wave.open(str(a)) as fa, wave.open(str(b)) as fb:
            x, y = array("h"), array("h")
            x.frombytes(fa.readframes(fa.getnframes()))
            y.frombytes(fb.readframes(fb.getnframes()))
    except (OSError, EOFError, wave.Error):
        return None
    if len(x) != len(y) or not x:
        return {"identical": False, "max_step": None, "differ_pct": None,
                "snr_db": None, "lengths": [len(x), len(y)]}
    diff = [p - q for p, q in zip(x, y)]
    err = sum(d * d for d in diff)
    sig = sum(p * p for p in x) or 1
    return {"identical": err == 0,
            "max_step": max(abs(d) for d in diff),
            "differ_pct": round(100.0 * sum(1 for d in diff if d) / len(diff), 2),
            "snr_db": None if err == 0 else round(10 * math.log10(sig / err), 1)}


def _run_variant(tag: str, env: dict, threads: int, texts, outdir: Path,
                 reference: Path | None, library_path: Path, log) -> dict:
    spec = {"threads": threads, "library": str(library_path), "tag": tag,
            "reference": str(reference) if reference else "",
            "sentences": texts, "outdir": str(outdir),
            "identity": IDENTITY_TEXT,
            "env": {**env, "QWEN3_TTS_OP_PROFILE": "1"}}
    err = outdir / f"stopwatch_{tag}.txt"
    with open(err, "w", encoding="utf-8") as fe:
        proc = subprocess.run(
            [sys.executable, "-m", "realme.engines.cpp_timing", "--worker",
             json.dumps(spec)],
            stdout=subprocess.PIPE, stderr=fe, text=True,
            encoding="utf-8", errors="replace")
    raw = err.read_text(encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        tail = "\n".join(raw.strip().splitlines()[-6:])
        log(f"    did not finish:\n{tail}")
        return {"error": tail}
    recs = parse(raw)[2:]          # identity and warm-up left out
    if not recs:
        log("    no stopwatch output found")
        return {"error": "no stopwatch output"}
    s = shares(recs)
    log(f"    {s['rtf']:.2f} seconds of work per second of speech")
    return {"sentences": recs, "vocoder_ops": parse_ops(raw),
            "switched_on": "F32 copy: on" in raw,
            "vocoder_f32": "vocoder F32: on" in raw,
            "tinyblas": "tinyBLAS yes" in raw, **s}


def compare(threads: int, sentences: int = 4, *, outdir: Path,
            reference: Path | None, library_path: Path, log=print,
            variants=VARIANTS) -> dict:
    """The same engine with the changes off, on, and with each one-pass
    snake. One sentence is recorded with sampling off in every run, so the
    runs can be compared sample by sample."""
    from realme.bench import DEFAULT_SCRIPT
    texts = (list(DEFAULT_SCRIPT) * 3)[:max(1, sentences)]
    outdir.mkdir(parents=True, exist_ok=True)
    results = {}
    names = {"before": "the changes off", "after": "the exact changes on",
             "snake": "plus the one-pass snake, library sine",
             "fast": "plus the one-pass snake, RealMe's sine",
             "voc": "as 'fast', with the 32-bit vocoder",
             "q8": "as 'fast', with the 8-bit model"}
    for tag, env in variants:
        log(f"\n  {tag}: {threads} threads, {names.get(tag, tag)}; "
            f"{len(texts)} sentences plus one to warm up and one to compare ...")
        results[tag] = _run_variant(tag, env, threads, texts, outdir,
                                    reference, library_path, log)
    if "before" in results and "after" in results:
        results["identical"] = same_audio(outdir / "identity_before.wav",
                                          outdir / "identity_after.wav")
    for tag in ("snake", "fast"):
        if tag in results and "after" in results:
            results[f"{tag}_vs_after"] = audio_difference(
                outdir / "identity_after.wav", outdir / f"identity_{tag}.wav")
    if "voc" in results and "fast" in results:
        results["voc_vs_fast"] = audio_difference(
            outdir / "identity_fast.wav", outdir / "identity_voc.wav")
    return results


def compare_report(res: dict) -> tuple[str, dict]:
    order = ("before", "after", "snake", "fast", "voc", "q8")
    cols = [(t, res[t]) for t in order if "rtf" in res.get(t, {})]
    if not cols:
        return "The comparison did not finish; see the stopwatch files.", {}
    part = lambda r, k: r["total_ms"] * r[k] / (r["audio_ms"] or 1.0)
    rows = [("small model", "small_compute"), ("main model", "main_compute"),
            ("vocoder", "vocoder")]
    lines = ["Seconds of work per second of speech:",
             f"  {'':14}" + "".join(f"{t:>8}" for t, _ in cols)]
    for name, k in rows:
        lines.append(f"  {name:14}" + "".join(f"{part(r, k):8.2f}" for _, r in cols))
    lines.append(f"  {'all of it':14}" + "".join(f"{r['rtf']:8.2f}" for _, r in cols))
    verdict: dict = {}
    b, a = res.get("before", {}), res.get("after", {})
    if "rtf" in b and "rtf" in a:
        speed = b["rtf"] / a["rtf"]
        lines.append("")
        lines.append(f"The changes make recording {speed:.2f} times as fast "
                     f"({100 * (1 - a['rtf'] / b['rtf']):.0f}% less time).")
        same = res.get("identical")
        lines.append("The comparison sentence came out " + (
            "identical, sample for sample." if same else
            "DIFFERENT -- do not rebuild the voice engine; send this file."
            if same is False else "-- could not be compared."))
        ops = a.get("vocoder_ops") or {}
        if ops:
            tot = sum(ops.values()) or 1.0
            lines += ["", "Inside the vocoder (after), by kind of calculation:"]
            for op, ms in sorted(ops.items(), key=lambda kv: -kv[1])[:6]:
                lines.append(f"  {op:22}{100 * ms / tot:5.0f}%")
        lines.append("")
        if same and speed >= 1.05:
            lines.append("Worth it: run _Rebuild_Voice_Engine.bat to give the "
                         "voice engine these changes.")
        elif same:
            lines.append("Little or no gain on this computer; rebuilding is "
                         "optional.")
        verdict.update({"speedup": round(speed, 3), "identical": same,
                        "rtf_before": round(b["rtf"], 3),
                        "rtf_after": round(a["rtf"], 3)})
    for tag, label, value in (("snake", "with the library's sine", "exact"),
                              ("fast", "with RealMe's sine", "1")):
        r, d = res.get(tag, {}), res.get(f"{tag}_vs_after")
        if "rtf" not in r or "rtf" not in a or not d:
            continue
        gain = a["rtf"] / r["rtf"]
        verdict[f"{tag}_speedup"] = round(gain, 3)
        verdict[f"{tag}_difference"] = d
        lines.append("")
        lines.append(f"One-pass snake {label}: {gain:.2f} times as fast as "
                     f"'after' ({100 * (1 - r['rtf'] / a['rtf']):.0f}% less "
                     f"time).")
        if d.get("identical"):
            lines.append("  Identical to 'after', sample for sample.")
        elif d.get("max_step") is not None:
            lines.append(f"  Differs from 'after' by at most {d['max_step']} "
                         f"steps of 32768 ({d['differ_pct']}% of samples), "
                         f"{d['snr_db']} dB below the speech.")
            lines.append(f"  Listen: identity_after.wav and identity_{tag}.wav "
                         f"in the results folder.")
        else:
            lines.append("  The comparison sentence came out a different "
                         "length; do not use it -- send this file.")
        if (d.get("identical") or (d.get("snr_db") or 0) >= 50) and gain >= 1.02:
            lines.append(f"  To use it, in 5_Command_Prompt.bat:  realme key "
                         f"QWEN3_TTS_FAST_SNAKE {value}"
                         + ("" if d.get("identical") else
                            "   (if you cannot hear a difference)"))
    v, fa, dv = res.get("voc", {}), res.get("fast", {}), res.get("voc_vs_fast")
    if "rtf" in v and "rtf" in fa:
        gain = fa["rtf"] / v["rtf"]
        vg = part(fa, "vocoder") / (part(v, "vocoder") or 1e-9)
        verdict.update({"voc_speedup": round(gain, 3),
                        "voc_vocoder_speedup": round(vg, 3),
                        "voc_tinyblas": v.get("tinyblas"),
                        "voc_difference": dv})
        lines.append("")
        if not v.get("vocoder_f32"):
            lines.append("The 32-bit vocoder did not switch on in this engine; "
                         "run _Rebuild_Voice_Engine.bat, then this again.")
        else:
            lines.append(f"The 32-bit vocoder: the vocoder {vg:.2f} times as "
                         f"fast, recording {gain:.2f} times as fast as 'fast' "
                         f"({100 * (1 - v['rtf'] / fa['rtf']):.0f}% less time).")
            if not v.get("tinyblas"):
                lines.append("  Without GGML's tinyBLAS here: run "
                             "_Rebuild_Voice_Engine.bat for the full gain.")
            if dv and dv.get("max_step") is not None and not dv.get("identical"):
                lines.append(f"  Differs from 'fast' by at most {dv['max_step']} "
                             f"steps of 32768 ({dv['differ_pct']}% of samples), "
                             f"{dv['snr_db']} dB below the speech: the same "
                             f"convolutions, rounded differently.")
                lines.append("  Listen: identity_fast.wav and identity_voc.wav, "
                             "and the voc_*.wav sentences against fast_*.wav.")
            elif dv and dv.get("identical"):
                lines.append("  Identical to 'fast' -- unexpected; send this file.")
            else:
                lines.append("  The comparison sentence came out a different "
                             "length; do not use it -- send this file.")
            if dv and (dv.get("snr_db") or 0) >= 40 and gain >= 1.03:
                lines.append("  If you cannot hear a difference, to use it, in "
                             "5_Command_Prompt.bat:  realme key "
                             "QWEN3_TTS_VOCODER_F32 1   (off: ... 0)")
    q = res.get("q8", {})
    if "rtf" in q:
        base_tag = next((t for t in ("fast", "snake", "after") if "rtf" in res.get(t, {})), None)
        lines.append("")
        if base_tag:
            base = res[base_tag]
            gain = base["rtf"] / q["rtf"]
            verdict["q8_speedup"] = round(gain, 3)
            verdict["q8_against"] = base_tag
            lines.append(f"The 8-bit model: {gain:.2f} times as fast as "
                         f"'{base_tag}' ({100 * (1 - q['rtf'] / base['rtf']):.0f}% "
                         f"less time).")
            lines.append(f"  Not comparable sample by sample: it is a different "
                         f"model, so it reads each sentence its own way. "
                         f"Listen: identity_{base_tag}.wav and identity_q8.wav, "
                         f"and the q8_*.wav sentences against {base_tag}_*.wav.")
            if gain >= 1.05:
                lines.append("  If it sounds as good, to use it, in "
                             "5_Command_Prompt.bat:  realme key "
                             "REALME_QWEN3CPP_QUANT q8_0   (back: ... f16)")
            else:
                lines.append("  Little or no gain on this computer; keep the "
                             "16-bit model.")
        else:
            lines.append(f"The 8-bit model: {q['rtf']:.2f} seconds of work per "
                         f"second of speech (run it beside 'fast' to compare).")
    lines.append("")
    lines.append("Measured once each on a few sentences: treat differences "
                 "under about 5% as noise.")
    if not verdict:
        verdict = {"measured": [t for t, _ in cols]}
    return "\n".join(lines), verdict


def compare_cli(threads: int, sentences: int, rebuild: bool, outdir: Path,
                reference: Path | None, log=print, variants=None) -> int:
    chosen = pick_variants(variants)
    lib = build(force=rebuild, log=log)
    if any(t == "q8" for t, _ in chosen):
        from realme.engines import cpp
        cpp.make_q8(library=lib, log=log)
    res = compare(threads, sentences, outdir=outdir, reference=reference,
                  library_path=lib, log=log, variants=chosen)
    text, verdict = compare_report(res)
    log("\n" + text)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    f = outdir / f"compare_{stamp}.json"
    f.write_text(json.dumps({"when": stamp, "cores": os.cpu_count(),
                             "threads": threads, "changes": CHANGES,
                             "variants": dict(chosen),
                             "results": res, "verdict": verdict,
                             "summary": text}, indent=2), encoding="utf-8")
    log(f"\n  results: {f}")
    return 0 if verdict else 1


def main_cli(threads, sentences: int, rebuild: bool, outdir: Path,
             reference: Path | None, log=print, build_only: bool = False) -> int:
    lib = build(force=rebuild, log=log)
    if build_only:
        log(f"\n  Built, not measured. To measure on another computer, copy\n"
            f"    {timing_home()}\n  into that computer's RealMe tools folder "
            f"and run _Engine_Timing.bat there.")
        return 0
    res = measure(threads, sentences, outdir=outdir, reference=reference,
                  library_path=lib, log=log)
    text, verdict = report(res)
    log("\n" + text)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    f = outdir / f"timing_{stamp}.json"
    f.write_text(json.dumps({"when": stamp, "cores": os.cpu_count(),
                             "results": {str(k): v for k, v in res.items()},
                             "verdict": verdict, "summary": text}, indent=2),
                 encoding="utf-8")
    log(f"\n  results: {f}\n  raw stopwatch output and audio: {outdir}")
    return 0 if verdict else 1


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        sys.exit(_worker(json.loads(sys.argv[2])))
    sys.exit("Run this as: realme engine-timing")
