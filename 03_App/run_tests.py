#!/usr/bin/env python3
"""
Run verify_tree and every test suite; exit 0 only if all pass.

    python 03_App/run_tests.py            live output, then a table
    python 03_App/run_tests.py --json     live output on stderr, then one
                                          JSON object on stdout
    python 03_App/run_tests.py t_agent    just the suites named

For a person, `00_Windows\\_Run_Tests.bat` does the same with the right
interpreter. This exists for a program (a script, CI, an AI agent) that
needs one exit code and one summary it can read without scraping.

An exit code of 1 from a suite means a check failed; anything else means it
crashed. They are reported apart, because a crash read as a failure sends
someone looking for the wrong thing.
"""
from __future__ import annotations
import json, subprocess, sys, time
from pathlib import Path

APP = Path(__file__).resolve().parent

#: Every suite, in the order they run. `verify_tree` checks that this list,
#: the files on disk and `_Run_Tests.bat` agree, so a new suite cannot be
#: written and then never run.
SUITES = ("t_tau", "t_pace", "t_migrate", "t_setup", "t_locate", "t_encoding",
          "t_voices", "t_lexicon", "t_render", "t_engines", "t_gemini",
          "t_agent", "t_tuning", "t_text", "t_inserts", "t_lean", "t_parbench", "t_prerecord", "t_engine_timing", "t_gemparallel",
          "t_writers_more", "t_engine_batch", "t_voice_server",
          # The CPU engine's: integrated source, packages, comparisons and recording checks.
          "t_acoustic_judge", "t_codebook_listening", "t_cpu_combinations", "t_engine_attention", "t_engine_cpu_compute", "t_engine_cpu_runtime",
          "t_engine_decode_async", "t_engine_embedding", "t_engine_embedding_validation", "t_engine_fewer_steps",
          "t_engine_text_mmap", "t_engine_vocoder_storage", "t_engine_weight_mmap", "t_lecture_cpu", "t_native_warmup", "t_rev_engine", "t_runaway", "t_weight_mmap_quality")


def run(name: str, script: Path, out) -> dict:
    t0 = time.perf_counter()
    proc = subprocess.Popen([sys.executable, str(script)], cwd=str(APP),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    fails = []
    for line in proc.stdout:
        out.write(line)
        out.flush()
        s = line.strip()
        if s.startswith(("FAIL ", "[FAIL]")):
            fails.append(s.split(None, 1)[1][:200] if " " in s else s)
    code = proc.wait()
    return {"name": name, "exit_code": code,
            "status": "pass" if code == 0 else "fail" if code == 1 else "crash",
            "failed_checks": fails, "seconds": round(time.perf_counter() - t0, 1)}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in argv
    names = [a for a in argv if not a.startswith("--")]
    unknown = [n for n in names if n not in SUITES and n != "verify_tree"]
    if unknown:
        print(f"Unknown suite(s): {', '.join(unknown)}. Known: "
              f"verify_tree, {', '.join(SUITES)}", file=sys.stderr)
        return 2
    out = sys.stderr if as_json else sys.stdout
    plan = (["verify_tree"] if not names or "verify_tree" in names else []) + \
           [n for n in SUITES if not names or n in names]
    results = []
    for name in plan:
        print(f"\n=== {name} ===", file=out, flush=True)
        r = run(name, APP / f"{name}.py", out)
        results.append(r)
        print(f"    exit={r['exit_code']}", file=out, flush=True)
        if name == "verify_tree" and r["status"] != "pass":
            break                        # fix the tree before the suites
    ok = bool(results) and all(r["status"] == "pass" for r in results)
    summary = {"ok": ok, "python": sys.executable,
               "passed": [r["name"] for r in results if r["status"] == "pass"],
               "failed": [r["name"] for r in results if r["status"] == "fail"],
               "crashed": [r["name"] for r in results if r["status"] == "crash"],
               "not_run": [n for n in plan if n not in {r["name"] for r in results}],
               "suites": results}
    if as_json:
        print(json.dumps(summary, indent=2))
    else:
        print()
        for r in results:
            print(f"  {r['status']:<6} {r['name']:<12} {r['seconds']:>6}s"
                  + (f"  {len(r['failed_checks'])} failed check(s)"
                     if r["failed_checks"] else ""))
        print("\n  " + ("All passed." if ok else "NOT all passed."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
