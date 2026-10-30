#!/usr/bin/env python3
"""
Text written to disk must say which encoding it is.

A render died on a deck containing an arrow. `Path.write_text(...)` with no
encoding uses the platform default, which on Windows is cp1252 -- so a
manifest holding "→", an en dash, a curly quote or "β" crashed at the moment
of writing, after the drafting work was done. None of those are exotic in an
academic deck.

The fix is one keyword argument in forty places, which means the real risk is
the forty-first. This checks the whole package rather than the places that
happened to break.
"""
from __future__ import annotations
import ast, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

#: Where this file lives. Every path below is relative to it, never to the
#: working directory: these suites are run from the project root as often as
#: from 03_App, and `Path("realme/...")` silently means a different thing in
#: each. `verify_tree.py` already anchored on __file__; the t_*.py files did
#: not, so t_render died with FileNotFoundError when run from the root.
HERE = Path(__file__).resolve().parent

FAILED = []
SKIP = ("qwen3cpp", "ov", "knnvc", "__pycache__")


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}"
          + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def offenders():
    """Every text read or write in the package that does not name an encoding."""
    out = []
    for p in sorted(Path(HERE / "realme").rglob("*.py")):
        if any(x in p.parts for x in SKIP):
            continue
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr in {"write_text", "read_text"}
                    and not any(k.arg == "encoding" for k in n.keywords)):
                out.append(f"{p}:{n.lineno} .{n.func.attr}()")
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id == "open"):
                mode = ""
                if len(n.args) > 1 and isinstance(n.args[1], ast.Constant):
                    mode = str(n.args[1].value)
                for k in n.keywords:
                    if k.arg == "mode" and isinstance(k.value, ast.Constant):
                        mode = str(k.value.value)
                if "b" in mode:
                    continue            # binary: encoding would be an error
                if not any(k.arg == "encoding" for k in n.keywords):
                    # a computed mode cannot be read here; those are checked
                    # by eye and are all binary downloads
                    if len(n.args) > 1 and not isinstance(n.args[1], ast.Constant):
                        continue
                    out.append(f"{p}:{n.lineno} open(mode={mode or 'r'!r})")
    return out


def main() -> int:
    print("\nthe failure, reproduced")
    tmp = Path(tempfile.mkdtemp(prefix="t_enc_"))
    # The characters that actually appear in slides: an arrow, an en dash, a
    # curly quote, a Greek letter, a non-breaking space.
    hard = "step 1 → step 2 – “adjusted” β̂ ± 1.96"
    f = tmp / "manifest.json"
    try:
        f.write_text(hard, encoding="cp1252")
        broke = False
    except UnicodeEncodeError:
        broke = True
    check("cp1252 cannot hold what a deck contains", broke,
          "this platform encoded it, so the test proves nothing here")

    f.write_text(hard, encoding="utf-8")
    check("utf-8 can", f.read_text(encoding="utf-8") == hard)

    print("\nthe package never leaves it to the platform")
    bad = offenders()
    check(f"every text read and write names an encoding", not bad,
          "\n         " + "\n         ".join(bad[:12]))

    # The other half of the same decision, and the one that was missing.
    #
    # Naming utf-8 on both sides is right for new files and fatal for old
    # ones: until September 2026 Windows wrote these files as cp1252 and read
    # them back as cp1252, and the pair agreed. An em dash in the narration is
    # byte 0x97, which is not valid utf-8, so every project drafted before the
    # change stopped opening -- with a UnicodeDecodeError from pathlib that
    # named neither the project nor the encoding.
    print("\na file written before the rule still opens")
    import json, os
    from realme.core import textio
    from realme.pipeline.lecture import load_script, save_script
    from realme.core.schema import Manifest, Segment, Prosody

    proj = tmp / "old_project"
    (proj / "_work").mkdir(parents=True)
    legacy = ("{\n  \"project_id\": \"deck\",\n  \"title\": \"T\",\n"
              "  \"script_source\": \"gemini:x\",\n  \"segments\": [\n"
              "    {\"segment_id\": 1, \"slide_index\": 0,\n"
              "     \"spoken_text\": \"Confounding \u2014 the third variable\"}\n"
              "  ]\n}\n")
    mf = proj / "_work" / "manifest.json"
    mf.write_bytes(legacy.encode("cp1252"))
    check("the fixture really is cp1252", b"\x97" in mf.read_bytes())
    old_mtime = mf.stat().st_mtime
    os.utime(mf, (old_mtime - 5000, old_mtime - 5000))
    old_mtime = mf.stat().st_mtime
    before = len(textio.REPAIRED)
    try:
        m = load_script(proj)
        loaded = m.segments[0].spoken_text
    except Exception as e:
        loaded = f"{type(e).__name__}: {e}"
    check("a cp1252 manifest loads", loaded == "Confounding \u2014 the third variable",
          str(loaded)[:80])
    check("and the em dash survived", "\u2014" in str(loaded))
    check("the repair is reported, not silent", len(textio.REPAIRED) == before + 1)
    check("the file on disk is now utf-8",
          mf.read_bytes().decode("utf-8").find("\u2014") > 0)
    check("its modification time is unchanged",
          abs(mf.stat().st_mtime - old_mtime) < 1,
          "a repair is not an edit; /api/projects sorts by this")
    before = len(textio.REPAIRED)
    load_script(proj)
    check("reading it again repairs nothing", len(textio.REPAIRED) == before)

    # A utf-8 file must not be touched at all.
    save_script(proj, Manifest(project_id="d", title="T", script_source="s",
                               segments=[Segment(segment_id=1, slide_index=0,
                                                 spoken_text="plain \u2192 text",
                                                 prosody=Prosody())]))
    before = len(textio.REPAIRED)
    check("a utf-8 manifest round-trips untouched",
          load_script(proj).segments[0].spoken_text == "plain \u2192 text"
          and len(textio.REPAIRED) == before)

    # The ledger is the other file every old project has.
    from realme.core.ledger import RenderLedger
    lg = proj / "_work" / "ledger.json"
    lg.write_bytes(json.dumps({"utt:1": {"path": str(lg), "duration": 1.0,
                                         "note": "caf\u00e9 \u2014 x"}}
                              ).encode("cp1252"))
    check("a cp1252 ledger loads", "utt:1" in RenderLedger(lg).state)

    # And the helper must decide from the bytes rather than assert an encoding.
    src = (HERE / "realme/core/textio.py").read_text(encoding="utf-8")
    check("the repair reads bytes, not a named encoding",
          "read_bytes()" in src and "def _decode" in src)

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: {', '.join(FAILED)}")
        return 1
    print("all good")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
