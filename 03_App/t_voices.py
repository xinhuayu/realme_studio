#!/usr/bin/env python3
"""
"Not installed" has to be a claim about a folder, and name it.

The Podcast page reported Amy and Cori missing on a machine where both were
plainly on disk. Nothing on screen said where it had looked, so there was
nothing to check the claim against -- and the two places that answer "is this
voice installed?" did not agree with each other.
"""
from __future__ import annotations
import ast, sys, tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

#: Where this file lives. Every path below is relative to it, never to the
#: working directory: these suites are run from the project root as often as
#: from 03_App, and `Path("realme/...")` silently means a different thing in
#: each. `verify_tree.py` already anchored on __file__; the t_*.py files did
#: not, so t_render died with FileNotFoundError when run from the root.
HERE = Path(__file__).resolve().parent

# Isolated from the real installation. See the note in t_pace.py: these
# resolve through `data_home()` and `find_tool()`, which read the environment
# at call time, so a suite that does not redirect them is testing whatever
# happens to be installed on the machine running it.
_ISO = tempfile.mkdtemp(prefix="t_voices_iso_")
os.environ["REALME_HOME"] = str(Path(_ISO) / "home")
os.environ.setdefault("REALME_TOOLS", str(Path(_ISO) / "tools"))

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}"
          + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def code_only(path: str) -> str:
    # `path` is relative to this file, never to the working directory.
    """Source with comments and docstrings removed.

    Two checks in the first draft of this file failed against comments that
    EXPLAIN the fix by naming the old behaviour. A test that greps prose
    reports a regression every time the reasoning is written down, which is
    the opposite of what it should encourage.
    """
    import io, tokenize
    out = []
    src = (HERE / path).read_text(encoding="utf-8")
    prev_end = (1, 0)
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in (tokenize.COMMENT,):
            continue
        if tok.type == tokenize.STRING and tok.start[1] == 0:
            continue                      # module/def docstring at column 0
        out.append(tok.string)
    return "\n".join(out)


def calls_in(path: str, func: str) -> set:
    """Every function or method called inside `func`."""
    tree = ast.parse((HERE / path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func:
            names = set()
            for n in ast.walk(node):
                if isinstance(n, ast.Call):
                    f = n.func
                    names.add(f.attr if isinstance(f, ast.Attribute)
                              else getattr(f, "id", ""))
            return names
    return set()


def main() -> int:
    called = calls_in("realme/app/server.py", "draft_voices")

    print("\none definition of installed")
    check("the endpoint asks piper_status, which checks the PAIR",
          "piper_status" in called,
          "it checks only the .onnx, so a voice with no .json reads as ready "
          "and then fails at load")
    check("and no longer asks piper_voice_file alone",
          "piper_voice_file" not in called, str(sorted(called)))

    print("\nit says where it looked")
    server = code_only("realme/app/server.py")
    for key in ("folder", "folder_exists", "present", "realme_tools"):
        check(f"the answer carries {key!r}", f'"{key}"' in server)
    check("the folder comes from piper_home, not the engine folder",
          "piper_home" in called,
          "tools/qwen3/piper is not where voices live")

    print("\nand so does the setup check")
    sc = code_only("realme/setup_check.py")
    check("setup names the voice folder from piper_home",
          "piper_home" in calls_in("realme/setup_check.py", "run_checks"),
          str(sorted(c for c in calls_in("realme/setup_check.py", "run_checks")
                     if "piper" in c or "home" in c)))
    check("and no longer prints the engine folder as the voice folder",
          'home / \'piper\'' not in sc and 'home / "piper"' not in sc,
          "it still says tools/qwen3/piper, a folder that has never existed")
    check("a missing voice is reported with what the folder does hold",
          "that folder holds" in sc)

    print("\nthe page shows it")
    ui = (HERE / "realme/app/ui.html").read_text(encoding="utf-8")
    check("the draft-voice note is rendered", "pdraft-note" in ui)
    check("and it names the folder", "draftVoiceNote" in ui
          and "r.folder" in ui)
    check("and REALME_TOOLS, which decides where it looks",
          "realme_tools" in ui)

    print("\nwhere voices are looked for does not depend on the cwd")
    import os, tempfile, shutil
    from realme.core import paths as P
    tmp = Path(tempfile.mkdtemp(prefix="t_voices_"))
    real_file, real_cwd = P.__file__, os.getcwd()
    had = os.environ.pop("REALME_TOOLS", None)
    try:
        # A package installed into site-packages: no checkout above it, so
        # `project_root()` finds nothing. The tools folder is a few levels up
        # all the same.
        # The layout that the older search could NOT reach: tools/ sits above
        # the directory the package was installed from, and there is no
        # `03_App` marker for project_root() to find. Neither tools_dir() nor
        # package_root()/"tools" lands on it, so only the upward walk does --
        # which is what makes this check test the new code rather than the
        # old. The first version of this fixture put tools/ inside
        # package_root and passed either way; a mutation caught that.
        deep = tmp / "installed" / "lib" / "site-packages"
        pkg = deep / "realme" / "core"
        pkg.mkdir(parents=True)
        (tmp / "installed" / "tools" / "piper").mkdir(parents=True)
        P.__file__ = str(pkg / "paths.py")
        os.chdir(tmp)                      # a cwd with no tools/ in it
        found = P.find_tool("piper")
        check("tools/ above the install is found",
              found == tmp / "installed" / "tools" / "piper", str(found))
        check("and not from the cwd", not str(found or "").startswith(
            str(tmp / "tools")), str(found))

        # And REALME_TOOLS still wins when it is set and real.
        elsewhere = tmp / "elsewhere"
        (elsewhere / "piper").mkdir(parents=True)
        os.environ["REALME_TOOLS"] = str(elsewhere)
        check("an explicit REALME_TOOLS still wins",
              P.find_tool("piper") == elsewhere / "piper",
              str(P.find_tool("piper")))
    finally:
        P.__file__, _ = real_file, os.chdir(real_cwd)
        os.environ.pop("REALME_TOOLS", None)
        if had is not None:
            os.environ["REALME_TOOLS"] = had
        shutil.rmtree(tmp, ignore_errors=True)

    print("\nthe vendored converter's noise is silenced, narrowly")
    vc = code_only("realme/adapters/vc.py")
    check("the deprecation warning is filtered by category and message",
          "FutureWarning" in vc and "weight_norm" in vc)
    check("and only around the load, not globally",
          "catch_warnings" in vc and "simplefilter" not in vc,
          "a global filter would hide warnings this project does want")
    check("the print from inside the vendored model is captured",
          "redirect_stdout" in vc)

    print("\nthe page's javascript still parses")
    i = ui.index("<script>") + 8
    j = ui.rindex("</script>")
    tmp = Path(tempfile.mkdtemp(prefix="t_voices_")) / "ui.js"
    tmp.write_text(ui[i:j], encoding="utf-8")
    import subprocess
    try:
        r = subprocess.run(["node", "--check", str(tmp)], capture_output=True,
                           text=True)
        check("node --check passes", r.returncode == 0,
              (r.stderr or "").splitlines()[0] if r.returncode else "")
    except FileNotFoundError:
        print("  --   node is not installed; javascript not checked")
    import shutil
    shutil.rmtree(tmp.parent, ignore_errors=True)

    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: {', '.join(FAILED)}")
        return 1
    print("all good")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
