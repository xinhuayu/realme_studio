#!/usr/bin/env python3
"""
Finding a program an installer did not put on PATH.

Written because LibreOffice on Windows installs to Program Files and does not
touch PATH, so `shutil.which("soffice")` reports missing on a machine where
LibreOffice is installed and working. The advice that follows -- install it --
is then something the user has already done, which is the most frustrating
kind of wrong answer a setup check can give.
"""
from __future__ import annotations
import os, sys, tempfile
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
_ISO = tempfile.mkdtemp(prefix="t_locate_iso_")
os.environ["REALME_HOME"] = str(Path(_ISO) / "home")
os.environ.setdefault("REALME_TOOLS", str(Path(_ISO) / "tools"))

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}"
          + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def main() -> int:
    from realme.core import media as M

    tmp = Path(tempfile.mkdtemp(prefix="t_locate_"))
    exe = "soffice.exe" if os.name == "nt" else "soffice"
    installed = tmp / "LibreOffice" / "program"
    installed.mkdir(parents=True)
    (installed / exe).write_bytes(b"MZ")

    real_which, real_known, real_bundled = M.shutil.which, M.KNOWN_LOCATIONS, M._bundled_dirs
    real_path = os.environ.get("PATH", "")
    try:
        M.shutil.which = lambda n: None          # nothing on PATH at all
        M._bundled_dirs = lambda: []

        print("\nnot on PATH, but installed")
        M.KNOWN_LOCATIONS = {"soffice": [str(installed)]}
        found = M.find_binary("soffice")
        check("it is found where the installer put it",
              found == str(installed / exe), str(found))
        check("and its folder is added to PATH for anything it launches",
              str(installed) in os.environ.get("PATH", ""))
        try:
            got, why = M.require("soffice"), ""
        except Exception as e:
            got, why = None, f"raised {type(e).__name__}: {e}"
        check("require() returns it rather than raising",
              got == str(installed / exe), why or str(got))

        print("\ngenuinely absent")
        M.KNOWN_LOCATIONS = {"soffice": [str(tmp / "nowhere")]}
        check("find_binary says so without raising",
              M.find_binary("soffice") is None)
        try:
            M.require("soffice")
            ok, msg = False, "require() did not raise"
        except M.MediaError as e:
            ok, msg = True, str(e)
        check("require() raises", ok, msg)
        check("and names LibreOffice, not ffmpeg, in the advice",
              "LibreOffice" in msg and "ffmpeg" not in msg, msg)
        check("and says a PDF deck needs none of it", "PDF deck" in msg, msg)

        print("\nan unexpanded variable is not mistaken for a folder")
        # A real directory whose name contains a percent sign, holding a real
        # soffice. Only the guard can stop it being found, so this tests the
        # guard rather than the absence of the directory.
        weird = tmp / "%NO_SUCH_VAR%" / "program"
        weird.mkdir(parents=True)
        (weird / exe).write_bytes(b"MZ")
        M.KNOWN_LOCATIONS = {"soffice": [str(weird)]}
        check("a path with an unexpanded variable is skipped",
              M.find_binary("soffice") is None,
              "it searched a folder whose name is still a variable")

        print("\nPATH still wins when it has an answer")
        M.shutil.which = lambda n: "/usr/bin/soffice" if n == "soffice" else None
        M.KNOWN_LOCATIONS = {"soffice": [str(installed)]}
        check("the copy on PATH is preferred over a guessed location",
              M.find_binary("soffice") == "/usr/bin/soffice")

        print("\nsetup and the renderer agree")
        import realme.setup_check as SC
        src = (HERE / "realme/setup_check.py").read_text(encoding="utf-8")
        check("setup asks find_binary, not shutil.which directly",
              "find_binary as _find" in src and '_find("soffice")' in src,
              "setup would report missing what the renderer can find")
    finally:
        M.shutil.which, M.KNOWN_LOCATIONS, M._bundled_dirs = (
            real_which, real_known, real_bundled)
        os.environ["PATH"] = real_path
        import shutil as _sh
        _sh.rmtree(tmp, ignore_errors=True)

    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: {', '.join(FAILED)}")
        return 1
    print("all good")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
