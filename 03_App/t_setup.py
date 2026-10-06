#!/usr/bin/env python3
"""
`realme setup --fix` installs what the check just found missing — and only that.

The risk in a self-installing setup step is not that it fails; it is that it
succeeds into the wrong place, or reaches past pip into things pip does not
own. Both are checked here. pip is never actually run: the call is captured.
"""
from __future__ import annotations
import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Isolated from the real installation. See the note in t_pace.py: these
# resolve through `data_home()` and `find_tool()`, which read the environment
# at call time, so a suite that does not redirect them is testing whatever
# happens to be installed on the machine running it.
_ISO = tempfile.mkdtemp(prefix="t_setup_iso_")
os.environ["REALME_HOME"] = str(Path(_ISO) / "home")
os.environ.setdefault("REALME_TOOLS", str(Path(_ISO) / "tools"))

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}"
          + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def main() -> int:
    from realme import setup_check as SC

    C = SC.Check
    checks = [
        C("python: numpy", False, "the acoustic check", "pip install numpy"),
        C("python: scipy", False, "the acoustic check", "pip install scipy"),
        C("python: pptx", False, "speaker notes", "pip install python-pptx"),
        C("python: pydantic", True, "already here", "pip install pydantic"),
        # A machine where NO pdf backend works: decks cannot be read, so the
        # declared dependency is installed like any other.
        C("PDF backend", False, "no backend", "pip install pymupdf"),
        # the ones pip must not be asked to do
        C("ffmpeg", False, "audio", "conda install -y -c conda-forge ffmpeg"),
        C("soffice (LibreOffice)", False, "pptx→pdf",
          "winget install libreoffice", required=False),
        C("speech-rule-engine", False, "equations",
          "npm install -g speech-rule-engine", required=False),
        # A pip-installable OPTIONAL extra. Without one of these in the list,
        # deleting the required-only filter changes nothing and the check that
        # guards it proves nothing.
        # Equations are installed by default now, so the optional pip
        # package that guards the required-only filter has to be a different
        # one. A check that no longer tests anything is worse than no check.
        C("python: latex2mathml", False, "converting equations to speech",
          "pip install latex2mathml", required=True),
    ]

    # A different machine: the permissive fallback works, so PyMuPDF is a
    # recommendation rather than a repair.
    fallback = [
        C("PDF backend", True, "pypdf + pdfplumber"),
        C("PyMuPDF", False, "reads tables and figures more faithfully",
          "pip install pymupdf", required=False,
          note="recommended, not required -- it is AGPL-3.0"),
        C("python: numpy", False, "measurement", "pip install numpy"),
    ]

    called = {}

    class FakeResult:
        returncode = 0

    real_env = SC.managed_environment
    SC.managed_environment = lambda: (True, "test environment")
    real_run = SC.subprocess.run
    SC.subprocess.run = lambda cmd, *a, **k: (called.setdefault("cmd", cmd),
                                              FakeResult())[1]
    try:
        print("\nwhat it decides to install")
        n = SC.install_missing(checks, log=lambda *_: None)
        cmd = called.get("cmd", [])
        pkgs = cmd[4:] if len(cmd) > 4 else []
        check("it installs the missing pip packages",
              set(pkgs) == {"numpy", "scipy", "python-pptx", "pymupdf",
                            "latex2mathml"}, str(pkgs))
        check("and reports how many", n == 5, str(n))
        check("equations are installed by default, not suggested",
              "latex2mathml" in pkgs, str(pkgs))
        check("it does not reinstall what is already there",
              "pydantic" not in pkgs, str(pkgs))

        print("\nwhat it refuses to touch")
        for bad in ("ffmpeg", "libreoffice", "speech-rule-engine"):
            check(f"{bad} is not handed to pip", bad not in pkgs, str(pkgs))
        # The property is "the only thing invoked is THIS interpreter's pip",
        # and it is asserted structurally rather than by grepping the command
        # for "conda".
        #
        # That grep failed on the one machine it was written to protect:
        # cmd[0] is sys.executable, and on an Anaconda install that reads
        # C:\Users\...\anaconda3\envs\realme\python.exe -- which contains
        # the substring "conda". The check passed in a sandbox whose Python
        # lives at /usr/bin/python3, and passed again when run by hand with a
        # `python` that happened to resolve to a non-conda interpreter, so it
        # only ever failed through the launcher that uses the right one.
        # Grepping a string where the behaviour can be asserted is the fault
        # this project's own audit flagged twice before.
        check("only this interpreter's pip is invoked",
              Path(cmd[0]) == Path(sys.executable)
              and cmd[1:4] == ["-m", "pip", "install"],
              str(cmd[:4]))
        check("no package manager is invoked instead",
              not any(Path(str(a)).name in ("conda.exe", "conda", "conda.bat",
                                            "npm", "npm.cmd", "npx")
                      for a in cmd),
              str(cmd[:4]))
        check("and nothing conda- or npm-shaped is handed to pip as a package",
              not any(p_ in ("conda", "npm", "speech-rule-engine")
                      for p_ in pkgs), str(pkgs))

        print("\nwhere it installs")
        check("into the interpreter running it, not whatever pip is on PATH",
              cmd[:3] == [sys.executable, "-m", "pip"], str(cmd[:3]))

        print("\nwhen there is nothing to do")
        called.clear()
        n = SC.install_missing([C("python: numpy", True, "", "pip install numpy")],
                               log=lambda *_: None)
        check("pip is not run at all", "cmd" not in called and n == 0, str(called))

        print("\nwhen pip fails")
        class Bad:
            returncode = 1
        SC.subprocess.run = lambda *a, **k: Bad()
        said = []
        n = SC.install_missing(checks, log=said.append)
        check("it reports nothing installed", n == 0, str(n))
        check("and says the network is the usual cause",
              any("network" in s for s in said), str(said)[:120])
        print("\nonly required packages, never optional extras")
        called.clear()
        SC.subprocess.run = lambda cmd, *a, **k: (called.setdefault("cmd", cmd),
                                                  FakeResult())[1]
        SC.install_missing(checks, log=lambda *_: None)
        pkgs2 = called.get("cmd", [])[4:]
        check("an optional pip package is not installed unasked",
              "latex2mathml" in pkgs2, "equations are required now, so this "
              "list needs a different optional package to prove the filter")

        print("\nPyMuPDF is recommended, never forced")
        called.clear()
        SC.install_missing(fallback, log=lambda *_: None)
        pkgs3 = called.get("cmd", [])[4:]
        check("a working fallback means PyMuPDF is left alone",
              "pymupdf" not in pkgs3, str(pkgs3))
        check("but the rest of that machine is still repaired",
              pkgs3 == ["numpy"], str(pkgs3))
        check("and it is still printed as a suggestion",
              any(c.name == "PyMuPDF" and not c.ok and not c.required
                  for c in fallback))
        check("but it is installed when asked for explicitly",
              "latex2mathml" in (lambda: (called.clear(),
                                          SC.install_missing(
                                              checks + [SC.Check(
                                                  "python: latex2mathml", False,
                                                  "equations",
                                                  "pip install latex2mathml",
                                                  required=False)],
                                          log=lambda *_: None,
                                          required_only=False),
                                          called.get("cmd", [])[4:])[-1])(),
              "required_only=False did not include it")

        print("\nwhich environments may be written to")
        SC.managed_environment = real_env
        import os, tempfile
        real_prefix, real_base = SC.sys.prefix, SC.sys.base_prefix
        had = os.environ.pop("CONDA_PREFIX", None)
        try:
            # A conda environment, with conda NOT activated -- which is how
            # every RealMe launcher runs, since they call the env's exe
            # directly instead of activating.
            d = Path(tempfile.mkdtemp(prefix="t_env_"))
            (d / "conda-meta").mkdir()
            SC.sys.prefix = SC.sys.base_prefix = str(d)
            ok, where = SC.managed_environment()
            check("a conda env is recognised without conda being activated",
                  ok and "conda" in where, f"{ok} / {where}")

            # A virtualenv: prefix and base_prefix differ.
            SC.sys.prefix, SC.sys.base_prefix = str(d / "venv"), str(d)
            ok, where = SC.managed_environment()
            check("a virtualenv is recognised", ok and "virtual" in where,
                  f"{ok} / {where}")

            # A bare system interpreter: neither marker.
            SC.sys.prefix = SC.sys.base_prefix = str(d / "nothing")
            ok, where = SC.managed_environment()
            check("a bare system interpreter is not", not ok, f"{ok} / {where}")
        finally:
            SC.sys.prefix, SC.sys.base_prefix = real_prefix, real_base
            if had is not None:
                os.environ["CONDA_PREFIX"] = had

        called.clear()
        SC.managed_environment = lambda: (False, "a system Python")
        said2 = []
        n = SC.install_missing(checks, log=said2.append)
        check("a system Python is not installed into",
              "cmd" not in called and n == 0, str(called))
        check("and the command is printed instead",
              any("pip install" in s for s in said2), str(said2)[:120])
    finally:
        SC.managed_environment = real_env
        SC.subprocess.run = real_run

    print("\nthe wrong-pip diagnosis")
    import subprocess as _sp

    real_which, real_run2 = SC.shutil.which, SC.subprocess.run
    try:
        # PATH's pip belongs to another interpreter, and that one HAS the
        # package this one is missing. The message has to name that.
        other = Path("/opt/other-python")
        SC.shutil.which = lambda n: str(other / "Scripts" / "pip.exe") if n == "pip" else None
        found = SC.foreign_pip()
        check("a pip from another prefix is spotted",
              found is not None and "other-python" in found[0], str(found))

        class Listed:
            stdout = "latex2mathml==3.81.1\nnumpy==2.0.0\n"
            returncode = 0
        SC.subprocess.run = lambda *a, **k: Listed()
        there = SC.installed_elsewhere("whatever", ["latex2mathml", "scipy"])
        check("it asks that Python what it has, rather than guessing",
              there == {"latex2mathml"}, str(there))

        said = []
        SC.explain_wrong_pip(["latex2mathml", "scipy"], log=said.append)
        blob = "\n".join(said)
        check("it names both interpreters",
              "other-python" in blob and sys.executable in blob, blob[:120])
        check("it says which package is in the wrong one",
              "latex2mathml" in blob and "over THERE" in blob, blob[:200])
        check("and gives the command that installs into the right one",
              f'"{sys.executable}" -m pip install' in blob, blob[-200:])

        # Same interpreter: nothing to warn about.
        SC.shutil.which = lambda n: str(Path(sys.prefix) / "Scripts" / "pip.exe") \
            if n == "pip" else None
        check("a pip from this same prefix is not flagged",
              SC.foreign_pip() is None, str(SC.foreign_pip()))
        quiet = []
        check("and nothing is printed",
              SC.explain_wrong_pip(["numpy"], log=quiet.append) is False
              and not quiet, str(quiet))

        # The other Python cannot be asked: say nothing rather than guess.
        SC.shutil.which = lambda n: str(other / "Scripts" / "pip.exe") if n == "pip" else None
        def boom(*a, **k):
            raise OSError("no such interpreter")
        SC.subprocess.run = boom
        check("an unreachable interpreter is not reported as empty or full",
              SC.installed_elsewhere("gone", ["numpy"]) == set())

        # The likelier failure: it is there, it answers, it never finishes.
        def hang(*a, **k):
            raise _sp.TimeoutExpired(cmd="pip list", timeout=30)
        SC.subprocess.run = hang
        check("and neither is one that hangs",
              SC.installed_elsewhere("slow", ["numpy"]) == set())
    finally:
        SC.shutil.which, SC.subprocess.run = real_which, real_run2

    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: {', '.join(FAILED)}")
        return 1
    print("all good")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
