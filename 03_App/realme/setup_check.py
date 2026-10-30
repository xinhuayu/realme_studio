"""
`realme setup` -- tell the user exactly what to run, for their machine.

`doctor` reports what is missing. That is not the same as being actionable: an
instructor reading "[MISS] ffmpeg" still has to go and find out what ffmpeg is
and how it gets onto Windows. This prints the command to paste.
"""
from __future__ import annotations
import os
import shutil
import subprocess
import sys
from pathlib import Path

WIN = sys.platform.startswith("win")
MAC = sys.platform == "darwin"
HAS_CONDA = bool(shutil.which("conda") or os.environ.get("CONDA_PREFIX"))


def _pkg_cmd(conda_name: str, apt_name: str, brew_name: str) -> str:
    if HAS_CONDA:
        return f"conda install -y -c conda-forge {conda_name}"
    if WIN:
        return f"winget install {conda_name}    (or: choco install {conda_name})"
    if MAC:
        return f"brew install {brew_name}"
    return f"sudo apt-get install -y {apt_name}"


class Check:
    def __init__(self, name, ok, detail="", fix="", required=True, note=""):
        self.name, self.ok, self.detail = name, ok, detail
        self.fix, self.required, self.note = fix, required, note


def run_checks(model: str = "") -> list[Check]:
    out: list[Check] = []

    v = sys.version_info
    out.append(Check("Python 3.10+", v >= (3, 10), f"{v.major}.{v.minor}.{v.micro}",
                     "Install Python 3.11 (conda create -n realme python=3.11)"))

    for b in ("ffmpeg", "ffprobe"):
        try:
            from realme.core.media import require as _require
            found = _require(b)
            out.append(Check(b, True, found))
        except Exception:
            out.append(Check(
                b, False, "not found",
                ("00_Windows\\_get_ffmpeg.bat   (self-contained, avoids the "
                 "conda DLL conflicts)") if WIN
                else _pkg_cmd("ffmpeg", "ffmpeg", "ffmpeg")))

    # PyMuPDF is recommended but never forced.
    #
    # It reads decks better -- tables and figures come through more faithfully
    # than with the permissive fallback -- and it is also AGPL-3.0, which is a
    # licensing decision rather than a convenience. Somebody who removed it on
    # purpose should not find it quietly reinstalled by a setup check.
    #
    # So: when SOME backend works, PyMuPDF is a suggestion in the optional
    # section. When none works, decks cannot be read at all and the declared
    # dependency is installed like any other.
    # `pdfdoc.backend()` names the fallback without checking it exists, so
    # the fallback's own parts are checked here.
    have_mupdf = False
    try:
        from realme.pipeline import pdfdoc
        have_mupdf = bool(getattr(pdfdoc, "HAVE_MUPDF", False))
    except Exception:
        pass
    if have_mupdf:
        out.append(Check("PDF backend", True, "pymupdf"))
    else:
        missing = []
        for mod in ("pdfplumber", "pypdf"):
            try:
                __import__(mod)
            except ImportError:
                missing.append(mod)
        try:
            from realme.core.media import find_binary as _fb
            if not _fb("pdftoppm"):
                missing.append("pdftoppm (poppler)")
        except Exception:
            missing.append("pdftoppm (poppler)")
        if missing:
            out.append(Check("PDF backend", False,
                             "no PDF reader: missing " + ", ".join(missing),
                             "pip install pymupdf", required=True,
                             note="pymupdf is AGPL-3.0; the permissive route "
                                  "is pip install pypdf pdfplumber plus "
                                  "poppler on PATH"))
        else:
            out.append(Check("PDF backend", True, "poppler+pdfplumber"))
            out.append(Check(
                "PyMuPDF", False,
                "reads tables and figures more faithfully than the fallback",
                "pip install pymupdf", required=False,
                note="recommended, not required -- it is AGPL-3.0, so this is "
                     "your call"))

    # `required` here has to mean the same thing `pyproject.toml` means, or
    # the install step and the dependency list disagree about what RealMe
    # needs. numpy and scipy are required: numpy generates the mask tone and
    # every enrolment measurement, scipy runs the acoustic check a lecture
    # render performs by default, at the last step.
    for mod, why, req in (("pydantic", "data contracts", True),
                          ("PIL", "image sizing", True),
                          ("fastapi", "the Studio web app", True),
                          ("uvicorn", "serving the Studio", True),
                          # Every upload in the Studio goes through this.
                          ("multipart", "Studio uploads", True),
                          ("pptx", "reading .pptx speaker notes", True),
                          ("numpy", "enrolment measurement and the mask tone", True),
                          ("scipy", "the acoustic check every render runs", True),
                          # Installed by default: decks with equations are
                          # common, and without it they are read as raw
                          # LaTeX. Speaking them ALSO wants
                          # speech-rule-engine, which is npm's and stays a
                          # printed suggestion below.
                          ("latex2mathml", "converting equations to speech", True),
                          ):
        try:
            __import__(mod)
            ok = True
        except ImportError:
            ok = False
        pipname = {"PIL": "pillow", "pptx": "python-pptx",
                   "multipart": "python-multipart"}.get(mod, mod)
        # The note, not the fix: `fix` is parsed by `install_missing` and by
        # `report`, which split on "pip install ". The note is for the reader,
        # and names the interpreter so the package lands where RealMe looks.
        out.append(Check(f"python: {mod}", ok, why, f"pip install {pipname}",
                         required=req,
                         note=("" if ok else
                               f"install it HERE with:  "
                               f'"{sys.executable}" -m pip install {pipname}')))

    # Asked the same way the renderer asks, not with `shutil.which`.
    # LibreOffice does not add itself to PATH on Windows, so `which` says
    # missing on a machine where it is installed and working.
    try:
        from realme.core.media import find_binary as _find
        soffice = _find("soffice")
    except Exception:
        soffice = shutil.which("soffice")
    out.append(Check("soffice (LibreOffice)", bool(soffice),
                     soffice or "not installed -- use PDF decks",
                     "winget install --id TheDocumentFoundation.LibreOffice -e"
                     if WIN else
                     _pkg_cmd("libreoffice", "libreoffice", "--cask libreoffice"),
                     required=False,
                     note="optional: PDF is the normal deck format and needs "
                          "nothing. Only a .pptx needs LibreOffice; without "
                          "it, upload the PDF version"))

    out.append(Check("speech-rule-engine", bool(shutil.which("sre")),
                     "speaks LaTeX equations",
                     "npm install -g speech-rule-engine", required=False))

    # voices
    #
    # Asks the same resolver the installer and the adapter use, so all three
    # agree on where voices live (tools/piper, not the working directory),
    # and checks every voice in DRAFT_VOICES.
    from realme.engines.install import piper_status, engine_home, piper_home
    from realme.adapters.tts import DRAFT_VOICES
    home = engine_home()
    # Where the voices actually are. `home` is the ENGINE folder
    # (tools/qwen3); the voices sit beside it in tools/piper, not in
    # `home / "piper"`.
    vdir = piper_home(home)
    states = {v: piper_status(home, v) for v in DRAFT_VOICES}
    have_piper = any(st["code"] for st in states.values())
    ready = [v for v, st in states.items() if st["ready"]]
    missing = [v for v, st in states.items() if not st["ready"]]

    on_disk = (sorted(f.name for f in Path(vdir).glob("*.onnx"))
               if Path(vdir).is_dir() else [])
    if not have_piper:
        detail = f"the piper package is not importable by {sys.executable}"
    elif ready:
        detail = f"{len(ready)} of {len(states)} voices in {vdir}"
        if missing:
            # Name the folder AND what is in it. "Missing" about a file the
            # user can see on disk is a claim about a path, and the path is
            # the part worth printing.
            detail += (f" -- missing {', '.join(missing)}; that folder holds "
                       + (", ".join(on_disk) if on_disk else "no .onnx files"))
    else:
        one = next(iter(states.values()))
        detail = (f"piper installed, no voice files under {vdir} "
                  f"-- a voice is the .onnx AND the .onnx.json"
                  if not one["voice_present"] else "voice files incomplete")
    out.append(Check("piper (draft voice)", have_piper and bool(ready), detail,
                     "realme engine install draft",
                     required=False,
                     note="the draft voice the Studio offers"))
    out.append(Check("espeak-ng", bool(shutil.which("espeak-ng")),
                     "fallback draft voice + phoneme helper",
                     _pkg_cmd("espeak-ng", "espeak-ng", "espeak-ng"),
                     required=False,
                     note="optional: without it the lexicon uses respellings, "
                          "which still fix pronunciation"))

    # keys
    out.append(Check("GEMINI_API_KEY", bool(os.environ.get("GEMINI_API_KEY")),
                     "writes the narration",
                     "Free key at aistudio.google.com, then:\n"
                     "  realme key GEMINI_API_KEY your-key-here\n"
                     "  (writes a .env file; takes effect immediately)"))
    # Your own voice: Gemini's replicated voice is the default; the local
    # Qwen3 engine is the optional alternative. Either one satisfies this.
    gvoice = ""
    try:
        from realme.app.profile import Profile
        from realme.core.env import data_home
        gvoice = ((Profile(data_home() / "profile").data.get("gemini_voice")
                   or {}).get("voice") or "")
    except Exception:
        pass
    local = False
    try:
        from realme.engines.cpp import status as _cpp_status
        _st = _cpp_status()
        local = bool(_st.get("built") and _st.get("weights_ready"))
    except Exception:
        pass
    said = ("Gemini replicated voice enrolled" if gvoice else
            "local Qwen3 engine installed" if local else
            "none yet -- drafts use the Piper voices")
    out.append(Check("your cloned voice", bool(gvoice or local), said,
                     "Studio: Twin Setup -> Your voice -> Gemini cloned voice\n"
                     "  (or, optional and local: 3_Get_Voice_Engine.bat)",
                     required=False,
                     note="without one, lectures are read by a draft voice, "
                          "which is fine for checking the script"))
    return out


# Launcher names that earlier versions of RealMe used. Renaming a .bat does not
# remove the old one -- an update extracts over the folder, it does not clean
# it -- so without this the folder accumulates old launcher names, and it stops
# being obvious which file to double-click.
#
# This lives in Python, not in 0_Update.bat, for a specific reason: the .bat
# that runs during an update is the OLD one, already on disk. Only code shipped
# *inside* the update can clean up after a rename on the very first run.
SUPERSEDED_LAUNCHERS = [
    "1b_Get_FFmpeg.bat", "2_Get_FFmpeg.bat",      # -> _get_ffmpeg.bat (a helper)
    "3_Set_Key.bat",                               # -> 2_Set_Key.bat
    "6_Get_Voice_Engine.bat",                      # -> 3_Get_Voice_Engine.bat
    "2_Start_Studio.bat",                          # -> 4_Start_Studio.bat
    "4_Command_Prompt.bat",                        # -> 5_Command_Prompt.bat
    "7_Free_Disk_Space.bat",                       # -> 6_Free_Disk_Space.bat
]


def windows_dir() -> Path | None:
    """The 00_Windows folder, if this is a checkout rather than a site install."""
    here = Path(__file__).resolve()
    for base in (here.parents[2], here.parents[3]):
        d = base / "00_Windows"
        if d.is_dir():
            return d
    return None


def prune_superseded(dry_run: bool = False) -> list[str]:
    """
    Delete launchers that a rename made obsolete. Returns what went.

    Deliberately a fixed list, never a pattern like "*.bat we don't ship any
    more": a wildcard here would delete a launcher someone wrote themselves.
    """
    d = windows_dir()
    if d is None:
        return []
    removed = []
    for name in SUPERSEDED_LAUNCHERS:
        f = d / name
        if not f.is_file():
            continue
        if dry_run:
            removed.append(name)
            continue
        try:
            f.unlink()
            removed.append(name)
        except OSError:
            pass
    return removed


#: Files that an older install may hold but the release no longer carries.
#: They are moved -- never deleted -- into `_Archive/<folder>` the next time
#: `realme setup` runs (which every update ends with). Paths are relative to
#: the RealMe folder. A fixed list, for the same reason as above.
#:
#: The animation engine: an animated explainer made in a separate tool joins a
#: lecture with `realme insert`.
RETIRED_CODE = {
    "animate_2026-10": [
        "03_App/realme/pipeline/animate.py",
        "03_App/realme/pipeline/scenes_guard.py",
        "03_App/t_animate.py",
        "03_App/examples/guide_v3",
    ],
    # Development files a colleague's copy should not carry: a guide-deck
    # build script, a README and notebook folder, older guide decks, notes and
    # videos with their render folder, and an audit report.
    "history_2026-10": [
        "03_App/make_guide_slides.py",
        "03_App/README.md",
        "03_App/notebooks",
        "RealMe_Guide_narrated.pptx",
        "RealMe_Guide_notes.txt",
        "RealMe_Guide_narrated_v2.pdf",
        "RealMe_Guide_narrated_v2.pptx",
        "RealMe_Guide_notes_v2.txt",
        "RealMe_Guide_narrated_v2.mp4",
        "RealMe_Guide_narrated_v3.mp4",
        "realme_guide",
        "Independent_Audit_September_2026.md",
    ],
}


def archive_retired(root: Path | None = None, dry_run: bool = False) -> list[str]:
    """Move retired code into _Archive. Returns what moved.

    An update extracts over the folder and never removes anything, so a file
    the release no longer carries stays on disk, and a stale module that
    still imports can confuse a later test run. Moved, not deleted:
    `_Archive` is never part of a release.
    """
    import shutil
    if root is None:
        d = windows_dir()
        if d is None:
            return []
        root = d.parent
    moved = []
    for folder, rels in RETIRED_CODE.items():
        for rel in rels:
            src = root / rel
            if not src.exists():
                continue
            dst = root / "_Archive" / folder / rel
            if dst.exists():
                continue                 # already archived once; leave both
            if not dry_run:
                try:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(dst))
                except OSError:
                    continue
            moved.append(rel)
    return moved


def normalise_future_mtimes(root: Path | None = None, log=None) -> int:
    """
    Pull future-dated files in the project tree back to now.

    ZIP timestamps carry no timezone. The update archive is built on a UTC
    machine, and Windows reads those naive stamps as *local* time, so west of
    UTC every extracted file lands hours ahead of the clock. Python does not
    care, but any build system that compares timestamps does -- Ninja gives up
    with "manifest still dirty after 100 tries, perhaps system time is not set",
    which reads like a broken clock and is not.

    The update archive is backdated, so this arises only from an older zip.
    Only future files are touched.
    """
    import time
    root = root or (windows_dir().parent if windows_dir() else None)
    if root is None:
        return 0
    now = time.time()
    cutoff = now + 60
    fixed = 0
    for f in root.rglob("*"):
        try:
            if f.is_file() and f.stat().st_mtime > cutoff:
                os.utime(f, (now, now))
                fixed += 1
        except OSError:
            pass
    return fixed


def _prefix_of_script(exe: str | Path) -> Path:
    """`<prefix>\\Scripts\\pip.exe` -> `<prefix>`; `<prefix>/bin/pip` -> `<prefix>`."""
    return Path(exe).resolve().parent.parent


def _interpreter_in(prefix: Path) -> Path:
    return prefix / ("python.exe" if WIN else "bin/python")


def foreign_pip() -> tuple[str, str] | None:
    """The `pip` on PATH, when it belongs to a different Python. Else None.

    Windows machines often have several Pythons -- a store install, a
    python.org install, Anaconda -- and `pip` resolves to whichever one is
    first on PATH. RealMe runs from a conda environment that its launchers
    deliberately never activate, so PATH almost never points at it.

    The result is that `pip install latex2mathml` reports success, and RealMe
    goes on saying the package is missing, because it was installed into a
    Python RealMe never loads.

    Returns (path to that pip, path to its interpreter).
    """
    exe = shutil.which("pip")
    if not exe:
        return None
    try:
        theirs = _prefix_of_script(exe)
        if theirs == Path(sys.prefix).resolve():
            return None
        return str(exe), str(_interpreter_in(theirs))
    except OSError:
        return None


def installed_elsewhere(python_exe: str, names) -> set:
    """Which of `names` that OTHER interpreter already has.

    One subprocess, not one per package. Asked rather than assumed, so
    "installed into the other Python" is only reported when it is true.
    """
    want = {n.lower().replace("_", "-") for n in names}
    if not want:
        return set()
    try:
        r = subprocess.run([python_exe, "-m", "pip", "list", "--format=freeze"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    except (OSError, subprocess.SubprocessError):
        return set()
    have = {line.split("==")[0].strip().lower().replace("_", "-")
            for line in (r.stdout or "").splitlines() if line.strip()}
    return {n for n in names if n.lower().replace("_", "-") in have}


def managed_environment() -> tuple[bool, str]:
    """May this interpreter be installed into, and what is it?

    Installing into the Python that came with the operating system often
    needs administrator rights and can break something else on the machine. A conda environment or a virtualenv is ours; a system interpreter
    is not.

    `conda-meta` rather than `CONDA_PREFIX`, because that variable is only set
    by `conda activate` and RealMe's launchers deliberately never activate --
    they run `...\\envs\\realme\\Scripts\\realme.exe` directly, which is why they
    start instantly. Testing the variable would call the right environment
    a system one whenever the app is launched normally.
    """
    prefix = Path(sys.prefix)
    if (prefix / "conda-meta").is_dir():
        return True, f"conda environment at {prefix}"
    if sys.prefix != sys.base_prefix:
        return True, f"virtual environment at {prefix}"
    return False, (f"{sys.executable}\n           looks like a system Python, "
                   f"not an environment of yours")


def install_missing(checks, log=print, required_only: bool = True) -> int:
    """Install the Python packages the checks just found missing. Returns how many.

    Driven by the SAME list `run_checks` walks, and by the `pip install ...`
    string it already prints, so there is no second inventory of what RealMe
    needs.

    Only pip, and only into the interpreter running this -- `sys.executable`,
    not whatever `pip` resolves to on PATH, which on a machine with several
    environments installs into the wrong one and reports success. Nothing else
    on the list is touched: ffmpeg, LibreOffice and the equation reader are
    not pip's to install, and no other package manager is guessed at.
    """
    wanted = []
    for c in checks:
        if c.ok or not str(c.fix).startswith("pip install "):
            continue
        # Optional extras are suggested, never installed unasked. Equation
        # reading is a choice, not a missing part.
        if required_only and not c.required:
            continue
        wanted += str(c.fix)[len("pip install "):].split()
    # Ordered, de-duplicated: two checks can name one package.
    seen, pkgs = set(), []
    for w in wanted:
        if w not in seen:
            seen.add(w)
            pkgs.append(w)
    if not pkgs:
        return 0
    ok, where = managed_environment()
    if not ok:
        log(f"  {len(pkgs)} Python package(s) are missing, and this is not an "
            f"environment\n  RealMe should install into:\n\n           {where}"
            f"\n\n  Install them yourself, or run this from the RealMe "
            f"environment:\n\n      pip install {' '.join(pkgs)}\n")
        return 0
    log(f"  Installing {len(pkgs)} missing package(s) into the {where}")
    log(f"      {' '.join(pkgs)}\n")
    r = subprocess.run([sys.executable, "-m", "pip", "install", *pkgs])
    if r.returncode != 0:
        log("\n  pip failed. The packages come from PyPI, so this needs a "
            "working network\n  connection the first time; a proxy or a "
            "blocked index is the usual cause.")
        return 0
    return len(pkgs)


def explain_wrong_pip(missing: list, log=print) -> bool:
    """Say, before anything else, that `pip` here is not RealMe's pip.

    Only when something is actually missing: the situation is harmless until
    somebody tries to fix it by hand, and a healthy run stays quiet.
    """
    other = foreign_pip()
    if other is None:
        return False
    pip_exe, their_python = other
    log("A different Python owns `pip` in this window\n" + "=" * 60)
    log(f"  pip on PATH : {pip_exe}")
    log(f"  RealMe runs : {sys.executable}")
    there = installed_elsewhere(their_python, missing)
    if there:
        log("")
        log(f"  {', '.join(sorted(there))} "
            f"{'is' if len(there) == 1 else 'are'} installed over THERE, not "
            f"here.")
        log("  That is why RealMe still says missing after a successful "
            "`pip install`.")
    log("")
    log("  Typing `pip install <package>` in this window installs into that "
        "other Python.")
    log("  To install where RealMe will find it, either let this command do "
        "it for you,")
    log("  or name the interpreter:")
    log("")
    log(f'      "{sys.executable}" -m pip install <package>')
    log("")
    return True


def install_here(package: str) -> str:
    """The command that installs `package` where RealMe will actually find it.

    `pip install matplotlib` typed into a Command Prompt installs into
    whichever Python owns `pip` on PATH, and RealMe deliberately runs from a
    conda environment its launchers never activate -- so on most machines
    those are two different Pythons, the install reports success, and RealMe
    goes on saying the package is missing. Naming the interpreter removes the
    question.
    """
    return f'"{sys.executable}" -m pip install {package}'


def report(model: str = "", install: bool = True) -> int:
    """Check the machine, and by default fix the part that can be fixed here.

    Installing by default rather than on a flag, because the person who most
    needs this is the one who has just unzipped the package and does not yet
    know that `--fix` exists. Everything that is pip's to install gets
    installed; everything that is not -- ffmpeg, LibreOffice, the equation
    reader -- is printed as a command to run, rather than guessing at a
    package manager.

    `--no-install` turns it back into a pure diagnostic.
    """
    checks = run_checks(model)
    missing_pip = [c for c in checks
                   if c.required and not c.ok
                   and str(c.fix).startswith("pip install ")]
    names = sorted({w for c in missing_pip
                    for w in str(c.fix)[len("pip install "):].split()})
    if missing_pip:
        explain_wrong_pip(names)
    if install and missing_pip:
        print("Python packages\n" + "=" * 60)
        n = install_missing(checks)
        if n:
            print(f"\n  Installed {n}. Checking again.\n")
            checks = run_checks(model)      # measure again, do not assume
        print()
    blocking = [c for c in checks if c.required and not c.ok]
    optional = [c for c in checks if not c.required and not c.ok]

    redated = normalise_future_mtimes()
    if redated:
        print(f"Re-dated {redated} file(s) that arrived stamped in the future "
              f"(a timezone artefact of the update zip, not your clock).\n")

    archived = archive_retired()
    if archived:
        print(f"Moved {len(archived)} retired file(s) into _Archive:")
        for name in archived:
            print(f"  - {name}")
        print()

    gone = prune_superseded()
    if gone:
        print(f"Removed {len(gone)} launcher(s) that a rename made obsolete:")
        for name in gone:
            print(f"  - {name}")
        print()

    print("RealMe setup check\n" + "=" * 60)
    print(f"  python   {sys.executable}")
    for c in checks:
        mark = "ok  " if c.ok else ("MISS" if c.required else "--  ")
        print(f"  [{mark}] {c.name:24} {c.detail}")
        if c.note and not c.ok:
            print(f"           {c.note}")
    print()

    # Every pip command printed for a person to TYPE names the interpreter.
    # `c.fix` keeps its plain form because `install_missing` parses it; what
    # is shown is the form that installs where RealMe looks, avoiding the
    # problem `explain_wrong_pip` warns about.
    def _typed(c):
        if str(c.fix).startswith("pip install "):
            return install_here(str(c.fix)[len("pip install "):])
        return c.fix

    if blocking:
        print("REQUIRED - run these:\n")
        for c in blocking:
            print(f"  # {c.name}\n  {_typed(c)}\n")
    else:
        print("Everything required is present.\n")

    if optional:
        print("OPTIONAL - each unlocks one thing:\n")
        for c in optional:
            print(f"  # {c.name}: {c.detail}\n  {_typed(c)}\n")

    # Repeat the verdict at the END.
    #
    # The table above is the actual answer, but it is printed first and
    # scrolls away. `0_Update.bat` finishes with a pause, so the tail is what
    # is read; it states what passes as well as what is missing.
    print("=" * 60)
    present = [c.name for c in checks if c.ok]
    print(f"  {len(present)} of {len(checks)} checks pass.")
    if blocking:
        print(f"  Blocking: {', '.join(c.name for c in blocking)}")
    else:
        print("  Nothing required is missing.")
    if optional:
        print(f"  Optional, not installed: "
              f"{', '.join(c.name for c in optional)}")
    print("  Everything not listed on those two lines is installed and working.")

    # Which release this is, and whether the zip in the folder is a different
    # one. By content, not by date: update entries are backdated on purpose
    # (realme.core.build), so extracted files are always "older" than the
    # archive they came from and mtimes cannot answer the question.
    from realme.core.paths import project_root
    from realme.core.build import installed_id, update_pending
    root = project_root()
    if root is not None:
        print(f"  Version installed: {installed_id(root) or 'unstamped (pre-dates versioning)'}")
        pending, why = update_pending(root)
        if pending:
            print(f"  ! RealMe_Update.zip is a different version ({why}).")
            print("    Run 0_Update.bat to apply it.")

    # GPU readiness, said here rather than only in `engine cpp --cpp-action
    # status`. The C++ engine is what actually records lectures, and on a new
    # machine the difference between five hours and something practical is
    # whether it was compiled for the GPU sitting in it.
    # What this machine needs doing to the engine, measured rather than assumed.
    try:
        from realme.engines.cpp import engine_advice
        print("")
        for line in engine_advice():
            print(f"  {line}")
    except Exception:
        pass          # a status extra must never be what breaks setup

    return 1 if blocking else 0
