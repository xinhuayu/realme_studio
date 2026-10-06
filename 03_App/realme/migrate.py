"""
Moving a working install to another machine.

The point is to arrive with nothing left to download and nothing to compile:
the code, the tools already built, the model weights, and the enrolled voice --
so the new machine records a lecture on the first afternoon rather than
repeating a week of setup.

**What is deliberately left out.** `tools/qwen3` is 4 GB of PyTorch weights and
a CPU-only virtual environment for the engine that measured slower and less like
the speaker; `tools/vc` is 1.3 GB of voice-conversion weights for a path that was
tried, measured and closed. Their adapters travel -- they are a few KB, and
`realme engine install` brings the payload back on demand -- but their downloads
do not. That is the difference between a 2.5 GB package and an 8 GB one.

**The one thing that can still go wrong.** ggml compiles for the CPU it is built
on. A binary built on a 2024 laptop can meet an older processor and die with an
illegal-instruction error, which looks like a corrupt download and is not. Build
with `--portable` before packaging, or rebuild on the target; both are in the
README this writes.
"""
from __future__ import annotations
import shutil
import sys
import zipfile
from pathlib import Path

#: Installed payload worth carrying. Everything else under tools/ is rebuildable
#: or was deliberately retired.
KEEP_TOOLS = ("ffmpeg", "piper", "qwen3cpp")

#: Engine source for paths that were closed. Excluded on request; the adapters
#: remain and will say plainly that the payload is missing.
#: Matched anywhere in the path, not as a prefix: `shipped_files` yields
#: `03_App/realme/engines/ov/...`, so a startswith test silently matched
#: nothing and packaged them anyway.
DROP_CODE = ("/realme/engines/ov/", "/realme/engines/knnvc/")

#: Build leavings. The binaries travel; the intermediates that made them do not.
JUNK_DIRS = {"CMakeFiles", "__pycache__", ".git", ".sisyphus", "tests"}
JUNK_SUFFIX = {".o", ".obj", ".ilk", ".pdb", ".exp", ".ninja_deps", ".ninja_log",
               ".tmp", ".part"}


def _skip(rel: Path) -> bool:
    if set(rel.parts) & JUNK_DIRS:
        return True
    return rel.suffix.lower() in JUNK_SUFFIX


def _walk(base: Path, prefix: str, keep=None) -> list[tuple[Path, str]]:
    out = []
    if not base.is_dir():
        return out
    for f in base.rglob("*"):
        if not f.is_file():
            continue
        rel = f.relative_to(base)
        if _skip(rel) or (keep is not None and not keep(rel)):
            continue
        out.append((f, f"{prefix}/{rel.as_posix()}"))
    return out


def _piper_keep(rel: Path) -> bool:
    """The draft voices the application actually offers, and no others.

    `tools/piper` accumulates. Auditioning voices installs them, and eleven of
    them is 795 MB -- most of it voices nobody chose, riding along in every
    package because the folder was copied wholesale. The list of voices the
    Studio offers already exists in one place; this asks it rather than keeping
    a second copy that would drift.

    Both files travel or neither does: a piper voice is the `.onnx` AND its
    `.onnx.json`, and a half-present one fails at load rather than at install.
    """
    from realme.adapters.tts import DRAFT_VOICES
    name = rel.name
    for suffix in (".onnx.json", ".onnx"):
        if name.endswith(suffix):
            return name[: -len(suffix)] in DRAFT_VOICES
    return False


def _qwen3cpp_keep(rel: Path) -> bool:
    """Weights, and the binaries -- not the tree that produced them.

    `realme engine cpp` re-copies its source from the vendored copy inside the
    package, so shipping the build tree as well would be 60 MB of duplicate.
    """
    parts = rel.parts
    if parts and parts[0] == "models":
        return rel.suffix == ".gguf"
    if parts and parts[0] == "build":
        return rel.suffix.lower() in {".exe", ".dll", ".so", ".dylib"}
    # What the binary was built FOR. Without it the recipient's setup says
    # "built for CPU target 'unknown'" and the crash advice recommends the
    # rebuild that was already done.
    if len(parts) == 1 and rel.name == "build_info.json":
        return True
    return False


def collect(root: Path, data: Path, include_key: bool = False,
            lean: bool = True, include_voice: bool = True) -> dict:
    """Everything the package holds, grouped so the plan can be read."""
    from realme.core.build import shipped_files
    groups: dict[str, list[tuple[Path, str]]] = {}

    code = []
    for rel in shipped_files(root):
        posix = rel.as_posix()
        if lean and any(d in f"/{posix}" for d in DROP_CODE):
            continue
        code.append((root / rel, f"RealMe/{posix}"))
    groups["code"] = code

    # Through the project's own resolver, not `root/"tools"`. An engine
    # installed in the legacy location is still the engine, and hard-coding the
    # path here would quietly package nothing -- which a dry run shows as three
    # empty rows and a package that arrives useless.
    from realme.core.paths import find_tool
    for name in KEEP_TOOLS:
        base = find_tool(name) or (root / "tools" / name)
        keep = {"qwen3cpp": _qwen3cpp_keep, "piper": _piper_keep}.get(name)
        groups[f"tools/{name}"] = _walk(base, f"RealMe/tools/{name}", keep)

    # `scripts/` is the sender's own working material: test passages, draft
    # narration, and the renders made from them. It belongs in a package going
    # to your own next machine and in none going to a person.
    groups["scripts"] = (_walk(root / "scripts", "RealMe/scripts")
                         if include_voice else [])

    # The enrolled voice: the tuned reference, its transcript, the raw take it
    # came from, and the profile that ties them together.
    #
    # Carried when the package is for ANOTHER MACHINE OF YOURS, and left out
    # when it is for another person. A cloned voice is not a setting: it is a
    # recording of someone, with a consent statement attached to it, and the
    # default -- carry everything so the new machine records on the first
    # afternoon -- is exactly wrong when the new machine belongs to a
    # colleague. `include_voice=False` is what `--no-voice` passes.
    if include_voice:
        voice = _walk(data / "profile", "profile_data/profile")
        voice += _walk(data / "voice", "profile_data/voice")
        groups["voice"] = voice
    else:
        groups["voice"] = []

    if include_key:
        env = data / ".env"
        groups["key"] = [(env, "profile_data/.env")] if env.is_file() else []

    return groups


def plan(groups: dict) -> dict:
    rows, total, count = [], 0, 0
    for name, items in groups.items():
        size = sum(f.stat().st_size for f, _ in items if f.is_file())
        rows.append({"group": name, "files": len(items), "bytes": size})
        total += size
        count += len(items)
    return {"rows": rows, "total_bytes": total, "files": count}


README = """# RealMe on this machine

Everything is here: the code, ffmpeg, the three draft voices, the C++ engine with its
binaries and weights, the test scripts, and the enrolled voice. Nothing to
download and nothing to compile.

The only thing this package cannot bring is Python itself.


## Setup, in order

**1. Put the folder somewhere permanent.** Copy `RealMe` out of wherever you
unzipped it -- `C:\\Users\\<you>\\Documents\\realme` is as good as anywhere. It
does not have to match the path on the old machine.

**2. Install Miniconda, if this machine has no Python.**
https://www.anaconda.com/download/success -- accept the defaults. Close and
reopen any Command Prompt afterwards so it picks up the new PATH.

**3. Run `RealMe\00_Windows\1_Install.bat`.** It creates the environment and
installs the package. It finds ffmpeg and the voices already sitting in
`tools\` and skips downloading them, so this is minutes rather than an evening.

It ends by running `realme setup`, which prints what this machine needs:

    CPU        : Intel(R) Core(TM) Ultra 7 268V  [AMD64]
    engine     : runs here (built for CPU target 'avx2')
    GPU        : Intel(R) Arc(TM) Graphics
      -> a VULKAN build is worth it here: --cpp-action build --force --gpu vulkan

Read those three lines. They are measured on this machine, not assumed: the
binary is actually executed and the GPU actually queried.

**4. Restore your voice.**

    realme migrate --restore

Run it from the folder you unzipped into and it finds `profile_data` itself;
give it the path if it cannot. Your reference recording, its transcript and your
profile land in `%USERPROFILE%\RealMeStudio`, and the profile's stored path is
rewritten to this machine. You do not record again.

**5. Put the API key back**, if you want script writing here. It was left out of
the package on purpose -- a key in a zip travels further than you intend.

    realme key GEMINI_API_KEY <your key>

**6. Confirm.**

    realme setup
    realme voice show

`setup` should show everything present, and `voice show` should point at
`...\RealMeStudio\profile\baked_assets\voice_reference.wav`.


## Prove it works before you trust it

    realme bench --engines piper,qwen3cpp --lines 3 --mode sentence ^
      --script RealMe\scripts\VR_Themes_test_script.txt --yes

Three utterances on each engine: piper in seconds, qwen3cpp in a few minutes.
Listen to the joined file it names. If that sounds like you, everything that
matters survived the move.


## If qwen3cpp crashes immediately

`realme setup` will say so plainly:

    engine     : built for 'avx2', and it CRASHES on this processor
      -> realme engine cpp --cpp-action build --force --target-cpu baseline

That is not a corrupt download. ggml compiles for a chosen instruction set, and
this processor does not have it. Rebuilding needs a C++ compiler -- Visual
Studio Build Tools or w64devkit; `--cpp-action status` says which it found.


## When a GPU arrives

    realme engine cpp --cpp-action build --force --gpu cuda      (NVIDIA)
    realme engine cpp --cpp-action build --force --gpu vulkan    (any vendor)

`realme setup` says which is worth it here, and integrated graphics that share
system memory are named as such rather than recommended.


## What was deliberately left out

`tools\qwen3` (the PyTorch engine, ~4 GB) and `tools\vc` (voice conversion,
~1.3 GB). Both were measured and set aside; their adapters travel, and
`realme engine install` fetches the payload if it is ever wanted again.
"""


#: The README for a package going to another PERSON.
#:
#: Deliberately not a variant of the one above with two sentences changed. The
#: first step differs -- there is no voice to restore, there is a voice to
#: record -- and that is the step everything else depends on, so the two
#: documents diverge from the top rather than at a footnote.
README_FRESH = """# RealMe

Slide decks become narrated lecture videos in your own voice.

Everything is in this package. **You do not need a C++ compiler and you do not
need a GPU.**

| | |
|---|---|
| **The speech engine** | qwen3-tts, compiled, with its model weights. This is what speaks in your voice. |
| **Three draft voices** | Amy (US, female), Ryan (US, male), Cori (UK, female). Fast, robotic-ish, for checking a script before spending hours on the real thing -- and for the second speaker in a podcast. |
| **ffmpeg and ffprobe** | every audio and video operation, and every duration measurement |
| **RealMe itself** | the application, the Studio, the launchers |

It does not carry a voice of a person, and it does not carry an API key. You
will record your own voice in step 4 -- that is the point of the tool -- and
get your own key in step 3.

The Python libraries RealMe uses -- PyMuPDF, NumPy, SciPy, python-pptx, the
equation converter and a handful of others -- are installed by step 2 from the
Python package index, so that step needs a working network connection once.
Step 4 checks each one and installs anything that did not arrive.


## What you need first

| | Why |
|---|---|
| **Windows 10 or 11** | |
| **Miniconda or Anaconda** | the one thing a zip cannot carry is Python. https://www.anaconda.com/download/success -- accept the defaults |
| **A Gemini API key** | drafts the narration from your slides. Free from https://aistudio.google.com |
| **~6 GB free disk** | the engine and weights, plus your renders |
| **A phone and a quiet room** | 30-60 seconds of your speech |


## Setup -- four steps

**1. Unzip it somewhere permanent.** Copy the `RealMe` folder out of wherever
you unzipped -- `C:\\Users\\<you>\\Documents\\RealMe` is as good as anywhere. The
path does not have to match the machine it came from. From here on, that folder
is "the RealMe folder".

**2. Double-click `00_Windows\\1_Install.bat`.** This is the only launcher you
need. It creates the environment and installs the package -- a few minutes.

It is quick because everything else is already here: it finds ffmpeg, the draft
voices and the speech engine sitting in `tools\\` and skips downloading them.
On a machine without this package that step is an evening.

It ends by running `realme setup`, which measures THIS machine -- processor,
engine, graphics -- and prints what it needs. Read those three lines. They are
measured, not assumed: the engine is actually run and the graphics actually
queried.

**3. Open a Command Prompt and set your key.**

    cd /d C:\\Users\\<you>\\Documents\\RealMe
    realme key GEMINI_API_KEY your-key-here

`realme` works from that folder without activating anything -- there is a small
shim there that finds the environment for you.

The key is stored in a `.env` file in the folder you are standing in, so run
that from the RealMe folder and start RealMe from the same place. If you would
rather not think about it, double-click `00_Windows\\2_Set_Key.bat` instead: it
asks for the key and stores it somewhere permanent. `realme key --where` says
which one is being read, if you ever wonder.

**4. Check it.**

    realme setup

This measures this machine and lists everything RealMe needs, marking what is
missing. Any required **Python package** that is absent is installed for you,
into this environment, and the check runs again -- so this step usually ends
with "Everything required is present."

It stops at pip on purpose. ffmpeg, LibreOffice and the equation reader are not
pip's to install, so those stay as commands printed for you to run: a setup
script that guesses at a package manager is how a machine gets broken by
something that was trying to help. It also will not install into a Python that
came with the operating system -- if that is what it finds, it prints the line
and leaves the machine alone. `realme setup --no-install` reports without
changing anything.

Two things it recommends rather than installs. **PyMuPDF** reads tables and
figures in a deck more faithfully than the permissive fallback, and it is
AGPL-3.0 -- if some PDF backend already works, that is your decision to make.
**speech-rule-engine** speaks equations and comes from npm rather than pip.
Both are printed with the command, if they are missing.

Read the three lines at the end about processor, engine and graphics. They are
measured here, not assumed.

**5. Start it, and record your voice.**

    realme studio

That opens the app at `http://127.0.0.1:8000`. (`00_Windows\\4_Start_Studio.bat`
does the same thing with a double-click.)

Go to **Twin Setup -> Your voice** and record yourself:

  - **30 to 60 seconds.** Longer does not help -- the engine builds a fixed
    description of your voice either way.
  - **Read something you have the text of,** and paste that text in as the
    transcript. Your speaking rate is measured from it.
  - **One sitting, one room, one microphone.** A phone at a steady distance is
    fine. On iPhone use Voice Memos set to **Lossless** (Settings -> Voice
    Memos -> Audio Quality).
  - **Read normally** -- not slowly, not in a performance voice. The clone
    reproduces what you give it.

The page tells you whether the recording is worth cloning -- length, noise
floor, signal-to-noise -- before you commit to it. Press **Save this voice**
when you are happy.

That is the whole setup. From now on it is only ever `realme studio`.


## Your first lecture

Work cheap to expensive. The full render is slow -- roughly five hours of
compute for a fifty-minute lecture on a laptop -- so every step below exists to
stop you spending that on a script you had not read.

1. **Draft the narration.** Lecture tab. Drop in a `.pdf` or `.pptx`, fill in
   **Course context** and **Your teaching voice**, press **Draft narration**.
   One model call, no audio, seconds.
2. **Read it and edit it.** Free, and it is the step that decides whether the
   video is any good.
3. **Hear one line.** The button under each slide. Checks a pronunciation in
   seconds.
4. **Render.** Leave it running. If it stops partway, start it again --
   finished segments are reused.

A deck with speaker notes narrates from the notes. A deck without them narrates
from the visible slide text, and it reads exactly like that.

The full guide is `START_HERE.md` in the RealMe folder, and the same material
is in the app's **Help** tab.


## Where things live

| | Where | |
|---|---|---|
| **The RealMe folder** | wherever you put it | code and launchers. Replaced wholesale by an update. |
| **Your data folder** | `%USERPROFILE%\\RealMeStudio` | your voice, projects and renders. **Never touched by an update.** |
| **The engine** | `RealMe\\tools\\` | ffmpeg, voices, weights -- about 3 GB. Also never touched by an update. |

Override the data location with `REALME_HOME` if you want it elsewhere.


## If something is wrong

    realme setup           what is missing; installs the Python packages it can
    realme setup --no-install   report only, change nothing
    realme key --where     which .env is read, and which keys are visible
    realme doctor          which speech engines can actually run here

**`'realme' is not recognized`** -- you are not in the RealMe folder, or step 2
has not been run. `cd` to the folder first; `00_Windows\\5_Command_Prompt.bat`
opens a prompt that works from anywhere.

**`pip install` succeeded and RealMe still says the package is missing.** This
machine has more than one Python, and `pip` installed into the wrong one --
almost certain on Windows, where PATH rarely points at the conda environment
RealMe runs from. Do not fight it: `realme setup` installs into its own
interpreter, and it will name both Pythons and tell you which one has the
package. By hand, name the interpreter instead of trusting PATH:

    realme setup


**The app says there is no Gemini key.** Run step 3 again from the RealMe
folder, then reload the page. Do not use Windows `setx` -- it does not affect
the window you typed it in.

**The engine crashes immediately.** `realme setup` will say
`built for 'avx2', and it CRASHES on this processor`. That is not a corrupt
download -- the engine is compiled for a chosen instruction set. Rebuilding
lower needs a C++ compiler (Visual Studio Build Tools or w64devkit):

    realme engine cpp --cpp-action build --force --target-cpu baseline

**Windows blocks the engine.** A freshly compiled unsigned binary trips
Defender's reputation check, and the symptom is misleading -- the file is there
at the right size but access is denied. `realme engine cpp --cpp-action status`
reports `BLOCKED by Windows`. Do not rebuild: allow it in Windows Security ->
Virus & threat protection -> Protection history. If it recurs on every rebuild,
add a folder exclusion for `tools\\qwen3cpp\\build` only.


## If you have a GPU

Optional, and only worth it if rendering time bothers you. Needs a C++ compiler
and the vendor's toolkit:

    realme engine cpp --cpp-action build --force --gpu cuda      NVIDIA - CUDA Toolkit, ~3 GB
    realme engine cpp --cpp-action build --force --gpu vulkan    any vendor - Vulkan SDK, ~400 MB

`realme setup` says which is worth it here. Integrated graphics that share
system memory are named rather than recommended: decoding is limited by memory
bandwidth, and sharing it with the CPU is not a win.


## What was deliberately left out

`tools\\qwen3` (a PyTorch speech engine, ~4 GB) and `tools\\vc` (voice
conversion, ~1.3 GB). Both were measured and set aside for being slower and
less like the speaker. Their adapters travel, and `realme engine install`
fetches the payload if it is ever wanted.
"""


def write_package(out: Path, groups: dict, log=print, readme: str = "") -> Path:
    """Write the zip. Entries are backdated for the same reason updates are."""
    import time
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.localtime(time.time() - 24 * 3600)[:6]
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for name, items in groups.items():
            if items:
                log(f"  {name}: {len(items)} files")
            for src, arc in items:
                info = zipfile.ZipInfo(arc, date_time=stamp)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                with open(src, "rb") as f:
                    z.writestr(info, f.read())
                n += 1
        z.writestr(zipfile.ZipInfo("README_MIGRATION.md", date_time=stamp),
                    readme or README)
    log(f"  wrote {n + 1} files to {out} ({out.stat().st_size / 1e9:.2f} GB)")
    return out


def personal_entries(zip_path: Path) -> list:
    """Anything in the finished package that belongs to a person, not the tool.

    Read back from the ARCHIVE, not from the plan that built it. A flag that
    was supposed to exclude something and a file that is actually absent are
    different claims, and only the second one is worth making to somebody
    before you send them your voice. Cheap: the central directory carries the
    names, so nothing is decompressed.
    """
    import zipfile as _z
    try:
        with _z.ZipFile(zip_path) as z:
            names = z.namelist()
    except (OSError, _z.BadZipFile):
        return ["could not be read back"]
    def personal(n: str) -> bool:
        base = n.rsplit("/", 1)[-1]
        if n.startswith("profile_data/"):
            return True
        # `.env` holds a key; `example.env` is the blank template and is meant
        # to travel. `core.build._excluded` already draws exactly this line,
        # and this did not -- so a correct package was refused for carrying
        # the file the packager deliberately includes. Two places, one
        # decision, and they disagreed.
        if base.endswith(".env") and base != "example.env":
            return True
        if "/voice_reference" in n or "consent" in base.lower():
            return True
        # Anything recorded, anywhere in the archive. Not because a rendered
        # clip is the enrolment reference -- it is not -- but because a .wav
        # of the cloned voice reading a draft is the same disclosure by
        # another route, and `scripts/` is where those accumulate.
        if base.lower().endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg",
                                  ".opus", ".aac")):
            return True
        # Video is deliberately NOT on that list. The one video that ships,
        # `RealMe_Guide_narrated_v2.mp4`, is narrated in the author's cloned
        # voice and travels in every package on purpose: it is the demo a
        # recipient checks their install against, and the README links to it
        # (preflight checks the pair). Decided 19 September 2026 for the
        # earlier `test_slides.mp4`, and kept when the narrated guide replaced
        # it in October. If a second video ever ships, decide again here
        # rather than by omission.
        return False

    return sorted(n for n in names if personal(n))


def find_package(start: Path | None = None) -> tuple[Path | None, list[Path]]:
    """
    Where the extracted package is, and everywhere that was looked.

    `--restore <folder>` means the folder you unzipped into -- the one holding
    `profile_data` and `RealMe`. That is obvious once you have seen the layout
    and not before, so the common places are searched when no folder is given,
    and the failure lists them rather than repeating the requirement.
    """
    from realme.core.paths import project_root
    looked: list[Path] = []
    roots = [Path(start)] if start else []
    if not start:
        cwd = Path.cwd()
        roots = [cwd, cwd.parent]
        root = project_root()
        if root is not None:
            roots += [root, root.parent]
    for r in roots:
        r = Path(r).resolve()
        for cand in (r, r / "profile_data"):
            if cand in looked:
                continue
            looked.append(cand.resolve() if not cand.is_absolute() else cand)
            if (cand / "profile").is_dir() or (cand / "profile_data").is_dir():
                return (cand, looked)
    return (None, looked)


def restore(package_root: Path | None, data: Path, log=print) -> dict:
    """Copy `profile_data` into the data directory on the new machine."""
    found, looked = find_package(package_root)
    if found is None:
        where = "\n  ".join(str(p) for p in looked)
        raise ValueError(
            "Could not find the extracted package. Give the folder you unzipped "
            "into --\nthe one holding `profile_data` and `RealMe`:\n"
            "  realme migrate --restore D:\\RealMe_Migration\n\nLooked in:\n  "
            + where)
    src = found / "profile_data" if (found / "profile_data").is_dir() else found
    data = Path(data)
    copied = []
    for f in src.rglob("*"):
        if not f.is_file():
            continue
        dst = data / f.relative_to(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Never overwrite a voice already enrolled here: arriving with a package
        # should not silently replace a recording made on this machine.
        if dst.exists() and dst.stat().st_size == f.stat().st_size:
            continue
        if dst.exists():
            backup = dst.with_suffix(dst.suffix + ".before_migrate")
            shutil.copy2(dst, backup)
            log(f"  kept the existing {dst.name} as {backup.name}")
        shutil.copy2(f, dst)
        copied.append(str(dst))
    fixed = _repoint_profile(data, log)
    log(f"  restored {len(copied)} files into {data}")
    return {"copied": copied, "data": str(data), "repointed": fixed}


def _repoint_profile(data: Path, log=print) -> list[str]:
    """
    Rewrite absolute paths in profile.json to this machine.

    The profile stores the reference clip by absolute path. That path names the
    OLD machine's user folder, and on a new PC with a different username it
    points at nothing -- the voice would be sitting right there, restored, while
    every engine reported no reference enrolled.
    """
    import json
    pj = Path(data) / "profile" / "profile.json"
    if not pj.is_file():
        return []
    try:
        from realme.core.textio import read_text
        prof = json.loads(read_text(pj))
    except (ValueError, OSError):
        return []
    changed = []
    for key in ("voice_reference", "consent_recording"):
        old_path = prof.get(key)
        if not old_path:
            continue
        if Path(old_path).is_file():
            continue                      # same layout; nothing to do
        # Separator-agnostic basename. `Path(r"C:\...\x.wav").name` under
        # POSIX returns the WHOLE string -- backslashes are ordinary characters
        # there -- so the rewritten path was nonsense and the check that it
        # exists quietly declined to fix anything. This function exists to cross
        # exactly that boundary, so it cannot assume the host's separator.
        base = str(old_path).replace("\\", "/").rsplit("/", 1)[-1]
        here = Path(data) / "profile" / "baked_assets" / base
        if here.is_file():
            prof[key] = str(here)
            changed.append(key)
    if changed:
        pj.write_text(json.dumps(prof, indent=2), encoding="utf-8")
        for k in changed:
            log(f"  repointed {k} to this machine's folder")
    return changed


def preflight(root: Path, data: Path, groups: dict) -> list[tuple[bool, str]]:
    """
    The "did you remember" list, run rather than recited.

    Every item here is something that has actually been forgotten while
    assembling one of these: packaging code that an unapplied update had already
    superseded, carrying a binary compiled for the wrong processor, shipping a
    reference the profile no longer points at. Each returns (ok, message) so the
    caller can print the whole list rather than stopping at the first problem --
    you want to fix them in one pass, not one round trip each.
    """
    out: list[tuple[bool, str]] = []

    from realme.core.build import installed_id, update_pending
    pending, why = update_pending(root)
    out.append((not pending,
                f"code is current ({installed_id(root) or 'unstamped'})" if not pending
                else f"an update is waiting and would NOT be packaged ({why})"
                     " -- run 0_Update.bat first"))

    try:
        from realme.engines.cpp import status as cpp_status
        st = cpp_status()
        built = st.get("built_cpu", "unknown")
        state = st["binary_state"]
        out.append((state == "ok",
                    f"engine binary runs here" if state == "ok"
                    else f"engine binary is '{state}' -- fix before packaging"))
        out.append((built in ("avx2", "baseline"),
                    f"engine CPU target '{built}' travels to another machine"
                    if built in ("avx2", "baseline")
                    else f"engine CPU target is '{built}' -- build --force "
                         f"--target-cpu avx2 so it runs elsewhere"))
        out.append((st["weights_ready"],
                    "model weights present" if st["weights_ready"]
                    else "model weights missing -- --cpp-action convert"))
    except Exception as e:
        out.append((False, f"could not inspect the C++ engine: {e}"))

    try:
        from realme.app.profile import Profile
        prof = Profile(Path(data) / "profile").data
        ref = prof.get("voice_reference")
        preset = prof.get("voice_preset") or "unrecorded"
        ok = bool(ref and Path(ref).is_file())
        out.append((ok, f"voice reference enrolled (preset '{preset}')" if ok
                    else "no voice reference enrolled -- realme voice bake ..."))
    except Exception:
        out.append((False, "could not read the profile"))

    # What is actually in the box, named rather than assumed.
    #
    # "tools/piper: 22 files" is not an answer to "will they have Amy?". These
    # read the collected entries back and say which voices, which binaries,
    # which weights -- so a missing one is visible before the zip is sent
    # rather than after it is installed.
    try:
        from realme.adapters.tts import DRAFT_VOICES
        names = {Path(arc).name for _, arc in groups.get("tools/piper", [])}
        have = sorted(v for v in DRAFT_VOICES
                      if f"{v}.onnx" in names and f"{v}.onnx.json" in names)
        missing = sorted(set(DRAFT_VOICES) - set(have))
        out.append((not missing,
                    f"draft voices: {', '.join(have) or 'NONE'}"
                    + (f" -- MISSING {', '.join(missing)}; "
                       f"`realme engine install` adds them" if missing else "")))
    except Exception as e:
        out.append((False, f"could not check the draft voices: {e}"))

    ff = {Path(arc).name.lower() for _, arc in groups.get("tools/ffmpeg", [])}
    need = {"ffmpeg.exe", "ffprobe.exe"} if sys.platform.startswith("win") \
        else {"ffmpeg", "ffprobe"}
    out.append((need <= ff,
                "ffmpeg and ffprobe both travel" if need <= ff
                else f"ffmpeg payload is incomplete: {sorted(need - ff)} "
                     f"-- every duration measurement needs ffprobe"))

    q = [Path(arc).name for _, arc in groups.get("tools/qwen3cpp", [])]
    gguf = [n for n in q if n.endswith(".gguf")]
    binaries = [n for n in q if n.endswith((".dll", ".exe", ".so", ".dylib"))]
    out.append((bool(gguf) and bool(binaries),
                f"voice engine: {len(gguf)} weight file(s), "
                f"{len(binaries)} binar{'y' if len(binaries) == 1 else 'ies'}"
                + ("" if gguf and binaries else " -- INCOMPLETE, it will not speak")))

    # The guide's slides and the video made from them are a pair: the README
    # links to both, and one without the other is a broken link in the first
    # thing a recipient opens. Checked in every package, --no-voice included.
    code_names = {Path(arc).name for _, arc in groups.get("code", [])}
    from realme.core.build import GUIDE_VIDEO
    deck, demo = "RealMe_Guide_narrated_v2.pdf", GUIDE_VIDEO
    if deck in code_names or demo in code_names:
        both = deck in code_names and demo in code_names
        out.append((both, "the guide's slides and its video both travel"
                    if both else
                    f"{deck if deck in code_names else demo} travels without "
                    f"{demo if deck in code_names else deck} -- the deck's "
                    f"link to the video would be broken"))

    for name, items in groups.items():
        if name.startswith("tools/") and not items:
            out.append((False, f"{name} found nothing -- the package would "
                               f"arrive without it"))
    if not groups.get("scripts"):
        out.append((True, "no scripts/ folder (optional)"))
    return out
