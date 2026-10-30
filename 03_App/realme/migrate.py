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

#: The developer's own launchers and the scripts only they run. A package for
#: a colleague leaves them out so 00_Windows holds only what a colleague uses;
#: a package for your own next machine keeps them.
DEV_ONLY = ("00_Windows/8_Expressive_Experiment.bat",
            "00_Windows/9_Archive_Old_Docs.bat",
            "00_Windows/_Make_GitHub_Package.bat",
            "00_Windows/_Bench_Parallel.bat",
            "00_Windows/_Engine_Timing.bat",
            "00_Windows/_Engine_Speedup_Check.bat",
            "00_Windows/_Bench_Batch.bat",
            "00_Windows/_Engine_Check_8bit.bat",
            "00_Windows/_Engine_Check_Vocoder.bat",
            "00_Windows/_Make_qwen3cpp_Release.bat",
            "03_App/explore_expressive.py")

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
            lean: bool = True, include_voice: bool = True,
            include_qwen: bool | None = None) -> dict:
    """Everything the package holds, grouped so the plan can be read.

    `include_qwen`: carry the local Qwen3 engine (about 2.2 GB of the package).
    None means yes: the local clone is a standard part of RealMe (October
    2026), and a package without it leaves the recipient a compiler to
    install. Packed only if it is installed here.
    """
    if include_qwen is None:
        include_qwen = True
    from realme.core.build import ensure_guide_video, shipped_files
    groups: dict[str, list[tuple[Path, str]]] = {}

    # Collect the guide video to the root first, exactly as the update
    # packager does. Its own docstring says it is called by both packagers,
    # and it was not called here -- so a migration package carried the
    # guide's deck with no video, and only a preflight line said so.
    # Best effort, not a refusal: migrate reports its problems and lets the
    # person decide, and the preflight below still names a missing video.
    ensure_guide_video(root, log=lambda *_: None)

    code = []
    for rel in shipped_files(root):
        posix = rel.as_posix()
        if lean and any(d in f"/{posix}" for d in DROP_CODE):
            continue
        if not include_voice and posix in DEV_ONLY:
            continue
        # `shipped_files` always lists the BUILD_ID stamp, which on a tree
        # that has never built an update zip does not exist yet.
        if not (root / rel).is_file():
            continue
        code.append((root / rel, f"RealMe/{posix}"))
    groups["code"] = code

    # Through the project's own resolver, not `root/"tools"`. An engine
    # installed in the legacy location is still the engine, and hard-coding the
    # path here would quietly package nothing -- which a dry run shows as three
    # empty rows and a package that arrives useless.
    from realme.core.paths import find_tool
    for name in KEEP_TOOLS:
        if name == "qwen3cpp" and not include_qwen:
            groups["tools/qwen3cpp"] = []
            continue
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

**3. Run `RealMe\\00_Windows\\1_Install.bat`.** It creates the environment and
installs the package. It finds ffmpeg and the voices already sitting in
`tools\\` and skips downloading them, so this is minutes rather than an evening.

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
profile land in `%USERPROFILE%\\RealMeStudio`, and the profile's stored path is
rewritten to this machine. You do not record again.

**5. Put the API key back**, if you want script writing here. It was left out of
the package on purpose -- a key in a zip travels further than you intend.

    realme key GEMINI_API_KEY <your key>

**6. Confirm.**

    realme setup
    realme voice show

`setup` should show everything present, and `voice show` should point at
`...\\RealMeStudio\\profile\\baked_assets\\voice_reference.wav`.


## Prove it works before you trust it

    realme bench --engines piper,qwen3cpp --lines 3 --mode sentence ^
      --script RealMe\\scripts\\VR_Themes_test_script.txt --yes

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

"""


#: The README for a package going to another PERSON.
#:
#: Deliberately not a variant of the one above with two sentences changed. The
#: first step differs -- there is no voice to restore, there is a voice to
#: record -- and that is the step everything else depends on, so the two
#: documents diverge from the top rather than at a footnote.
def readme_fresh(include_qwen: bool = False) -> str:
    """The README inside a package for a colleague.

    Written for the default package: Gemini speaks in their cloned voice,
    piper drafts, and the local Qwen3 engine is an optional extra they can add
    with one launcher. When the package does carry the engine, the engine
    paragraph says so instead.
    """
    return README_FRESH.replace("{ENGINE_ROW}", _ENGINE_ROW[include_qwen]) \
                       .replace("{ENGINE_STEP}", _ENGINE_STEP[include_qwen]) \
                       .replace("{DISK}", "~6 GB" if include_qwen else "~2 GB")


_ENGINE_ROW = {
    False: ("| **Not included in this package: the free local voice clone** | "
            "`qwen3cpp`, about 3 GB -- see *The free local voice clone* below. |"),
    True: ("| **The free local voice clone** | `qwen3cpp`, ready to run: your "
           "cloned voice made on this computer, nothing uploaded, nothing to pay. "
           "Gemini is the default. |"),
}
_ENGINE_STEP = {
    False: """## The free local voice clone

Gemini clones your voice at Google: the clip it is cloned from, and every
sentence it speaks, go to Google, at about a dollar an hour of lecture. The
free local voice clone, `qwen3cpp`, records at no cost and nothing leaves your
computer, at about real time on a 6-core PC (an hour of lecture in about
an hour; longer on a laptop). This package does not carry it: build it with
`00_Windows\\3_Get_Voice_Engine.bat` (it needs Visual Studio Build Tools,
*Desktop development with C++*, and about 6 GB of disk; START_HERE.md section
12 has the steps), or ask Xinhua Yu, the RealMe developer, for a package that
carries it ready to run. Then choose `qwen3cpp` under **Twin Setup -> Voice
for lectures**, or for one render in the Lecture tab's **Recording** menu.

""",
    True: """## The free local voice clone

This package also carries `qwen3cpp`, the free local voice clone, ready to
run. It clones your voice on this computer, with the network unplugged,
nothing uploaded and nothing to pay, at about real time on a 6-core PC
(an hour of lecture in about an hour; longer on a laptop). The faster 8-bit
copy of its model is made from the one included the first time it records
(a minute or two). Choose
`qwen3cpp` under **Twin Setup -> Voice for lectures** to make it the default,
or pick it for one render in the Lecture tab's **Recording** menu.

""",
}


README_FRESH = """# RealMe

Slide decks become narrated lecture videos in your own voice.

| | |
|---|---|
| **Your voice** | cloned by Google's Gemini from 30-60 seconds of you reading. This is the voice a lecture uses by default. |
| **Three draft voices** | Amy (US, female), Ryan (US, male), Cori (UK, female). Free and local, for checking a script before recording it in your voice -- and the second speaker in a podcast. |
{ENGINE_ROW}
| **ffmpeg and ffprobe** | every audio and video operation, and every duration measurement |
| **RealMe itself** | the application, the Studio, the launchers |

It does not carry anyone's voice, and it does not carry an API key. You record
your own voice and set your own key below.

The Python libraries RealMe uses are installed by step 2 from the Python
package index, so that step needs a network connection once.


## What you need first

| | Why |
|---|---|
| **Windows 10 or 11** | |
| **Miniconda or Anaconda** | the one thing a zip cannot carry is Python. https://www.anaconda.com/download/success -- accept the defaults |
| **A Gemini API key on a paid Google project** | drafts the narration from your slides, and speaks in your cloned voice. Key from https://aistudio.google.com; link it to a billing account. Speaking costs about a dollar per hour of lecture. A free-tier key drafts narration, but Google's free-tier terms allow human review of what you send -- read the note in Twin Setup before cloning a voice on one. |
| **{DISK} free disk** | RealMe, the draft voices, and your renders |
| **A quiet room and a decent microphone** | 30-60 seconds of your speech, plus one consent sentence |


## Setup

**1. Unzip it somewhere permanent.** Copy the `RealMe` folder out of wherever
you unzipped -- `C:\\Users\\<you>\\Documents\\RealMe` is as good as anywhere.
From here on, that folder is "the RealMe folder".

**2. Double-click `00_Windows\\1_Install.bat`.** It creates the Python
environment and installs RealMe -- a few minutes. ffmpeg and the draft voices
are already in `tools\\`, so nothing large is downloaded. You
do not need a C++ compiler. It ends by running `realme setup`, which lists
anything still missing.

**3. Double-click `00_Windows\\2_Set_Key.bat`** and paste your Gemini key. It
is saved in your RealMe data folder (`%USERPROFILE%\\RealMeStudio\\.env`), not
in the RealMe folder, so an update never touches it. `realme key --where` says
which file is being read.

**4. Double-click `00_Windows\\4_Start_Studio.bat`.** The Studio opens in your
browser at `http://127.0.0.1:8000`. Nothing is on the internet: the page is
served by a program on your own machine. (From a command prompt opened with
`5_Command_Prompt.bat`, `realme studio` does the same.)

At the end of step 2 the installer asks once whether you want your own cloned
voice: **1** Gemini, **2** the free local clone, **3** no clone -- the free
draft voices read your lectures. You can change it any time in the Studio:
Twin Setup -> Voice for lectures.

**5. Make your cloned voice (optional)** -- skip this if you chose the draft
voices. Twin Setup tab:

  - **Your voice:** record or upload 30 to 60 seconds of yourself reading
    something you have the text of, and paste that text as the transcript.
    Read normally -- the clone reproduces what you give it. One room, one
    microphone. The page says whether the recording is good enough.
  - **Gemini cloned voice** (if you chose Gemini): tick that the key bills to a paid project, record
    the consent sentence Google requires (same microphone, same room), and
    press **Create the cloned voice**.

That is the whole setup. From now on it is only ever `4_Start_Studio.bat`.

{ENGINE_STEP}## Your first lecture

Work cheap to expensive. A render in your Gemini voice takes minutes and costs
about a dollar an hour of lecture; a draft in Amy's or Ryan's voice costs
nothing. Every step below exists to stop you spending either on a script you
had not read.

1. **Draft the narration.** Lecture tab. Drop in your slides as a `.pdf` --
   PowerPoint's *File -> Save As -> PDF* makes one; a `.pptx` works directly
   only if LibreOffice is installed. Fill in
   **Course context** and **Your teaching voice**, press **Draft narration**.
   One model call, no audio, seconds.
2. **Read it and edit it.** Free, and it is the step that decides whether the
   video is any good.
3. **Hear one line.** The button under each slide. Checks a pronunciation in
   seconds.
4. **Render.** Pick a draft voice first if the words are not settled (it is
   saved separately and cannot overwrite the real one), then your voice. If a
   render stops partway, start it again -- finished sentences are reused and
   not paid for twice.

A PDF carries no speaker notes, so the draft is written from what the slides
show. If you have already written your narration, use **New with my notes**
instead and give it the PDF and a text file with one section per slide.

`RealMe_Guide_v4.pdf` and `RealMe_Guide_notes_v4.txt`, in the RealMe folder,
make a good first lecture: they are the guide to RealMe itself.

The full guide is `START_HERE.md` in the RealMe folder, and the same material
is in the app's **Help** tab.


## Where things live

| | Where | |
|---|---|---|
| **The RealMe folder** | wherever you put it | code and launchers. Replaced wholesale by an update. |
| **Your data folder** | `%USERPROFILE%\\RealMeStudio` | your voice, projects and renders. **Never touched by an update.** |
| **Tools** | `RealMe\\tools\\` | ffmpeg, the draft voices, and the local engine if you add it. Also never touched by an update. |
| **Your cloned voice** | your Google project | Gemini keeps it there for a year from its last use; Twin Setup can delete it. |

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

The next two only concern the optional local voice clone.

**The local engine crashes immediately.** `realme setup` will say
`built for 'avx2', and it CRASHES on this processor`. That is not a corrupt
download -- the engine is compiled for a chosen instruction set. Rebuilding
lower needs a C++ compiler (Visual Studio Build Tools or w64devkit):

    realme engine cpp --cpp-action build --force --target-cpu baseline

**Windows blocks the local engine.** A freshly compiled unsigned binary trips
Defender's reputation check, and the symptom is misleading -- the file is there
at the right size but access is denied. `realme engine cpp --cpp-action status`
reports `BLOCKED by Windows`. Do not rebuild: allow it in Windows Security ->
Virus & threat protection -> Protection history. If it recurs on every rebuild,
add a folder exclusion for `tools\\qwen3cpp\\build` only.


## If you use the local engine and have a GPU

Optional, and only worth it if local rendering time bothers you. Needs a C++ compiler
and the vendor's toolkit:

    realme engine cpp --cpp-action build --force --gpu cuda      NVIDIA - CUDA Toolkit, ~3 GB
    realme engine cpp --cpp-action build --force --gpu vulkan    any vendor - Vulkan SDK, ~400 MB

`realme setup` says which is worth it here. Integrated graphics that share
system memory are named rather than recommended: decoding is limited by memory
bandwidth, and sharing it with the CPU is not a win.


## Not in this package

Anyone's voice and any API key, always. You record your own voice and set your
own key.
"""


#: The archive names, one per kind of package.
PACKAGE_NAME = {True: "RealMe_Migration.zip",    # for another machine of yours
                False: "RealMe_Package.zip"}     # for another person


def resolve_out(out, default_dir: Path, name: str) -> Path:
    """Where `--out` means the archive to go.

    `--out D:\\Transfer` named a folder that already existed, and the
    command tried to open the folder itself as a zip and crashed. A folder
    is the natural thing to type -- the launchers ask "where to write it"
    -- so a folder gets the default name inside it.

      - nothing given            -> default_dir / name
      - an existing folder       -> that folder / name
      - ends in a separator      -> that folder (created) / name
      - ends in .zip             -> exactly that file
      - anything else            -> treated as a folder (created) / name

    Double quotes are removed first. A Windows path cannot contain one, and
    they arrive by two routes: a pasted "Copy as path" value, and cmd's own
    rule that `"D:\\Transfer\\"` -- a quoted path ending in a backslash --
    reaches the program as `D:\\Transfer"`.
    """
    if out is None:
        return Path(default_dir) / name
    s = str(out).replace('"', "").strip()
    if not s:
        return Path(default_dir) / name
    trailing = s.endswith(("/", "\\"))
    p = Path(s).expanduser()
    if p.is_dir() or trailing:
        return p / name
    if p.suffix.lower() == ".zip":
        return p
    if p.exists():
        raise ValueError(f"--out {s} is an existing file that is not a .zip; "
                         f"give a folder or a name ending in .zip")
    return p / name


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


class _NotCarried(Exception):
    """The local engine is not in this package, so its checks do not apply."""


def preflight(root: Path, data: Path, groups: dict,
              include_voice: bool = True) -> list[tuple[bool, str]]:
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

    carries_qwen = bool(groups.get("tools/qwen3cpp"))
    if not carries_qwen:
        out.append((True, "free local clone (qwen3cpp) left out -- the "
                          "recipient can build it with 3_Get_Voice_Engine.bat "
                          "(Visual Studio Build Tools), or use Gemini or the "
                          "draft voices"))
    try:
        if not carries_qwen:
            raise _NotCarried
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
    except _NotCarried:
        pass
    except Exception as e:
        out.append((False, f"could not inspect the C++ engine: {e}"))

    try:
        if not include_voice:
            raise _NotCarried         # a colleague records their own
        from realme.app.profile import Profile
        prof = Profile(Path(data) / "profile").data
        ref = prof.get("voice_reference")
        preset = prof.get("voice_preset") or "unrecorded"
        ok = bool(ref and Path(ref).is_file())
        out.append((ok, f"voice reference enrolled (preset '{preset}')" if ok
                    else "no voice reference enrolled -- realme voice bake ..."))
    except _NotCarried:
        out.append((True, "no voice in this package -- the recipient records "
                          "their own, or uses the draft voices"))
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
    if carries_qwen:
        out.append((bool(gguf) and bool(binaries),
                    f"voice engine: {len(gguf)} weight file(s), "
                    f"{len(binaries)} binar{'y' if len(binaries) == 1 else 'ies'}"
                    + ("" if gguf and binaries
                       else " -- INCOMPLETE, it will not speak")))

    # The guide's slides and narration travel in every package; its video
    # only once it has been rendered.
    code_names = {Path(arc).name for _, arc in groups.get("code", [])}
    from realme.core.build import GUIDE_VIDEO, GUIDE_DECK, GUIDE_NOTES
    if GUIDE_DECK in code_names or GUIDE_NOTES in code_names:
        both = GUIDE_DECK in code_names and GUIDE_NOTES in code_names
        out.append((both, "the guide's slides and narration both travel"
                    + (f", with its video" if GUIDE_VIDEO in code_names else "")
                    if both else
                    f"the guide travels incomplete -- {GUIDE_DECK} and "
                    f"{GUIDE_NOTES} belong together"))

    for name, items in groups.items():
        if name == "tools/qwen3cpp" and not carries_qwen:
            continue                      # left out on purpose, said above
        if name.startswith("tools/") and not items:
            out.append((False, f"{name} found nothing -- the package would "
                               f"arrive without it"))
    if not groups.get("scripts"):
        out.append((True, "no scripts/ folder (optional)"))
    return out
