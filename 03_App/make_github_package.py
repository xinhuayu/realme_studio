#!/usr/bin/env python3
"""
Build the public source tree -- the thing that goes on GitHub.

This is NOT the update zip and not the colleague package. Those two carry a
working installation: a compiled engine, 2.5 GB of model weights, ffmpeg, the
piper voices. A source repository carries none of it. Someone who clones this
runs two commands and their machine fetches or builds its own, which is both
smaller and more honest -- the binaries in a package are built for one
machine's instruction set and were never redistributable in the first place.

What this adds over `git init && git add .`, and why each part exists:

  * **It refuses to publish a secret.** `.gitignore` already excludes `.env`,
    which protects the file nobody was going to commit by hand. It does
    nothing about a key pasted into a document, a test fixture or a comment --
    and a published API key is not a mistake you can take back by deleting the
    commit. Every shipped file is scanned for key-shaped strings.
  * **It refuses to publish the author.** A tree written on one machine is
    full of that machine: `C:\\Users\\<name>\\...` in an example, a path in a
    docstring, a name in a fixture. On a private install that is helpful. In a
    public repository it is someone's home directory and, usually, their real
    name.
  * **It leaves out the installed payload and the big media** -- `tools/`, the
    demo video, the compiled engine -- by reusing the same exclusion list the
    update packager uses, so the two can never drift apart.
  * **It writes INSTALL.md**, which is where "with commands to install the
    engines" lives. Generated rather than hand-kept so that it names the
    launchers and the CLI verbs that actually exist in the tree it ships.

Usage:
    python 03_App/make_github_package.py [outdir]     (default: ../RealMe_Source)
    python 03_App/make_github_package.py --zip        also write RealMe_Source.zip
"""
from __future__ import annotations
import argparse
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent

from realme.core.build import (GUIDE_VIDEO, SHIP_DIRS, SHIP_FILES,  # noqa: E402
                                _excluded, ensure_guide_video)

#: The superseded introduction: a deck and demo video describing a tool with
#: one voice that ran only locally. Neither ships any more and neither belongs
#: in a repository.
NOT_IN_GIT = {"test_slides.mp4", "RealMe_Introduction.pdf"}

#: The guide video DOES go in, by decision rather than by default, and it is
#: worth being clear about the cost. It is ~26 MB of already-compressed
#: binary: git cannot diff it, so every re-render adds another 26 MB to the
#: history for ever, and every clone pays for all of them. The usual answer
#: is to attach it to a GitHub Release instead and link that from the README,
#: which keeps the repository small and the video one click away. `--no-video`
#: does it that way; the default is to include it, because a reader who has to
#: build the demo to see the demo usually does not.
INCLUDE_VIDEO = True

#: Files whose whole purpose is to look like a key.
KEY_FIXTURES = {"example.env"}

#: Shapes of secret. Narrow on purpose: a pattern that fires on every long
#: hex string makes the check noise, and noise gets switched off.
SECRET_PATTERNS = (
    ("Google API key", re.compile(r"AIza[0-9A-Za-z_\-]{30,}")),
    ("OpenAI-style key", re.compile(r"\bsk-[A-Za-z0-9]{20,}")),
    ("ElevenLabs key", re.compile(r"\bxi-[A-Za-z0-9]{24,}")),
    # A real id always carries digits; `voice_fingerprint`, which is a method
    # name in six files here, does not. A scanner that cries wolf on its own
    # source is a scanner somebody passes --force to by reflex.
    ("Gemini voice id",
     re.compile(r"\bvoice_(?=[a-z0-9]*\d)[a-z0-9]{10,}\b")),
    ("Gemini voice key", re.compile(r"\bvoicekey_[A-Za-z0-9_\-]{16,}")),
    ("AWS key id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("assigned key", re.compile(r"(?i)(api[_-]?key|secret|token)\s*[=:]\s*"
                                r"['\"][A-Za-z0-9_\-]{20,}['\"]")),
)

#: Text that identifies the machine this was built on. Filled in from the
#: environment rather than hard-coded, so this works for whoever runs it.
def personal_patterns() -> list[tuple[str, re.Pattern]]:
    import getpass
    out = []
    try:
        user = getpass.getuser()
    except Exception:
        user = ""
    if user and len(user) >= 3 and user.lower() not in ("user", "root", "admin"):
        out.append((f"the username {user!r}",
                    re.compile(rf"(?i)\busers[\\/]{re.escape(user)}\b")))
        out.append((f"a home path for {user!r}",
                    re.compile(rf"(?i)/home/{re.escape(user)}\b")))
        out.append((f"the username {user!r}",
                    re.compile(rf"(?i)(?<![A-Za-z0-9]){re.escape(user)}(?![A-Za-z0-9])")))
    out.append(("an email address",
                re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")))
    return out

TEXT_SUFFIXES = {".py", ".md", ".txt", ".bat", ".html", ".json", ".toml",
                 ".cfg", ".ini", ".yml", ".yaml", ".env", ".ps1", ".sh",
                 ".ipynb", ".csv", ".xml", ".cmake", ".in", ".c", ".cpp",
                 ".h", ".mm", ".js", ".css", ".srt", ".vtt"}

#: Third-party source vendored unmodified (see its PROVENANCE.md). Its own
#: authors' addresses are part of the upstream licence notices, so the email
#: check does not apply inside it; keys and home paths still do.
VENDORED = ("03_App/realme/engines/qwen3cpp/",)

#: Retired code that no package carries; nor does the repository.
from realme.migrate import DROP_CODE  # noqa: E402

INSTALL_MD = """\
# Installing RealMe from source

RealMe turns a slide deck (PDF) into a narrated lecture video in your own
cloned voice. It is Python plus things it does not contain: **ffmpeg**, the
**voices**, and the compiled **voice engine** with its model. The commands
below fetch or build them on the machine that will use them, and every step
checks for a working copy first instead of downloading again.

## 1. Python and the package

Python 3.11 or newer.

```
conda create -n realme python=3.12     # or python -m venv .venv
conda activate realme
pip install -e 03_App
```

On Windows, `00_Windows\\1_Install.bat` does all of this, fetches ffmpeg and
the draft voices, and then asks once which voice lectures should use.

## 2. ffmpeg and ffprobe

```
00_Windows\\_get_ffmpeg.bat          # Windows: a static build into tools\\
brew install ffmpeg                  # macOS
sudo apt install ffmpeg              # Debian/Ubuntu
```

## 3. Voices

**Draft voices** (Piper: Amy, Ryan, Cori) are free, local and not your voice.
They are enough to use the whole system, and they are the second speaker in a
podcast.

```
realme engine install draft
```

**Your cloned voice** -- choose one, or none:

```
realme voice choose                  # asks: Gemini, the local clone, or none
```

- **The free local clone** (`qwen3cpp`) clones your voice on your own
  computer: nothing uploaded, nothing to pay, about real time on a small
  6-core PC (a 51-slide, 52-minute lecture rendered in 57 minutes on an
  Intel i7-10710U with 64 GB of DDR4-2667 and no graphics card). It runs on RealMe's revised
  qwen3-tts.cpp: the source of qwen3-tts.cpp (predict-woo, MIT) and GGML
  (MIT) is vendored under `03_App/realme/engines/qwen3cpp`, and RealMe's
  changes (`03_App/realme/engines/qwen3cpp_batch/REALME_CHANGES.md`) are
  applied to the copy that is built. It is compiled on the machine that uses
  it. On Windows that needs **Visual Studio 2022 Build Tools** with
  *Desktop development with C++* (it brings CMake and Ninja); on Linux or
  macOS, a C++17 compiler and CMake.

  ```
  realme engine install clone        # weights (~2.5 GB), conversion, the build,
                                     # and the 8-bit model lectures use
  realme engine status
  realme engine cpp --cpp-action rebuild-all    # after an update that changes it
  ```

  On Windows, `00_Windows\\3_Get_Voice_Engine.bat` and
  `00_Windows\\_Rebuild_Voice_Engine.bat` do the same. The revised engine is
  also published on its own as **realme-qwen3-tts.cpp**
  (`realme engine cpp --cpp-action export-release --export-to <folder>`
  writes it). `DESIGN.md` explains how it is built and why it is fast.

- **Gemini** (`gemini-tts`) clones your voice in your Google project; your
  reference clip and every sentence are sent there, at about a dollar an
  hour of lecture. Use a **paid** project: on the free tier Google's terms
  say human reviewers may read API input and output.

  ```
  realme key GEMINI_API_KEY <your-key>
  realme voice enroll my_voice.m4a --transcript-file what_i_said.txt
  realme voice gemini --acknowledge-paid
  realme voice gemini --enroll --consent-wav consent.wav
  ```

  Record the reference and Google's consent sentence on the same microphone
  in the same room; Google refuses a pair that do not match.

A script writer drafts narration from a deck: a Gemini key by default, or
Claude or OpenAI (`realme key ANTHROPIC_API_KEY ...` / `OPENAI_API_KEY`).
Without one, write the narration yourself and import it
(`realme import-script`).

## 4. Check it and start

```
python 03_App/verify_tree.py         # invariants, imports, launcher checks
python 03_App/run_tests.py           # every test suite
realme setup                         # what this machine has, what is missing
realme studio                        # the Studio, at http://127.0.0.1:8000
```

`START_HERE.md` is the user guide.

## What is deliberately not in this repository

| | why |
|---|---|
| `tools/` | ffmpeg, voice files, model weights and compiled engines: large, machine-specific, re-fetchable |
| any key, voice recording, profile or project | they live in `%USERPROFILE%\\RealMeStudio` (`~/RealMeStudio`), outside the repository |
| `_Archive/` and retired code | the development history |
"""


def _is_text(path: Path) -> bool:
    """Suffix first; a file with none (AUTHORS, LICENSE) is text unless it
    holds a NUL byte in its first 8 KB."""
    if path.suffix.lower() in TEXT_SUFFIXES:
        return True
    if path.suffix:
        return False
    try:
        return b"\0" not in path.read_bytes()[:8192]
    except OSError:
        return False


def scan(path: Path, rel: Path, problems: list[str]) -> None:
    if path.name in KEY_FIXTURES or not _is_text(path):
        return
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    vendored = rel.as_posix().startswith(VENDORED)
    patterns = SECRET_PATTERNS + tuple(
        (lb, pt) for lb, pt in personal_patterns()
        if not (vendored and lb == "an email address"))
    for label, pattern in patterns:
        for m in pattern.finditer(text):
            line = text[:m.start()].count("\n") + 1
            found = m.group(0)
            if len(found) > 48:
                found = found[:45] + "..."
            problems.append(f"{rel.as_posix()}:{line}  {label}: {found}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir", nargs="?", type=Path,
                    default=ROOT.parent / "RealMe_Source")
    ap.add_argument("--zip", action="store_true", help="also write a .zip beside it")
    ap.add_argument("--no-video", dest="video", action="store_false",
                    help="leave the ~26 MB guide video out, to be attached to "
                         "a GitHub Release instead of committed")
    ap.add_argument("--force", action="store_true",
                    help="copy even though the scan found something. For when "
                         "you have READ each line it printed and each one is a "
                         "false positive.")
    a = ap.parse_args()

    ensure_guide_video(ROOT, log=lambda m: print(f"  {m.strip()}"))
    skip = set(NOT_IN_GIT) | ({GUIDE_VIDEO} if not a.video else set())
    files: list[Path] = []
    for d in SHIP_DIRS:
        base = ROOT / d
        if not base.is_dir():
            continue
        for f in sorted(base.rglob("*")):
            if f.is_file():
                rel = f.relative_to(ROOT)
                if (not _excluded(rel) and f.name not in skip
                        and not any(dc in f"/{rel.as_posix()}" for dc in DROP_CODE)):
                    files.append(rel)
    for name in SHIP_FILES:
        if (ROOT / name).is_file() and name not in skip:
            files.append(Path(name))

    print(f"  {len(files)} files from {ROOT}")

    problems: list[str] = []
    for rel in files:
        scan(ROOT / rel, rel, problems)
    if problems:
        print(f"\n  {len(problems)} thing(s) that should not be published:\n")
        for p_ in problems[:40]:
            print(f"    {p_}")
        if len(problems) > 40:
            print(f"    ... and {len(problems) - 40} more")
        print("\n  A key in a public repository is not undone by deleting the "
              "commit.\n  Fix them, or --force if every line above is a false "
              "positive.")
        if not a.force:
            return 1
        print("\n  --force given; continuing anyway.\n")

    out = a.outdir.resolve()
    if out.exists():
        if not (out / "03_App").is_dir() and any(out.iterdir()):
            print(f"\n  {out} exists and does not look like a previous build. "
                  f"Not touching it.")
            return 1
        # A repository already published (GitHub Desktop): everything but
        # its `.git` is replaced, so the next commit shows exactly what
        # changed. Otherwise the old build goes.
        if (out / ".git").exists():
            for p in out.iterdir():
                if p.name != ".git":
                    (shutil.rmtree if p.is_dir() else Path.unlink)(p)
        else:
            shutil.rmtree(out)
    for rel in files:
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dst)
    (out / "INSTALL.md").write_text(INSTALL_MD, encoding="utf-8")

    # The repository's own .gitignore, not the install's: a clone starts with
    # none of the installed payload, and should not acquire it in a commit.
    (out / ".gitignore").write_text(
        (ROOT / ".gitignore").read_text(encoding="utf-8")
        + "\nRealMe_Source/\nRealMe_Package.zip\ngemini_probe/\n*.wav\n*.mp4\n",
        encoding="utf-8")
    # Launchers need CRLF on Windows and shell scripts LF everywhere, for
    # whoever clones it.
    (out / ".gitattributes").write_text(
        "* text=auto\n*.bat text eol=crlf\n*.sh text eol=lf\n"
        "*.pdf binary\n*.pptx binary\n*.mp4 binary\n*.png binary\n",
        encoding="utf-8")

    total = sum((out / rel).stat().st_size for rel in files)
    print(f"\n  wrote {out}")
    print(f"  {len(files) + 1} files, {total / 1e6:.1f} MB")
    print("\n  No key, no voice reference, no model and no compiled binary.")
    print("  INSTALL.md tells a reader how to fetch each of those.")
    if a.video and (out / GUIDE_VIDEO).is_file():
        print(f"\n  {GUIDE_VIDEO} is in it (~26 MB). git cannot diff a video,")
        print("  so each re-render adds that again to the history for ever.")
        print("  --no-video leaves it out; attach it to a Release instead.")

    if a.zip:
        archive = out.with_suffix(".zip")
        shutil.make_archive(str(out), "zip", root_dir=out)
        print(f"  and {archive}")

    print("\n  To publish it with GitHub Desktop, the first time:")
    print(f"    1. File - Add local repository - {out}")
    print("       It is not a repository yet: click \"create a repository\".")
    print("       Name it (realme-studio, say); leave README, .gitignore and")
    print("       licence at None -- the folder has its own. Create repository.")
    print("    2. Publish repository; untick \"Keep this code private\" for a")
    print("       public one.")
    print("  Later: run this again; GitHub Desktop shows what changed. Commit,")
    print("  then Push origin. The folder's git history is kept.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
