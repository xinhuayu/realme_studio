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
    out.append(("an email address",
                re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")))
    return out

TEXT_SUFFIXES = {".py", ".md", ".txt", ".bat", ".html", ".json", ".toml",
                 ".cfg", ".ini", ".yml", ".yaml", ".env", ".ps1", ".sh"}

INSTALL_MD = """\
# Installing RealMe from source

RealMe is Python plus two things it does not contain: **ffmpeg**, and a
**speech engine**. Neither is vendored here. Both are fetched or built by the
commands below, onto the machine that will use them — which is the only place
a compiled engine is meaningful, since a build is tied to the instruction set
it was compiled for.

Nothing here downloads anything twice: every step checks for a working copy
first and says so.

## 1. Python and the package

Python 3.11 or newer.

```
conda create -n realme python=3.12     # or python -m venv .venv
conda activate realme
pip install -e 03_App
```

On Windows, `00_Windows\\1_Install.bat` does the above and reports what it
found rather than assuming.

## 2. ffmpeg and ffprobe

Every duration in RealMe is measured, never estimated, and ffprobe is what
measures it. Both binaries are required.

```
00_Windows\\_get_ffmpeg.bat          # Windows: fetches a static build into tools\\
brew install ffmpeg                  # macOS
sudo apt install ffmpeg              # Debian/Ubuntu
```

## 3. A voice

Pick one. They are a real choice, not a ranking.

### gemini-tts — hosted, no GPU, about 1.7 cents a minute of audio

Your clone lives in a Google project; your reference clip and every sentence
of every lecture are sent there. Use a **paid** project: on the free tier
Google's terms say human reviewers may read API input and output and that the
content is used to improve Google products, and a recording of your voice is
personal information.

```
realme key gemini <your-key>              # or put GEMINI_API_KEY in a .env
realme voice enroll my_voice.m4a --transcript-file what_i_said.txt
realme voice gemini                       # prints Google's consent sentence
realme voice gemini --acknowledge-paid
realme voice gemini --enroll --consent-wav consent.wav
```

Record the reference and the consent clip **on the same microphone in the same
room**. Google compares them and refuses the pair if its speaker check
disagrees; a desk mic for one and a phone for the other fails that check even
though both are plainly you.

### qwen3cpp — local, free, offline, slower

Nothing about your voice leaves the machine. Roughly six times slower than
real time on a laptop CPU, so a fifty-minute lecture is several hours of
compute — and it will still render that lecture identically in five years.

```
realme engine install clone        # weights, GGUF conversion and the C++ build
realme engine status               # what is installed, what is missing
```

`00_Windows\\3_Get_Voice_Engine.bat` is the same thing with the paths filled
in. `realme engine cpp --target-cpu avx2` builds a binary that runs on any
Core chip since 2013, rather than only on the machine that compiled it.

### piper — free draft voices, not your voice

Fast, costs nothing, and the right tool for getting the words and the slide
timing right before you spend anything on the real recording. It is also the
second speaker in a podcast, and it is a perfectly good way to use the whole
system for free.

```
realme engine install draft        # amy, ryan, cori
```

## 4. Check it

```
python 03_App/verify_tree.py       # invariants, imports, launcher references
00_Windows\\_Run_Tests.bat          # the eleven suites
realme doctor                      # what this machine actually has
realme studio                      # the web interface, at 127.0.0.1:8765
```

## What is deliberately not in this repository

| | why |
|---|---|
| `tools/` | ~3 GB of ffmpeg, model weights and a compiled engine. Machine-specific, re-fetchable, and not ours to redistribute |
| the demo video and introduction deck | they travel in the package a colleague receives (`realme migrate --no-voice`), where the deck's "the video ships beside this file" is true |
| `_Archive/` | the development history. Worth keeping, not worth publishing |
| any voice, profile or project | yours lives in `%USERPROFILE%\\RealMeStudio`, which no update and no package touches unless you ask it to |
"""


def scan(path: Path, rel: Path, problems: list[str]) -> None:
    if path.suffix.lower() not in TEXT_SUFFIXES or path.name in KEY_FIXTURES:
        return
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    for label, pattern in SECRET_PATTERNS + tuple(personal_patterns()):
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
                if not _excluded(rel) and f.name not in skip:
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

    total = sum((out / rel).stat().st_size for rel in files)
    print(f"\n  wrote {out}")
    print(f"  {len(files) + 1} files, {total / 1e6:.1f} MB")
    print("\n  No key, no voice reference, no model and no compiled binary.")
    print("  INSTALL.md tells a reader how to fetch each of those.")
    if a.video:
        print(f"\n  {GUIDE_VIDEO} is in it (~26 MB). git cannot diff a video,")
        print("  so each re-render adds that again to the history for ever.")
        print("  --no-video leaves it out; attach it to a Release instead.")

    if a.zip:
        archive = out.with_suffix(".zip")
        shutil.make_archive(str(out), "zip", root_dir=out)
        print(f"  and {archive}")

    print("\n  To publish:")
    print(f"    cd {out}")
    print("    git init && git add -A && git commit -m \"RealMe\"")
    print("    gh repo create realme --public --source=. --push")
    return 0


if __name__ == "__main__":
    sys.exit(main())
