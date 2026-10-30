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

On Windows, `00_Windows\1_Install.bat` does all of this, fetches ffmpeg and
the draft voices, and then asks once which voice lectures should use.

## 2. ffmpeg and ffprobe

```
00_Windows\_get_ffmpeg.bat          # Windows: a static build into tools\
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

  On Windows, `00_Windows\3_Get_Voice_Engine.bat` and
  `00_Windows\_Rebuild_Voice_Engine.bat` do the same. The revised engine is
  also published on its own as **realme-qwen3-tts.cpp**
  (`realme engine cpp --cpp-action export-release --export-to <folder>`
  writes it). `DESIGN.md` explains how it is built and why it is fast.
  `00_Windows\_Make_Voice_Server.bat` packs it as a voice server for other
  tools (an OpenAI-compatible `/v1/audio/speech` on this computer).

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
| any key, voice recording, profile or project | they live in `%USERPROFILE%\RealMeStudio` (`~/RealMeStudio`), outside the repository |
| `_Archive/` and retired code | the development history |
