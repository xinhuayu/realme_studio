# Installing RealMe from source

RealMe turns a slide deck (PDF) into a narrated lecture video, optionally in
your own cloned voice. It is Python plus two things it does not contain:
**ffmpeg**, and the **voices**. Neither is in this repository. The commands
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
the draft voices, and then asks once whether you want a cloned voice.

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

**A cloned voice is optional.** Choose one, or none:

```
realme voice choose                  # asks: Gemini, the local clone, or none
```

- **Gemini** (`gemini-tts`) clones your voice in your Google project; your
  reference clip and every sentence are sent there, at about 1.7 cents a
  minute of audio. Use a **paid** project: on the free tier Google's terms say
  human reviewers may read API input and output.

  ```
  realme key GEMINI_API_KEY <your-key>
  realme voice enroll my_voice.m4a --transcript-file what_i_said.txt
  realme voice gemini --acknowledge-paid
  realme voice gemini --enroll --consent-wav consent.wav
  ```

  Record the reference and Google's consent sentence on the same microphone
  in the same room; Google refuses a pair that do not match.

- **qwen3cpp** clones your voice on your own computer: free, offline, much
  slower. Its source is vendored under `03_App/realme/engines/qwen3cpp`
  (MIT; see `PROVENANCE.md` there) and is compiled on the machine that uses
  it, which needs a C++ compiler:

  ```
  realme engine install clone        # model weights, conversion and the build
  realme engine status
  ```

A Gemini key is also what drafts narration from a deck. Without one, write the
narration yourself and import it (`realme import-script`).

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
