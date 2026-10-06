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

On Windows, `00_Windows\1_Install.bat` does the above and reports what it
found rather than assuming.

## 2. ffmpeg and ffprobe

Every duration in RealMe is measured, never estimated, and ffprobe is what
measures it. Both binaries are required.

```
00_Windows\_get_ffmpeg.bat          # Windows: fetches a static build into tools\
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

`00_Windows\3_Get_Voice_Engine.bat` is the same thing with the paths filled
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
00_Windows\_Run_Tests.bat          # the eleven suites
realme doctor                      # what this machine actually has
realme studio                      # the web interface, at 127.0.0.1:8765
```

## What is deliberately not in this repository

| | why |
|---|---|
| `tools/` | ~3 GB of ffmpeg, model weights and a compiled engine. Machine-specific, re-fetchable, and not ours to redistribute |
| the demo video and introduction deck | they travel in the package a colleague receives (`realme migrate --no-voice`), where the deck's "the video ships beside this file" is true |
| `_Archive/` | the development history. Worth keeping, not worth publishing |
| any voice, profile or project | yours lives in `%USERPROFILE%\RealMeStudio`, which no update and no package touches unless you ask it to |
