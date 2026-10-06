# RealMe Studio

Built around one rule: **an adapter either does the real thing, or it raises.
It never silently substitutes fake output for real output.**

## Inline delivery controls

```
Right. [[pause:600ms]] The estimate was [[emphasis]]substantially[[/emphasis]] larger.
[[hedging]] The reviewer called it [[mask]]complete nonsense[[/mask]], which is unhelpful.
[[slow]]Read this part carefully.[[/slow]]  As they say, [[fr]]c'est la vie[[/fr]].
```

| Marker | Effect |
|---|---|
| `[[pause:600ms]]` `[[breath]]` | Silence exactly there (capped at 2 s) |
| `[[hedging]]` | A natural hesitation, language-aware |
| `[[emphasis]]…[[/emphasis]]` | Modest emphasis |
| `[[slow]]` `[[fast]]` `[[soft]]` `[[clear]]` | Bounded delivery changes |
| `[[mask:450]]…[[/mask]]` | Words **never sent to the engine**; a soft tone replaces them |
| `[[fr]]…[[/fr]]` | Explicit language routing |

A marker you mistype (`[[pasue:400]]`) is **left visible in the text** rather
than silently dropped, so you can find it. Masked words never reach the speech
engine and never appear in the captions.

## Writing a whole script at once

```bash
realme import-script lecture.md --project ./out
realme lecture deck.pdf -o ./out --script imported
```

Mark slides with `## Slide 1: Title` or `[[slide 1: Title]]`. Labels must form a
complete 1..N sequence — a gap is refused rather than guessed at, since guessing
maps narration onto the wrong slide silently.

## Slide timing

Each slide gets a **3 s silent lead-in** and a **3 s outro hold** by default.
The lead-in stops the slide cut and the voice onset landing on the same frame,
which reads as a burst; the outro is where a point lands before the slide turns.
Both are per-slide settings, and `[[pause:400ms]]` gives finer control inside
the narration. Captions start when speech does; chapter markers mark when the
slide appears.

## Your voice

```bash
realme voice check   reference.wav        # should you re-record?
realme voice compare reference.wav        # renders four presets to listen to
realme voice bake    reference.wav --preset natural
realme voice consent --name "Your Name"
```

`check` measures duration, peak, noise floor and signal-to-noise and gives
actionable advice — "turn off the air conditioning and get closer to the mic",
not a number. `bake` polishes the reference once and installs it as the
profile's voice, so the cloning backend learns the improved version and every
lecture inherits it.

Presets: `off`, `clean`, `natural` (default), `warm`. All are ffmpeg — highpass,
FFT denoise, corrective EQ, de-esser, gentle compression, EBU R128 to −16 LUFS,
limiter. Nothing to install and nothing exotic.

Pick conservatively: a clone trained on an over-processed reference sounds
processed in every lecture you make, and that is not undoable.

## API keys

```bash
realme key GEMINI_API_KEY your-key-here     # writes .env, effective at once
realme key --where                          # what is read, and what is visible
```

Keys are read from `./.env`, then `$REALME_HOME/.env`, then
`~/RealMeStudio/.env`. A real environment variable always wins. On Windows
prefer this to `setx`, which does not affect the terminal you type it in.

## ffmpeg

RealMe looks for `ffmpeg` and `ffprobe` in `tools/ffmpeg/` before falling back to
PATH, so a private static copy can live beside the app. On Windows this avoids
installing ffmpeg through conda, which mixes channels and produces DLL errors
unrelated to video (`gdk_pixbuf-2.0-0.dll not found` and friends).
`00_Windows/_get_ffmpeg.bat` fetches one. `REALME_TOOLS` overrides the location.

## Install

```bash
apt-get install -y ffmpeg poppler-utils libreoffice espeak-ng
pip install -e .
realme doctor      # exactly which adapters can run here, and why not
```

## The app

```bash
realme studio                 # http://127.0.0.1:8000
realme studio --colab         # opens through Colab's built-in port proxy
```

Three tabs:

- **Lecture** — upload a deck, draft narration, **read and edit it**, then
  render. The edit step is the point: everything before it is one cheap model
  call, everything after it costs money and minutes.
- **Podcast & Debate** — topic (optionally with a source paper), two voices,
  rendered turn by turn and mastered to −16 LUFS with a timestamped transcript.
- **Twin Setup** — voice reference, consent recording, teaching-voice notes,
  course context, adapter bindings. Captured once, reused all term.

Data lives in `~/RealMeStudio` (override with `REALME_HOME`).

## Command line

```bash
export GEMINI_API_KEY=...     # free tier at aistudio.google.com

realme lecture my_deck.pdf -o ./out \
  --script gemini --tts elevenlabs \
  --context "PUBH 6002, second-year MPH. They know DAGs but not IPTW." \
  --style   "Dry and precise. Worked examples over abstraction."

realme dialogue "Does target trial emulation solve immortal time bias?" \
  -s paper.pdf --mode debate --turns 10 --tts qwen3cpp --guest-tts piper

realme verify ./out/my_deck.mp4
```

## Synchronised signalling

While the narration discusses a region of the slide, that region is cued. The
boxes are **read out of the PDF text layer**, not estimated by a model, so they
land on the exact words; the timing comes from measured utterance boundaries, so
it cannot drift out of sync with the audio.

```bash
realme lecture deck.pdf --signalling highlight   # translucent box (default)
realme lecture deck.pdf --signalling underline   # coloured bar beneath
realme lecture deck.pdf --signalling focus       # dim everything else
realme lecture deck.pdf --signalling off
```

Cues are derived deterministically — phrases that appear both on the slide and
in the narration — so it degrades to *no cues* rather than *wrong cues*, which
is the right failure direction for something overlaid on a lecture. Slide titles
are excluded: cueing the title while the narrator reads the title is signal with
no information in it. The cue list is written to `<project>_cues.json`.

## Hearing one line before you render twenty minutes

```bash
realme preview "The OR was 2.3 with p < 0.05." --tts piper
```

Prints the normalized text, the engine text, which lexicon entries fired and any
warnings — and writes the WAV. In the Studio there is a **Hear first line**
button on every segment. This is the cheap loop for the thing that actually goes
wrong: you hear "the *or* was 2.3", add a lexicon entry, hear it again.

## Mathematics

```bash
realme math '\hat{\beta}_1 \sim \chi^2'      # -> beta hat sub 1 distributed as chi squared
realme math --pending                          # expressions awaiting approval
```

`$...$` spans in narration are converted via `latex2mathml` → Speech Rule Engine
(`clearspeak`), then corrected for domain semantics no rule engine can know:
`\beta^{\top}` is "beta transpose", not "beta raised to the down tack power";
`\perp` is "independent of"; `\mid` is "given". Approved strings are cached
against a hash of the LaTeX, so a deck's recurring expressions are fixed once.

Requires `pip install latex2mathml` and `npm install -g speech-rule-engine`.
Without them, math spans are left alone and reported as a warning.

## Speaker notes

A `.pptx` carries real speaker notes and they are now read from it — they are
what you *meant to say*, whereas the on-slide text is only what the audience can
already read. Feeding the model the wrong one produces narration that recites
the slide. Notes are placed above the slide text in the prompt and the model is
told they outrank it.

## Does the audio actually say it correctly?

```bash
realme check-audio segment.wav IPTW OR MCAR
realme lecture deck.pdf --no-audio-check      # skip it
```

Runs automatically after every render. For each lexicon term it synthesizes both
the pronunciation you intended and the one the engine produces unaided, slides
each across the rendered audio, and reports which one the audio matches. Closed-
set, so no threshold-tuning and no language model — and no model downloads.

Verified in both directions: on correctly-rendered audio all terms come back
`intended`; on audio rendered the way v2 did it, IPTW, OR and MCAR are all
caught. Terms whose templates are too short to discriminate are reported
`not checkable` rather than guessed at — a wrong "this is fine" is worse than
"I could not tell".

## The local voice engine

```bash
realme engine status
realme engine install                       # download runtime + weights
realme engine install --from /path/to/real_voice_qwen3   # copy instead
realme engine install --force               # rebuild a broken install
```

The worker is vendored at `realme/engines/qwen3/`; the runtime (~500 MB) and
weights (~1.9 GB) install into `tools/qwen3/`. The runtime is the CPU build of
PyTorch with gradio excluded — `qwen-tts` declares gradio but references it only
in `demo.py`, which is never imported. `--full` disables both economies if the
lean install ever misbehaves. Same principle as ffmpeg and
Python: the repository carries our code and instructions for everything else,
so it stays publishable without shipping other people's binaries.

Every step is skip-if-present. A venv that exists but cannot import `qwen_tts`
is repaired rather than deleted — deleting it would re-download PyTorch to fix
something much smaller.

## Local Qwen3-TTS

Drives the worker from a `real_voice_qwen3` checkout rather than reimplementing
it, so it inherits that project's clone-prompt cache, synthesis cache, warm-up
and CPU tuning.

```bash
realme key REALME_QWEN3_ROOT "/path/to/real_voice_qwen3"
realme key REALME_QWEN3_REF_TEXT "the exact words spoken in your reference"
realme lecture deck.pdf --tts qwen3
```

Qwen3-TTS conditions on the reference audio **and its exact transcript**, so the
second key is required, not optional. See
`01_Architecture/Adopted_From_real_voice_qwen3.md`.

## PDF backend

PyMuPDF is preferred: one dependency replaces pdfplumber, pypdf **and the
poppler system binary**, and it is faster. Note it is AGPL-3.0 (or a paid
Artifex licence) where those are permissive; obligations attach to network
distribution, not to local use. Uninstall it and the permissive fallback takes
over automatically — `realme doctor` reports which is active.

## Voices

| `--tts` | What | Cost | Use when |
|---|---|---|---|
| `qwen3` | **Local Qwen3-TTS, clones your voice, CPU** | **$0** | **The default if you have real_voice_qwen3 installed** |
| `espeak` | *Retired September 2026* | — | The placeholder from before cloning worked. `espeak-ng` the program is still needed: the lexicon asks it whether a term reads as a word or as letters. |
| `elevenlabs` | Hosted clone, commercial licence on paid tiers | ~$22/mo | Fastest path to a real twin |
| `google_chirp3` | Chirp 3 Instant Custom Voice | ~$3.24/hr | Staying inside Google — once allowlisted |
| `chatterbox` | MIT, self-hosted, clones from ~5s | $0 + GPU | The voice model must never leave your machine |

## Why timing is measured, never estimated

Every duration comes from `ffprobe` on real audio. Captions, chapters and video
length all derive from those measurements; `render_segment` asserts each
segment is within 0.25s of its audio, and `realme verify` re-probes the finished
file independently. Measured drift on the demo deck: **0.049s over 121s**.

## Resuming

The ledger is content-hashed. Edit slide 7 of 20 and only slide 7 re-renders;
everything else is reused. Editing narration clears that segment's measured
duration so captions can never trust a stale number.

## Not built, on purpose

- **Avatar** — deferred. `compose.render_segment()` already takes `avatar_mp4`
  and implements `pip` and `side_by_side`; adding one `BaseAvatar` adapter is
  the whole job. EchoMimicV3 (Apache 2.0, 12 GB VRAM) on hardware you control.
- **YouTube auto-publish** — uploads from an unaudited API project lock to
  *private* with no appeal. Use the generated chapter file and set visibility
  by hand.
