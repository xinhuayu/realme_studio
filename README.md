# RealMe

Slide decks become narrated lecture videos in your own voice. Documents are
read aloud. A paper or a topic becomes a two-voice podcast.

Voice cloning is optional, and the installer asks. **Gemini** (`gemini-tts`)
clones your voice in your Google project: no graphics card needed, a lecture
records in about a third of its length (100 minutes in 30–40), about a dollar
an hour of lecture — your recording and the
narration are sent to Google, so use a **paid** API project. **The free local
clone** (`realme-rev-qwen3-tts`) keeps everything on your computer at no
cost, on RealMe's revised voice engine: faster than real time on a 6-core PC
(a 52-minute lecture recorded in 30 minutes on a 2019 NUC). It comes in a
RealMe package, or `3_Get_Voice_Engine.bat` builds it (Visual Studio Build
Tools); `_Rebuild_Voice_Engine.bat` builds it again for a new processor. Or use the
three free draft voices for everything.

Slides go in as a **PDF**. A `.pptx` works only where LibreOffice is
installed.

**New here? Open [START_HERE.md](START_HERE.md).** It is the whole guide:
install, make your voice, and a first lecture.

## The guide

Nineteen slides walk through the Studio — install, your cloned voice, the
workflow, the editor, voice and timing, revising, tools, reading aloud and
podcasts:

- **The slides** — [`RealMe_Guide_v4.pdf`](RealMe_Guide_v4.pdf), and the
  editable [`RealMe_Guide_v4.pptx`](RealMe_Guide_v4.pptx) with the narration in
  its speaker notes.
- **The narration** — [`RealMe_Guide_notes_v4.txt`](RealMe_Guide_notes_v4.txt).

Render it in your own voice — it makes a good first lecture: Lecture tab →
**New with my notes**, choose the PDF and the notes file, then Render. Or:

```
realme import-script RealMe_Guide_notes_v4.txt --project guide_v4
realme lecture RealMe_Guide_v4.pdf --script imported -o guide_v4
```

## The research report

[Running AI models locally on a small budget: how software design made a
cloned-voice lecture narrator practical on an ordinary CPU](03_App/realme/engines/qwen3cpp_batch/release/paper/realme-qwen3-tts-report.pdf)
(November 2026) describes what limits the voice engine on a laptop processor
without a graphics card, which changes made it about eight times as fast,
which did not pay, and how a full lecture compares with Gemini. Its sources
are beside it in `paper/source/`.

## What is in this folder

```
00_Windows\     The launchers. 1, 2 and 4 once; after that only 4.
03_App\         The application — a Python package.
tools\          ffmpeg and the draft voices (and the local clone, if added).
START_HERE.md   The guide.
DESIGN.md       How the local voice engine is built, and why it is fast.
HANDOVER.md     How to package RealMe for a colleague.
realme.bat      Runs the command line without activating the environment.
```

Your own material does not live here. It lives in `%USERPROFILE%\RealMeStudio`
— your key, your voice, your projects and renders — and no update touches it.

## What it does

- Slide deck → narrated video, with captions and chapters
- Revise a recorded deck; only what changed is spoken again
- A video played between slides, remembered by the lecture
- Split a lecture at slide boundaries; join sections across lectures
- Any document read aloud, with captions
- Two-voice podcast, Socratic dialogue, debate
- A saved speed per voice, and slide gaps — re-timing, never re-recording
- Pronunciation lexicon, LaTeX equations, delivery markers

## Two rules the code follows

1. **A voice either does the real thing, or it says why not.** Nothing
   silently substitutes a placeholder for your voice, and nothing reports
   success it has not checked.
2. **Every word passes through the text layer before it is spoken.** Symbols,
   equations, money and markup are turned into words first; anything with no
   spoken form is reported, never silently dropped.

## For a maintainer

```
python 03_App\verify_tree.py         checks that must pass before packaging
python 03_App\run_tests.py --json    every check and suite; one exit code and a JSON summary
python 03_App\make_update_zip.py     build RealMe_Update.zip
```

All three want the environment's own Python; `verify_tree.py` prints the right path
if run under another one.

**[DESIGN.md](DESIGN.md)** describes how the free local voice clone is built
and why it is fast: the revised engine, the recording pipeline around it,
the defaults and the switches. Every change to the engine, with its switch
and how it was checked, is in
`03_App\realme\engines\qwen3cpp_batch\REALME_CHANGES.md`.

### The local voice engine

- **The local voice is `realme-rev-qwen3-tts`** (DESIGN.md section 0). Studio
  builds it from the engine source below for each computer's processor:
  `_Rebuild_Voice_Engine.bat` (`realme engine rev`; `force` to compile again
  anyway) into `tools\realme-rev-qwen3-tts`, reading the 8-bit model files
  in `tools\qwen3cpp\models_q8` as they are. Its settings are fixed in
  `engines/rev.py` (`realme engine rev --threads N --cp-steps 12|15` changes
  this computer's), and every take is checked as it is recorded. The previous
  engine (`qwen3cpp`, described below) is no longer offered as a voice; it
  still converts and quantizes the model files (`_Rebuild_Voice_Engine.bat
  legacy`) and is the portable build a colleague's package carries.
- RealMe's engine source is `03_App/realme/engines/qwen3cpp_realme`: the
  original qwen3-tts.cpp with every RealMe change already made, RealMe's own
  `src/realme_*.inc`, and GGML's one gated file under `ggml_overlay/`. Every
  build compiles it; nothing is patched while building. The original
  (`engines/qwen3cpp`, with GGML) is kept unchanged as the record and as
  GGML's source, and is built only when asked for by name (`realme engine
  cpp --cpp-action build --source original`, into `tools\qwen3cpp_original`;
  RealMe never uses that build). A change to an upstream file is made as a
  patch in `engines/cpp.py` (`PATCHES`, `apply_sgemm_patch`), and `realme
  engine cpp --cpp-action integrate` writes `qwen3cpp_realme` again from the
  original.
- `3_Get_Voice_Engine.bat` downloads and converts the weights, builds the
  engine and makes the 8-bit model. `_Rebuild_Voice_Engine.bat` (`realme
  engine cpp --cpp-action rebuild`; `rebuild-all` also restages GGML and
  builds the timing copy) rebuilds for the same processor and GPU setting,
  keeping the working engine aside and restoring it if the build fails.
  Packages do not carry `models_q8\`; a build or the first recording makes it.
- Defaults: the 8-bit model once made (`REALME_QWEN3CPP_QUANT=f16` for the
  16-bit one), one thread per core up to 6 (`REALME_ENGINE_THREADS`), the
  fast snake and the 32-bit vocoder with RealMe's own convolution kernel
  (`QWEN3_TTS_FAST_SNAKE`; `QWEN3_TTS_VOCODER_F32=3` on an engine built with
  it -- AVX2 on x86, chosen at run time, NEON on 64-bit ARM -- else `2`, else
  `1` with `QWEN3_TTS_VOCODER_F32_MAX_FRAMES` by memory), RealMe's 8-bit
  multiply for a pass (AVX2/NEON, checked against GGML's at first use;
  `QWEN3_TTS_Q8_BATCH=0` off), one pool of engine threads kept between steps
  (`QWEN3_TTS_POOL=0` off), 8 sentences per pass (`REALME_RECORD_BATCH`,
  1 = off), each finished
  sentence handing its place to the next (`REALME_RECORD_STREAM=0` for fixed
  passes, made of sentences of similar length: `REALME_RECORD_LOOKAHEAD`,
  1 = in order). A computer with cores
  to spare records with more than one engine at once: one per 6 cores, at
  most 4, as free memory allows (`REALME_RECORD_WORKERS`). Slide videos are
  made on the threads the engine leaves free (`REALME_EARLY_VIDEOS`).
  `realme setup` says what this computer will use.
- Measuring on the real voice: `_Tune_Voice_Engine.bat`
  (`03_App\run_cpu_combinations.py`) times every combination of the local
  voice's switches and pass sizes, checks they give the same audio, and says
  the fastest; `03_App\run_codebook_listening.py` renders the predictor-step
  and 8-bit-vocoder variants side by side to listen to. For the previous
  engine, `realme engine-timing --compare`, `realme bench-batch` and
  `realme bench-parallel` remain on the command line (their launchers were
  retired in November 2026). None changes the working engine.

### The voice server

`00_Windows\_Make_Voice_Server.bat` (`realme engine cpp --cpp-action
export-server --export-to <folder> [--with-my-voice]`, `cpp.export_server_bundle`)
writes `..\realme-qwen3-tts-server`: the engine as built here (`bin/`), the
8-bit model, and `server/qwen3_tts_server.py` from
`engines/qwen3cpp_batch/release/server/` -- an OpenAI-compatible
`/v1/audio/speech` in Python's standard library, voices as WAVs in
`voices/` (`realme.wav` is the default voice "realme"), several sentences
(and simultaneous requests) per pass. Settings in `voice_server.json`
(`host`, `port` -- 8800 unless a tool needs another --, `voice`, `api_key`,
`batch`, `pause`, `model_aliases`), overridden by command-line options. It
answers only to the model `realme-qwen3-tts`; other names (`tts-1`) only
when listed in `model_aliases`. It runs without RealMe; re-exporting keeps
its `voices/` and `voice_server.json`. Tests: `t_voice_server.py`.

### realme-qwen3-tts.cpp, the engine on its own

RealMe's revised qwen3-tts.cpp is also published as a standalone source
release, **realme-qwen3-tts.cpp** (MIT; predict-woo's and the ggml authors'
credit and licences kept). `00_Windows\_Make_qwen3cpp_Release.bat` (or
`realme engine cpp --cpp-action export-release --export-to <new folder>`)
writes it from RealMe's engine source (`engines/qwen3cpp_realme`), with GGML
from `engines/qwen3cpp` and its tinyBLAS gate -- upstream's README as
README_UPSTREAM.md, and the release's own README, CHANGES
(`engines/qwen3cpp_batch/REALME_CHANGES.md`), NOTICE, RELEASE_NOTES, LICENSE, `build.bat`/`build.sh`, `tools/quantize_q8.py`, the
voice server and `voice_server.json` (from `engines/qwen3cpp_batch/release/`)
-- plus a source zip beside it. The launcher writes them beside the RealMe
folder (`..\realme-qwen3-tts.cpp`), so the two repositories never nest, and
ends with the GitHub Desktop steps that publish it. Run again, it brings that
folder up to date in place and keeps its git history. Version: 1.3.0
(`cpp.RELEASE_VERSION`).

### Driving RealMe from a script

- `--json` after any command prints one object on stdout (`ok`, `exit_code`,
  `report`, `files`, `warnings`, `error`) and sends the readable lines to
  stderr: `realme lecture deck.pdf --json`.
- `--max-spend 2.00` on `lecture` and `narrate` (or `REALME_MAX_SPEND_USD`)
  stops a paid render before the call that would pass the cap; the same
  command with a higher cap continues.
- `realme insert LECTURE VIDEO --after 12` plays a video after slide 12 in
  this and every later render; `--apply` re-joins the last render now.
- `REALME_LEAN=1` keeps every finished lecture as its video and text files
  without pressing *Keep only the video and text files*; editing it then
  records only the changed slides.
- Narration can be drafted by Gemini, Claude or OpenAI
  (`realme lecture --script claude|openai`, or *Script writer* in the
  Studio). `adapters/script_writer_more.py` reuses the Gemini writer's
  prompts and replaces only its network call; replies are streamed. Keys:
  `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`; default models
  `REALME_CLAUDE_MODEL`, `REALME_OPENAI_MODEL`. A deck longer than 15
  slides is drafted in blocks of about 12 (`REALME_DRAFT_BLOCK`, 0 = whole
  deck), one request after another; each block gets the deck outline, the
  end of the narration before it and the next slide's heading. Drafting
  waits four minutes plus ten seconds a slide (at most fifteen) and asks
  once more after a timeout.
- A Gemini render records 2 sentences at once, fewer when Google asks to
  slow down, and makes slide videos while later slides are still recording.
  `00_Windows\_Probe_Gemini_Parallel.bat` / `realme gemini-parallel`
  measures how many the Google project takes in a short burst and saves it
  as a ceiling (`gemini_parallel.json` in the data folder); it does not
  raise the number. `realme gemini-parallel --use N` does, up to that
  ceiling.
- The Studio's routes declare their request bodies; `/docs` lists them.
- The Studio answers only to `127.0.0.1` / `localhost`. `REALME_ALLOWED_HOSTS`
  (comma-separated) adds a name. Binding to a network address needs
  `--allow-network`, because the Studio has no login.

## Author and licence

RealMe Studio is written by **Xinhua Yu**, and released under the
[MIT licence](LICENSE): anyone may use, change and share it, keeping the
copyright notice. To cite it, use [`CITATION.cff`](CITATION.cff) (GitHub's
*Cite this repository* button reads it).

Third-party code keeps its own licence: the local voice clone's original
source, `03_App/realme/engines/qwen3cpp` and its `ggml` library, is MIT,
with its notices retained unchanged (see `PROVENANCE.md` there); RealMe's
revised source, `engines/qwen3cpp_realme`, keeps upstream's `LICENSE` and
adds RealMe's changes under the same licence. Python libraries are
installed from PyPI under their own licences. PyMuPDF is AGPL-3.0; with it
uninstalled, RealMe reads PDFs through a permissively licensed fallback
(pdfplumber, pypdf and poppler) instead.

