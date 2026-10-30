# RealMe

Slide decks become narrated lecture videos in your own voice. Documents are
read aloud. A paper or a topic becomes a two-voice podcast.

Voice cloning is optional, and the installer asks. **Gemini** (`gemini-tts`)
clones your voice in your Google project: no graphics card needed, a lecture
records in about a third of its length (100 minutes in 30–40), about a dollar
an hour of lecture — your recording and the
narration are sent to Google, so use a **paid** API project. **The free local
clone** (`qwen3cpp`) keeps everything on your computer at no cost, on
RealMe's revised voice engine: about real time on a 6-core PC (an hour
of lecture in about an hour). It comes in a RealMe package, or
`3_Get_Voice_Engine.bat` builds it (Visual Studio Build Tools). Or use the
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

## What is in this folder

```
00_Windows\     The launchers. 1, 2 and 4 once; after that only 4.
03_App\         The application — a Python package.
tools\          ffmpeg and the draft voices (and the local clone, if added).
START_HERE.md   The guide.
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

Both want the environment's own Python; `verify_tree.py` prints the right path
if run under another one.

**[DESIGN.md](DESIGN.md)** describes how the free local voice clone is built
and why it is fast: the revised engine, the recording pipeline around it,
the defaults and the switches. Every change to the engine, with its switch
and how it was checked, is in
`03_App\realme\engines\qwen3cpp_batch\REALME_CHANGES.md`.

### The local voice engine

- The vendored upstream (`03_App/realme/engines/qwen3cpp`, with GGML) is
  never edited. `engines/cpp.py` applies RealMe's changes (`PATCHES`, and
  `apply_sgemm_patch` for GGML) to the copy it builds.
- `3_Get_Voice_Engine.bat` downloads and converts the weights, builds the
  engine and makes the 8-bit model. `_Rebuild_Voice_Engine.bat` (`realme
  engine cpp --cpp-action rebuild`; `rebuild-all` also restages GGML and
  builds the timing copy) rebuilds for the same processor and GPU setting,
  keeping the working engine aside and restoring it if the build fails.
  Packages do not carry `models_q8\`; a build or the first recording makes it.
- Defaults: the 8-bit model once made (`REALME_QWEN3CPP_QUANT=f16` for the
  16-bit one), one thread per core up to 6 (`REALME_ENGINE_THREADS`), the
  fast snake and the copy-free 32-bit vocoder (`QWEN3_TTS_FAST_SNAKE`,
  `QWEN3_TTS_VOCODER_F32=2` on an engine built with it, else `1` with
  `QWEN3_TTS_VOCODER_F32_MAX_FRAMES` by memory), several
  sentences per pass (`REALME_RECORD_BATCH`, 1 = off). A computer with cores
  to spare records with more than one engine at once: one per 6 cores, at
  most 4, as free memory allows (`REALME_RECORD_WORKERS`). Slide videos are
  made on the threads the engine leaves free (`REALME_EARLY_VIDEOS`).
  `realme setup` says what this computer will use.
- Checks on the real voice, each leaving the audio to listen to:
  `_Engine_Speedup_Check.bat` (`realme engine-timing --compare`),
  `_Engine_Check_8bit.bat`, `_Engine_Check_Vocoder.bat`; `_Bench_Batch.bat`
  (`realme bench-batch`) measures the best pass size and saves it;
  `_Engine_Timing.bat` reports where the time goes; `_Bench_Parallel.bat`
  measures whether more than one engine at once helps. None changes the
  working engine.

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
writes it from the vendored source with every change applied -- GGML with its
tinyBLAS gate, upstream's README as README_UPSTREAM.md, and the release's own
README, CHANGES (`engines/qwen3cpp_batch/REALME_CHANGES.md`), NOTICE,
RELEASE_NOTES, LICENSE, `build.bat`/`build.sh`, `tools/quantize_q8.py`, the
voice server and `voice_server.json` (from `engines/qwen3cpp_batch/release/`)
-- plus a source zip beside it. The launcher writes them beside the RealMe
folder (`..\realme-qwen3-tts.cpp`), so the two repositories never nest, and
ends with the GitHub Desktop steps that publish it. Run again, it brings that
folder up to date in place and keeps its git history. Version:
`cpp.RELEASE_VERSION`.

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

Third-party code keeps its own licence: the local voice clone's source,
`03_App/realme/engines/qwen3cpp` and its `ggml` library, is MIT, with its
notices retained unchanged (see `PROVENANCE.md` there). Python libraries are
installed from PyPI under their own licences. PyMuPDF is AGPL-3.0; with it
uninstalled, RealMe reads PDFs through a permissively licensed fallback
(pdfplumber, pypdf and poppler) instead.

