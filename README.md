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
the defaults and the switches.

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
its `voices/` and `voice_server.json`. Tests:
`t_voice_server.py`.

### realme-qwen3-tts.cpp, the engine on its own

RealMe's revised qwen3-tts.cpp is also published as a standalone source
release, **realme-qwen3-tts.cpp** (MIT; predict-woo's and the ggml authors'
credit and licences kept). `00_Windows\_Make_qwen3cpp_Release.bat` (or
`realme engine cpp --cpp-action export-release --export-to <new folder>`)
writes it from the vendored source with every change applied -- GGML with its
tinyBLAS gate, upstream's README as README_UPSTREAM.md, and the release's own
README, CHANGES (`engines/qwen3cpp_batch/REALME_CHANGES.md`), NOTICE,
RELEASE_NOTES, LICENSE, `build.bat`/`build.sh` and `tools/quantize_q8.py`
(from `engines/qwen3cpp_batch/release/`) -- plus a source zip beside it.
The launcher writes them beside the RealMe folder (`..\realme-qwen3-tts.cpp`),
so the two repositories never nest, and ends with the GitHub Desktop steps
that publish it. Run again after an update, it brings that folder up to date
in place and keeps its git history. Version: `cpp.RELEASE_VERSION`.

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
- The free local clone uses one thread per core, up to 6
  (`REALME_ENGINE_THREADS` overrides it). A computer with cores to spare
  records with more than one copy of the engine at once: one per 6 cores, at
  most 4, and only as many as free memory allows (about 3 GB each). On a
  6-core machine that is one; a second measured only 7-8% faster there.
  Engines built before the thread fix run on 4 threads and keep two copies on
  any machine with 8 or more logical processors. `REALME_RECORD_WORKERS`
  sets the number (1 turns it off).
  `realme setup` (5_Command_Prompt.bat) says what this computer will use.
- Narration can be drafted by Gemini, Claude or OpenAI
  (`realme lecture --script claude|openai`, or *Script writer* in the
  Studio). `adapters/script_writer_more.py` reuses the Gemini writer's
  prompts and replaces only its network call; replies are streamed. Keys:
  `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`; default models
  `REALME_CLAUDE_MODEL`, `REALME_OPENAI_MODEL`. A deck longer than 15
  slides is drafted in blocks of about 12 (`REALME_DRAFT_BLOCK`, 0 = whole
  deck), one request after another; each block gets the deck outline, the
  end of the narration before it and the next slide's heading. Top-ups and
  additions on a long deck send only the affected slides' pictures and their
  neighbours. Drafting waits four minutes plus ten seconds a slide (at most
  fifteen) and asks once more after a timeout instead of three times.
- A Gemini render records 2 sentences at once, fewer when Google asks to
  slow down, and makes slide videos while later slides are still recording.
  `00_Windows\_Probe_Gemini_Parallel.bat` / `realme gemini-parallel`
  measures how many the Google project takes in a short burst and saves it
  as a ceiling (`gemini_parallel.json` in the data folder); it does not
  raise the number. `realme gemini-parallel --use N` does, up to that
  ceiling.
- `00_Windows\_Rebuild_Voice_Engine.bat` (or `realme engine cpp
  --cpp-action rebuild`) rebuilds the local clone so it uses one thread per
  core, at most 6. Engines built before October 2026 ignore their thread
  setting and run on 4. Measured about a quarter faster on a 6-core machine.
  The working engine is kept aside and restored if the build fails.
  `--cpp-action rebuild-all` (the launcher's "rebuild everything", or
  `_Rebuild_Voice_Engine.bat all`) also stages GGML again from the vendored
  copy and builds the stopwatch copy the `_Engine_Check_*` files use. A build
  or rebuild then makes the 8-bit model too (`cpp.ensure_q8`; skipped, with
  a message, when the disk is short or the engine lacks the copier), and a
  recording that asks for it makes it if it is still missing. Packages do
  not carry `models_q8\`.
- `00_Windows\_Engine_Timing.bat` (or `realme engine-timing`) builds a
  stopwatch copy of the local clone in `tools\qwen3cpp_timing` and reports
  where its time goes, at several thread counts. The working engine is not
  touched.
- Two speed changes to the local clone, applied to the staged copy at build
  time (`engines/cpp.py`), never to the vendored source. Each gives the same
  numbers as before, so the sound is identical:
  - **ffn_down converted once.** Upstream converts each layer's `ffn_down`
    weight from F16 to F32 inside every step's graph: 33 layers, about 1,240
    conversions per second of speech. RealMe converts them once at load
    (+0.4 GB). Measured on a stand-in model with the real shapes: one step
    is 30-37% faster, and the generated codes are identical.
    `QWEN3_TTS_FFN_F32=0` switches it off.
  - **No per-channel copies in the vocoder.** The snake activation and layer
    scale no longer copy a per-channel value out to the full signal length
    before multiplying (`ggml_mul` broadcasts). Measured with the real
    vocoder weights: about 5% of vocoder time, identical samples.
    `QWEN3_TTS_BROADCAST=0` switches it off.

  - **The vocoder's snake in one pass** (`engines/qwen3cpp_batch/realme_snake.inc`,
    a GGML custom op). Upstream does x + exp(-beta)*sin(exp(alpha)*x)^2 as
    five passes over the signal, with the C library's sinf one number at a
    time (17% of vocoder time on Windows). Off in the engine; RealMe sets
    `1` (`tts_qwen3_dll.VOCODER_DEFAULTS`) unless told otherwise;
    `QWEN3_TTS_FAST_SNAKE=exact` uses the library's sine in one pass (same
    samples as upstream, checked with the real vocoder weights);
    `QWEN3_TTS_FAST_SNAKE=1` uses RealMe's AVX2 sine (Cephes polynomial,
    about float precision). Any sine other than the library's own, even a
    correctly rounded one, moves the output about 70 dB below the speech:
    the vocoder's layers spread a one-ulp difference. Set it with
    `realme key QWEN3_TTS_FAST_SNAKE exact` (or `1`).

  - **The vocoder's convolutions in 32 bits** (`engines/qwen3cpp_batch/realme_vocoder.inc`).
    Profiled with the real weights, the vocoder was about a third of a
    sentence's time one at a time, and more with several sentences per pass
    (passes share the transformers' weight reads; the vocoder gains
    nothing): matrix multiply 63%, im2col 17%, transposed convolution 13%.
    Upstream rounds the signal to F16 for every convolution and does the
    transposed convolutions with GGML's direct kernel (one-thread permute,
    a dot product per output sample). RealMe converts the convolution
    weights to F32 once at load (~220 MB; working memory about 1.8x
    upstream's per second, so sentences over 250 codes / 20 s -- RealMe sets
    375 / 30 s with 32 GB or more, 250 / 20 s with 16, 125 / 10 s below
    (`tts_qwen3_dll.FRAMES_BY_MEMORY`),
    `QWEN3_TTS_VOCODER_F32_MAX_FRAMES` to set it -- take upstream's path), unrolls in
    F32 (output padded to a multiple of 4 so tinyBLAS takes every conv), does each
    transposed convolution as one matrix multiply plus an overlap-add custom
    op (crop and bias folded in), and names those multiplies
    `realme_sgemm`, which a RealMe build of GGML -- `GGML_LLAMAFILE=ON`,
    with `ggml-cpu.c` gated by name (`cpp.apply_sgemm_patch`) -- sends to
    tinyBLAS. Nothing else uses tinyBLAS, so the codes are unchanged and,
    with the switch off, the samples are upstream's. Measured on 2 cores:
    the vocoder 1.37x as fast. The samples move about 50 dB below the
    speech (as reordering an F32 sum alone does). On the real voice
    (Windows, 6 cores): the vocoder 0.85 -> 0.55 s of work per s of speech,
    62 dB apart, not heard. Off in the engine; RealMe sets
    `QWEN3_TTS_VOCODER_F32=1` unless told otherwise (Twin Setup: Vocoder);
    `QWEN3_TTS_SGEMM=0` keeps GGML's own multiply. Since October 2026 the
    8-bit model is also the default where it has been made
    (`quant_setting`; `REALME_QWEN3CPP_QUANT=f16` keeps the 16-bit one).
    `00_Windows\_Engine_Check_Vocoder.bat` (`engine-timing --compare
    --variants fast,voc`) measures it and leaves the audio to listen to.

  `00_Windows\_Engine_Speedup_Check.bat` (`realme engine-timing --compare`)
  records the same sentences with the changes off, on, and with each
  one-pass snake, using the timing copy; compares one sentence sample by
  sample (or reports how far apart, in 16-bit steps and dB); and profiles the
  vocoder by kind of calculation. `_Rebuild_Voice_Engine.bat` gives the working engine
  the changes.
- **Several sentences per pass** (`engines/qwen3cpp_batch/realme_batch.inc`,
  added to the staged C API source with `friend` lines in two classes;
  exported as `qwen3_tts_synthesize_batch_with_embedding`). Each sentence
  keeps its own KV caches, attention, sampling and stopping; the weight
  matrices are multiplied for all of them at once. With sampling off the
  codes, and the audio, are identical to one-at-a-time (checked on a
  stand-in model with the real shapes, through the Python adapter and the
  real vocoder). Measured there for the generation part: 1.5x at 2 per pass,
  2.0x at 4, 2.4x at 8. `_Bench_Batch.bat` (`realme bench-batch`) checks
  identity and speed on the real voice and saves the pass size to
  `qwen3_batch.json` in the data folder; the prerecorder then gives one
  engine process passes of that size. `REALME_RECORD_BATCH` overrides it
  (1 = off). Each sentence in a pass needs about 0.5 GB more memory for its
  caches. Slide videos are made during local recording too where the CPU has
  threads the engine leaves free (`lecture.early_video_plan`: one or two
  ffmpeg encoders, `-threads` capped to the spare count);
  `REALME_EARLY_VIDEOS=0` makes them afterwards, `1` always alongside. The
  render report gives `videos_after_audio_s`, the video time left after the
  last slide's audio.
- **RealMe's changes to qwen3-tts.cpp** are listed, with how each was checked
  and its switch, in `03_App\realme\engines\qwen3cpp_batch\REALME_CHANGES.md`.
  The vendored upstream is never edited; `cpp.py` applies the changes
  (`PATCHES`) to every built copy. `realme engine cpp --cpp-action
  export-source --export-to <new folder>` writes the revised source as a
  standalone tree with upstream's MIT licence and that file.
- **An 8-bit copy of the model** (`engines/qwen3cpp_batch/realme_quant.inc`,
  exported as `qwen3_tts_realme_quantize_q8`; `realme engine cpp
  --cpp-action quantize`). Made from the 16-bit GGUF by the engine's own
  library -- no PyTorch -- into `tools\qwen3cpp\models_q8\` (the engine would
  pick an 8-bit file up automatically from `models\`). Q8_0 for the 247
  matrices each step multiplies by; embeddings, norms, the text projection
  and the speaker encoder unchanged. 8-bit `ffn_down` is multiplied as it is
  (no F32 copy; GGML scales its input per block, so no F16 overflow).
  `REALME_QWEN3CPP_QUANT=q8_0` switches lectures to it, with its own
  fingerprint so its takes and the 16-bit model's are kept apart.
  `00_Windows\_Engine_Check_8bit.bat` (`realme engine-timing --compare
  --variants fast,q8`, then `realme bench-batch --quant q8_0`) measures it
  and leaves the audio to listen to. Batch sizes are saved per model. On a
  2-core stand-in with the real shapes, code generation took 45% less time.
- `00_Windows\_Bench_Parallel.bat` (or `realme bench-parallel`) measures
  whether two or three local-clone engines at once record faster than one on
  this computer. It changes nothing; it reports.
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

