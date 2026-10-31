# The free local voice clone: design

How RealMe records a lecture in the lecturer's cloned voice on their own
computer, and why it runs faster than real time on an ordinary CPU. For using it,
see START_HERE.md section 12; for every engine change with its switch and
check, `03_App/realme/engines/qwen3cpp_batch/REALME_CHANGES.md`.

## 1. The pieces

```
Studio / CLI ──► lecture.render ──► prerecord (ahead of the render loop)
                      │                  │  one engine process, N sentences per pass
                      │                  ▼
                      │          adapters/tts_qwen3_dll.py  (ctypes)
                      │                  │
                      │                  ▼
                      │          qwen3tts.dll / libqwen3tts.so  = RealMe's revised qwen3-tts.cpp
                      │             text tokenizer ─► talker (28 layers) + code predictor (5 layers)
                      │                              ─► 16 codes per frame, 12.5 frames a second
                      │             vocoder (pre-transformer, upsampling, 4 decoder blocks) ─► 24 kHz audio
                      ▼
               slide videos (ffmpeg), made alongside where threads are free ─► concat, captions, chapters
```

- **The engine** is RealMe's revised qwen3-tts.cpp (predict-woo's, on GGML)
  in `03_App/realme/engines/qwen3cpp_realme/`, with every change already
  made, built on the user's computer as it is -- nothing is patched while
  building. The original, unchanged and with GGML in its `ggml/`, is kept in
  `engines/qwen3cpp/` (section 4).
- **The adapter** (`Qwen3DllTTS`) loads the library, keeps the speaker
  embedding of the lecturer's reference recording, decides the model (16- or
  8-bit) once per engine, and sets the engine's switches before it loads.
- **The prerecorder** (`pipeline/prerecord.py`) plans every sentence of the
  lecture, records them ahead of the render loop in a separate process (for a
  hosted voice, in threads), and enters each take in the render's ledger.
  The render loop waits only for the slide it is on.
- **The ledger** keys each take by its text, delivery and the engine's
  *voice fingerprint* (reference recording, settings, model). A take is
  reused only for the same fingerprint, so changing the model re-records,
  and nothing else does.

## 2. Where the time goes

A second of speech is 12.5 frames of 16 codes each. Each frame takes one
step of the 28-layer **talker** (the first code) and then 15 runs of the
5-layer **code predictor** (the other 15 codes, one after another; each run
needs the code before it). The next frame's talker step needs all 16. So
within one sentence nothing can run ahead: every frame is about 190
dependent transformer layers' worth of matrix-vector products.

What the model holds, with the 8-bit model:

| Part | Shape | Weights | Use per frame |
|---|---|---|---|
| talker | 28 layers; width 1024, 16 query / 8 key-value heads of 128, feed-forward 3072 | 16.7 MB a layer, 470 MB | each layer once |
| code predictor | 5 layers, the same size as the talker's; 15 output heads (1024 x 2048) and 15 code tables | 84 MB of layers, 33 MB of heads, 60 MB of tables | each layer 15 times; one head per run; the tables are looked up, not multiplied |
| text side | text embedding and a two-layer projection to 1024 | | once per text token, then added to each frame's input |
| vocoder | code tables; an 8-layer, 512-wide transformer at 12.5 Hz; two upsampling blocks to 50 Hz; a 1024 -> 1536 convolution; four stages up by 8, 5, 4 and 3 to 24,000 Hz (768, 384, 192, 96 channels), each a transposed convolution and three residual units of a 7-tap and a 1-tap convolution with snake activations | 218 MB (16-bit) | once per sentence, over the whole signal |

Per frame, one sentence reads about **1.76 GB of weights** and does about
3.3 billion multiply-adds (41 GFLOP per second of speech), and 73% of both
is the code predictor: its 5 layers run 75 times a frame against the
talker's 28. Measured, the code predictor took 72% of the generation time.
One sentence alone needs 22 GB a second of memory reads for real time --
about what a dual-channel DDR4 laptop delivers -- so one sentence at a time
is **memory-bound**. A pass reads each weight once for all its sentences;
from about 4 per pass the arithmetic becomes the limit.

The vocoder is the opposite: about **61 GFLOP per second of speech**, almost
all in the four stages' convolutions (13, 17, 17 and 12 GFLOP), with small
weights used over and over -- **compute-bound** large matrix products. It
does not share anything between sentences, so it costs the same per
sentence at any pass size, and at 8 per pass it is about half the time.

The original engine, on a 6-core PC (where it uses 4 threads), takes 3.79
seconds of work per second of speech, most of it in the transformers and
the rest in the vocoder. Each change below attacks one of the two.

## 3. The engine changes

| Target | Change | Gain |
|---|---|---|
| everything | the thread count applied (upstream ran on 4) | 1.26x (16-bit model, 4 -> 6 threads, before the other changes) |
| everything | one pool of worker threads kept for the engine's life, instead of one per graph (about 16 per step of speech) | identical; no gain measured on 2 cores, for measuring on larger machines |
| transformers | `ffn_down` converted to F32 once at load, not per step | with threads and broadcast: 3.79 → 2.82 |
| vocoder | per-channel values broadcast, not copied | |
| vocoder | one-pass snake activation, AVX2 sine | → 2.59 |
| transformers | several sentences per pass: the weights read once per step for all of them | 4 per pass: 1.66; 8: 1.51 |
| transformers | 8-bit model: half the bytes per step | 4 per pass: 1.35 |
| vocoder | F32 convolutions through GGML's tinyBLAS (gated to the vocoder), transposed convolutions as matrix products, outputs padded so tinyBLAS takes all of them | the vocoder alone 0.80 → 0.43 s per s; 4 per pass: 1.07 |
| vocoder | the decoder stage copy-free: the signal laid out channel by channel, so each convolution reads its taps through overlapping rows instead of an unrolled copy; bias, snake, padding and regrouping by dilation fused into one pass | 1 at a time: 1.49 → 1.31; 4 per pass: 0.79; 8: 0.74; 16: 0.72. A third of the vocoder's working memory; the same samples as the line above (about 120 dB apart) |
| transformers | a pass's 8-bit products in one kernel: each weight row loaded once for all the sentences, with GGML's exact per-sentence arithmetic | identical samples; on a 2-core machine a pass of 8 took 16% less time, a pass of 4 about 6% |
| transformers | opt-in: tinyBLAS for every product (`QWEN3_TTS_SGEMM_ALL=1`) | a few per cent in passes; slightly different samples |
| vocoder | RealMe's own convolution kernel: the signal channel by channel throughout, taps read in place (dilation included), weights packed at load, 6 x 16 tiles in registers; bias, residual add and the next snake applied as rows leave the kernel, so a residual unit is one operation and a transposed convolution of stride s is s two-tap products | the vocoder alone 0.26 → 0.19 s per s on the 6-core PC (27% less time than the copy-free stage); the same samples (about 110 dB apart) |
| transformers | passes whose places are refilled: a finished sentence is decoded and handed back at once, the next takes its place | 9% on a run of mixed sentences, 4 at a time (2-core machine); identical samples |
| build | the engine's compiler flags follow the processor target: upstream's `-march=native` kept under `REALME_NATIVE`, off for a portable build | a portable build runs on the processors it says it does |

All of it, on the 6-core PC with the 8-bit model: 1.25 seconds of work per
second of speech one at a time (1.14 in a later run of 8 sentences), 0.68
at 4 per pass, 0.58 at 8, 0.57 at 16, against upstream's 3.79.

Principles kept throughout:

- **Identical where it can be.** A change that should not alter the sound is
  checked sample for sample against an unmodified build, with sampling off.
  Those changes are on by default in the engine. Changes that alter the
  rounding (the fast sine, the 32-bit vocoder and its kernel) are off in
  the engine, and the 8-bit model is used only where its file is, measured for how far apart they are (dB below the speech),
  listened to, and then turned on by RealMe's own defaults.
- **One switch per change**, so any of them can be turned off at run time and
  compared (`realme engine-timing --compare`).
- **CPU only where only the CPU was checked.** The batch, broadcast, snake
  and vocoder changes switch themselves off on a GPU backend.
- **No unnamed product changes.** tinyBLAS is compiled in but used only for
  products named `realme_sgemm`, so the transformers compute exactly as
  upstream's GGML does -- unless `QWEN3_TTS_SGEMM_ALL=1` is set, which is
  never a default. The 8-bit pass kernel reproduces GGML's arithmetic and
  checks itself against GGML at first use.

### Universal, with processor-specific additions

Most of the gain is in how the work is arranged, and that is plain C++ or
GGML on every processor. The processor-specific code is a small set of
innermost loops, each with a portable version beside it and each chosen at
run time where the processor can differ:

| Universal (any processor) | Added where the processor has it |
|---|---|
| threads applied, one thread pool; `ffn_down` once; no broadcast copies | |
| passes of several sentences, refilled as sentences end; length grouping | 8-bit pass products: AVX2 (x86, checked at run time), NEON (64-bit ARM; the dot-product instruction where the build has it); elsewhere GGML's own products |
| 8-bit model | |
| one-pass snake (scalar sine) | AVX2 sine, 8 at a time |
| 32-bit vocoder; its copy-free decoder stage (mode 2, GGML's products) | GGML's tinyBLAS for the vocoder (x86 AVX/AVX2/AVX-512, ARM NEON) |
| the vocoder kernel's method (mode 3): layout, packing, blocking, fused steps; a plain C++ tile and snake | the tile in AVX2/FMA (x86, at run time) or NEON (64-bit ARM). With only the plain tile, mode 3 gives way to mode 2: on the 2-core test machine (a build tuned for it) GGML's products were faster than it (vocoder 4.3 s against 3.2 for mode 2 and 2.5 for the AVX2 tile, on 4.8 s of speech) |

So a processor RealMe has no kernel for still gets everything in the left
column. On the test PC the copy-free vocoder without RealMe's kernels ran at
0.74 seconds of work per second of speech at 8 per pass (still with the AVX2
sine and tinyBLAS) against 0.58 with all of it -- roughly 80% of the speed;
a processor without those will be somewhat slower. Each
addition gives the same samples as the universal version it replaces
(identical for the 8-bit pass products; 110--120 dB apart for the vocoder
tiles), and a pass equals one sentence at a time with each.

Checked on 64-bit ARM with a cross compiler and an emulator (no ARM
hardware here, so not timed), for ARMv8.2 with the dot-product
instruction and for plain ARMv8.0: a pass equals one sentence at a time,
the 8-bit kernel passes its self-check, the NEON tile comes within about 110
dB of mode 2, and the plain tile gives the NEON tile's samples exactly.
An engine built for an older ARM level than its GGML failed the self-check
and handed the 8-bit products back to GGML, as intended. On a Mac, GGML
builds its Metal (GPU) and Accelerate (BLAS) backends by default, and the
engine would prefer either to the CPU, where RealMe's changes run; RealMe's
build turns both off unless Metal is asked for, and the release's `build.sh`
does the same. Not measured on a Mac.

### On a real lecture

51 slides, 578 sentences, on an Intel NUC (i7-10710U, 15 W, 6 cores / 12
threads, 64 GB DDR4-2667, no GPU; AVX2 build), 8-bit, 8 per pass, 6
threads, slide videos on the spare threads. With everything (refilled
passes, the vocoder kernel, the 8-bit pass products -- today's defaults):
recorded in 37.2 minutes and rendered in 38.0, 0.72 and 0.73 of the video's
51.8 minutes; a second run, 35.3 and 36.5 minutes, 0.68 and 0.71 of its 51.6. Before those three (fixed passes, the copy-free vocoder):
recorded in 44.4 minutes and rendered in 45.6 (0.86 and 0.89 of that take's
51.5 minutes). 16% less time; the voice not told apart.

## 4. Building

Two source folders, side by side in `engines/`:

- `qwen3cpp/` -- **the original**, upstream qwen3-tts.cpp and GGML exactly
  as vendored (`PROVENANCE.md`). The record of what upstream wrote, and
  GGML's source. Never built unless asked for by name.
- `qwen3cpp_realme/` -- **RealMe's engine source**: the original with every
  change already made, RealMe's own files (`src/realme_*.inc`), the change
  list, and GGML's one changed file (`ggml_overlay/`, the tinyBLAS gate).
  This is what is built, exported and read.

`engines/cpp.py` builds the engine from RealMe's source:

1. Copy `qwen3cpp_realme/` into `tools/qwen3cpp/` (never build in place),
   writing only the files that differ, so only they are recompiled. Nothing
   is patched while building.
2. Stage GGML from the original; put the gated `ggml-cpu.c` in its place
   only when GGML's own file is the one the gate was made from (both
   fingerprints in `REALME_INTEGRATED.json`), and build GGML with
   `-DGGML_LLAMAFILE=ON` -- otherwise with it explicitly OFF.
3. Build with CMake and Ninja (MSVC environment loaded on Windows), copy the
   GGML DLLs beside the library, check the library exports its C API, and
   record what was built in `build_info.json` (`"source": "realme"`).
4. Make the 8-bit model beside the 16-bit one (`ensure_q8`), using the
   engine's own copier; never fails the build.

RealMe's source is written from the original by `integrate()` (`realme
engine cpp --cpp-action integrate`): it applies `PATCHES` in order (the
processor-target option for the engine's compiler flags, threads, the
thread pool, `ffn_down`, broadcast, batch, snake, 8-bit copier, Unicode
paths, 32-bit vocoder, its copy-free decoder stage, the vocoder kernel),
each a set of exact-text anchors that must each be found once -- a source
that has changed is refused, not half-patched. So a change to an upstream
file is made as a patch in `cpp.py` and the folder written again; RealMe's
own `.inc` files are edited in the folder itself. The tests check that the
folder is exactly what `integrate()` writes. On a new upstream version the
same patches are made again on it.

The original can still be built, for comparison:
`realme engine cpp --cpp-action build --source original` builds it
unchanged (upstream's GGML, no gate) into `tools/qwen3cpp_original/`,
beside the working engine. RealMe never loads that build. It has
upstream's behaviour, including what RealMe's changes fix (threads not
applied, non-ASCII folders not opened on Windows, and the rest in
`REALME_CHANGES.md`).

`3_Get_Voice_Engine.bat` does this after downloading and converting the
weights. `_Rebuild_Voice_Engine.bat` rebuilds for the same processor and GPU
setting, keeping the working engine aside and putting it back if the build
fails; *rebuild everything* (`rebuild_all`) also stages GGML afresh and builds
the stopwatch copy the checks use. The checks' copy (`engines/cpp_timing.py`)
is built from RealMe's source plus a per-operation stopwatch for the vocoder.

`export_release` writes RealMe's source, with GGML and its gate, as
the standalone **realme-qwen3-tts.cpp** release (README, CHANGES, NOTICE,
RELEASE_NOTES, LICENSE with upstream's copyright kept, upstream's README as
README_UPSTREAM, build scripts, the 8-bit tool and the voice server).

## 5. Defaults and who decides

The engine reads its switches from the environment once, when it loads. The
adapter sets RealMe's defaults just before that, and only where nothing is
set:

| Setting | Default | Set by the user in |
|---|---|---|
| model | 8-bit once made, else 16-bit (`quant_setting`) | Twin Setup *model*; `REALME_QWEN3CPP_QUANT` |
| sentences per pass (up to 16) | measured by `_Bench_Batch.bat` or Twin Setup's *Measure this computer* on the engine as built (1, 2, 4, 8, 12, 16; the smallest within 6% of the fastest), else 8, lowered where free memory is short | Twin Setup; `REALME_RECORD_BATCH` (1 = off) |
| fast snake, 32-bit vocoder | on (`vocoder_defaults`): the kernel (`3`) on an engine built with it, else the copy-free form (`2`), else `1` | Twin Setup *Vocoder*; `QWEN3_TTS_FAST_SNAKE`, `QWEN3_TTS_VOCODER_F32` |
| 32-bit vocoder's longest sentence | none with `2` or `3`; with `1`, 30 s with 32 GB of memory, 20 s with 16, 10 s below (`FRAMES_BY_MEMORY`) | `QWEN3_TTS_VOCODER_F32_MAX_FRAMES` |
| threads | one per physical core, up to 6 | `QWEN3_TTS_THREADS` |
| how passes are filled | a stream on an engine built with it: sentences in order, each finished one handing its place to the next | `REALME_RECORD_STREAM` (0 = fixed passes) |
| which sentences share a fixed pass | the oldest waiting, and the nearest to it in length among 3 passes' worth (a pass lasts as long as its longest sentence); none waits more than 6 passes | `REALME_RECORD_LOOKAHEAD` (1 = in order) |
| 8-bit products of a pass | RealMe's kernel (identical to GGML's) | `QWEN3_TTS_Q8_BATCH=0` |
| tinyBLAS for every product | off | `QWEN3_TTS_SGEMM_ALL=1` |

A value from the shell wins over the settings file (`.env` in the data
folder, written by `realme key` and by Twin Setup), which wins over RealMe's
defaults. Twin Setup's *Automatic defaults* takes the voice-engine settings
(`ENGINE_KEYS` in `app/server.py`) out of the settings file and sets RealMe's
defaults for the engine as built, at once. *Measure this computer* is
separate and optional: `realme bench-batch` in its own process, as a job;
preview, render, narration and dialogue are refused while it runs. A saved
measurement older than the engine's `build_info.json` is not used
(`batch_measured`, `batch_stale`), so a rebuild falls back to the default
until it is measured again. RealMe's own defaults are marked as such (`DEFAULTED`, and owned by
the settings loader), so a later re-read of the settings file replaces them
and Twin Setup does not report them as set elsewhere. Twin Setup refuses
changes while a render runs, and a running engine keeps the model it started
with, so one lecture is never recorded half in one model and half in the
other.

## 6. Recording a lecture

1. **Plan.** Every sentence of the manifest is planned as it will be spoken
   (text preparation, delivery, pause), keyed with the fingerprint; sentences
   already in the ledger are skipped.
2. **Record ahead.** One engine process (more on a machine with 12 or more
   cores and the memory for them) takes sentences in passes of the chosen
   size. Each process has its own copy of the engine; the render's own engine
   stays free for anything the loop must record itself. A pass that fails is
   recorded one sentence at a time instead.
3. **Make the slide videos alongside.** Where the processor has threads the
   engine leaves free, one or two ffmpeg encoders, each capped to the spare
   threads, make each slide's video as soon as its audio is in
   (`early_video_plan`; `REALME_EARLY_VIDEOS`).
4. **Assemble**: concatenate, mux captions and chapters, check the audio.
   The report gives recording time, passes, engines, and the video time left
   after the last slide's audio (`videos_after_audio_s`).

Memory is checked before the recorder starts: about 3 GB per engine plus
0.5 GB per sentence in a pass, with 2 GB spare; engines, then pass size, are
reduced to fit.

## 7. How it was checked

- `t_engine_batch.py`, `t_engine_timing.py`, `t_prerecord.py` and the rest of
  `run_tests.py`: patches against stand-ins of the source (and against the
  real vendored source when present), the adapter's batch path with a fake
  library, settings and defaults, rebuild and restore, the recorder with
  stand-in engines, early videos, and the release export.
- Against the real model through the C API: identity with every switch off,
  with the exact fixes, in a GGML with the gate; batch equals one at a time;
  sentences over the vocoder's length limit equal upstream; the 8-bit copier;
  all under AddressSanitizer and UndefinedBehaviorSanitizer; the exported
  release built standalone gives the same samples. The copy-free decoder
  stage against the 32-bit vocoder: about 120 dB apart with each snake mode,
  batch equal to one at a time, without tinyBLAS (GGML's own multiply) about
  58 dB apart, off in low-memory mode, and clean under both sanitizers.
  The vocoder kernel against the copy-free stage: about 110 dB apart, the
  same samples on one thread and two, batch equal to one at a time, clean
  under both sanitizers. Refilled passes: every sentence identical to one
  at a time with 1, 2, 4 and 8 places and out-of-order finishing, through
  the C API and through the adapter.
- On the lecturer's computer: `_Engine_Speedup_Check.bat`,
  `_Engine_Check_8bit.bat`, `_Engine_Check_Vocoder.bat` (the same sentences
  each way, measured and left to listen to) and `_Bench_Batch.bat`.

## 8. The engine for other tools

The voice server (`engines/qwen3cpp_batch/release/server/`) is part of the
standalone release and has nothing to do with RealMe's code: Python's
standard library, the engine through ctypes, OpenAI's `/v1/audio/speech`.
It sets the same engine defaults (fast snake; the vocoder kernel, `3`, or
the copy-free `2` on an engine without it, neither with a length limit;
threads), splits long input into sentences, and
gathers sentences -- of one request, or of requests that arrive together for
the same voice and language -- into passes of the batch entry point.
`_Make_Voice_Server.bat` packs it with the engine built here and the 8-bit
model into a folder that runs on its own.

It is one model, `realme-qwen3-tts`, with RealMe's engine and defaults: a
request naming any other model is refused rather than quietly served by
something else, and other names (OpenAI's `tts-1`, for a tool that cannot be
told otherwise) are accepted only when listed in `model_aliases`. The default
voice is `realme` (`voices/realme.wav`): the bundle carries the owner's
recording there only when he chooses to include it (`--with-my-voice`);
otherwise `voices/` holds what the user puts there. The address belongs to the tool
using it: `voice_server.json` holds the host and port (8800 unless set), the
default voice, a key, the pass size, the pause and the aliases, and
command-line options override it.

## 9. Limits and next steps

- **GPU builds are untested.** CUDA and Vulkan should work as upstream does,
  with RealMe's CPU changes switched off.
- **The vocoder** was about half the time per sentence at the best pass
  sizes. With its own kernel it runs at roughly 80% of the processor's peak
  arithmetic on the 2-core test machine; what is left is the arithmetic
  itself (61 GFLOP per second of speech). Fewer bits do not pay here:
  simulated in the kernel, 8-bit weights and signal put the output 24 dB
  below the speech, 4.8 dB apart in spectrum (8-bit signal alone 4.6 dB;
  16-bit both 0.07 dB), and on AVX2 without VNNI an 8-bit kernel gains at
  most about a third over 32-bit floats, a 16-bit one nothing.
- **Threads.** On the 6-core NUC, with the 8-bit model and the other changes
  in, 3 threads -> 6 gave only 1.15x one sentence
  at a time and 1.28x at 8 per pass: the 15-watt limit and memory speed
  bound it there, not mainly the barriers. Machines without that limit gain
  more from less waiting, so the engine keeps one thread pool for its whole
  life (identical output), and RealMe can build GGML with its own threads
  instead of OpenMP (`REALME_GGML_OPENMP=0`), to be measured per machine --
  on Windows the OpenMP runtime is MSVC's. GGML's own threads spin while
  waiting, so they are not the default: with more threads than free cores
  (the slide-video encoders, a second engine) they can slow down badly.
- **Generation's thread coordination.** Each sentence in a pass has its own
  attention operations in every layer, so a frame step at 8 per pass is
  several thousand small operations, each ending at a barrier where all
  threads meet. On 2 cores generation ran 1.79 times as fast as on one (the
  vocoder 1.94); on 6 cores the share lost to the barriers is larger and is
  for measuring there. Batching the sentences' attention into one padded
  operation would change GGML's summation order for short contexts and so
  break "a pass equals one sentence"; fusing operations with GGML's exact
  arithmetic would not, at more work.
- **Passes wait for their longest sentence.** A pass runs until its
  longest sentence is done, its slots emptying as shorter ones finish. On a
  real script (sentences of 73 characters on average, SD 40), passes filled
  in slide order cost about 20% of the generation time against slots that
  never empty. On an engine built with it the recorder now refills a place
  as soon as its sentence ends (a stream); fixed passes group sentences of
  similar length from three passes' worth, which leaves about 6%. Neither
  changes any sentence's samples, since a sentence comes out the same in
  any pass.
- **Overlapping the vocoder with generation** uses what one of them leaves
  idle (memory stalls, thread barriers, sampling between steps). Two engine
  copies side by side on the 6-core PC, which overlap in the same way, were
  7-8% faster than one, so this is a modest gain.
- **Two engines at 3 threads, each in passes**, against one at 6: the
  recorder can already do it (`REALME_RECORD_WORKERS=2` splits the cores
  between the engines), and `realme bench-parallel --layouts 1x6,2x3
  --batch 8` measures it per machine; it recommends a layout that is 5% or
  more faster in every run.
- **Reusing the voice prompt between sentences** (an LLM's prompt cache) was
  measured and not built: the whole prefill of a 4.8-second sentence was
  186 ms of 9.4 s (2%), and the part every sentence shares is a few of its
  positions, so under 1% to gain.
- The upstream CLI's `-j` is still not applied (upstream behaviour);
  `QWEN3_TTS_THREADS` is.
