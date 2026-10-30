# The free local voice clone: design

How RealMe records a lecture in the lecturer's cloned voice on their own
computer, and why it runs at about real time on an ordinary CPU. For using it,
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

- **The engine** is qwen3-tts.cpp by predict-woo on GGML, vendored unmodified
  in `03_App/realme/engines/qwen3cpp/` (with GGML in its `ggml/`), and built
  on the user's computer with RealMe's changes applied to a staged copy.
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

A second of speech is 12.5 frames. Each frame is one step of the 28-layer
talker and 15 steps of the 5-layer code predictor: about 190 small
transformer steps a second, each a matrix-vector product per weight matrix.
Those steps are **memory-bound**: each reads the model's weights from memory
and does little arithmetic with them. The vocoder is the opposite: about 30
convolutions over the whole signal (up to 24,000 samples a second at the
last stage), **compute-bound** large matrix products.

The original engine, on a 6-core PC (where it uses 4 threads), takes 3.79
seconds of work per second of speech, about two thirds in the transformers
and one third in the vocoder. Each change below attacks one of the two.

## 3. The engine changes

| Target | Change | Gain |
|---|---|---|
| everything | the thread count applied (upstream ran on 4) | 1.26x |
| transformers | `ffn_down` converted to F32 once at load, not per step | with threads and broadcast: 3.79 → 2.82 |
| vocoder | per-channel values broadcast, not copied | |
| vocoder | one-pass snake activation, AVX2 sine | → 2.59 |
| transformers | several sentences per pass: the weights read once per step for all of them | 4 per pass: 1.66; 8: 1.51 |
| transformers | 8-bit model: half the bytes per step | 4 per pass: 1.35 |
| vocoder | F32 convolutions through GGML's tinyBLAS (gated to the vocoder), transposed convolutions as matrix products, outputs padded so tinyBLAS takes all of them | vocoder 0.85 → 0.55 s per s; 4 per pass: 1.07 |

Principles kept throughout:

- **Identical where it can be.** A change that should not alter the sound is
  checked sample for sample against an unmodified build, with sampling off.
  Those changes are on by default in the engine. Changes that alter the
  rounding (the fast sine, the 32-bit vocoder, the 8-bit model) are off in
  the engine, measured for how far apart they are (dB below the speech),
  listened to, and then turned on by RealMe's own defaults.
- **One switch per change**, so any of them can be turned off at run time and
  compared (`realme engine-timing --compare`).
- **CPU only where only the CPU was checked.** The batch, broadcast, snake
  and vocoder changes switch themselves off on a GPU backend.
- **No unnamed product changes.** tinyBLAS is compiled in but used only for
  products named `realme_sgemm`, so the transformers compute exactly as
  upstream's GGML does.

### On a real lecture

51 slides, 52 minutes, 578 sentences (564 different): recorded in 56.2
minutes, rendered in 56.9 -- 1.10 times the lecture's length -- on an Intel
NUC (i7-10710U, 15 W, 6 cores / 12 threads, 64 GB DDR4-2667, no GPU; AVX2
build), with the defaults (8-bit, 4 per pass, faster vocoder, 6 threads,
slide videos on the spare threads; 10.8 s of video left after the audio).

## 4. Building

`engines/cpp.py` builds the engine:

1. Stage the vendored source into `tools/qwen3cpp/` (never build in place;
   the vendored copy stays diffable against upstream).
2. Apply `PATCHES` in order: threads, `ffn_down`, broadcast, batch, snake,
   8-bit copier, Unicode paths, 32-bit vocoder. Each is a set of exact-text
   anchors that must each be found once; a source that has changed is
   refused, not half-patched. The `.inc` files in `qwen3cpp_batch/` carry the
   larger pieces of new code.
3. Stage GGML, apply the tinyBLAS gate (`apply_sgemm_patch`) and build it with
   `-DGGML_LLAMAFILE=ON` -- or, if the gate does not fit, with it explicitly
   OFF.
4. Build with CMake and Ninja (MSVC environment loaded on Windows), copy the
   GGML DLLs beside the library, check the library exports its C API, and
   record what was built in `build_info.json`.
5. Make the 8-bit model beside the 16-bit one (`ensure_q8`), using the
   engine's own copier; never fails the build.

`3_Get_Voice_Engine.bat` does this after downloading and converting the
weights. `_Rebuild_Voice_Engine.bat` rebuilds for the same processor and GPU
setting, keeping the working engine aside and putting it back if the build
fails; *rebuild everything* (`rebuild_all`) also stages GGML afresh and builds
the stopwatch copy the checks use. The checks' copy (`engines/cpp_timing.py`)
is built from the same patches plus a per-operation stopwatch for the vocoder.

`export_release` writes the same patched source, with GGML and its gate, as
the standalone **realme-qwen3-tts.cpp** release (README, CHANGES, NOTICE,
LICENSE with upstream's copyright kept, build scripts, the 8-bit tool).

## 5. Defaults and who decides

The engine reads its switches from the environment once, when it loads. The
adapter sets RealMe's defaults just before that, and only where nothing is
set:

| Setting | Default | Set by the user in |
|---|---|---|
| model | 8-bit once made, else 16-bit (`quant_setting`) | Twin Setup *model*; `REALME_QWEN3CPP_QUANT` |
| sentences per pass | measured by `_Bench_Batch.bat`, else 8 (16-bit) / 4 (8-bit), lowered where free memory is short | Twin Setup; `REALME_RECORD_BATCH` (1 = off) |
| fast snake, 32-bit vocoder | on (`VOCODER_DEFAULTS`) | Twin Setup *Vocoder*; `QWEN3_TTS_FAST_SNAKE`, `QWEN3_TTS_VOCODER_F32` |
| 32-bit vocoder's longest sentence | 30 s with 32 GB of memory, 20 s with 16, 10 s below (`FRAMES_BY_MEMORY`) | `QWEN3_TTS_VOCODER_F32_MAX_FRAMES` |
| threads | one per physical core, up to 6 | `QWEN3_TTS_THREADS` |

A value from the shell wins over the settings file (`.env` in the data
folder, written by `realme key` and by Twin Setup), which wins over RealMe's
defaults. RealMe's own defaults are marked as such (`DEFAULTED`, and owned by
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
  release built standalone gives the same samples.
- On the lecturer's computer: `_Engine_Speedup_Check.bat`,
  `_Engine_Check_8bit.bat`, `_Engine_Check_Vocoder.bat` (the same sentences
  each way, measured and left to listen to) and `_Bench_Batch.bat`.

## 8. The engine for other tools

The voice server (`engines/qwen3cpp_batch/release/server/`) is part of the
standalone release and has nothing to do with RealMe's code: Python's
standard library, the engine through ctypes, OpenAI's `/v1/audio/speech`.
It sets the same engine defaults (fast snake, 32-bit vocoder with the
memory-based length limit, threads), splits long input into sentences, and
gathers sentences -- of one request, or of requests that arrive together for
the same voice and language -- into passes of the batch entry point.
`_Make_Voice_Server.bat` packs it with the engine built here and the 8-bit
model into a folder that runs on its own.

It is one model, `realme-qwen3-tts`, with RealMe's engine and defaults: a
request naming any other model is refused rather than quietly served by
something else, and other names (OpenAI's `tts-1`, for a tool that cannot be
told otherwise) are accepted only when listed in `model_aliases`. The default
voice is `realme` (`voices/realme.wav`). The address belongs to the tool
using it: `voice_server.json` holds the host and port (8800 unless set), the
default voice, a key, the pass size, the pause and the aliases, and
command-line options override it.

## 9. Limits and next steps

- **GPU builds are untested.** CUDA and Vulkan should work as upstream does,
  with RealMe's CPU changes switched off.
- **The vocoder is still about half the time per sentence** at the best pass
  sizes. It is compute-bound and the transformers memory-bound, so running
  the vocoder for one pass alongside the generation of the next, on a
  processor with hyperthreads, is the next step worth measuring.
- **The 32-bit vocoder's memory** is why long sentences fall back to the
  original vocoder; decoding long sentences in overlapping chunks would lift
  that limit at the cost of boundary checks.
- The upstream CLI's `-j` is still not applied (upstream behaviour);
  `QWEN3_TTS_THREADS` is.
