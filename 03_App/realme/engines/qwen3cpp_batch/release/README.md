# realme-qwen3-tts.cpp

A revised [qwen3-tts.cpp](https://github.com/predict-woo/qwen3-tts.cpp) — C++
inference for [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)
voice cloning on the [GGML](https://github.com/ggml-org/ggml) tensor library —
made **about six and a half times as fast on an ordinary CPU** (three
times one sentence at a time), with the same sound.

It is the local voice engine of [RealMe Studio](#about-realme-studio), which
turns slide decks into narrated lecture videos in the lecturer's own cloned
voice, and is released here on its own so anyone can use it.

**All credit for the engine itself goes to predict-woo**, who wrote
qwen3-tts.cpp, to the **ggml authors**, and to the **Qwen team** at Alibaba for
the model. This repository adds a set of changes on top of upstream commit
`b3ba14077cf1` (GGML `3af5f5760e19`); every one is listed in
[CHANGES.md](CHANGES.md) with how it was checked and the switch that turns it
off. Upstream's own README is kept as [README_UPSTREAM.md](README_UPSTREAM.md).

## Report

[paper/realme-qwen3-tts-report.pdf](paper/realme-qwen3-tts-report.pdf)
describes what limits this model on an ordinary CPU, what was changed, what
did not pay, and how a full lecture compares with a hosted voice. Since the
table below was measured, the code predictor's attention for a whole pass
became one operation with its graphs kept, and RealMe Studio records with 12
predictor steps (15 for a take that fails its check): in passes of 16 the
engine needs 0.48 seconds of work per second of speech (a short test), and the
same 51.7-minute lecture was recorded in 30.2 minutes (0.59 of its length).
Its sources (Markdown, LaTeX, figures) are in `paper/source/`.

## What changed, and what it bought

Measured on a small PC with a 6-core, 12-thread laptop processor (Intel i7-10710U, 15 W; 64 GB DDR4-2667; AVX2 build, no GPU),
cloning a real voice, as **seconds of work per second of speech** (lower is faster; 1.0 is
real time):

| | 1 sentence at a time | 4 per pass | 8 per pass | 16 per pass |
|---|---|---|---|---|
| upstream | 3.79 | — | — | — |
| + exact fixes (threads, `ffn_down`, broadcast) | 2.82 | — | — | — |
| + one-pass snake | 2.59 | — | — | — |
| + several sentences per pass | 2.54 | 1.66 | 1.51 | — |
| + 8-bit model | 1.77 | 1.35 | 1.30 | — |
| + 32-bit vocoder | 1.49 | 1.07 | 1.02 | — |
| + the vocoder copy-free | 1.31 | 0.79 | 0.74 | 0.72 |
| + 8-bit pass products, the vocoder kernel (everything) | **1.25** | **0.68** | **0.58** | **0.57** |

An hour of speech now takes about 35 minutes of work on that 15-watt laptop
processor, against almost four hours. (The last two rows were measured on 16
sentences, the others on 8.) Beyond 8 per pass the gain is small: 12 per
pass gave 0.58. A real 51.8-minute lecture video (578 sentences), 8 per
pass, with everything on, was recorded in 37 minutes (0.72 of its length)
and finished in 38 (0.73), the slide videos made during the recording (a
second run: 35 and 36.5 minutes, 0.68 and 0.71);
before the last row's changes and refilled passes, 44 and 46. Refilled passes
come on top of the last row: on a 2-core test machine they took 9% off a run
of mixed sentences, which a measurement of equal-length passes does not
show.

**Other hardware.** The transformers' steps mostly wait on memory, so faster
memory (DDR5, more channels) speeds them up; the vocoder is arithmetic, so
more and faster cores speed that up. A desktop processor that holds a high
clock under load should do better than this 15-watt chip, a smaller laptop
worse. GPU builds have not been measured; the batch and vocoder changes are
CPU-only and switch themselves off there, so whether a GPU is faster for this
model on a given machine is for measuring. On a Mac, GGML builds its Metal
(GPU) and Accelerate (BLAS) backends by default and the engine prefers
either to the CPU, which would switch RealMe's changes off; `build.sh`
builds without them there (or run with `QWEN3_TTS_BACKEND=cpu`). Not
measured on a Mac.

| Change | Sound | Default |
|---|---|---|
| **Thread count applied.** Upstream stores `n_threads` and never passes it to the CPU backend, so everything ran on 4 threads. | identical | `QWEN3_TTS_THREADS=N` (without it, 4 as upstream; the CLI's `-j` is still not applied) |
| **One thread pool for the engine's life**, instead of one made for every graph (about 16 per step of speech). | identical | on (`QWEN3_TTS_POOL=0` off) |
| **`ffn_down` converted to F32 once at load**, not in every step's graph (about 1,240 conversions per second of speech). | identical | on (`QWEN3_TTS_FFN_F32=0` off) |
| **No per-channel copies in the vocoder**: multiplies broadcast instead of first copying a value out to the signal's length. | identical | on (`QWEN3_TTS_BROADCAST=0` off) |
| **Several sentences per pass**: each keeps its own caches, sampling and stopping; the weights are read once for all of them. New C entry point. | identical with sampling off | API: `qwen3_tts_synthesize_batch_with_embedding` |
| **8-bit (Q8_0) copy of the model**, written by the engine's own library from the 16-bit GGUF, no PyTorch. | a different model; not told apart in listening | API: `qwen3_tts_realme_quantize_q8`; `tools/quantize_q8.py` |
| **One-pass snake activation** in the vocoder, with an AVX2 sine. | `exact`: identical; `1`: about 70 dB below the speech | off (`QWEN3_TTS_FAST_SNAKE=1`, or `exact`) |
| **32-bit vocoder**: convolutions unrolled and multiplied in F32 instead of rounding the signal to F16 at every layer; transposed convolutions as one matrix multiply plus an overlap-add; these products only through GGML's tinyBLAS. | about 50–60 dB below the speech; not told apart in listening | off (`QWEN3_TTS_VOCODER_F32=1`) |
| **The vocoder's decoder stage without copies**: the signal laid out channel by channel, so each convolution reads its taps through overlapping rows instead of an unrolled copy; bias, snake, padding and residual adds fused. About a third less vocoder time than the line above and a third of its memory -- less than upstream's at any length. | the same samples as `1` (about 120 dB apart) | off (`QWEN3_TTS_VOCODER_F32=2`) |
| **RealMe's own vocoder kernel**: the signal stays channel by channel throughout; each convolution reads its taps in place and multiplies them by weights packed once at load, 6 time steps by 16 channels at a time; bias, residual add and the next snake are applied as rows leave the kernel. About a quarter less vocoder time than `2`. The innermost tile in AVX2/FMA (x86, chosen at run time) or NEON (64-bit ARM), with a plain C++ one beside it; `2` where only the plain one would run. | the same samples as `2` (about 110 dB apart) | off (`QWEN3_TTS_VOCODER_F32=3`) |
| **Passes whose places are refilled**: a sentence that ends is decoded and handed back at once and the next takes its place, so a pass never waits half empty on its longest sentence. New C entry point. | identical with sampling off | API: `qwen3_tts_synthesize_stream_with_embedding` |
| **8-bit products for a pass**: each 8-bit weight row is loaded once for every sentence in the pass, with exactly the arithmetic of GGML's dot product per sentence (AVX2 on x86, NEON on 64-bit ARM); checked against GGML's own at first use and switched off on any difference. | identical | on (`QWEN3_TTS_Q8_BATCH=0` off) |
| **tinyBLAS for every product** it takes (opt-in). | a sentence comes out as a slightly different take of itself; listen first | off (`QWEN3_TTS_SGEMM_ALL=1`) |
| **Unicode paths on Windows**: model and audio files opened with `ggml_fopen`. | identical | on |
| **The engine's compiler flags follow the processor target**: upstream's `-march=native` kept under `REALME_NATIVE` (`-mcpu=native` on 64-bit ARM), so a portable build runs where it says it does. | identical | on, as upstream (`-DREALME_NATIVE=OFF` for a portable build; `build.sh` does) |

Every change that gives identical sound was checked sample for sample against
an unmodified build; the batch path against one sentence at a time; all of it
under AddressSanitizer and UndefinedBehaviorSanitizer. GPU builds (CUDA,
Vulkan, Metal) have not been tested: there the batch, broadcast, snake and
vocoder changes switch themselves off.

### The one change to GGML

GGML's tinyBLAS matrix multiply (`GGML_LLAMAFILE`) is used **only** for the
products this engine names `realme_sgemm` — the vocoder's convolutions — by a
gate added to `ggml/src/ggml-cpu/ggml-cpu.c`. Every other product in the
engine is computed exactly as upstream GGML computes it, so the two
transformers' output does not change. Build GGML with `-DGGML_LLAMAFILE=ON`
only with this gated file; `QWEN3_TTS_SGEMM=0` turns tinyBLAS off at run time.
`QWEN3_TTS_SGEMM_ALL=1` opens the gate to every product tinyBLAS takes: a
little faster in passes, but the transformers' results move slightly. One
sentence and a pass still give the same samples while the 8-bit pass
products are on (they take a pass's 8-bit products before tinyBLAS sees
them); without them, they do not.

## Build

Needs CMake 3.14+, Ninja (or make) and a C++17 compiler. On Windows: Visual
Studio 2022 Build Tools with **Desktop development with C++** (it brings
CMake and Ninja); run the build from an **x64 Native Tools Command Prompt**.

```bat
build.bat                 :: Windows: GGML, then the engine, for AVX2 CPUs
```

```bash
./build.sh                # Linux / macOS
```

Both do these two steps (`build.sh` builds for the machine it is on when
that is 64-bit ARM, and without Metal and Accelerate on a Mac), which you
can also run by hand:

```bash
cmake -S ggml -B ggml/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
      -DGGML_LLAMAFILE=ON -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF
cmake --build ggml/build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=ON -DREALME_NATIVE=OFF
cmake --build build
```

`-DGGML_NATIVE=ON` instead of the five CPU flags, and `-DREALME_NATIVE=ON`
(the default, as upstream), build for the machine you are on (faster there,
may not run elsewhere). `REALME_NATIVE` only matters with GCC or Clang:
upstream tunes the engine with `-march=native` whatever GGML is built for;
RealMe keeps that under the option (`-mcpu=native` on 64-bit ARM). On Windows, copy the GGML DLLs
from `ggml\build\bin` next to `build\qwen3-tts-cli.exe` and `qwen3tts.dll`
(`build.bat` does). For a GPU, add `-DGGML_CUDA=ON` or `-DGGML_VULKAN=ON` to
the GGML step, as upstream describes.

## Models

As upstream: `scripts/setup_pipeline_models.py` downloads Qwen3-TTS and
writes `models/qwen3-tts-0.6b-f16.gguf` and
`models/qwen3-tts-tokenizer-f16.gguf` (see README_UPSTREAM.md, *Model
Setup*). Then, for the faster 8-bit model:

```bash
python tools/quantize_q8.py build/libqwen3tts.so models models_q8     # Linux
python tools\quantize_q8.py build\qwen3tts.dll models models_q8       # Windows
python tools/quantize_q8.py build/libqwen3tts.dylib models models_q8  # macOS
```

It writes `models_q8/` with the 8-bit transformer and the vocoder file beside
it. Point `-m` at `models_q8` to use it. (The engine also picks up an 8-bit
file placed in `models/` by itself.)

## Use

The CLI is upstream's. The fast settings are switches, read when the engine
starts:

```bat
set QWEN3_TTS_THREADS=6
set QWEN3_TTS_FAST_SNAKE=1
set QWEN3_TTS_VOCODER_F32=3
build\qwen3-tts-cli -m models_q8 -r voices\realme.wav -t "Hello." -o hello.wav
```

| Switch | Effect |
|---|---|
| `QWEN3_TTS_THREADS=N` | CPU threads (one per physical core, up to about 6, is best for this model) |
| `QWEN3_TTS_FAST_SNAKE=1` / `exact` | one-pass snake with RealMe's sine / with the C library's (identical) |
| `QWEN3_TTS_VOCODER_F32=3` | the 32-bit vocoder with RealMe's convolution kernel (fastest; with an AVX2 or NEON tile, else as `2`; `2` is the copy-free form and `1` the first version, all with the same samples) |
| `QWEN3_TTS_VOCODER_F32_MAX_FRAMES=N` | longest sentence for it, in codes at 12.5 a second; longer ones use the original vocoder. Default: none with `2` or `3`; 250 (20 s) with `1`, which needs about 1.8 times the original's working memory (20 s: 4.6 GB against 2.5) |
| `QWEN3_TTS_SGEMM=0` | GGML's own multiply for the vocoder too |
| `QWEN3_TTS_SGEMM_ALL=1` | tinyBLAS for every product it takes (opt-in; slightly different sound) |
| `QWEN3_TTS_Q8_BATCH=0` | GGML's own 8-bit products in a pass (identical, slower) |
| `QWEN3_TTS_V3_GENERIC=1` | the vocoder kernel's plain C++ tile even where an AVX2 or NEON one exists (to compare) |
| `QWEN3_TTS_POOL=0` | GGML's thread pool made per graph, as upstream, instead of one for the engine's life |
| `QWEN3_TTS_FFN_F32=0`, `QWEN3_TTS_BROADCAST=0` | the exact fixes off, for comparison |
| `QWEN3_TTS_LOW_MEM=1` | upstream's low-memory mode; the copies and batches stay off in it |
| `QWEN3_TTS_BACKEND=cpu` | upstream's: the CPU even where a GPU or Accelerate backend was built (RealMe's changes are CPU-only) |

Several sentences per pass are a C API call (declared in
`src/realme_batch.inc`, not in `qwen3tts_c_api.h`; declare them as below):

```c
int32_t qwen3_tts_synthesize_batch_with_embedding(
    Qwen3Tts * tts, const char ** texts, int32_t n,
    const float * embedding, int32_t embedding_size,
    const Qwen3TtsParams * params, Qwen3TtsAudio ** out);
/* n on success (free each out[i] with qwen3_tts_free_audio), -1 on failure */
```

Four to eight per pass is the useful range on such a PC; each sentence in a
pass needs about 0.5 GB more for its caches.

A pass whose places are refilled, for a run of sentences of different
lengths (each finished sentence hands its place to the next):

```c
typedef int32_t (*qwen3_tts_next_fn)(void * user, const char ** text);
typedef int32_t (*qwen3_tts_done_fn)(void * user, int32_t index, const float * samples,
                                     int32_t n_samples, int32_t sample_rate);
int32_t qwen3_tts_synthesize_stream_with_embedding(
    Qwen3Tts * tts, int32_t slots, qwen3_tts_next_fn next, qwen3_tts_done_fn done,
    void * user, const float * embedding, int32_t embedding_size,
    const Qwen3TtsParams * params);
/* next: 1 and a sentence to start, 0 if none is waiting now, -1 if none will
   come; asked whenever a place is free. done: each finished sentence by the
   order next() gave them (samples valid during the call; return 0 to stop).
   Returns how many were handed back, -1 on failure. */
```

## Voice server

`server/qwen3_tts_server.py` puts the engine behind **OpenAI's speech API**,
on your own computer, so any tool that can use OpenAI's text-to-speech -- or
"an OpenAI-compatible voice server" -- speaks in a voice cloned here instead:
nothing uploaded, nothing to pay. It needs only Python 3.9 or newer (no
packages) and the engine built above.

1. Put a recording of each voice in `voices/` as a WAV file:
   `voices/realme.wav` is the voice `realme`, the default (10–60 seconds of
   ordinary speech; `voices/anna.wav` would be the voice `anna`). Each is
   made into a speaker embedding once and kept.
2. Set the port your tool expects in `voice_server.json` (8800 unless you
   change it), or pass `--port`.
3. Start it: `server\start_server.bat` (Windows) or `server/start_server.sh`.
   It listens on this computer only.
4. Point the tool at it: base address `http://127.0.0.1:<port>/v1`, model
   `realme-qwen3-tts`, voice `realme`.

```bash
curl http://127.0.0.1:8800/v1/audio/speech -H "Content-Type: application/json" \
     -d '{"model": "realme-qwen3-tts", "input": "Hello from my own voice.", "voice": "realme"}' \
     -o hello.mp3
```

(MP3 needs ffmpeg on the PATH; without it the answer is WAV -- add
`"response_format": "wav"`.)

`voice_server.json`, read at start; command-line options override it:

```json
{"host": "127.0.0.1", "port": 8800, "voice": "realme", "api_key": "",
 "batch": 4, "pause": 0.25, "model_aliases": []}
```

The server is one model, `realme-qwen3-tts`: this engine with these
defaults. A request that names another model is refused, so a tool set up
for a different engine finds out rather than getting this one by surprise.
If a tool can send only another name (OpenAI's `tts-1`, say), list it:
`"model_aliases": ["tts-1"]`.

| | |
|---|---|
| `POST /v1/audio/speech` | `input`, `voice`, `response_format` (`mp3` by default, as OpenAI; `wav`, `pcm`, `flac`, `opus`, `aac` -- all but `wav` and `pcm` need ffmpeg on the PATH, and come back as WAV without it), `speed` (0.25–4, with ffmpeg), and two extras: `language` (`en`, `zh`, `ja`, `de`, `fr`, `es`, `it`, `pt`, `ko`, `ru`) and `temperature` |
| `GET /v1/models`, `GET /v1/voices`, `GET /health` | the model, the voices, what is loaded |
| options | as in `voice_server.json`: `--port`, `--voice` (the default voice), `--api-key` (then ask for `Authorization: Bearer …`), `--batch` (sentences per pass, default 4), `--pause`, `--host` (to listen beyond this computer; set a key as well -- the server warns without one); and `--config` (another settings file), `--threads`, `--exact` (the original engine's numbers) |

Long input is split into sentences and recorded several to a pass; requests
that arrive together for the same voice, language and temperature are
recorded in the same pass. The fast settings (`QWEN3_TTS_FAST_SNAKE=1`,
`QWEN3_TTS_VOCODER_F32=3` -- the vocoder kernel, or `2` with an engine built
before it -- and `QWEN3_TTS_THREADS`) are turned on unless set already. Clone only your own voice, or one you have permission to use.

## Licence and credit

MIT, as upstream: see [LICENSE](LICENSE), which keeps predict-woo's copyright
and adds one for the changes. GGML is MIT (`ggml/LICENSE`, the ggml authors).
The Qwen3-TTS weights are not included and carry their own licence. See
[NOTICE.md](NOTICE.md).

## About RealMe Studio

RealMe Studio, by Xinhua Yu, turns a slide deck into a narrated lecture video
in the lecturer's own voice: it drafts the narration, records it in a cloned
voice — hosted (Gemini) or entirely on the lecturer's own computer with this
engine — and assembles the video, captions and chapters.
