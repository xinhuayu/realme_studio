# realme-qwen3-tts.cpp

A revised [qwen3-tts.cpp](https://github.com/predict-woo/qwen3-tts.cpp) — C++
inference for [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)
voice cloning on the [GGML](https://github.com/ggml-org/ggml) tensor library —
made **about five times as fast on an ordinary CPU** (2.9 times one sentence
at a time), with the same sound.

It is the local voice engine of [RealMe Studio](#about-realme-studio), which
turns slide decks into narrated lecture videos in the lecturer's own cloned
voice, and is released here on its own so anyone can use it.

**All credit for the engine itself goes to predict-woo**, who wrote
qwen3-tts.cpp, to the **ggml authors**, and to the **Qwen team** at Alibaba for
the model. This repository adds a set of changes on top of upstream commit
`b3ba14077cf1` (GGML `3af5f5760e19`); every one is listed in
[CHANGES.md](CHANGES.md) with how it was checked and the switch that turns it
off. Upstream's own README is kept as [README_UPSTREAM.md](README_UPSTREAM.md).

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
| + the vocoder copy-free (everything) | **1.31** | **0.79** | **0.74** | **0.72** |

An hour of speech now takes about three quarters of an hour of work on that
15-watt laptop processor, against almost four. (The last row was measured on
16 sentences, the others on 8.) Beyond 8 per pass the gain is small: 12 per
pass gave 0.73.

**Other hardware.** The transformers' steps mostly wait on memory, so faster
memory (DDR5, more channels) speeds them up; the vocoder is arithmetic, so
more and faster cores speed that up. A desktop processor that holds a high
clock under load should do better than this 15-watt chip, a smaller laptop
worse. GPU builds have not been measured; the batch and vocoder changes are
CPU-only and switch themselves off there, so whether a GPU is faster for this
model on a given machine is for measuring.

| Change | Sound | Default |
|---|---|---|
| **Thread count applied.** Upstream stores `n_threads` and never passes it to the CPU backend, so everything ran on 4 threads. | identical | on (`QWEN3_TTS_THREADS=N` sets it) |
| **`ffn_down` converted to F32 once at load**, not in every step's graph (about 1,240 conversions per second of speech). | identical | on (`QWEN3_TTS_FFN_F32=0` off) |
| **No per-channel copies in the vocoder**: multiplies broadcast instead of first copying a value out to the signal's length. | identical | on (`QWEN3_TTS_BROADCAST=0` off) |
| **Several sentences per pass**: each keeps its own caches, sampling and stopping; the weights are read once for all of them. New C entry point. | identical with sampling off | API: `qwen3_tts_synthesize_batch_with_embedding` |
| **8-bit (Q8_0) copy of the model**, written by the engine's own library from the 16-bit GGUF, no PyTorch. | a different model; not told apart in listening | API: `qwen3_tts_realme_quantize_q8`; `tools/quantize_q8.py` |
| **One-pass snake activation** in the vocoder, with an AVX2 sine. | `exact`: identical; `1`: about 70 dB below the speech | `QWEN3_TTS_FAST_SNAKE=1` (or `exact`) |
| **32-bit vocoder**: convolutions unrolled and multiplied in F32 instead of rounding the signal to F16 at every layer; transposed convolutions as one matrix multiply plus an overlap-add; these products only through GGML's tinyBLAS. | about 50–60 dB below the speech; not told apart in listening | `QWEN3_TTS_VOCODER_F32=1` |
| **The vocoder's decoder stage without copies**: the signal laid out channel by channel, so each convolution reads its taps through overlapping rows instead of an unrolled copy; bias, snake, padding and residual adds fused. About a third less vocoder time than the line above and a third of its memory -- less than upstream's at any length. | the same samples as `1` (about 120 dB apart) | `QWEN3_TTS_VOCODER_F32=2` |
| **Unicode paths on Windows**: model and audio files opened with `ggml_fopen`. | identical | on |

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

Both do the same two steps, which you can also run by hand:

```bash
cmake -S ggml -B ggml/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
      -DGGML_LLAMAFILE=ON -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF
cmake --build ggml/build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=ON
cmake --build build
```

`-DGGML_NATIVE=ON` instead of the five CPU flags builds for the machine you
are on (faster there, may not run elsewhere). On Windows, copy the GGML DLLs
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
set QWEN3_TTS_VOCODER_F32=2
build\qwen3-tts-cli -m models_q8 -r voices\realme.wav -t "Hello." -o hello.wav
```

| Switch | Effect |
|---|---|
| `QWEN3_TTS_THREADS=N` | CPU threads (one per physical core, up to about 6, is best for this model) |
| `QWEN3_TTS_FAST_SNAKE=1` / `exact` | one-pass snake with RealMe's sine / with the C library's (identical) |
| `QWEN3_TTS_VOCODER_F32=2` | the 32-bit vocoder, copy-free (fastest; `1` is the first version, with the same samples) |
| `QWEN3_TTS_VOCODER_F32_MAX_FRAMES=N` | longest sentence for it, in codes at 12.5 a second; longer ones use the original vocoder. Default: none with `2`; 250 (20 s) with `1`, which needs about 1.8 times the original's working memory (20 s: 4.6 GB against 2.5) |
| `QWEN3_TTS_SGEMM=0` | GGML's own multiply for the vocoder too |
| `QWEN3_TTS_FFN_F32=0`, `QWEN3_TTS_BROADCAST=0` | the exact fixes off, for comparison |
| `QWEN3_TTS_LOW_MEM=1` | upstream's low-memory mode; the copies and batches stay off in it |

Several sentences per pass are a C API call (`src/realme_batch.inc`):

```c
int32_t qwen3_tts_synthesize_batch_with_embedding(
    Qwen3Tts * tts, const char ** texts, int32_t n,
    const float * embedding, int32_t embedding_size,
    const Qwen3TtsParams * params, Qwen3TtsAudio ** out);
/* n on success (free each out[i] with qwen3_tts_free_audio), -1 on failure */
```

Four to eight per pass is the useful range on such a PC; each sentence in a
pass needs about 0.5 GB more for its caches.

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
| options | as in `voice_server.json`: `--port`, `--voice` (the default voice), `--api-key` (then ask for `Authorization: Bearer …`), `--batch` (sentences per pass, default 4), `--pause`, `--host` (beyond this computer only with a key); and `--config` (another settings file), `--threads`, `--exact` (the original engine's numbers) |

Long input is split into sentences and recorded several to a pass; requests
that arrive together for the same voice are recorded in the same pass. The
fast settings (`QWEN3_TTS_FAST_SNAKE`, `QWEN3_TTS_VOCODER_F32` with its length
limit by the machine's memory, `QWEN3_TTS_THREADS`) are turned on unless set
already. Clone only your own voice, or one you have permission to use.

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
