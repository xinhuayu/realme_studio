# RealMe's changes to qwen3-tts.cpp

**qwen3-tts.cpp** by predict-woo (https://github.com/predict-woo/qwen3-tts.cpp,
MIT) runs Qwen3-TTS voice cloning in C++ on the **GGML** tensor library
(https://github.com/ggml-org/ggml, MIT, the ggml authors). RealMe Studio uses
it as its local voice engine, and changed it to run about five times as fast
on an ordinary CPU (2.9 times one sentence at a time) with the same sound. This file lists every change, the
switch that turns it off, and how it was checked. The base is upstream commit
`b3ba14077cf1` with GGML `3af5f5760e19`. The Qwen3-TTS weights are the Qwen
team's and carry their own licence; nothing here changes them.

In RealMe Studio the changes are applied to a copy of the unmodified
upstream source each time the engine is built; in the standalone release,
realme-qwen3-tts.cpp, they are already applied in the source.

## Results

On a small PC with a 6-core, 12-thread laptop processor (Intel i7-10710U, 15 W; 64 GB DDR4-2667; AVX2 build, no GPU), a real cloned voice;
seconds of work per second of speech (1.0 is real time):

| | 1 at a time | 4 per pass | 8 per pass | 16 per pass |
|---|---|---|---|---|
| upstream | 3.79 | | | |
| + threads, `ffn_down`, broadcast | 2.82 | | | |
| + one-pass snake | 2.59 | | | |
| + several sentences per pass | 2.54 | 1.66 | 1.51 | |
| + 8-bit model | 1.77 | 1.35 | 1.30 | |
| + 32-bit vocoder | 1.49 | 1.07 | 1.02 | |
| + the vocoder copy-free (all of it) | 1.31 | 0.79 | 0.74 | 0.72 |

(The last row on 16 sentences, the others on 8; 2 per pass 0.92, 12 per
pass 0.73.)

## The changes

| Change | Files | Sound | Switch |
|---|---|---|---|
| **Thread count applied.** Upstream stores it and never passes it to the CPU backend, so everything ran on 4 threads. | `gguf_loader.cpp` | identical | `QWEN3_TTS_THREADS=N` |
| **`ffn_down` converted to F32 once at load**, not inside every step's graph (33 layers, about 1,240 conversions per second of speech). 8-bit weights are multiplied as they are. Not in low-memory mode, which reloads the model per sentence. About 0.4 GB more memory. | `tts_transformer.{h,cpp}` | identical | `QWEN3_TTS_FFN_F32=0` turns it off |
| **No per-channel copies in the vocoder**: it multiplies by per-channel values directly instead of first copying them out to the signal's length. CPU backend only. | `audio_tokenizer_decoder.cpp` | identical | `QWEN3_TTS_BROADCAST=0` |
| **Several sentences in one pass.** Each keeps its own caches, attention, sampling and stopping; the weights are read once for all. CPU backend only; refused in low-memory mode. About 0.5 GB more per sentence in a pass. | `realme_batch.inc`, `friend` lines in `tts_transformer.h` and `qwen3_tts.h` | identical to one at a time with sampling off (with sampling on, each sentence draws its random numbers in a different order, so takes differ as any two takes do) | new entry point `qwen3_tts_synthesize_batch_with_embedding` |
| **One-pass snake activation** in the vocoder, with the C library's sine (`exact`) or RealMe's AVX2 sine (`1`). CPU backend only. | `realme_snake.inc` | `exact`: identical; `1`: about 70 dB below the speech | `QWEN3_TTS_FAST_SNAKE` (off in the engine; RealMe sets `1`) |
| **An 8-bit (Q8_0) copy of the model**, written by the engine's own library from the 16-bit GGUF: the matrices each step multiplies by, in 8 bits; embeddings, norms, the text projection, the speaker encoder and the vocoder as they are. | `realme_quant.inc` | a different model; not told apart in listening | `qwen3_tts_realme_quantize_q8` (RealMe uses it by default once made; `REALME_QWEN3CPP_QUANT=f16` for the 16-bit model) |
| **32-bit vocoder.** Convolutions unrolled and multiplied in F32 (weights converted once at load, about 220 MB) instead of rounding the signal to F16 at every layer; each transposed convolution as one matrix multiply plus an overlap-add that also trims and adds the bias, instead of GGML's one-thread rearrangement and per-sample dot products; outputs padded to a multiple of 4 samples so tinyBLAS takes every product. Working memory is about 1.8 times upstream's per second of speech (20 s: 4.6 GB against 2.5), so sentences longer than `QWEN3_TTS_VOCODER_F32_MAX_FRAMES` codes (12.5 a second; default 250, 20 s) are decoded as upstream does. CPU backend only; off in low-memory mode. | `realme_vocoder.inc`, `audio_tokenizer_decoder.{h,cpp}` | the same convolutions rounded differently: 50–60 dB below the speech (reordering a 32-bit sum alone gives about that through the vocoder's ~30 layers); the codes the transformers produce are unchanged | `QWEN3_TTS_VOCODER_F32=1` (off in the engine; RealMe sets it, with the length limit by the machine's memory: 30 s from 32 GB, 20 s from 16 GB, 10 s below) |
| **The vocoder's decoder stage without copies** (`QWEN3_TTS_VOCODER_F32=2`, building on the line above): the signal feeding each convolution is laid out channel by channel within each time step, so a 7-tap convolution's inputs are adjacent and the multiply reads them through a view whose rows overlap instead of through a sevenfold unrolled copy; a dilated convolution's input is first grouped by time step mod the dilation. Bias, snake, causal padding and that grouping are one pass; conv2's bias and the residual add another; the last convolution (one output channel), its bias and tanh a third. About a third less vocoder time than the line above, and about a third of its working memory -- less than upstream's at any length, so no length limit. Code embeddings, pre-transformer and upsampling as with `1`. CPU backend only; off in low-memory mode. | `realme_vocoder2.inc`, `audio_tokenizer_decoder.{h,cpp}` | the same samples as `1` (about 120 dB apart: the same products, summed in a different order) | `QWEN3_TTS_VOCODER_F32=2` (off in the engine; RealMe sets it on an engine built with it) |
| **GGML's tinyBLAS (`GGML_LLAMAFILE`) for the vocoder's products only**: a gate in GGML's `ggml-cpu.c` lets tinyBLAS take a product only when it is named `realme_sgemm`. Every other product in the engine is computed as upstream GGML computes it. Build GGML with `-DGGML_LLAMAFILE=ON` only with this gate; on an ungated GGML that flag changes every product. | GGML's `src/ggml-cpu/ggml-cpu.c` | with the 32-bit vocoder off: identical to upstream | `QWEN3_TTS_SGEMM=0` |
| **Unicode paths on Windows**: model and audio files opened with `ggml_fopen` instead of `fopen`, so a folder with non-ASCII letters (a user name in Chinese, say) opens. | `gguf_loader.cpp`, `tts_transformer.cpp`, `qwen3_tts.cpp` | identical | none |

## How they were checked

- **Against an unmodified build**, through the C API with sampling off: the
  changed engine with every switch off, with the exact fixes on, and with the
  `exact` snake gave upstream's samples; so did a GGML built with the gated
  tinyBLAS, with the 32-bit vocoder off. An independent check confirmed that
  no unnamed product changes in the gated GGML.
- **Batches** gave the same samples as one sentence at a time, including
  sentences that end at different points, a sentence recorded after a batch,
  the 8-bit model, and the 32-bit vocoder with the one-pass snake.
- **The 32-bit vocoder's arithmetic** was checked against a double-precision
  transposed and ordinary convolution (errors of about 2e-5 on sums of 1,536
  terms), with 1 to 16 threads; sentences over the length limit gave
  upstream's samples exactly.
- **The copy-free decoder stage** against the 32-bit vocoder, with sampling
  off: about 120 dB apart with each snake mode and on a 30-second sentence;
  batches equal to one sentence at a time; with tinyBLAS switched off
  (GGML's own multiply reading the overlapping rows) about 58 dB apart; off
  in low-memory mode. On a 2-core test machine, a 30-second sentence: the
  vocoder 32.0 -> 21.4 s, the engine's peak memory 4.2 -> 2.3 GB.
- **Sanitizers**: the same runs under AddressSanitizer and
  UndefinedBehaviorSanitizer, plus low-memory mode, the 8-bit copier and the
  copy-free decoder stage: no reports.
- **Builds**: the shared library, the CLI and upstream's tests, both inside
  RealMe and as an exported standalone tree; the new code also compiles
  cleanly with clang's `-Wall -Wextra -Wpedantic`.
- **On the real voice** (Windows, 6 cores): the results above; identical
  sound for the exact fixes; no audible difference for the fast snake, the
  8-bit model or the 32-bit vocoder (the vocoder alone 0.85 -> 0.55 seconds
  of work per second of speech, 62 dB below the speech apart).
- **GPU builds** (CUDA, Vulkan, Metal) have not been tested. There the batch,
  broadcast, snake and vocoder changes switch themselves off; the `ffn_down`
  change stays on, using about 0.4 GB more GPU memory, and falls back to
  upstream's per-step conversion if that memory is not there.
