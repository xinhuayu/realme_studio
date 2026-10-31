# RealMe's changes to qwen3-tts.cpp

**qwen3-tts.cpp** by predict-woo (https://github.com/predict-woo/qwen3-tts.cpp,
MIT) runs Qwen3-TTS voice cloning in C++ on the **GGML** tensor library
(https://github.com/ggml-org/ggml, MIT, the ggml authors). RealMe Studio uses
it as its local voice engine, and changed it to run about six and a half
times as fast on an ordinary CPU (three times one sentence at a time) with
the same sound. This file lists every change, the
switch that turns it off, and how it was checked. The base is upstream commit
`b3ba14077cf1` with GGML `3af5f5760e19`. The Qwen3-TTS weights are the Qwen
team's and carry their own licence; nothing here changes them.

In RealMe Studio the changes are made once, in RealMe's engine source
(`engines/qwen3cpp_realme`), which is what is built; the original is kept
unchanged beside it (`engines/qwen3cpp`) as the record of upstream and is
built only when asked for by name. The standalone release,
realme-qwen3-tts.cpp, is that same source.

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
| + the vocoder copy-free | 1.31 | 0.79 | 0.74 | 0.72 |
| + 8-bit pass products, the vocoder kernel (all of it) | 1.25 | 0.68 | 0.58 | 0.57 |

(The last two rows on 16 sentences, the others on 8; with the last row, 2
per pass 0.86, 12 per pass 0.58; one at a time 1.14 in a later run of 8
sentences.) A whole lecture on the same PC, 8 per pass: a 51.8-minute
lecture video (51 slides, 578 sentences) recorded with every change below in
37 minutes (0.72 of its length) and finished in 38 (0.73), the slide videos
made during the recording, and in a second run 35 and 36.5 (0.68 and 0.71); with the copy-free vocoder and fixed passes, a
51.5-minute take of it had taken 44 and 46 (0.86 and 0.89). The vocoder alone, one
sentence at a time (16-bit model): 0.80 seconds of work per second of
speech as upstream, 0.43 in 32 bits, 0.26 copy-free, 0.19 with the kernel;
the four not told apart in listening.

## The changes

| Change | Files | Sound | Switch |
|---|---|---|---|
| **Thread count applied.** Upstream stores it and never passes it to the CPU backend, so everything ran on 4 threads. | `gguf_loader.cpp` | identical | `QWEN3_TTS_THREADS=N` |
| **One pool of worker threads for the engine**, kept as long as its CPU backend (GGML otherwise makes one for every graph, about 16 per step of speech). With GGML built without OpenMP (`REALME_GGML_OPENMP=0` when RealMe builds it) the pool's threads are GGML's own, waiting at each barrier on an atomic counter instead of in the compiler's OpenMP runtime. Thread scheduling does not change the arithmetic: every output is computed whole by one thread. Which is faster depends on the machine and its compiler's OpenMP (on Windows, MSVC's), so it is measured there; OpenMP stays the default, as GGML's spinning threads slow down badly when other work shares the cores. | `gguf_loader.cpp` | identical | `QWEN3_TTS_POOL=0`; GGML's own threads: `-DGGML_OPENMP=OFF` |
| **`ffn_down` converted to F32 once at load**, not inside every step's graph (33 layers, about 1,240 conversions per second of speech). 8-bit weights are multiplied as they are. Not in low-memory mode, which reloads the model per sentence. About 0.4 GB more memory. | `tts_transformer.{h,cpp}` | identical | `QWEN3_TTS_FFN_F32=0` turns it off |
| **No per-channel copies in the vocoder**: it multiplies by per-channel values directly instead of first copying them out to the signal's length. CPU backend only. | `audio_tokenizer_decoder.cpp` | identical | `QWEN3_TTS_BROADCAST=0` |
| **Several sentences in one pass.** Each keeps its own caches, attention, sampling and stopping; the weights are read once for all. CPU backend only; refused in low-memory mode. About 0.5 GB more per sentence in a pass. | `realme_batch.inc`, `friend` lines in `tts_transformer.h` and `qwen3_tts.h` | identical to one at a time with sampling off (with sampling on, each sentence draws its random numbers in a different order, so takes differ as any two takes do) | new entry point `qwen3_tts_synthesize_batch_with_embedding` |
| **A pass whose places are refilled**: when a sentence in a pass ends, it is decoded and handed back at once and the next sentence takes its place in the following step, so a pass does not run on half empty while its longest sentence finishes. Each sentence keeps its own frame count, caches and stopping, as in a fixed pass. | `realme_batch.inc` | identical to one at a time with sampling off, whatever the order in which places are refilled | new entry point `qwen3_tts_synthesize_stream_with_embedding` |
| **One-pass snake activation** in the vocoder, with the C library's sine (`exact`) or RealMe's AVX2 sine (`1`). CPU backend only. | `realme_snake.inc` | `exact`: identical; `1`: about 70 dB below the speech | `QWEN3_TTS_FAST_SNAKE` (off in the engine; RealMe sets `1`) |
| **An 8-bit (Q8_0) copy of the model**, written by the engine's own library from the 16-bit GGUF: the matrices each step multiplies by, in 8 bits; embeddings, norms, the text projection, the speaker encoder and the vocoder as they are. | `realme_quant.inc` | a different model; not told apart in listening | `qwen3_tts_realme_quantize_q8` (RealMe uses it by default once made; `REALME_QWEN3CPP_QUANT=f16` for the 16-bit model) |
| **32-bit vocoder.** Convolutions unrolled and multiplied in F32 (weights converted once at load, about 220 MB) instead of rounding the signal to F16 at every layer; each transposed convolution as one matrix multiply plus an overlap-add that also trims and adds the bias, instead of GGML's one-thread rearrangement and per-sample dot products; outputs padded to a multiple of 4 samples so tinyBLAS takes every product. Working memory is about 1.8 times upstream's per second of speech (20 s: 4.6 GB against 2.5), so sentences longer than `QWEN3_TTS_VOCODER_F32_MAX_FRAMES` codes (12.5 a second; default 250, 20 s) are decoded as upstream does. CPU backend only; off in low-memory mode. | `realme_vocoder.inc`, `audio_tokenizer_decoder.{h,cpp}` | the same convolutions rounded differently: 50–60 dB below the speech (reordering a 32-bit sum alone gives about that through the vocoder's ~30 layers); the codes the transformers produce are unchanged | `QWEN3_TTS_VOCODER_F32=1` (off in the engine; RealMe now sets `3` or `2`, below; with `1` it sets the length limit by the machine's memory: 30 s from 32 GB, 20 s from 16 GB, 10 s below) |
| **The vocoder's decoder stage without copies** (`QWEN3_TTS_VOCODER_F32=2`, building on the line above): the signal feeding each convolution is laid out channel by channel within each time step, so a 7-tap convolution's inputs are adjacent and the multiply reads them through a view whose rows overlap instead of through a sevenfold unrolled copy; a dilated convolution's input is first grouped by time step mod the dilation. Bias, snake, causal padding and that grouping are one pass; conv2's bias and the residual add another; the last convolution (one output channel), its bias and tanh a third. About a third less vocoder time than the line above, and about a third of its working memory -- less than upstream's at any length, so no length limit. Code embeddings, pre-transformer and upsampling as with `1`. CPU backend only; off in low-memory mode. | `realme_vocoder2.inc`, `audio_tokenizer_decoder.{h,cpp}` | the same samples as `1` (about 120 dB apart: the same products, summed in a different order) | `QWEN3_TTS_VOCODER_F32=2` (off in the engine; RealMe sets it on an engine built with it) |
| **The vocoder's decoder stage with RealMe's own convolution kernel** (`QWEN3_TTS_VOCODER_F32=3`): the signal stays channel by channel from the first convolution to the last; each convolution reads its taps' rows in place (dilated ones too, with no regrouping) and multiplies them, 6 time steps by 16 channels at a time, by weights packed once at load; bias, residual add and the next convolution's snake are applied as rows leave the kernel, written straight into the next convolution's padded input. A residual unit is one operation, and a transposed convolution (kernel 2s, stride s) s two-tap products with no overlap-add. Each output is summed in one fixed order whatever the thread count. About a quarter less vocoder time than `2`, with about the same memory. The method is the same on every processor; only the innermost tile and the snake are processor-specific: AVX2/FMA on x86 (chosen at run time), NEON on 64-bit ARM, each with a plain C++ version beside it (4-wide vectors of the compiler's vector extension). Where only the plain tile would run, `2` is used -- GGML's products, tuned for what GGML was built for, were faster than it where both were timed -- unless `QWEN3_TTS_V3_GENERIC=1`, which also forces the plain tile on AVX2 or NEON, to compare. CPU backend only; off in low-memory mode. | `realme_vocoder3.inc`, `audio_tokenizer_decoder.{h,cpp}`, a line in `realme_vocoder2.inc` (3 also enables 2, as its fallback) | the same samples as `2` (about 110 dB apart: the same products, summed in a different order; the plain and NEON tiles likewise, 110--120 dB); the snakes and biases computed exactly as there; a pass equals one sentence with every tile | `QWEN3_TTS_VOCODER_F32=3` (off in the engine; RealMe sets it on an engine built with it) |
| **GGML's tinyBLAS (`GGML_LLAMAFILE`) for the vocoder's products only**: a gate in GGML's `ggml-cpu.c` lets tinyBLAS take a product only when it is named `realme_sgemm`. Every other product in the engine is computed as upstream GGML computes it. Build GGML with `-DGGML_LLAMAFILE=ON` only with this gate; on an ungated GGML that flag changes every product. | GGML's `src/ggml-cpu/ggml-cpu.c` | with the 32-bit vocoder off: identical to upstream | `QWEN3_TTS_SGEMM=0` |
| **8-bit products for a pass, several sentences at once**: each 8-bit weight row is loaded once and multiplied against every sentence in the pass, with exactly the arithmetic of GGML's own 8-bit dot product for each sentence (on x86 its AVX2 one: the same integer products, the same scale rounding, the same eight-lane accumulation and final sum). A self-check against GGML's own dot product at first use switches it off on any difference. Only in a pass (two or more sentences), for the transformers' 8-bit matrices; attention is GGML's. On x86 with AVX2 (checked at run time) it reproduces GGML's AVX2 dot product; on 64-bit ARM, GGML's NEON one (with the dot-product instruction where the build has it, else GGML's emulation of it) -- there, on processors with the 8-bit matrix instructions (i8mm), GGML takes two rows at a time for two or more columns, so without this kernel a pass and one sentence differ slightly. Elsewhere GGML's own. CPU backend. | `realme_batch.inc` | identical to GGML's (and so to one sentence at a time) | `QWEN3_TTS_Q8_BATCH=0` |
| **tinyBLAS for every product** (opt-in), through the same gate. | GGML's `src/ggml-cpu/ggml-cpu.c` | the transformers' results move slightly, so a sentence comes out as a different take of itself; one sentence and a pass still match while the line above is on (without it they do not, and RealMe's measurement then turns passes off). Listen before keeping it. | `QWEN3_TTS_SGEMM_ALL=1` (off unless set) |
| **The engine's compiler flags follow the processor target.** Upstream adds `-march=native` for GCC and Clang whatever GGML is built for, so a portable build still had its engine tuned for, and only runnable on, the building computer; older Apple compilers refuse it on 64-bit ARM. Kept under an option, on by default (`-mcpu=native` on 64-bit ARM, as GGML uses there); RealMe switches it off for any build not meant for this computer. RealMe's own AVX2 code does not need it: it checks the processor at run time. | `CMakeLists.txt` | identical | `-DREALME_NATIVE=OFF` |
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
- **The 8-bit pass products**: the same samples as before, one sentence and
  passes of 3 and 16, with the kernel on and off; the self-check compares
  every column count from 1 to 11 bit for bit. On a 2-core test machine, a
  pass of 8 took 16% less time, a pass of 4 about 6%; one sentence at a time
  is unchanged (it does not use the kernel). With `QWEN3_TTS_SGEMM_ALL=1` as
  well, a pass of 8 a further 4% (on 2 cores), and one sentence and a pass
  still matched.
- **The vocoder kernel** against the copy-free decoder stage, with sampling
  off: 111 dB apart on a 5-second sentence and 112 dB on a 30-second one;
  one thread and two give the same samples; a pass equal to one sentence at
  a time; the copy-free stage unchanged by the fallback edit (the same
  samples as before). On a 2-core test machine the vocoder took 27% less
  time (3.57 -> 2.61 s for a 5-second sentence, four runs each; a 30-second
  sentence 21.2 -> 16.0 s), peak memory 2.32 -> 2.41 GB.
- **Other processors.** The vocoder kernel's plain tile, forced on x86
  (`QWEN3_TTS_V3_GENERIC=1`): a pass equal to one sentence at a time, 121 dB
  from the AVX2 tile. x86 output is unchanged by the ARM and plain versions
  (the same samples as before). On 64-bit ARM, built with a cross compiler
  and run under an emulator (not timed), engine and GGML both for ARMv8.2
  with the dot-product instruction: the 8-bit pass products pass their
  self-check; a pass equals one sentence at a time with vocoder 2 and with
  vocoder 3; vocoder 3 is 110--112 dB from vocoder 2; the plain tile gives
  the same samples as the NEON tile (the same fused multiply-adds in the
  same order). With the engine built for ARMv8.0 against that GGML, the
  self-check found that its products were not GGML's (GGML's dot-product
  instruction groups the sums differently from its emulation) and handed
  them back to GGML: same samples, a pass still equal to one sentence. Both
  built for ARMv8.0 (no dot-product instruction): the self-check passes
  (the kernel's emulation is GGML's), and a pass equals one sentence. The
  engine and GGML are meant to be built for the same processor level, as
  RealMe's build and `build.sh` do.
- **Refilled passes**: seven sentences of different lengths (one stopping
  after 7 codes, one after 13, the rest at the limit), with 1, 2, 4 and 8
  places, the longest first so the short ones cycle through the other
  places: every sentence identical to one at a time. Through RealMe's own
  adapter, with a sentence in another language in the middle (it waits for
  a stream of its own): identical. On the 2-core machine, 16 sentences of
  mixed length, 4 at a time: 9% less time than fixed passes of 4.
- **Sanitizers**: the same runs under AddressSanitizer and
  UndefinedBehaviorSanitizer, plus low-memory mode, the 8-bit copier, the
  copy-free decoder stage, the 8-bit pass products, the vocoder kernel and
  refilled passes: no reports.
- **Builds**: the shared library, the CLI and upstream's tests, both inside
  RealMe and as an exported standalone tree; the new code also compiles
  cleanly with clang's `-Wall -Wextra -Wpedantic`.
- **On the real voice** (Windows, 6 cores): the results above; identical
  sound for the exact fixes; no audible difference for the fast snake, the
  8-bit model or the 32-bit vocoder (the vocoder alone 0.85 -> 0.55 seconds
  of work per second of speech in a first measurement, 62 dB below the speech
  apart; the later side-by-side check gave 0.80 -> 0.43, above).
- **GPU builds** (CUDA, Vulkan, Metal) have not been tested. On a Mac,
  GGML's Metal and Accelerate backends are built by default and preferred
  to the CPU; RealMe's build and the release's `build.sh` leave them out, and
  `QWEN3_TTS_BACKEND=cpu` keeps RealMe's CPU changes on in any build. There the batch,
  broadcast, snake and vocoder changes switch themselves off; the `ffn_down`
  change stays on, using about 0.4 GB more GPU memory, and falls back to
  upstream's per-step conversion if that memory is not there.
