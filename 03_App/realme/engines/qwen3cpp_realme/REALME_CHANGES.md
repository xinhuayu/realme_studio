# RealMe's changes to qwen3-tts.cpp

## CPU attention copy controls

Optional bits 2048/4096/8192 select contiguous sentence views, ordered F32 gather,
and single-query causal-mask elision when no future key exists. All default off.
Matrix/softmax arithmetic and persistent V-cache layout are unchanged. Gather
uses at most GGML_MAX_SRC-1 operands per custom node; strided/unsupported inputs
retain legacy concat. The production attention core is an owned testable helper.
Current mask is 16383; there are still 19 upstream-derived patches. Native tests
add 2,100 exact attention comparisons and 64 copy/mask cases. See root
`CPU_ATTENTION_HANDOFF.md` for boundaries, copy accounting, short voice evidence,
portability and the read-only weight-compressibility survey. No runtime weight
codec or full lecture test is introduced.

## Predictor scale storage and CPU matrix scheduling

Optional runtime bits 256/512/1024 select predictor-only F32 scales, shape-based
custom-product task requests, and aligned four-row worker boundaries. All default
off. All 247 current Q8 matrices are consumed; retaining expanded scales only for
50 predictor matrices reduces array storage from 69,206,016 to 13,762,560 bytes.
Packed/original model bytes are unchanged. Task policy uses bounded proxy work
and existing GGML thread limits; alignment preserves per-matrix tile pairs.
No timing gain is claimed. Owned helpers plus `cpp.py` remain reproducible with
19 upstream-derived patches; current runtime mask is 2047. See root
`CPU_SCHEDULE_HANDOFF.md` for tests, portability and optional batch comparisons.

## CPU embedding access/cache and four-row packed kernel

Three optional exact-output controls add runtime bits 32/64/128: direct CPU
embedding-row access, an 8 MiB converted-row payload cache with bounded metadata,
and four-row products using existing RMQ8PK1 two-row tiles. All default off.
The row owner resets on model unload; cache misses can retain the original
backend copy independently of direct access. Four-row work never crosses worker
or matrix boundaries and uses two-row tails. Original GGUF/cache bytes stay
unchanged. `cpp.py` patch 19 reproduces embedding header/CPP changes; rebuild
all affected libraries because the internal TTSTransformer layout changes.
See root `CPU_EMBEDDING_TILES_HANDOFF.md` and `CPU_FEATURES.json` for controls,
portability, test records, and optional short timing/quality batches.

## Incremental CPU runtime controls and exact packed scales

`realme_runtime.inc` provides per-engine call snapshots and busy rejection.
`qwen3_tts_set_cpu_options` overrides five compute flags between calls: top-k,
shared preparation, grouped products, packed weights, and optional exact F32
packed scales. Scale expansion retains additional private memory and removes
repeated half conversions; default off, no speed claim. Original GGUF/RMQ8PK1
files stay unchanged. C API edits are reproducible patch 18 in `cpp.py`.
Native checks now cover 2,880 tensor combinations plus sampler/RNG cases.
See root `CPU_RUNTIME_HANDOFF.md`, `CPU_FEATURES.json`, and
`03_App/check_cpu_matrix.py` for incremental/pair/factorial plans, portability,
accuracy evidence and pending hardware checks. Full lectures remain deferred.

**qwen3-tts.cpp** by predict-woo (https://github.com/predict-woo/qwen3-tts.cpp,
MIT) runs Qwen3-TTS voice cloning in C++ on the **GGML** tensor library
(https://github.com/ggml-org/ggml, MIT, the ggml authors). RealMe Studio uses
it as its local voice engine, and changed it to run about six and a half
times as fast on an ordinary CPU (three times one sentence at a time) with
the same sound. This file lists every change, the
switch that turns it off, and how it was checked. The base is upstream commit
`b3ba14277cf1` with GGML `3af5f5760e19`. The Qwen3-TTS weights are the Qwen
team's and carry their own licence; nothing here changes them.

In RealMe Studio the changes are made once, in RealMe's engine source
(`engines/qwen3cpp_realme`), which is what is built; the original is kept
unchanged beside it (`engines/qwen3cpp`) as the record of upstream and is
built only when asked for by name. The standalone release,
realme-qwen3-tts.cpp, is that same source.

## Results

### CPU preparation, grouped projections, and RMQ8PK1

The owned sampling helper selects the top-k cutoff without sorting score/index
pairs, with `QWEN3_TTS_TOPK_SELECT=0` preserving the legacy path. All three
single-transformer sampling sites and the batch sampler use it. Optional
`QWEN3_TTS_SEED` supports reproducible sampled checks and is reflected in the
adapter's sampled-voice fingerprint.

Opt-in `QWEN3_TTS_Q8_PREP=1` shares prepared input columns through a GGML dependency;
`QWEN3_TTS_Q8_GROUP=1` groups Q/K/V and gate/up products into contiguous output
views. `QWEN3_TTS_Q8_PACK=1` uses a validated, read-only RMQ8PK1 derived file with
two-row integer tiles and original half-scale bits. Originals are never rewritten.
The packed CPU kernel retains per-block arithmetic/reduction order and includes
singleton batch tails; unsupported/cache-failure paths retain original kernels.
The internal model gains a shared owner, so native consumers must rebuild.
See root `CPU_COMPUTE_HANDOFF.md` for the 1,440 tensor tests, seeded sampler/short
voice checks, format/rollback details, memory tradeoff, and remaining validation.

### Opt-in read-only transformer weight mapping

`QWEN3_TTS_WEIGHT_MMAP=1` extends the existing CPU text map to every tensor in
the loaded transformer context. Original GGUF values/types/layouts are used
without conversion or repacking. This can reduce private committed weight
storage and allow OS file-page sharing, while retaining hot-weight computation
and bandwidth costs. The default remains text-only mapping. Native Q8/F16 checks
compare every mapped byte, protection, lifetime, and fallback; bounded short
waveform comparisons are recorded in root `WEIGHT_MMAP_HANDOFF.md`.
`apply_weight_mmap_patch` reproduces the mapped-tensor copy skip and handles
GGML's expected null allocation when all tensors already have backing storage.
No full lecture or speed gain is claimed.

### Reuse CPU generation scratch vectors

`realme_batch.inc` now uses the owned `realme_batch_scratch.inc` helper to retain
input, position, index, embedding-sum, hidden-state, and logit vector capacity
within one fixed batch or streaming call. Every reset restores the previous
fresh-vector initialization; model operations and graph scheduling are unchanged.
`QWEN3_TTS_SCRATCH_REUSE=0` disables this capacity reuse. A model-free native
check verifies matching bytes during shrink/refill, zero allocations after
warming tested shapes, and call-local cleanup/ownership. No speed or waveform
claim is made; later comparisons are optional. Root `SCRATCH_REUSE_HANDOFF.md`
records the implementation, memory tradeoff, rollback snapshot, and cross-checks.

### Speaker embedding bounds and CPU runtime routing

Embedding synthesis now rejects buffers whose declared length differs from the
loaded model's hidden size, before reading speaker data. This covers single,
reference-encoder, batch, and stream synthesis. The idempotent
`apply_embedding_validation_patch` reproduces the upstream-file edits.
The adapter permits CPU batches on a GPU-capable build when CPU is explicitly
selected, and preserves an explicit environment backend override in auto mode.
Valid-input arithmetic and weights are unchanged. XPS checks: 21 guard-page
boundary rejections, 142 batch checks, patch reproduction/idempotence/drift checks.
New comparative timing, listening, lecture stress, and NUC installation are
deferred. See root `CPU_TESTING_HANDOFF.md` and optional
`03_App/check_lecture_cpu.py` for staged, recorded runs.

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
| **Lazy mode-3 fallback weights.** Prepare the packed decoder weights first, retain only the three earlier regular F32 convolutions, and prepare the missing older F32 layout only if the packed decoder is rejected. A failed fallback allocation returns a clear error and can be retried. | `realme_vocoder.inc`, generated edits in `audio_tokenizer_decoder.{h,cpp}` | identical float samples in modes 0–3, frame-cap and shape fallback, short input, low-memory mode, and allocation failure/retry; see `VOCODER_STORAGE_HANDOFF.md` | `QWEN3_TTS_VOCODER_LAZY=0` retains eager copies |
| **Read-only mapped text embeddings on CPU.** The F16 vocabulary table stays in the existing GGUF and is paged in on demand. GGML wraps its original bytes instead of making a private table and a full-size read buffer. Mapping failures and other backends use the copying loader. | `realme_text_mmap.inc`, generated edits in `tts_transformer.{h,cpp}` | every table byte checked against the file; identical greedy float samples against the cache-enabled engine; see `TEXT_MMAP_HANDOFF.md` in RealMe for measurements and limits | `QWEN3_TTS_TEXT_MMAP=0` selects copying |
| **Growing and reused sentence caches.** CPU batches start with the prompt plus 128 audio positions and the existing allowance, grow by copying the used F16 key and value prefixes before the next graph is built, and keep the original generation limit. Refilled streams reuse completed slots' allocations within the stream; all sentence state is reset. Single-sentence allocation is unchanged. The recorder's memory guard stays conservative for long sentences. | `realme_batch.inc` | identical float samples in forced growth and refill checks against the prechange library; see `CACHE_GROWTH_HANDOFF.md` in RealMe for measurements and limits | `QWEN3_TTS_CACHE_GROW=0` restores eager allocation and no reuse; `QWEN3_TTS_CACHE_INITIAL` defaults to 128; `QWEN3_TTS_CACHE_TRACE=1` logs events |
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

## Optional Q8 model-data experiments (2026-10-31)

Q8 remains the production baseline. No trained Q8, F16 or F32 values are changed.
New controls default off and retain the previous paths. The current opaque C API
runtime mask is 1048575; all controls except profiling are enabled by 786431.
Bits 16384/32768/65536 select native four-row RMQ8PK2 panels, predictor-only panel
layout and read-only persisted predictor F32 scales. Bit 131072 selects an
8MiB/64-entry whole-token-sequence text projection cache, bit 262144 enables
packed-product counters, and bit 524288 selects resolved tensor-pointer dispatch.
The environment suffixes are Q8_PANEL4, Q8_ROLE_LAYOUT, Q8_MAPPED_SCALES,
TEXT_PROJECT_CACHE, Q8_PROFILE and Q8_INDEX, each prefixed by QWEN3_TTS_.

RMQ8PK1 is unchanged. PK2 descriptors identify per-matrix row layout and kernel
ABI; strict semantic source/payload checks and atomic publication are retained.
Original GGUF files remain required for fallback. Packed consumers are currently
x86 AVX2/FMA CPU code; other backends keep their original routes. Layout choices
are experiments, not measured universal winners. Profiling reports summed worker
time and logical payload visits, not elapsed time or hardware memory traffic.
Current XPS evidence includes 14,400 exact new-format matrix comparisons and
short greedy/seeded-sampled speech checks against the saved engine. Controlled
timing, broader listening, NUC and ARM/GPU checks remain separate.

## Complete CPU-native bundle, PK3 interleaving and complete vocabulary

RMCPU001 is an opt-in mapped container with both original Q8/F16 GGUF model
sections, PK2 and PK3 prepared layouts, full F32 projected text vocabulary and
provenance. Transformer weights attach directly to mapped sections; encoder
and decoder copy selected weights to existing backend buffers. Empty model
directory and direct .rmcpu input checks require no extraction. CPU backend
must be explicit; existing GPU/auto routes retain their defaults.

PK3 physically interleaves Q/K/V and gate/up row panels in layer order with
per-panel q/d/F32-scale offsets. Dot-product/FMA/reduction order is unchanged.
Bit 1048576 (QWEN3_TTS_Q8_INTERLEAVE) selects this prepared route. Bit 2097152
(QWEN3_TTS_TEXT_VOCAB) copies compatible rows from all 151,936 x 1,024 F32
projected results. Default off; runtime mask 4194303. Vocabulary guards bind
source, producer DLLs, CPU execution fingerprint, FP mode and settings. A
guard failure uses the source projection graph. Original precision is kept.

Model-bound mapping ownership, lazy per-section SHA256 validation, optional
eager validation, busy-guarded offline asset APIs and atomic fresh conversion
are recorded in CPU_NATIVE_BUNDLE_HANDOFF.md. Required source sections are
validated before selection; corrupt optional assets retain source routes.
The bundle is immutable for a loaded model's lifetime. Original GGUFs and old
packs remain available; a direct invalid bundle has no implicit fallback dir.

Final XPS validation: 23,040 exact PK2/PK3 matrix comparisons, seven short
voice profiles, live switching, seeded sampling, public single synthesis,
corrupt-data fallback, 5,115 vocabulary row comparisons, exact 1,024-value
speaker embedding and eight Python check scripts. Source integration is
reproducible through apply_cpu_bundle_patch (21 patches total). This is Windows
x86 CPU validation, not a NUC/GPU/ARM result or controlled speedup claim.
See CPU_NATIVE_BUNDLE_XPS_CHECKS.json and the frozen producer under
_review/cpu_native_bundle/validated/. Full lectures and broad listening remain
optional; the user's report has not been edited.

## Stage timing, batched predictor attention, decoding beside the pass (2026-11-01)

Added by Claude after reviewing the CPU work above. All three are off by
default; with them off the engine computes exactly what it did.

One frame routine. The fixed pass (`generate`) and the stream each carried a
copy of the per-frame code: the code predictor's 15 runs, keeping the frame,
and the talker step. Both now call `frame_rest`, so a change to one frame is
made once. Each sentence uses its own frame count, as the stream already did;
in the fixed pass that count equals the shared one. Greedy codes from the
refactored engine equal the previous engine's code for code (1, 4, 8 and 16
sentences, including a sentence cache growing from 146 to 238 positions).

Stage timing (QWEN3_TTS_STAGE_TIMING=1). Each pass or stream reports where
its time went: prompt, first code, predictor graph building and computing,
sampling, embeddings, cache growth, talker building and computing, vocoder
and hand-back, with graph nodes per frame step. Off, each mark is one branch.
On a synthetic model with the real shapes (random weights, two threads) the
code predictor took 64-71% of generation and the talker 26-34%; graph building
was 2-3%. The predictor's graphs had about 2,100 nodes each at 16 sentences,
almost all of them per-sentence attention.

Batched code predictor attention (QWEN3_TTS_CP_BATCH_ATTN=1). The predictor's
sentences all stand at the same step of the same frame and its cache holds
one frame only, so one cache of [head_dim, kv_heads, 16, sentences] per layer
serves the pass, and each layer's attention is one set of operations for all
sentences (`realme_attention_batched`) instead of one per sentence. The ops
are those of `realme_attention_graph`, applied slice by slice in GGML's third
and fourth dimensions, so the numbers are the same: codes identical for 1, 2,
5 and 16 sentences (test_cp_batch_attention). Predictor nodes per frame step
fell from 31,640 to 3,440 at 16 sentences; on two threads a 16-sentence pass
took 4-6% less time in alternating runs, while at two to four sentences the
difference was within run-to-run noise. The saving is
the per-node work and thread synchronisation, which grows with threads, so a
six-core machine may gain more; that is for its own measurement.

Decoding beside the pass (QWEN3_TTS_DECODE_ASYNC=1). In the stream a finished
sentence was decoded inside the loop while every other sentence waited. With
the switch on, the vocoder takes a CPU backend of its own
(`apply_decode_async_patch`, `realme_decoder_backend`) with the engine's
thread count, and one worker thread decodes finished sentences while the pass
continues; during that stream the vocoder uses QWEN3_TTS_DECODE_THREADS
threads when set, and the engine's count again afterwards. Audio is still handed back on the
caller's thread, in the order sentences finished, with the decoder's own
samples. The transformer and vocoder otherwise share one CPU backend, which
runs one graph at a time, so the switch must be set when the engine loads; if
the vocoder has no backend of its own the stream decodes in the loop as
before. Whether overlap gains anything depends on idle cores: the two pools
can compete for the same cores. test_decode_worker checks order, samples,
empty sentences, a failure reported in place and joining with work queued
(also under ThreadSanitizer); identical audio on the real model is part of
run_cpu_combinations.py.

Smaller changes. Cache growth copies CPU caches with one memcpy per tensor
instead of through a 1 MiB staging buffer (test_cache_growth). SHA-256 has a
portable implementation where the Windows provider is absent (NIST vectors and
301 lengths against hashlib, test_sha256); the bundle is still mapped only on
Windows, so this lets its format checks run elsewhere, not the bundle itself.

Measuring them together. run_cpu_combinations.py runs every combination of
the batched predictor attention, decoding beside the pass, the talker's
ordered attention gather (runtime option 4096) and two pass widths, each
twice in a shuffled and then reversed order; requires identical greedy audio
from every run; estimates each switch's effect and each pair's interaction
from all runs; and confirms the fastest combination against the current
setting in alternating pairs (each at least 3% faster). It changes no
setting; it prints the RealMeStudio/.env lines for a final lecture render.

## Kept code predictor graphs (2026-11-01, later)

With the batched predictor attention, a predictor graph depends only on how
many sentences are in the pass and on the pass cache, not on which sentences.
So its 15 graphs (prefill and 14 steps) are built once for a width, each in a
metadata buffer of its own with its own GGML allocator, and run on the CPU
backend directly frame after frame (`cp_graphs`, `cp_reused`, `cp_compute`).
A new width, a grown pass cache or new packing builds them again; the cache
growing clears them first, since they refer to it. On by default with
QWEN3_TTS_CP_BATCH_ATTN=1; QWEN3_TTS_CP_GRAPH_REUSE=0 turns it off. Same
graphs, same operations: codes identical with and without, while widths go
2, 4, 3, 6, 1, 5 and members change at one width (test_cp_graph_reuse; 150
graph builds instead of 210 there, 15 instead of 600 in a 40-frame pass;
also under ASan/UBSan). On the XPS quick test graph building took 5-7% of the
time; its saving there is for the next measurement. About 1 MB of graph
metadata per kept graph.

Decoding beside the pass, full threads when alone (2026-11-01). The NUC
combination run showed the switch costing time: its corpus had one sentence
that, under greedy decoding, never ended and ran alone to the 4096-frame limit
(328 s of audio), and that sentence was then decoded on QWEN3_TTS_DECODE_THREADS
(two) threads while nothing else computed: vocoder 150 s against 90 s in the
loop. Each decode now carries its own thread count, set on the worker thread
before that sentence's vocoder graph: the lowered count while other sentences
are speaking, the engine's full count when the finished sentence was the last
one in the pass. test_decode_worker passes (also under ThreadSanitizer).

## A frame limit per sentence, and run-on takes recorded again (2026-11-01)

Qwen3-TTS ends a sentence by choosing its end code. Now and then it never does
and talks on alone to the call's limit: 4096 frames, 5.5 minutes of audio,
about 10 minutes of computing on the NUC and a vocoder graph of about 12 GB
(the NUC combination run met this with greedy decoding on a 33-word item).

Engine: each sentence in a pass or stream has its own limit, 40 frames plus
QWEN3_TTS_FRAMES_PER_TOKEN (default 15) per prompt token, never above the
call's (`frame_limit`, `seq::max_frames`). That is several times a slow
reading pace, so a sentence that ends by itself never meets it and its codes
are unchanged (test_frame_limit: a sentence that never ends stops at its
limit with every frame up to there identical; greedy codes otherwise equal
the previous engine's). 0 turns it off.

Studio (tts_qwen3_dll.py, speak.py, prerecord.py): a single call's limit is
set the same way from the text (`frame_cap`); a take longer than twice the
text said slowly (1.5 words, 10 characters or 3 CJK characters a second, plus
2 s) is a run-on (`ran_on`). A run-on take is never cached. Recorded ahead
(stream, pass or single) it is handed back as failed and recorded again in
order; in order it is said again once (a fresh draw), then split into shorter
calls, and a piece that cannot be split further is kept with a note to
listen. REALME_RUNAWAY_CHECK=0 turns the check and the single-call limit
off. t_runaway.py; t_prerecord.py and the batch/stream adapter tests pass.


## Kept talker graph, AVX-VNNI kernels, fewer codebooks (2026-11-01, evening)

Kept talker graph (QWEN3_TTS_TALK_GRAPH_REUSE; on with QWEN3_TTS_CP_BATCH_ATTN=1
as the kept predictor graphs, =0 off, =1 on without it). A talker step's graph
differs from the previous step's only in where each sentence's attention
writes and reads its own cache: one position further on. So it is built once
for a set of sentences, at each sentence's last cache position (every
intermediate then has room for any step before), allocated with its own GGML
allocator, and before each step moved to the real positions
(`realme_attention_move`: the two cache writes and their copies placed, the
cache views, their permutes and the transpose given the new length, the
attention products, scale, mask and softmax made that length again, the mask's
position set where the mask is used). The CPU backend plans every compute from
the tensors as they are, so the same ops run on the same numbers. Another
sentence, a grown cache or new packing builds it again (`talk_graph`, keyed by
each sentence's id and cache buffer). test_talk_graph_reuse: 2816 codes
identical to talker graphs built every step and to the path with no kept
graphs at all, across changing members and widths, caches growing from 4
positions, with and without the mask copy (also under ASan/UBSan; two
deliberately wrong moves, one length and one stale write position, both
caught). Talker graph building took 2.8-3.0% of the XPS quick test (7 ms a
frame step); kept, about 0.2 ms.

AVX-VNNI kernels (QWEN3_TTS_Q8_VNNI=0 off). Intel Core 12th generation and
Core Ultra, and AMD Zen 5, add vpdpbusd: the four-byte sums that AVX2 makes
with vpmaddubsw then vpmaddwd, in one instruction. RealMe's 8-bit kernels (the
row kernel, the two- and four-row packed tiles) are now in
`realme_q8_kernels.inc`, included twice: once with AVX2's pair, once with
vpdpbusd, chosen at first use from CPUID 7.1 EAX bit 4. Each lane gets the same
integer both ways (Q8_0 values lie in -127..127, so nothing saturates), and
everything after it is the same code, so the numbers are identical bit for bit;
the self-checks against GGML's own dot product run on the computer itself.
test_q8_vnni: 1584 row and 2880 packed-tile products equal between the two
and to GGML's, random and extreme values; test_cpu_compute's 46080 exact
packed comparisons and the batch tests pass with VNNI active. In cache, the
VNNI row kernel is about 1.2 times as fast (Sapphire Rapids); in a pass,
where weights come from memory, less. The local build (rev_build.py) compiles
and runs a probe: where MSVC knows the instruction, RealMe's kernels are built
in (CL=/DREALME_AVXVNNI); where the processor also has it, GGML is built with
GGML_AVX_VNNI=ON, whose single-sentence dot product makes the same integers.
REALME_BUILD_VNNI=0 builds without. The i7-8550U and i7-10710U have no VNNI and
keep the AVX2 kernels.

Fewer codebooks (experimental, lossy; `realme_codebooks.inc`, batch and stream
only). QWEN3_TTS_CP_STEPS=K (1..15, default 15) stops the code predictor after
codebook K; the talker's next input adds each skipped codebook's mean
embedding (QWEN3_TTS_CP_FILL=mean, default; made once per table) or nothing
(=zero), and the vocoder hears codebooks 0..K. QWEN3_TTS_VOCODER_CODEBOOKS=N
(2..16) has the vocoder hear the first N only, with the transformers
unchanged: what fewer codebooks can sound like at best. The vocoder edit is
apply_codebooks_patch (the sum of the rest codebooks runs to the limit; inputs
left out are skipped). Unset, both are exactly as before. test_codebooks (a
synthetic vocoder, make_synthetic_vocoder.py, and the synthetic transformer):
unset or 15/16, codes and samples as before; codebooks past N are ignored;
K=12 runs 12 predictor graphs a frame, the first frame's codes 0..12 are the
full run's and 13..15 are 0; the mean embedding is the table's mean (also
under ASan/UBSan). run_codebook_listening.py renders the same sentences full,
vocoder-only at 13/11/9 and with 12/10/8 steps (mean and zero fill), saves the
takes side by side, and reports times, lengths and the vocoder stage's
signal-to-difference. Not for Studio's production profile until listened to.

## Rows shared out on hybrid processors; the vocoder on a GPU (2026-11-01, night)

Rows shared out (`realme_share.inc`; QWEN3_TTS_Q8_SHARE=1 on, =0 off, unset on
where CPUID 7.0 EDX bit 15 says the processor is hybrid -- Intel Core 12th
generation and Core Ultra; not the i7-8550U or i7-10710U). RealMe's 8-bit
products split rows evenly between threads, so on a processor with fast and
slow cores the fast ones wait at the end of every product. Shared out, each
thread takes the next chunk of rows (about total / (8 x threads)) as it
finishes the last; each row is still computed whole by one thread with the
same kernel, so the numbers are the same. The two chunk counters sit in the
product node's op_params after GGML's custom-op header; every graph thread
calls a custom op once per compute, the last to finish resets them, and the
barrier after the node orders that before the next compute.
test_q8_share: 835380 products equal to GGML's dot product shared out and split
evenly, 1..8 threads, two products in a row, the same graph computed three
times (counters back at zero each time); test_cpu_compute's packed and grouped
comparisons and the transformer tests pass with sharing forced on; the codes
of a 6-sentence, 12-frame run are the same with sharing on or off, 2 or 3
threads, VNNI on or off, and on the path with no kept graphs.

The vocoder on a GPU (QWEN3_TTS_VOCODER_BACKEND=gpu; `realme_decoder_backend`).
The integrated GPU first, else a discrete one, as the vocoder's own backend;
the transformers stay on the CPU (RealMe's revised engine sets
QWEN3_TTS_BACKEND=cpu). With QWEN3_TTS_DECODE_ASYNC=1 a finished sentence is
decoded on the GPU while the next frames are computed on the CPU. RealMe's
CPU vocoder kernels and the one-pass snake step aside on a non-CPU backend
(as before); ops the GPU lacks fall back to the decoder's CPU backend. Needs
GGML built with a GPU backend: the local build adds GGML_VULKAN=ON when
REALME_BUILD_VULKAN=1 and the Vulkan SDK's glslc is found (VULKAN_SDK), and
then ships ggml-vulkan.dll with the other four. Off unless both are set.
[Corrected 2026-11-02, see the review section below: as first written the
vocoder's weights stayed in host memory, so GGML's scheduler ran every op on
the CPU and nothing reached the GPU; the "53 dB" was the CPU vocoder at two
thread counts. test_vocoder_beside passed because the CPU fallback is exact.] Speed on a
real integrated GPU is for the new laptop to measure
(run_cpu_combinations.py --splits 6+0,6+gpu).

Thread splits (run_cpu_combinations.py --splits "6+0,6+2,5+1,6+6,6+gpu"):
the confirmed switches with the threads divided between transformers and
vocoder; GPU takes are compared with GPU takes only.

## The vocoder's convolutions in 8 bits (2026-11-02; experimental, lossy)

QWEN3_TTS_VOCODER_Q8=8 (or 1), =7; off unless set, read at load
(`realme_vocoder3.inc`, vocoder 3 only). Each output channel's weights per
tap are rounded to 8 bits with their own scale; each input time step to 8
bits (=8) or 7 bits (=7) with its own scale as it enters a convolution. The
integer products are exact on every processor -- AVX2 as GGML's 8-bit dot
product (sign steps), AVX-VNNI with the input offset by 128 and the offset
taken off; in mode 7 the input is stored plus 64 and AVX2 needs no sign
steps -- then one multiply per tap returns to floats; the final
one-channel convolution, snakes and biases stay 32-bit. The weights are made
at load beside the 32-bit packing (about 50 MB more; found under it,
`realme_voc3_pk`).

Synthetic vocoder (random weights, which magnify rounding through 30 layers;
the real one must be listened to): the sound is 23 dB (=8) or 19 dB (=7)
above its difference from the 32-bit vocoder; a fake-quantisation sweep
found per-32-channel scales worth 1-3 dB more and 7-bit weights 4 dB less.
Whole-vocoder time on this machine's Sapphire Rapids, 2 threads: 32-bit
3.5-3.8 s; =8 2.1-2.3 s with AVX-VNNI but 3.5 s with AVX2 alone (the sign
steps eat the gain); =7 2.1 s with AVX-VNNI, 2.7-3.0 s with AVX2 alone.
So on the NUC (AVX2) only =7 can help, at its lower quality; on a Core Ultra
=8 is the one to try. test_vocoder_q8: 2688 tile products exactly equal to a
plain loop for AVX2 and AVX-VNNI in both modes (1..6 rows, extreme values),
the decoded sound finite and 23.2 / 19.3 dB from 32 bits, the same with AVX2
and AVX-VNNI (also under ASan/UBSan); unset, vocoder 3's samples are as
before. Listening: run_codebook_listening.py --stage vocoder adds vocq8 and
vocq7.

## Review fixes (2026-11-02; an independent code review of the changes above)

GPU vocoder, made real. GGML's scheduler runs an op where its weights are,
and RealMe loads the vocoder's weights into host memory (QWEN3_TTS_BACKEND=
cpu), so with QWEN3_TTS_VOCODER_BACKEND=gpu every node still ran on the CPU
(GGML_SCHED_DEBUG=2: one CPU split). Now (apply_vocoder_gpu_patch,
`realme_vocoder_weights_to`) the weights, codebooks already normalised, are
moved into a buffer of the GPU backend once it is chosen; the decoder's CPU
fallback (for ops the GPU lacks, e.g. the transposed convolutions) gets the
vocoder's thread count instead of GGML's default 4, also when the stream
lowers or raises it. On llvmpipe: 7 Vulkan splits and 6 CPU ones; samples
25 dB above their difference from the 32-bit CPU vocoder on the synthetic
model (the GPU path is upstream's 16-bit graph, and that GPU's arithmetic);
upstream's 16-bit CPU graph is 53 dB from the 32-bit one there. So the GPU
vocoder belongs with the listening-tested options, not the exact ones.
test_vocoder_beside passes with the decode on the (software) GPU.

Sharing self-check. Rows shared out rely on GGML calling a custom op on every
graph thread once per compute. The first time sharing would be used (decided
while a graph is built, in realme_q8_eligible), two products are computed
twice on three threads; the numbers must equal GGML's dot product and the
counters be back at zero, or sharing stays off ("8-bit rows shared out
between cores: on/off").

Smaller: the VNNI build probe compiles RealMe's VNNI kernels in only when
its test program ran correctly or reported no VNNI (a crash or wrong sum now
means none); one GGML_VULKAN flag instead of OFF then ON; the stream joins
its decode worker before returning on a tokenizer failure (it could change
the decoder's threads while the worker was decoding); the predictor-mean
cache is keyed by the table's bytes too (a reloaded model at the same
address made its means again); the 8-bit vocoder packs only convolutions
with input channels in fours. All engine tests pass again (sharing forced on
and off).

## One local voice; what was removed (2026-11-02)

`realme-rev-qwen3-tts` is now RealMe's one local voice, built by Studio from
this source for each computer (`engines/rev_build.py: studio_build`;
`_Rebuild_Voice_Engine.bat`) and reading the 8-bit GGUF pair as it is (mapped,
`QWEN3_TTS_WEIGHT_MMAP=1`). Its adapter sets every runtime switch: batched
predictor attention with kept graphs, attention join and top-k selection
(`qwen3_tts_set_cpu_options` mask 4194303, enabled 4097), 12 predictor steps
with mean fill in passes and streams (a sentence on its own: 15), 16
sentences a pass with 32 GB or more.

Removed, with their patches, files and tests, because the voice no longer
uses them and none had shown a gain on the NUC: the CPU bundle (RMCPU001,
`realme_cpu_bundle.inc`, `apply_cpu_bundle_patch`, the direct-bundle model
path), the packed 8-bit layouts (RMQ8PK1/2/3, `realme_q8_pack.inc`, the
two- and four-row tile kernels and their dispatch), the projected text
vocabulary and projection cache (`realme_projection.inc`,
`apply_projection_cache_patch`), and the offline asset API
(`realme_assets.inc`: `qwen3_tts_cpu_dimensions`, `_prepare_cpu_weights`,
`_cpu_projection_layout`, `_project_cpu_tokens`, `_cpu_fingerprint`). The
runtime-option bits they used (8, 16, 128, 256, 1024, 16384, 32768, 65536,
131072, 262144, 524288, 1048576, 2097152) are still accepted and do nothing.
The 8-bit products now take unpacked rows only (`realme_q8_row`, AVX2 or
AVX-VNNI), as the voice always did.

Checked: the codes of 6 synthetic sentences over 12 frames hash the same
before and after the removal (with and without batched attention, attention
join and row sharing, and at 12 steps); every native test passes
(`test_cpu_compute` without the packed controls: 1440 exact tensor
comparisons; `test_q8_vnni` row kernels; `test_q8_share`; the transformer,
cache, graph, vocoder and mapping tests); the source is what `integrate()`
writes.

Also: a take check on every take recorded in a pass or stream
(`adapters/tts_qwen3_dll.py: take_problem`: run-on, invalid samples, too
short for its words, mostly silent, a stop of over 2 s, clipped), a take that
fails being recorded again on its own -- with all 15 steps.
