# realme-qwen3-tts.cpp 1.3.0

- **Universal, with processor-specific additions.** The vocoder kernel
  (`QWEN3_TTS_VOCODER_F32=3`) now has a plain C++ tile and snake beside
  its AVX2 ones, and a NEON tile for 64-bit ARM; on x86 the AVX2 versions
  are chosen at run time. Where only
  the plain tile would run, the copy-free stage (`2`) is used, as GGML's
  products were faster than it; `QWEN3_TTS_V3_GENERIC=1` forces the plain
  tile, to compare. The 8-bit pass products have a NEON version too (the
  dot-product instruction where the build has it, else GGML's emulation of
  it), so on ARM a pass still equals one sentence at a time -- on processors
  with the 8-bit matrix instructions GGML alone takes two rows at a time
  and differs slightly. Checked on 64-bit ARM (Linux, ARMv8.0 and ARMv8.2)
  under an emulator only (not timed). x86 samples unchanged.
- **On a Mac**, `build.sh` builds GGML without its Metal and Accelerate
  backends, which the engine would otherwise prefer to the CPU, switching
  RealMe's changes off (or run with `QWEN3_TTS_BACKEND=cpu`). Not measured
  on a Mac.
- **One thread pool for the engine's life** instead of one per graph
  (identical samples; `QWEN3_TTS_POOL=0` for upstream's).
- **The engine's compiler flags follow the processor target**: upstream's
  `-march=native` is kept under `REALME_NATIVE` (on by default;
  `-mcpu=native` on 64-bit ARM), and `build.sh` switches it off for its
  portable x86 build, so that build runs on any AVX2 processor.
- A real 51.8-minute lecture video (578 sentences, 8 per pass, everything
  on) recorded in 35-37 minutes (0.68-0.72 of its length, two runs) and
  finished in 36.5-38 on the 6-core test PC; 44 and 46 before the 8-bit
  pass products, the vocoder kernel and refilled passes.

# realme-qwen3-tts.cpp 1.2.0

- **The vocoder's decoder stage without copies** (`QWEN3_TTS_VOCODER_F32=2`):
  each convolution reads its inputs in place, through overlapping rows of a
  channel-by-channel layout, instead of through a sevenfold unrolled copy;
  bias, snake, padding and residual adds are fused. With the 8-bit model on
  the 15-watt, 6-core test PC: 1.31 seconds of work per second of speech one
  sentence at a time (from 1.49), 0.79 at 4 per pass (from 1.07), 0.74 at 8,
  0.72 at 16 -- about five times as fast as upstream. A third of the
  vocoder's working memory, with the same samples as `1` (about 120 dB
  apart). It needs less memory than the original vocoder at any length, so
  it has no length limit. The voice server uses it by default.
- The transposed convolutions' overlap-add reads its input in order (the
  same sums, faster).
- **8-bit products for a pass, several sentences at once**: each weight row
  is loaded once for the whole pass, with GGML's exact arithmetic for each
  sentence, so the samples do not change. On a 2-core test machine a pass
  of 8 took 16% less time, a pass of 4 about 6%. `QWEN3_TTS_Q8_BATCH=0`
  turns it off.
- **RealMe's own vocoder kernel** (`QWEN3_TTS_VOCODER_F32=3`): the
  decoder stage's signal stays channel by channel throughout, each
  convolution reads its taps in place against weights packed at load, and
  bias, residual add and snake are applied as rows leave the kernel; one
  operation per residual unit. On the 6-core test PC the vocoder 0.26 ->
  0.19 seconds of work per second of speech (27% less), with the same
  samples (about 110 dB apart; not told apart in listening). With the 8-bit
  pass products: 1.25 one sentence at a time, 0.68 at 4 per pass, 0.58 at
  8 -- about six and a half times as fast as upstream. Needed AVX2 in
  this version; `2` where it could not run. The voice server uses it when
  its engine has it.
- **Passes whose places are refilled**
  (`qwen3_tts_synthesize_stream_with_embedding`): each finished sentence is
  handed back at once and the next takes its place. Identical to one at a
  time with sampling off; 9% less time than fixed passes of 4 on a run of
  mixed sentences (2-core test machine).
- `QWEN3_TTS_SGEMM_ALL=1` (opt-in): GGML's tinyBLAS for every product it
  takes; a few per cent faster in passes, with slightly different samples.
- A real 51-minute lecture video (578 sentences, 8 per pass, before the
  8-bit pass products) was recorded in 44 minutes on the 6-core test PC.

# realme-qwen3-tts.cpp 1.1.0

- **A voice server** (`server/`): the engine behind OpenAI's speech API on
  your own computer, so any tool that can use OpenAI's text-to-speech speaks
  in a voice cloned here. Python 3.9+ only, no packages; voices are WAV files
  in `voices/` (`realme.wav` is the default voice, "realme"); long text and
  simultaneous requests are recorded several sentences to a pass. The port
  and other settings are in `voice_server.json`, set to suit the tool. It
  answers to one model, `realme-qwen3-tts`; other names only when listed as
  `model_aliases`. See the README, *Voice server*.
- Measured on a real lecture: 52 minutes of speech (564 sentences) recorded
  in 56 minutes on a 15-watt, 6-core laptop processor.

# realme-qwen3-tts.cpp 1.0.0

The first standalone release of the local voice engine used by RealMe
Studio: qwen3-tts.cpp (predict-woo, upstream `b3ba14077cf1`) on GGML
`3af5f5760e19`, with RealMe's changes.

**3.5 to 3.7 times as fast on an ordinary CPU**: on a small 6-core Windows PC (a 15-watt laptop processor),
cloning a real voice, 3.79 seconds of work per second of speech before, 1.07
with four sentences per pass and 1.02 with eight -- about real time.

- Thread count applied (upstream ran on 4 threads whatever was asked).
- `ffn_down` converted to F32 once at load instead of in every step.
- No per-channel copies in the vocoder.
- Several sentences per pass: `qwen3_tts_synthesize_batch_with_embedding`.
- An 8-bit model written by the engine itself: `qwen3_tts_realme_quantize_q8`, `tools/quantize_q8.py`.
- One-pass snake activation with an AVX2 sine (`QWEN3_TTS_FAST_SNAKE`).
- 32-bit vocoder through GGML's tinyBLAS, gated to the vocoder alone (`QWEN3_TTS_VOCODER_F32`).
- Unicode model paths on Windows.

The first three, and the snake's `exact` mode, give sample-identical output to
upstream; batches give output identical to one sentence at a time. See
CHANGES.md for every change, its switch and how it was checked. GPU builds
are untested; the CPU-only changes switch themselves off there.
