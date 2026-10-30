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
