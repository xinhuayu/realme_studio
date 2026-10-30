# Notices

realme-qwen3-tts.cpp is a revised copy of two MIT-licensed projects. Their
licences are kept with them, unmodified.

| Component | Where | Licence | Copyright |
|---|---|---|---|
| qwen3-tts.cpp, upstream commit `b3ba14077cf1b3e11b86e5f84aa9184605c89b28` (2026-07-18) | everything outside `ggml/`, except the RealMe files below | MIT (`LICENSE`) | predict-woo |
| GGML, commit `3af5f5760e19a96427f5f7a93b79cbdf3d4b265b` (2026-06-12) | `ggml/` | MIT (`ggml/LICENSE`) | the ggml authors |
| RealMe's changes | `src/realme_*.inc`, the marked edits listed in `CHANGES.md` (each begins with a `RealMe` comment), the gate in `ggml/src/ggml-cpu/ggml-cpu.c`, `tools/`, `server/`, `voice_server.json`, `voices/README.md`, `build.*` and these notes | MIT (`LICENSE`) | Xinhua Yu |

The Qwen3-TTS model weights are not part of this repository. They are
published by the Qwen team at Alibaba under their own licence; read it before
using or redistributing them (https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base).

The vocoder architecture follows WavTokenizer
(https://github.com/jishengpeng/WavTokenizer), as upstream notes.
