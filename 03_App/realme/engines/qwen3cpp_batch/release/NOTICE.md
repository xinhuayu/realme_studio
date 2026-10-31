# Notices

realme-qwen3-tts.cpp is a revised copy of two MIT-licensed projects.
Upstream's licence is kept in `LICENSE` with its text and copyright line
unchanged and one line added for RealMe's changes; GGML's is kept unmodified
in `ggml/LICENSE`.

| Component | Where | Licence | Copyright |
|---|---|---|---|
| qwen3-tts.cpp, upstream commit `b3ba14077cf1b3e11b86e5f84aa9184605c89b28` (2026-07-18) | everything outside `ggml/`, except the RealMe files below | MIT (`LICENSE`) | predict-woo |
| GGML, commit `3af5f5760e19a96427f5f7a93b79cbdf3d4b265b` (2026-06-12) | `ggml/` | MIT (`ggml/LICENSE`) | the ggml authors |
| RealMe's changes | `src/realme_*.inc`, the edits to upstream's files listed in `CHANGES.md` (marked by a `RealMe` comment or a `realme_` name; a diff against upstream commit `b3ba140` shows every one), the gate in `ggml/src/ggml-cpu/ggml-cpu.c`, `tools/`, `server/`, `voice_server.json`, `voices/README.md`, `build.*` and these notes | MIT (`LICENSE`) | Xinhua Yu |

In a voice server folder (realme-qwen3-tts-server), `bin/` holds the two
compiled: qwen3-tts.cpp with RealMe's changes (`LICENSE`) and GGML
(`LICENSE_GGML`); their source is the realme-qwen3-tts.cpp release.

The Qwen3-TTS model weights are not part of this repository. They are
published by the Qwen team at Alibaba under their own licence; read it before
using or redistributing them (https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base).

The vocoder architecture follows WavTokenizer
(https://github.com/jishengpeng/WavTokenizer), as upstream notes.
