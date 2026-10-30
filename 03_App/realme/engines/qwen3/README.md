# Vendored Qwen3-TTS worker

`worker.py` and `prosody.py` come from the `real_voice_qwen3` project, moved
here so RealMe is self-contained and that project can be retired. They are
unmodified: the worker owns model loading, the cached clone prompt, the
per-profile synthesis cache, the bounded codec-token budget, warm-up and
cancellation, and there is no reason to rewrite any of it.

**What is here and what is not, on purpose.** Code lives in the repository.
The two heavy things do not:

| | Size | Where it comes from |
|---|---|---|
| worker code | ~42 KB | here, in the repo |
| Python runtime | **~500 MB** | installed into `tools/qwen3/venv` |
| model weights | ~1.9 GB | copied from an existing install, or downloaded |

### Why the runtime is ~500 MB and not ~2.5 GB

Two things, both measured rather than assumed:

**CPU-only PyTorch.** Every CUDA dependency torch declares is marked
`platform_system == "Linux"`, so the Windows PyPI wheel is already CPU-only.
The installer asks for the CPU channel explicitly so Linux behaves the same
way — there is no reason to fetch ~2 GB of cuDNN and cuBLAS to run inference on
a laptop with no GPU.

**No gradio.** `pip install qwen-tts` pulls in gradio, a full web UI framework,
with FastAPI, pandas and Pillow behind it. Inspecting the package shows gradio
is referenced in exactly one file — `demo.py` — which the model classes never
import. So the installer uses `--no-deps` and names the real dependencies
itself. If `import qwen_tts` then fails, it installs the full set and says so;
the saving is never taken on trust.

That split is deliberate and is the same reason ffmpeg and Python are not
vendored either: a repository should carry the parts that are ours and give
instructions for the parts that are not. `realme engine install` does the
fetching, so a fresh clone is still one command away from working.

## Licence

Qwen3-TTS weights are Apache 2.0. Review the licence of whichever model
directory you point at before redistributing anything built with it.
