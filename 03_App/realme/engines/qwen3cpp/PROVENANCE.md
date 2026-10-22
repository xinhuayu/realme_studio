# Vendored: qwen3-tts.cpp

This is a copy of a third-party project, kept here **because it might not be
there tomorrow.**

RealMe's usual rule is the opposite: carry the parts that are ours, fetch the
parts that are not. That rule is about *binaries* — ffmpeg, PyTorch, model
weights. Things that are large, and that come from organisations with mirrors
and release archives and a lot to lose by disappearing.

This is neither. It is about a megabyte of C++ from a single author's
repository, last touched in July 2026. A one-person repo can be renamed,
made private, or deleted on a bad afternoon, and nothing about MIT obliges
anyone to keep hosting it. Fetching it at install time would mean a build step
that works today and fails silently in a year.

So the source lives here. It is compiled on the machine that needs it — no
binaries are committed, and the licence below is preserved unmodified as MIT
requires.

## What this is

| | |
|---|---|
| Upstream | https://github.com/predict-woo/qwen3-tts.cpp |
| Commit | `b3ba14077cf1b3e11b86e5f84aa9184605c89b28` |
| Commit date | 2026-07-18 |
| Vendored | 2026-08-25 |
| Licence | MIT — see `LICENSE`, retained verbatim |

Removed from the copy: `.git`, `build/`, `.venv`, `models/`, and the sample
`.wav` and `.png` files. The `ggml` submodule is vendored too — see below. Nothing was edited. Any
RealMe-specific change belongs in `realme/engines/cpp.py`, not in here, so that
a future `git diff` against upstream stays meaningful.

## GGML

| | |
|---|---|
| Upstream | https://github.com/ggml-org/ggml |
| Commit | `3af5f5760e19a96427f5f7a93b79cbdf3d4b265b` |
| Commit date | 2026-06-12 |
| Vendored | 2026-08-25 |
| Licence | MIT — see `ggml/LICENSE`, retained verbatim |
| Size | ~25 MB of source |

The tensor library the C++ engine runs on, in `ggml/`. Removed from the copy:
`.git` and `build/`. Nothing else was touched, so `git diff` against upstream
stays meaningful.

Vendoring this was a judgement call and it went the other way at first. The
argument against: GGML is maintained by an organisation, forked thousands of
times, and twenty-five times the size of the thing it protects — a pinned commit
hash is enough to recover the exact tree from any of those forks.

The argument that won: 25 MB is nothing on a local disk, and it removes the last
thing a build needs from the network. A pinned hash is only a recovery
*procedure*; it still assumes someone is there to run it, with git installed and
GitHub reachable. A directory does not assume anything. The whole point of this
folder is that a build works in five years without asking the internet's
permission, and half-vendoring would have left that promise conditional.

`realme/engines/cpp.py` still carries the URL and the pin, and still knows how to
fetch — but only as a fallback for a tree where this directory has been pruned.
Both routes land on the same commit, so they build identical code.

Most of those 25 MB are backends this project never compiles: CUDA, Vulkan,
SYCL, OpenCL, HIP, WebGPU, Metal, CANN. They are kept anyway. Pruning them would
mean editing upstream's CMake, which turns a clean vendored copy into a fork
somebody has to maintain — for a saving that does not matter.

## What this project is, in one line

A C++17/GGML reimplementation of Qwen3-TTS inference: same weights, converted
to GGUF, run without Python or PyTorch.

**Read `OPTIMIZATION.md` before believing the speed claims.** The README's
"4.07x faster" is measured on Apple Silicon with CoreML and Metal
(`docs/benchmark_pytorch_vs_cpp.json` records `"hardware": "Darwin / arm"`).
The project's own x86 CPU number, on a Ryzen 5 3600 with four threads, is a
real-time factor of **1.94** — still faster than PyTorch on that machine, but
not by four times. On a CPU-only Windows laptop, expect the smaller number.
