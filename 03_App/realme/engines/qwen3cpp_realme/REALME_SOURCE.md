# RealMe's qwen3-tts.cpp source

This folder is the voice engine's source as RealMe builds it: the original
qwen3-tts.cpp (predict-woo, MIT; `LICENSE` kept) with every RealMe change
already made. Nothing is patched while building.

- `src/realme_*.inc` -- RealMe's own code (passes, the 8-bit copier and
  kernel, the vocoders, the snake). Edited here.
- The upstream files (`src/*.cpp`, `src/*.h`, `CMakeLists.txt`) carry
  RealMe's changes made by `engines/cpp.py`. A change to one of them is made
  there, as a patch, and this folder written again
  (`realme engine cpp --cpp-action integrate`), so the same change can be
  made again on a new upstream version.
- `ggml_overlay/src/ggml-cpu/ggml-cpu.c` -- GGML's file with the tinyBLAS
  gate. A build copies it over GGML's own only when GGML's own is the file
  it was made from (`REALME_INTEGRATED.json` keeps both fingerprints).
- `REALME_CHANGES.md` -- every change, its switch, and how it was checked.

The original source, unchanged, is in `engines/qwen3cpp` with its GGML: the
record of what upstream wrote. It is never built unless asked for by name
(`realme engine cpp --cpp-action build --source original`), into a folder
of its own beside the working engine; RealMe never uses that build.
