#!/usr/bin/env sh
# Build GGML (with the gated tinyBLAS), then the engine. Pass "native" to
# build for this machine's processor instead of any AVX2 processor.
set -e
cd "$(dirname "$0")"
if [ "$1" = "native" ] || [ "$(uname -m)" = "arm64" ] || [ "$(uname -m)" = "aarch64" ]; then
  CPU="-DGGML_NATIVE=ON"
else
  CPU="-DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON"
fi
GEN=""
command -v ninja >/dev/null 2>&1 && GEN="-G Ninja"
cmake -S ggml -B ggml/build $GEN -DCMAKE_BUILD_TYPE=Release $CPU \
      -DGGML_LLAMAFILE=ON -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF
cmake --build ggml/build --config Release -j
cmake -S . -B build $GEN -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
echo "built: build/qwen3-tts-cli and the shared library in build/"
