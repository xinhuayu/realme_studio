#!/usr/bin/env sh
# Build GGML (with the gated tinyBLAS), then the engine. Pass "native" to
# build for this machine's processor instead of any AVX2 processor; on 64-bit
# ARM it always builds for this machine.
set -e
cd "$(dirname "$0")"
if [ "$1" = "native" ] || [ "$(uname -m)" = "arm64" ] || [ "$(uname -m)" = "aarch64" ]; then
  CPU="-DGGML_NATIVE=ON"
  ENG="-DREALME_NATIVE=ON"
else
  CPU="-DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON"
  ENG="-DREALME_NATIVE=OFF"   # RealMe's AVX2 kernels are chosen at run time
fi
# A Mac: GGML would build Metal and Accelerate and the engine would prefer
# them to the CPU, where RealMe's changes run.
[ "$(uname -s)" = "Darwin" ] && CPU="$CPU -DGGML_METAL=OFF -DGGML_BLAS=OFF"
GEN=""
command -v ninja >/dev/null 2>&1 && GEN="-G Ninja"
cmake -S ggml -B ggml/build $GEN -DCMAKE_BUILD_TYPE=Release $CPU \
      -DGGML_LLAMAFILE=ON -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF
cmake --build ggml/build --config Release -j
cmake -S . -B build $GEN -DCMAKE_BUILD_TYPE=Release $ENG
cmake --build build --config Release -j
echo "built: build/qwen3-tts-cli and the shared library in build/"
