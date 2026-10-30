@echo off
REM Build GGML (with the gated tinyBLAS), then the engine, for any AVX2
REM processor. Run from an "x64 Native Tools Command Prompt for VS 2022"
REM (Visual Studio Build Tools, "Desktop development with C++"), which puts
REM the compiler, CMake and Ninja on the PATH. "build.bat native" builds for
REM this machine's processor only.
setlocal
cd /d "%~dp0"
set "CPU=-DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON"
if /i "%~1"=="native" set "CPU=-DGGML_NATIVE=ON"
where cl >nul 2>nul || (echo Run this from an x64 Native Tools Command Prompt for VS. & exit /b 1)
cmake -S ggml -B ggml\build -G Ninja -DCMAKE_BUILD_TYPE=Release %CPU% -DGGML_LLAMAFILE=ON -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF || exit /b 1
cmake --build ggml\build || exit /b 1
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=ON || exit /b 1
cmake --build build || exit /b 1
REM Windows finds DLLs beside the program, not where they were built.
for /r ggml\build %%f in (*.dll) do copy /y "%%f" build\ >nul
echo built: build\qwen3-tts-cli.exe and build\qwen3tts.dll
