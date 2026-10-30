@echo off
REM ===================================================================
REM  Builds the free local voice clone (qwen3cpp): your cloned voice made
REM  on this computer, nothing sent out, nothing to pay -- about real time
REM  on a 6-core PC. It is one of RealMe's two cloned voices (Gemini is
REM  the other, and the default); RealMe packages usually carry it ready to
REM  run, and this builds it where one did not.
REM
REM  What it does: downloads the Qwen3-TTS weights (~2.5 GB), converts them
REM  to GGUF, builds RealMe's revised qwen3-tts.cpp from the source inside
REM  RealMe (every speed change applied), and makes the 8-bit model the
REM  lectures use. Needs Visual Studio 2022 Build Tools with "Desktop
REM  development with C++", and about 6 GB of disk.
REM
REM  Two heavy pieces, deliberately not stored in the repository:
REM      Python runtime       ~500 MB  (CPU PyTorch, no gradio)
REM      model weights        ~2.5 GB
REM
REM  If you already have a real_voice_qwen3 folder, this COPIES the weights
REM  from it - local disk, no download. The runtime is copied only if that
REM  folder's own runtime is already lean; a full one (gradio, pandas, CUDA)
REM  is skipped and a lean runtime built instead, which is both smaller and
REM  faster than copying several spare gigabytes.
REM
REM  Safe to re-run: anything already installed is skipped, not redone.
REM ===================================================================
setlocal
cd /d "%~dp0.."
call "%~dp0_env.bat"
if not defined REALME_ENV (
  echo  Run 1_Install.bat first.
  pause & exit /b 1
)
set "R=%REALME_ENV%\Scripts\realme.exe"
REM  From here on, paths are read as !VAR! inside ( ) blocks. cmd expands
REM  %VAR% when it reads the whole block, so a folder like "Documents (work)"
REM  ends the block in the middle of the path. Delayed expansion happens
REM  after parsing. Turned on only now, after every %~dp0, so a `!` in the
REM  folder name is not eaten.
setlocal EnableDelayedExpansion

echo.
"!R!" engine status
echo.
echo  The free local voice clone: your cloned voice made on this computer,
echo  nothing sent out, nothing to pay. It takes about 6 GB of disk and
echo  needs Visual Studio 2022 Build Tools ^("Desktop development with C++"^).
echo  It downloads the voice model, builds RealMe's revised engine and makes
echo  the 8-bit model the lectures use. Already-installed parts are skipped,
echo  so saying yes again is harmless.
echo.
set "GO="
set /p GO="  Build the local voice clone now? (y/N): "
if /i not "!GO!"=="y" goto :skipped

REM  An old engine folder can be offered as the default by setting
REM  REALME_LEGACY_QWEN3 in the environment before running this.

set "SRC="
if defined REALME_LEGACY_QWEN3 (
  echo  Found your existing engine folder:
  echo      !REALME_LEGACY_QWEN3!
  echo.
  echo  Press Enter to copy the weights from there ^(fast, no download^),
  echo  or paste a different folder, or type  none  to download instead.
  echo.
  set /p SRC="  Folder [Enter = the one above]: "
) else (
  echo  If you have an existing real_voice_qwen3 folder, paste its path to copy
  echo  from it instead of downloading ^(much faster^). Press Enter to download.
  echo.
  set /p SRC="  Existing folder (optional): "
)

REM  Explorer's "Copy as path" wraps the path in quotes. Removed with delayed
REM  expansion, after cmd has parsed the line, so a `)` or `&` in the path is
REM  never seen by the parser. A for /f "%%~A" idiom puts the quoted value
REM  inside a second pair of quotes, which UN-quotes the path and lets
REM  "C:\x (work)\real_voice_qwen3" end the for-list at its `)`.
if defined SRC set "SRC=!SRC:"=!"

REM  Parentheses matter: `if cond a & b` runs b unconditionally in cmd.
if /i "!SRC!"=="none" (
  set "SRC="
  set "REALME_LEGACY_QWEN3="
)
if "!SRC!"=="" set "SRC=!REALME_LEGACY_QWEN3!"
REM  "C:\x\" would reach realme as C:\x" - the \" reads as an escaped quote.
if defined SRC if "!SRC:~-1!"=="\" set "SRC=!SRC!."

if not "!SRC!"=="" if not exist "!SRC!" (
  echo.
  echo  That folder does not exist:
  echo      !SRC!
  echo  Nothing was changed.
  pause & exit /b 1
)

if "!SRC!"=="" (
  "!R!" engine install
) else (
  echo.
  echo  Copying from: !SRC!
  "!R!" engine install --from "!SRC!"
)
set "RC=!ERRORLEVEL!"
echo.
"!R!" engine status
echo.
if not "!RC!"=="0" (
  echo  ============================================================
  echo   Something did not install. The three parts are independent,
  echo   so whatever DID work is kept - re-running only does the rest.
  echo.
  echo   The full log, including the actual error, is in your data
  echo   folder as engine-install.log. To see it:
  echo.
  echo     5_Command_Prompt.bat   then   realme key --where
  echo.
  echo   That prints the data folder path; the log is beside the .env.
  echo  ============================================================
  echo.
  echo   A RealMe package built with the local clone ^(the usual kind^)
  echo   carries it ready to run, with nothing to build.
)
if "!RC!"=="0" echo  To use it: Studio - Twin Setup - Voice for lectures - qwen3cpp.
echo.
pause
exit /b 0

:skipped
echo.
echo  Nothing installed. Your voice settings are unchanged.
echo  Run this again any time if you change your mind.
echo.
pause
