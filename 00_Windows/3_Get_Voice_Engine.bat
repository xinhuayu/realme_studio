@echo off
REM ===================================================================
REM  OPTIONAL. Installs the local Qwen3 voice engine into  tools\qwen3\
REM
REM  RealMe speaks in your cloned voice through Gemini by default, which
REM  needs only the key from 2_Set_Key.bat. This engine is for people who
REM  want the cloned voice made on their own computer instead: no
REM  per-minute cost and nothing sent out, at the price of ~3 GB of disk
REM  and slower rendering on a laptop without a graphics card.
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
REM  ended the block in the middle of the path. Delayed expansion happens
REM  after parsing. Turned on only now, after every %~dp0, so a `!` in the
REM  folder name is not eaten.
setlocal EnableDelayedExpansion

echo.
"!R!" engine status
echo.
echo  This engine is OPTIONAL. Gemini is the default voice and needs only
echo  the key from 2_Set_Key.bat. The local engine takes about 3 GB and lets
echo  your cloned voice be made on this computer, with no per-minute cost.
echo  Already-installed parts are skipped, so saying yes again is harmless.
echo.
echo  Building it can need developer tools. If that is not for you, ask
echo  Xinhua Yu, the RealMe developer, for a RealMe package that already
echo  includes it - nothing to build.
echo.
set "GO="
set /p GO="  Install the local engine now? (y/N): "
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
REM  never seen by the parser. The old for /f "%%~A" idiom put the quoted
REM  value inside a second pair of quotes, which UN-quoted the path and let
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
  echo   Or ask Xinhua Yu, the RealMe developer, for a package with it.
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
