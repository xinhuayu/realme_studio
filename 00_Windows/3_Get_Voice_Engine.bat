@echo off
REM ===================================================================
REM  Installs the local Qwen3 voice engine into  tools\qwen3\
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

echo.
"%R%" engine status
echo.

REM  An old engine folder can be offered as the default by setting
REM  REALME_LEGACY_QWEN3 in the environment before running this.

if defined REALME_LEGACY_QWEN3 (
  echo  Found your existing engine folder:
  echo      %REALME_LEGACY_QWEN3%
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

REM  Explorer's "Copy as path" wraps the path in quotes. Strip them with the
REM  for/%%~A idiom rather than substring replacement: `set "V=%V:"=%"` embeds a
REM  bare quote in the pattern, which cmd's own parser mis-tokenises.
if defined SRC for /f "tokens=* delims=" %%A in ("%SRC%") do set "SRC=%%~A"

REM  Parentheses matter: `if cond a & b` runs b unconditionally in cmd.
if /i "%SRC%"=="none" (
  set "SRC="
  set "REALME_LEGACY_QWEN3="
)
if "%SRC%"=="" set "SRC=%REALME_LEGACY_QWEN3%"

if not "%SRC%"=="" if not exist "%SRC%" (
  echo.
  echo  That folder does not exist:
  echo      %SRC%
  echo  Nothing was changed.
  pause & exit /b 1
)

if "%SRC%"=="" (
  "%R%" engine install
) else (
  echo.
  echo  Copying from: %SRC%
  "%R%" engine install --from "%SRC%"
)
set "RC=%ERRORLEVEL%"
echo.
"%R%" engine status
echo.
if not "%RC%"=="0" (
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
)
echo.
pause
