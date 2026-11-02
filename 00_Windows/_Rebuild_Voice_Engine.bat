@echo off
setlocal
REM ===================================================================
REM  Build the voice engine. Two engines live in tools\:
REM
REM    realme-rev-qwen3-tts   RealMe's own local voice (the default here).
REM                           Built from this Studio's engine source
REM                           (03_App\realme\engines\qwen3cpp_realme) for
REM                           this processor, reading the 8-bit model files
REM                           in tools\qwen3cpp\models_q8 as they are.
REM                           Rebuilt only when the source or this computer
REM                           changed; "force" rebuilds anyway.
REM    qwen3cpp               the free local clone, the same source built the
REM                           older way with upstream's settings. Kept as
REM                           the reference and the fallback, and it makes
REM                           the 8-bit model files: "legacy" (or "all")
REM                           rebuilds it.
REM
REM    _Rebuild_Voice_Engine.bat           RealMe's own voice
REM    _Rebuild_Voice_Engine.bat force     the same, compiled again anyway
REM    _Rebuild_Voice_Engine.bat legacy    the free local clone
REM    _Rebuild_Voice_Engine.bat all       the free local clone, from scratch
REM
REM  Needs the C++ compiler (Visual Studio Build Tools, "Desktop development
REM  with C++"). Takes 5-10 minutes. The working engine stays selected if
REM  anything fails. Recordings already made are kept.
REM
REM  Exit codes are compared against 0, never with `if errorlevel 1`,
REM  which is ">= 1" and so reads a crash as success.
REM ===================================================================
cd /d "%~dp0.."
call "%~dp0_env.bat"
if not defined REALME_ENV (
  echo  The "realme" environment was not found. Run 1_Install.bat first.
  pause & exit /b 1
)
set "PY=%REALME_ENV%\python.exe"
REM  Delayed expansion only from here, after every %~dp0 above, so a `!` in
REM  the folder name is not eaten; paths are read as !PY! from here on.
setlocal EnableDelayedExpansion

if /i "%~1"=="legacy" goto :legacy
if /i "%~1"=="all" goto :legacy
goto :revised

:legacy

echo.
echo  Close the Studio first if it is open: it holds the engine's files.
echo.
set "MODE=rebuild"
if /i "%~1"=="all" set "MODE=rebuild-all"
if "!MODE!"=="rebuild" (
  set "ALL="
  set /p ALL="  Rebuild everything from scratch, including the checks' copy? (y/N): "
  if /i "!ALL!"=="y" set "MODE=rebuild-all"
)
"!PY!" -m realme engine cpp --cpp-action !MODE!
set "RC=!ERRORLEVEL!"
echo.
echo      exit=!RC!
if not "!RC!"=="0" (
  echo  The rebuild did not finish. The working engine was kept.
  pause & exit /b 1
)
echo.
echo  Done. Start the Studio again. Lectures use the 8-bit model ^(unless a
echo  message above says it was skipped^) and the faster vocoder; Twin Setup
echo  changes either.
echo.
pause
exit /b !RC!

:revised
echo.
echo  Close the Studio first if it is open: it holds the engine's files.
echo.
set "FORCE="
if /i "%~1"=="force" set "FORCE=--force"
if not exist "tools\qwen3cpp\models_q8\qwen3-tts-0.6b-q8_0.gguf" (
  echo  The 8-bit model files are not there yet. Building the free local clone
  echo  first, which makes them.
  "!PY!" -m realme engine cpp --cpp-action rebuild
  if not "!ERRORLEVEL!"=="0" (
    echo  The free local clone did not build; see above.
    pause & exit /b 1
  )
)
"!PY!" -m realme engine rev !FORCE!
set "RC=!ERRORLEVEL!"
echo.
echo      exit=!RC!
if not "!RC!"=="0" (
  echo  The build did not finish. The voice engine built before, if any, is still selected.
  echo  The build log is in tools\realme-rev-qwen3-tts\local\build_*\build.log
  pause & exit /b 1
)
echo.
echo  Done. Start the Studio again; choose Your voice ^(realme-rev-qwen3-tts^)
echo  in Twin Setup if it is not chosen yet.
echo.
pause
exit /b 0
