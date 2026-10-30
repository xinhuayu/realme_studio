@echo off
setlocal
REM ===================================================================
REM  Rebuild the free local clone (qwen3cpp) so it uses its threads.
REM
REM  Engines built before October 2026 run every calculation on 4 threads,
REM  whatever the setting says: the engine stored the number and never used
REM  it. Measured on a 6-core computer, using one thread per core records
REM  about a quarter faster. This builds the engine again with that fixed,
REM  for the same processors and GPU setting as before, then makes the
REM  8-bit copy of the model beside the 16-bit one (a minute or two; about
REM  1 GB) if it is not there yet. Lectures then use the 8-bit model and
REM  the faster vocoder (Twin Setup changes either).
REM
REM  Needs the C++ compiler (Visual Studio Build Tools, "Desktop development
REM  with C++"). Takes 5-10 minutes. The working engine is kept aside while
REM  it builds and put back if anything fails, so the voice keeps working
REM  either way. Recordings already made are kept; a lecture recorded with
REM  the 16-bit model is recorded again in the 8-bit one the next time it
REM  renders (choose 16-bit in Twin Setup to reuse its takes instead).
REM
REM  Rebuild everything ("y" at the question, or _Rebuild_Voice_Engine.bat
REM  all): also stages GGML's source again from the copy inside RealMe and
REM  builds the stopwatch copy the _Engine_Check_* files use, so nothing is
REM  left from an older RealMe. About twice as long.
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
