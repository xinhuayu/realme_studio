@echo off
setlocal
REM ===================================================================
REM  Do RealMe's speed changes to the free local clone (qwen3cpp) help
REM  on this computer, and is the sound exactly the same?
REM
REM  Uses the stopwatch copy of the engine in tools\qwen3cpp_timing
REM  (built, or rebuilt, first if needed; the engine the Studio uses is
REM  not touched). Records the same few sentences four ways at this
REM  computer's thread count: the changes off, on, and with each of the
REM  two one-pass versions of the vocoder's snake step. Compares one
REM  sentence sample for sample, or says how far apart it is. Takes about
REM  20-30 minutes. Needs the C++ compiler that built the engine. Leave the
REM  computer alone while it runs.
REM
REM  If it says the changes help and the sound is identical, run
REM  _Rebuild_Voice_Engine.bat to give the voice engine the changes.
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
"!PY!" -m realme engine-timing --compare %*
set "RC=!ERRORLEVEL!"
echo.
echo      exit=!RC!
if not "!RC!"=="0" (
  echo  The measurement did not finish. Nothing was changed.
  pause & exit /b 1
)
echo.
echo  Send the "results:" file named above to the RealMe developer.
echo.
pause
