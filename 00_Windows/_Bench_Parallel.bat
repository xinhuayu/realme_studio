@echo off
setlocal
REM ===================================================================
REM  Would two or three local voice engines at once record a lecture
REM  faster than one?
REM
REM  For the free local clone (qwen3cpp) only. Gemini is one call at a
REM  time whatever this says.
REM
REM  Starts one engine as the Studio does today, then two and three side
REM  by side, each with its own copy of the model (about 3-4 GB each),
REM  and times the same sentences through every layout, twice. Loading
REM  is reported separately and never counted as recording time.
REM
REM  It changes nothing: rendering stays as it is. At the end it says
REM  whether parallel recording would be worth building for THIS
REM  computer. Takes roughly 15-30 minutes; leave the computer alone
REM  while it runs, because anything else using it skews the times.
REM
REM  Extra options pass through, for example:
REM      _Bench_Parallel.bat --layouts 1x8,2x8,4x4 --repeat 3
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
"!PY!" -m realme bench-parallel %*
set "RC=!ERRORLEVEL!"
echo.
echo      exit=!RC!
if not "!RC!"=="0" (
  echo  The measurement did not finish. Nothing was changed.
  pause & exit /b 1
)
echo.
echo  Send the "results:" file named above to the RealMe developer, or
echo  read the last paragraph: it says whether parallel recording helps.
echo.
pause
