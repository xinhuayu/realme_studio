@echo off
setlocal
REM ===================================================================
REM  Does the free local clone (qwen3cpp) record faster with several
REM  sentences in one pass, on this computer?
REM
REM  Records 16 sentences in your voice one at a time, then 2, 4, 8, 12 and
REM  16 per pass (as memory allows), after checking that a pass gives the
REM  same sound sample for sample. Saves the best size; lectures then use
REM  it. Needs a voice engine built with batches: run
REM  _Rebuild_Voice_Engine.bat first. Takes about 20-40 minutes. Leave the
REM  computer alone while it runs.
REM
REM  Options pass through, for example:
REM      _Bench_Batch.bat --sizes 1,2,4 --sentences 8
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
"!PY!" -m realme bench-batch %*
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
