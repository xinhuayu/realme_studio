@echo off
setlocal
REM ===================================================================
REM  Does the free local clone (qwen3cpp) record faster with several
REM  sentences in one pass, on this computer?
REM
REM  Records a few sentences in your voice one at a time, then 2, 4 and 8
REM  per pass (as memory allows), after checking that a pass gives the same sound sample
REM  for sample. Saves the best size; lectures then use it. Needs a voice
REM  engine built with batches: run _Rebuild_Voice_Engine.bat first.
REM  Takes about 15-25 minutes. Leave the computer alone while it runs.
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
