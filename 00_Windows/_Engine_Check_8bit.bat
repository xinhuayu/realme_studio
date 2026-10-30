@echo off
setlocal
REM ===================================================================
REM  Is the 8-bit copy of the free local clone (qwen3cpp) faster on this
REM  computer, and does it sound as good?
REM
REM  Uses the 8-bit copy that _Rebuild_Voice_Engine.bat made (tools\
REM  qwen3cpp\models_q8), making it only if it is missing; the 16-bit model
REM  is not touched. Records the same sentences with the 16-bit model and the
REM  8-bit one (the same speed changes in both; the 32-bit vocoder off in
REM  both, so only the model differs) and leaves the audio
REM  side by side to listen to, then measures several sentences per pass
REM  with the 8-bit model. Needs an engine rebuilt with
REM  _Rebuild_Voice_Engine.bat. Takes about 30 minutes.
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
"!PY!" -m realme engine-timing --compare --variants fast,q8 %*
set "RC=!ERRORLEVEL!"
if "!RC!"=="0" (
  echo.
  echo  Now several sentences per pass with the 8-bit model ...
  "!PY!" -m realme bench-batch --quant q8_0
  set "RC=!ERRORLEVEL!"
)
echo.
echo      exit=!RC!
if not "!RC!"=="0" (
  echo  The measurement did not finish. The 8-bit model, if it was made,
  echo  is kept; lectures use it unless Twin Setup says 16-bit.
  pause & exit /b 1
)
echo.
echo  Send the "results:" file named above to the RealMe developer.
echo.
pause
