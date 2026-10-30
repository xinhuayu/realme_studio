@echo off
setlocal
REM ===================================================================
REM  Are the 32-bit vocoders faster on this computer, and can you hear them?
REM
REM  The vocoder turns the voice model's codes into sound; on the CPU it
REM  is a large share of the recording time. RealMe can run its
REM  convolutions in 32-bit floats through GGML's fast matrix multiply
REM  (QWEN3_TTS_VOCODER_F32). That is the same calculation rounded
REM  differently, so the sound differs slightly from the default: listen.
REM  The copy-free form (QWEN3_TTS_VOCODER_F32=2, RealMe's default on an
REM  engine built with it) gives the same samples as the first in less
REM  time and memory.
REM
REM  Run _Rebuild_Voice_Engine.bat first (the fast multiply comes with
REM  the rebuild). Then this records the same sentences with the default
REM  vocoder ('fast'), in 32 bits ('voc') and copy-free ('voc2'), using the stopwatch
REM  copy of the engine in tools\qwen3cpp_timing (built first if needed;
REM  the engine the Studio uses is not touched), and leaves them side by
REM  side to listen to. Takes about 30 minutes. Leave the computer alone
REM  while it runs.
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
"!PY!" -m realme engine-timing --compare --variants fast,voc,voc2 %*
set "RC=!ERRORLEVEL!"
echo.
echo      exit=!RC!
if not "!RC!"=="0" (
  echo  The measurement did not finish. Nothing was changed.
  pause & exit /b 1
)
echo.
echo  Listen to identity_fast.wav, identity_voc.wav and identity_voc2.wav in the results
echo  folder, then send the "results:" file named above to the RealMe
echo  developer.
echo.
pause
