@echo off
setlocal
REM ===================================================================
REM  How much text can Gemini TTS be given in ONE call?
REM
REM  Not what the API permits (8,192 input tokens and about eight minutes
REM  of audio out), but the size at which the audio stops matching the
REM  text, e.g. a sentence silently dropped from the middle of a slide.
REM
REM  The probe synthesizes each sentence on its own, adds up the
REM  durations, then synthesizes the same sentences together in one call
REM  and compares. Short means text was dropped. Long means it stalled or
REM  repeated. Every wav is kept so the two sizes that matter can be
REM  listened to: duration flags a problem, listening confirms it.
REM
REM  It calls a paid API. It prints the estimated cost first -- about
REM  twenty cents at current prices -- and the actual cost at the end.
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
echo  Measuring on your own narration if a project is on this machine,
echo  otherwise on a built-in sample. Pass --text yourfile.txt to choose.
echo.
"!PY!" 03_App\probe_gemini_tts.py %*
set "RC=!ERRORLEVEL!"
echo.
echo      exit=!RC!
if not "!RC!"=="0" (
  echo  The probe did not finish. Nothing was changed.
  pause & exit /b 1
)
echo.
echo  Nothing has changed yet: the renderer still uses the pipeline
echo  default until you set the measured size, either with
echo    realme voice gemini --max-chars NNN
echo  or in Twin Setup under "Gemini cloned voice".
echo.
pause
