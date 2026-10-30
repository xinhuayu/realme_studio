@echo off
setlocal
REM ===================================================================
REM  How many Gemini requests at once does your Google project take?
REM
REM  A lecture in your Gemini voice records faster with several sentences
REM  in flight, because each request is mostly waiting for Google. How
REM  many is set by Google, per project and tier, and not published for
REM  the voice models, so this measures it: a few short sentences at 1, 2,
REM  3 and 4 at a time, timed, watching for Google asking to slow down.
REM
REM  The answer is saved as an upper limit. Lectures still record 2 at a
REM  time; `realme gemini-parallel --use N` raises that, up to the limit.
REM  It costs a few cents and about 22 requests of today's allowance; both
REM  are printed first and it asks before sending anything.
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
"!PY!" -m realme gemini-parallel %*
set "RC=!ERRORLEVEL!"
echo.
echo      exit=!RC!
if not "!RC!"=="0" (
  echo  The measurement did not finish. Nothing was changed.
  pause & exit /b 1
)
echo.
pause
