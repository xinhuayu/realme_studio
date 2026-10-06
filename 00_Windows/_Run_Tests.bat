@echo off
setlocal enabledelayedexpansion
REM ===================================================================
REM  Run every check, from anywhere.
REM
REM  Two things went wrong doing this by hand, and both are removed here:
REM
REM    1. The wrong Python. `python t_render.py` finds whichever one is
REM       first on PATH, which on a machine with several is usually not the
REM       one RealMe is installed into -- forty import failures with one
REM       cause. _env.bat points at the right interpreter.
REM
REM    2. The wrong folder. The suites live in 03_App and used to read
REM       their fixtures relative to wherever the shell was standing, so
REM       running them from the project root failed late and confusingly.
REM       They are anchored on their own location now, and this file cd's
REM       to the root anyway so the question never arises.
REM
REM  Three things about how the result is read, each of them a scar:
REM
REM    `if errorlevel 1` is NOT used. It means "errorlevel >= 1", so a
REM    process that dies with a negative code -- an access violation is
REM    -1073741819 -- passes it. A crash reported as a pass is the worst
REM    thing a test runner can do. The code is captured and compared
REM    against 0, and printed either way, so a failed check (1) and a
REM    crash (anything else) are told apart on sight.
REM
REM    Nothing is piped. `cmd | more` leaves %ERRORLEVEL% holding the exit
REM    code of `more`, not of the program, so the pipe that made the output
REM    readable also made the verdict meaningless.
REM
REM    The failure list is `name=code`, with no parentheses in it. A close
REM    paren inside a FOR block ends the block -- even inside a quoted value
REM    and even inside a REM comment, which is how the comment EXPLAINING
REM    this broke the loop it was written in. A caret in a quoted `set` is a
REM    literal caret, not an escape, so the obvious spelling produced
REM    "t_setup^(1^)".
REM
REM    Output stays live. A suite can run for a minute and a runner that
REM    buffers it looks hung, which is the complaint this project already
REM    had about the render log. A suite that FAILS is re-run into
REM    tests.log, so there is one file to paste without losing the live view.
REM ===================================================================
cd /d "%~dp0.."
call "%~dp0_env.bat"
if not defined REALME_ENV (
  echo  The "realme" environment was not found. Run 1_Install.bat first.
  pause & exit /b 1
)
set "PY=%REALME_ENV%\python.exe"
set "LOG=%CD%\tests.log"
echo RealMe test run > "%LOG%"
echo python: %PY% >> "%LOG%"
echo cwd   : %CD% >> "%LOG%"

echo.
echo  Using %PY%
echo  Failures are written to %LOG%
echo.
echo  === verify_tree ===
"%PY%" 03_App\verify_tree.py
set "RC=%ERRORLEVEL%"
echo      exit=!RC!
if not "!RC!"=="0" (
  echo === verify_tree === >> "%LOG%"
  "%PY%" 03_App\verify_tree.py >> "%LOG%" 2>&1
  echo.
  echo  verify_tree FAILED ^(exit !RC!^). Fix that before running anything else.
  echo  Details in %LOG%
  pause & exit /b 1
)

set "FAILED="
for %%T in (t_tau t_pace t_migrate t_setup t_locate t_encoding t_voices t_lexicon t_render t_engines t_gemini) do (
  echo.
  echo  === %%T ===
  "%PY%" 03_App\%%T.py
  set "RC=!ERRORLEVEL!"
  echo      exit=!RC!
  if not "!RC!"=="0" (
    set "FAILED=!FAILED! %%T=!RC!"
    echo. >> "%LOG%"
    echo === %%T === exit=!RC! >> "%LOG%"
    "%PY%" 03_App\%%T.py >> "%LOG%" 2>&1
  )
)

echo.
if defined FAILED (
  echo  FAILED:!FAILED!
  echo  The number after each name is its exit code: 1 means a check
  echo  failed, any other code means it crashed.
  echo  Full output of the failures: %LOG%
  pause & exit /b 1
)
echo  All checks and all suites passed.
echo  Safe to build the update zip with _Make_Update_Zip.bat
echo.
pause
