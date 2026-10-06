@echo off
setlocal enabledelayedexpansion
REM ===================================================================
REM  Build the public source tree, for GitHub.
REM
REM  Not the update zip and not the colleague package. Those two carry a
REM  working installation -- a compiled engine, 2.5 GB of weights, ffmpeg,
REM  the piper voices. A repository carries none of it: whoever clones it
REM  runs two commands and their own machine fetches or builds its own,
REM  which is smaller, more honest, and the only way a compiled engine is
REM  meaningful (a build is tied to the instruction set it was made for).
REM
REM  Before it copies anything it reads every file it is about to publish
REM  and refuses on a key, a token, a home directory or an email address.
REM  A key in a public repository is not undone by deleting the commit.
REM
REM  Exit codes are compared against 0, never with `if errorlevel 1`,
REM  which means ">= 1" and so reads a crash as success.
REM ===================================================================
cd /d "%~dp0.."
call "%~dp0_env.bat"
if not defined REALME_ENV (
  echo  The "realme" environment was not found. Run 1_Install.bat first.
  pause & exit /b 1
)
set "PY=%REALME_ENV%\python.exe"

echo.
echo  === verify_tree ===
"%PY%" 03_App\verify_tree.py
set "RC=%ERRORLEVEL%"
echo      exit=!RC!
if not "!RC!"=="0" (
  echo.
  echo  verify_tree FAILED. Nothing was written.
  pause & exit /b 1
)

echo.
echo  === building the source tree ===
"%PY%" 03_App\make_github_package.py %*
set "RC=%ERRORLEVEL%"
echo      exit=!RC!
if not "!RC!"=="0" (
  echo.
  echo  Refused. Read the lines above: each one is something that would
  echo  have been published. Nothing was written.
  pause & exit /b 1
)

echo.
echo  The source tree is ready. It has no key, no voice, no model and no
echo  binary in it; INSTALL.md tells a reader how to fetch each of those.
echo.
pause
