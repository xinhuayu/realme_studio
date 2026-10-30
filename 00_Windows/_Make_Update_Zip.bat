@echo off
setlocal
REM ===================================================================
REM  Build RealMe_Update.zip, with the checks that must come first.
REM
REM  The sequence is three commands whose order matters; this launcher
REM  runs them in that order.
REM
REM  make_update_zip.py parses every shipped .py and refuses on a syntax
REM  error, but checks nothing more. verify_tree imports the modules,
REM  checks the invariants and the launcher cross-references, and -- once
REM  the zip exists -- that the archive carries a BUILD_ID and no tools\.
REM  So it runs on both sides of the build.
REM
REM  Exit codes are compared against 0, never with `if errorlevel 1`, which
REM  is ">= 1" and so treats a crash (a negative code) as success.
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
echo  === verify_tree (before) ===
"!PY!" 03_App\verify_tree.py
set "RC=!ERRORLEVEL!"
echo      exit=!RC!
if not "!RC!"=="0" (
  echo.
  echo  verify_tree FAILED. Nothing was packaged.
  pause & exit /b 1
)

echo.
echo  Who is this update for?
echo     1  your own computers - everything
echo     2  colleagues - without your developer launchers
set "WHO="
set /p WHO="  Choice [1]: "
set "FOR="
if "!WHO!"=="2" set "FOR=--for-others"

echo.
echo  === building ===
"!PY!" 03_App\make_update_zip.py !FOR!
set "RC=!ERRORLEVEL!"
echo      exit=!RC!
if not "!RC!"=="0" (
  echo.
  echo  Packaging refused. Nothing was written.
  pause & exit /b 1
)

echo.
echo  === verify_tree (after, now checking the archive) ===
"!PY!" 03_App\verify_tree.py
set "RC=!ERRORLEVEL!"
echo      exit=!RC!
if not "!RC!"=="0" (
  echo.
  echo  The archive did not pass. Do not send it.
  pause & exit /b 1
)

echo.
echo  RealMe_Update.zip is built and checked.
echo  Your own tree is already current; only 03_App\realme\BUILD_ID changed.
echo  It is in the RealMe folder. Whoever receives it puts it in their own
echo  RealMe folder and double-clicks 0_Update.bat.
echo.
pause
