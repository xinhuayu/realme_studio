@echo off
setlocal
REM ===================================================================
REM  Package this install for SOMEBODY ELSE: the code, ffmpeg and the
REM  Piper draft voices, with your enrolled voice and your profile left
REM  out. They record their own; the instructions inside tell them how.
REM
REM  The free local voice (realme-rev-qwen3-tts: its model files and a
REM  portable engine build, about 3 GB) goes in unless you answer n below.
REM  Without it the package is far smaller, and they can still add it later
REM  with 3_Get_Voice_Engine.bat.
REM
REM  This is the one you send. 7_Make_Migration_Package.bat makes the
REM  other one, which carries your voice and is for another machine of
REM  yours.
REM
REM  `migrate --no-voice` leaves out your API key as well, and after
REM  writing the archive it reads it back and stops if any personal file
REM  is still inside. That is why this launcher has no check of its own.
REM
REM  verify_tree runs first, as it does before an update zip.
REM
REM  Exit codes are compared against 0, never with `if errorlevel 1`,
REM  which is ">= 1" and so treats a crash (a negative code) as success.
REM ===================================================================
cd /d "%~dp0.."
call "%~dp0_env.bat"
if not defined REALME_ENV (
  echo  The "realme" environment was not found. Run 1_Install.bat first.
  pause & exit /b 1
)
set "PY=%REALME_ENV%\python.exe"
set "R=%REALME_ENV%\Scripts\realme.exe"
REM  Delayed expansion is turned on only here, after every %~dp0 above, so
REM  a `!` in the folder name is not eaten. From here on paths are read as
REM  !VAR!: inside a ( ) block cmd expands %VAR% when it reads the whole
REM  block, and a folder like "Documents (work)" ends it mid-path.
setlocal EnableDelayedExpansion

echo.
echo  === verify_tree ===
"!PY!" 03_App\verify_tree.py
set "RC=!ERRORLEVEL!"
echo      exit=!RC!
if not "!RC!"=="0" (
  echo.
  echo  verify_tree FAILED. Nothing was packaged.
  pause & exit /b 1
)

echo.
echo  The free local voice clone ^(realme-rev-qwen3-tts^) is a standard part of RealMe and
echo  travels ready to run, adding about 3 GB. Without it, your colleague
echo  needs Visual Studio Build Tools to build it themselves.
echo.
set "QQ="
set /p QQ="  Include the local voice clone? (Y/n): "
set "QFLAG=--with-qwen"
if /i "!QQ!"=="n" set "QFLAG=--no-qwen"

echo.
echo  What would be packaged (your voice and profile left out):
echo.
"!R!" migrate --no-voice !QFLAG! --dry-run
set "RC=!ERRORLEVEL!"
if not "!RC!"=="0" (
  echo.
  echo  The dry run failed, exit=!RC!. Nothing was written.
  pause & exit /b 1
)
echo.
set "GO="
set /p GO="  Build the package to send now? (y/N): "
if /i not "!GO!"=="y" (
  echo  Nothing written.
  pause & exit /b 0
)
REM  Explorer's "Copy as path" wraps the path in quotes: removed with
REM  delayed expansion, after cmd has parsed the line, so a `)` or `&` in it
REM  is never seen by the parser. A trailing backslash gets a dot, because
REM  "D:\Transfer\" reaches the program as D:\Transfer" - the \" is read
REM  as an escaped quote. A folder is fine: the zip is written inside it.
set "DEST="
set /p DEST="  Where to write it (a folder; Enter for one folder above this one): "
if defined DEST set "DEST=!DEST:"=!"
if defined DEST if "!DEST:~-1!"=="\" set "DEST=!DEST!."
if defined DEST goto :write_to_dest
"!R!" migrate --no-voice !QFLAG!
goto :written
:write_to_dest
"!R!" migrate --no-voice !QFLAG! --out "!DEST!"
:written
set "RC=!ERRORLEVEL!"
echo.
if not "!RC!"=="0" (
  echo  Packaging FAILED or the archive was refused, exit=!RC!.
  echo  Do NOT send anything that was written.
  pause & exit /b 1
)
echo  Done. This archive is RealMe_Package.zip, with no voice and no key.
echo  It is the one to send. The README inside walks them through setup.
echo.
pause
