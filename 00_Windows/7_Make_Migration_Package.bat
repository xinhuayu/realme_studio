@echo off
setlocal
REM ===================================================================
REM  Package this working install for ANOTHER MACHINE OF YOURS: the
REM  code, the tools already built, the model weights, and your
REM  enrolled voice.
REM
REM  For somebody else, use _Make_Package_For_Others.bat instead. That
REM  one leaves your voice out, and it is the difference between a
REM  colleague installing RealMe and a colleague holding a model of
REM  your voice.
REM
REM  Left out on purpose: tools\qwen3 (the PyTorch engine, ~4 GB) and
REM  tools\vc (voice conversion, ~1.3 GB). Both were measured and set
REM  aside; `realme engine install` brings them back if ever wanted.
REM
REM  Your .env is NOT included. Set the key on the new machine.
REM
REM  verify_tree runs first, as it does before an update zip. Packaging
REM  a tree that does not import produces an archive that installs and
REM  then fails on the new machine, where there is nobody to debug it.
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
echo  What would be packaged:
echo.
"!R!" migrate --dry-run
set "RC=!ERRORLEVEL!"
if not "!RC!"=="0" (
  echo.
  echo  The dry run failed, exit=!RC!. Nothing was written.
  pause & exit /b 1
)
echo.
set "GO="
set /p GO="  Build the package now? (y/N): "
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
"!R!" migrate
goto :written
:write_to_dest
"!R!" migrate --out "!DEST!"
:written
set "RC=!ERRORLEVEL!"
echo.
if not "!RC!"=="0" (
  echo  Packaging FAILED, exit=!RC!. Do not send anything.
  pause & exit /b 1
)
echo  Done. This archive is RealMe_Migration.zip and it CONTAINS YOUR VOICE.
echo  It is for another machine of yours, not for another person.
echo.
pause
