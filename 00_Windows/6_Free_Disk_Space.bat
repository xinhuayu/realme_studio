@echo off
REM ===================================================================
REM  Frees disk space the voice engine no longer needs. Close the Studio
REM  first. Two steps; neither deletes anything until you say so.
REM
REM  1. Older builds of the local voice (realme-rev-qwen3-tts) except the
REM     one in use, the CPU bundles the old separate engine package
REM     prepared, that package's own DLLs, packed-weight caches and scratch
REM     copies. Lists them with sizes; type YES to delete. Never your model
REM     files, voice, lectures or settings.
REM  2. Optional: an engine folder from before RealMe carried its own (an
REM     old real_voice_qwen3). Press Enter to skip. It checks that RealMe's
REM     engine loads and the copied weights are complete, lists what has
REM     become a second copy, and deletes only after you answer YES -
REM     archiving anything RealMe did not adopt beside the folder first.
REM ===================================================================
setlocal
cd /d "%~dp0.."
call "%~dp0_env.bat"
if not defined REALME_ENV (
  echo  Run 1_Install.bat first.
  pause & exit /b 1
)
set "R=%REALME_ENV%\Scripts\realme.exe"
REM  From here on, paths are read as !VAR! inside ( ) blocks. cmd expands
REM  %VAR% when it reads the whole block, so a folder like "Documents (work)"
REM  ends the block in the middle of the path. Delayed expansion happens
REM  after parsing. Turned on only now, after every %~dp0, so a `!` in the
REM  folder name is not eaten.
setlocal EnableDelayedExpansion

echo.
echo  === 1. old engine builds and data ===
"%REALME_ENV%\python.exe" "03_App\cleanup_engine_space.py"
echo.
echo  === 2. an engine folder from before RealMe ^(optional^) ===

REM  An old engine folder can be offered as the default by setting
REM  REALME_LEGACY_QWEN3 in the environment before running this.

echo.
set "OLD="
if defined REALME_LEGACY_QWEN3 (
  echo  Old engine folder:
  echo      !REALME_LEGACY_QWEN3!
  echo.
  set /p OLD="  Press Enter to use it, or paste another folder: "
) else (
  set /p OLD="  Old engine folder (press Enter to skip): "
)
REM  Explorer's "Copy as path" wraps the path in quotes. Removed with delayed
REM  expansion, after cmd has parsed the line, so a `)` or `&` in the path is
REM  never seen by the parser. A for /f "%%~A" idiom puts the quoted value
REM  inside a second pair of quotes, which UN-quotes the path and lets
REM  "C:\x (work)\real_voice_qwen3" end the for-list at its `)`.
if defined OLD set "OLD=!OLD:"=!"
if "!OLD!"=="" set "OLD=!REALME_LEGACY_QWEN3!"
REM  "C:\x\" would reach realme as C:\x" - the \" reads as an escaped quote.
if defined OLD if "!OLD:~-1!"=="\" set "OLD=!OLD!."
if "!OLD!"=="" (
  echo  Skipped.
  pause & exit /b 0
)
if not exist "!OLD!" (
  echo  That folder does not exist: !OLD!
  pause & exit /b 1
)

echo.
"!R!" engine retire "!OLD!"
if errorlevel 1 (
  echo.
  echo  Nothing was changed.
  pause & exit /b 1
)

echo.
set "OK="
set /p OK="  Delete the second copies listed above? (type YES): "
if /i not "!OK!"=="YES" (
  echo  Cancelled. Nothing was changed.
  pause & exit /b 0
)
echo.
"!R!" engine retire "!OLD!" --delete
echo.
pause
