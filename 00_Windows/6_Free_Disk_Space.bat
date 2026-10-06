@echo off
REM ===================================================================
REM  Reclaims the disk space held by an older engine folder, once
REM  RealMe carries its own copy.
REM
REM  This runs in two passes and the first one deletes nothing. It
REM  checks that RealMe's engine actually loads and that the copied
REM  weights are the full size, lists what has become a second copy,
REM  and stops. Only if you then answer YES does anything go - and
REM  anything RealMe did not adopt (your voice database, profiles,
REM  settings) is archived beside the folder before that happens.
REM ===================================================================
setlocal
cd /d "%~dp0.."
call "%~dp0_env.bat"
if not defined REALME_ENV (
  echo  Run 1_Install.bat first.
  pause & exit /b 1
)
set "R=%REALME_ENV%\Scripts\realme.exe"

REM  An old engine folder can be offered as the default by setting
REM  REALME_LEGACY_QWEN3 in the environment before running this.

echo.
if defined REALME_LEGACY_QWEN3 (
  echo  Old engine folder:
  echo      %REALME_LEGACY_QWEN3%
  echo.
  set /p OLD="  Press Enter to use it, or paste another: "
) else (
  set /p OLD="  Old engine folder: "
)
REM  Explorer's "Copy as path" wraps the path in quotes. Strip them with the
REM  for/%%~A idiom rather than substring replacement: `set "V=%V:"=%"` embeds a
REM  bare quote in the pattern, which cmd's own parser mis-tokenises.
if defined OLD for /f "tokens=* delims=" %%A in ("%OLD%") do set "OLD=%%~A"
if "%OLD%"=="" set "OLD=%REALME_LEGACY_QWEN3%"
if "%OLD%"=="" exit /b 0
if not exist "%OLD%" (
  echo  That folder does not exist: %OLD%
  pause & exit /b 1
)

echo.
"%R%" engine retire "%OLD%"
if errorlevel 1 (
  echo.
  echo  Nothing was changed.
  pause & exit /b 1
)

echo.
set /p OK="  Delete the second copies listed above? (type YES): "
if /i not "%OK%"=="YES" (
  echo  Cancelled. Nothing was changed.
  pause & exit /b 0
)
echo.
"%R%" engine retire "%OLD%" --delete
echo.
pause
