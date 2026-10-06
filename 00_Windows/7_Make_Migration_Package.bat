@echo off
REM ===================================================================
REM  Packages this working install for another computer: the code, the
REM  tools already built, the model weights, and your enrolled voice.
REM
REM  Left out on purpose: tools\qwen3 (the PyTorch engine, ~4 GB) and
REM  tools\vc (voice conversion, ~1.3 GB). Both were measured and set
REM  aside; `realme engine install` brings them back if ever wanted.
REM
REM  Your .env is NOT included. Set the key on the new machine.
REM ===================================================================
setlocal
cd /d "%~dp0.."
call "%~dp0_env.bat"
if not defined REALME_ENV (
  echo  Run 1_Install.bat first.
  pause & exit /b 1
)
set "R=%REALME_ENV%\Scripts\realme.exe"

echo.
echo  What would be packaged:
echo.
"%R%" migrate --dry-run
echo.
set /p GO="  Build the package now? (y/N): "
if /i not "%GO%"=="y" (
  echo  Nothing written.
  pause & exit /b 0
)
set /p DEST="  Where to write it (Enter for one folder above this one): "
if "%DEST%"=="" (
  "%R%" migrate
) else (
  for /f "tokens=* delims=" %%A in ("%DEST%") do set "DEST=%%~A"
  "%R%" migrate --out "%DEST%"
)
echo.
pause
