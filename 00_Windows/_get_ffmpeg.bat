@echo off
REM ===================================================================
REM  Downloads a self-contained ffmpeg into  tools\ffmpeg\
REM
REM  A HELPER, not a step. 1_Install.bat calls it. It is the ONLY place that
REM  decides whether ffmpeg needs fetching - callers must not add their own
REM  "is it already there" check, because two guards testing different files is
REM  how you get a half-installed toolchain reported as complete.
REM
REM  Pass any argument to suppress the pause when calling it from a script.
REM
REM  Why not conda: installing ffmpeg from conda-forge into an Anaconda
REM  environment mixes channels, and on Windows that reliably produces DLL
REM  errors with names like "gdk_pixbuf-2.0-0.dll not found" - nothing to do
REM  with video, everything to do with conflicting dependency trees.
REM
REM  A static build has no such dependencies. It is one folder, it is not on
REM  the system PATH, and it is found by RealMe automatically.
REM ===================================================================
setlocal
cd /d "%~dp0.."
set "DEST=%CD%\tools\ffmpeg"

if exist "%DEST%\ffmpeg.exe" if exist "%DEST%\ffprobe.exe" (
  echo  ffmpeg is already here: %DEST%
  "%DEST%\ffmpeg.exe" -version 2>nul | findstr /B "ffmpeg version"
  if "%~1"=="" pause
  exit /b 0
)

echo.
echo  Downloading a self-contained ffmpeg (about 30 MB)...
echo  Destination: %DEST%
echo.

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop';" ^
  "$tmp = Join-Path $env:TEMP 'realme-ffmpeg.zip';" ^
  "$urls = @('https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip'," ^
  "         'https://github.com/BtbN/FFmpeg-Builds/releases/latest/download/ffmpeg-master-latest-win64-gpl.zip');" ^
  "$ok=$false;" ^
  "foreach ($u in $urls) { try { Write-Host ('  trying ' + $u); Invoke-WebRequest -Uri $u -OutFile $tmp -UseBasicParsing; $ok=$true; break } catch { Write-Host ('  failed: ' + $_.Exception.Message) } };" ^
  "if (-not $ok) { throw 'Could not download ffmpeg from either source.' };" ^
  "$ex = Join-Path $env:TEMP 'realme-ffmpeg-x';" ^
  "if (Test-Path $ex) { Remove-Item $ex -Recurse -Force };" ^
  "Expand-Archive -LiteralPath $tmp -DestinationPath $ex -Force;" ^
  "$dest = '%DEST%';" ^
  "New-Item -ItemType Directory -Force -Path $dest | Out-Null;" ^
  "Get-ChildItem -Path $ex -Recurse -Include ffmpeg.exe,ffprobe.exe,ffplay.exe | ForEach-Object { Copy-Item $_.FullName -Destination $dest -Force };" ^
  "Remove-Item $tmp,$ex -Recurse -Force -ErrorAction SilentlyContinue"

if not exist "%DEST%\ffmpeg.exe" (
  echo.
  echo  [X] Download or extraction failed.
  echo.
  echo      Manual alternative, takes two minutes:
  echo        1. Open https://www.gyan.dev/ffmpeg/builds/
  echo        2. Download "ffmpeg-release-essentials.zip"
  echo        3. Extract it, find ffmpeg.exe and ffprobe.exe in its bin folder
  echo        4. Copy BOTH into:
  echo             %DEST%
  echo.
  if "%~1"=="" pause
  exit /b 1
)

echo.
echo  Installed:
"%DEST%\ffmpeg.exe" -version 2>nul | findstr /B "ffmpeg version"
if exist "%DEST%\ffprobe.exe" (echo    ffprobe.exe present) else (echo    [!] ffprobe.exe MISSING - RealMe needs it)
echo.
echo  RealMe finds this automatically. It is not added to your system PATH.
echo.
if "%~1"=="" pause
