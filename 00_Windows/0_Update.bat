@echo off
REM  Extracts RealMe_Update.zip over this folder, replacing old files.
REM  Safe to run repeatedly. Your .env and any renders are untouched.
setlocal
cd /d "%~dp0.."
if not exist "RealMe_Update.zip" (
  echo  RealMe_Update.zip is not in this folder.
  echo  Expected: %CD%\RealMe_Update.zip
  pause & exit /b 1
)
REM  Refuse to extract an update that would write into tools\.
REM
REM  tools\ holds ~3 GB that was downloaded or copied once: ffmpeg, the voice
REM  engine's virtual environment, and the model weights. Expand-Archive only
REM  overwrites paths present in the archive, so today this cannot happen -- but
REM  "cannot happen because nobody put it in the zip" is a convention, and this
REM  turns it into a check. An update that tried would be stopped here, not
REM  discovered afterwards by a voice engine that no longer loads.
echo  Checking the update ...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop';" ^
  "Add-Type -AssemblyName System.IO.Compression.FileSystem;" ^
  "$z = [System.IO.Compression.ZipFile]::OpenRead((Resolve-Path 'RealMe_Update.zip'));" ^
  "$bad = $z.Entries | Where-Object { $_.FullName -match '(^|/)tools/' };" ^
  "$z.Dispose();" ^
  "if ($bad) { Write-Host ''; Write-Host '  [X] This update contains tools/ entries:';" ^
  "  $bad | Select-Object -First 5 | ForEach-Object { Write-Host ('      ' + $_.FullName) };" ^
  "  Write-Host '  Refusing to extract - it would overwrite your installed engine.';" ^
  "  exit 1 }"
if errorlevel 1 ( echo. & echo  Nothing was changed. & pause & exit /b 1 )

echo  Updating RealMe in %CD% ...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Expand-Archive -LiteralPath 'RealMe_Update.zip' -DestinationPath '.' -Force"
if errorlevel 1 ( echo  Extract failed. & pause & exit /b 1 )
REM  Superseded launchers are removed by `realme setup` at the end of this
REM  script, NOT here.
REM
REM  Two reasons. First, the .bat running during an update is the OLD copy,
REM  already on disk - it cannot know about a rename that arrives in the very
REM  update it is applying. Code shipped inside the zip can. Second, the list
REM  belongs in one place, and a cmd `for (...)` list cannot span lines: the
REM  version that used to live here was silently broken for exactly that
REM  reason, which is how the folder collected three numbering schemes.
echo.
echo  Updated. 'pip install -e' points at this folder, so new code is live
echo  without reinstalling.
echo.
call "%~dp0_env.bat"
if defined REALME_ENV (
  "%REALME_ENV%\Scripts\realme.exe" setup
) else (
  echo   Environment not found yet - run 1_Install.bat
)
pause
