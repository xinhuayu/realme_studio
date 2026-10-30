@echo off
REM  Extracts RealMe_Update.zip over this folder, replacing old files.
REM  Safe to run repeatedly. Your .env and any renders are untouched.
REM
REM  This script RELAUNCHES ITSELF FROM %TEMP% before it extracts anything.
REM
REM  cmd does not load a .bat into memory: it reads the next line from the
REM  file, by byte offset, each time it needs one. The update zip carries
REM  00_Windows\0_Update.bat, so extracting it replaces the running file and
REM  every later line would be read from the middle of the NEW file. So the
REM  copy in this folder only copies itself out and hands over; the copy in
REM  %TEMP% does the work, and it can be overwritten here freely.
REM
REM  No %VAR% or %~dp0 inside ( ) blocks anywhere below: cmd expands those
REM  when it reads the whole block, and a folder like "Documents (work)" ends
REM  the block in the middle of the path. Branches use goto instead.
setlocal
if /i "%~1"=="--relaunched" goto :relaunched

REM  ---- first pass: running from 00_Windows ------------------------------
copy /y "%~f0" "%TEMP%\RealMe_0_Update.bat" >nul 2>nul
if errorlevel 1 goto :nocopy
REM  Invoked WITHOUT `call`, so control never comes back to this file - which
REM  is about to be replaced - and cmd never reads another line of it.
"%TEMP%\RealMe_0_Update.bat" --relaunched "%~dp0.."
exit /b

:nocopy
echo  Could not copy the updater to:
echo      "%TEMP%"
echo  It must run from a copy, because the update replaces this very file.
echo  Nothing was changed.
pause
exit /b 1
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
goto :extracted
REM  The colon line above is a landing pad, reached only by the PREVIOUS
REM  0_Update.bat (no relaunch) while it applies the update that brings
REM  this file. Its extract overwrites it with this file, and cmd resumes
REM  at its old byte offset, 1760 (LF copy) or 1794 (CRLF copy). Any tail
REM  of a colon line is a label, so it lands harmlessly and jumps to the
REM  post-extract steps. Keep the pad spanning those offsets.
REM
REM  ---- second pass: running from %TEMP% ---------------------------------
REM  %~dp0 is %TEMP% here, so the project root comes in as %2. Nothing below
REM  may use %~dp0.
:relaunched
REM  Tested with a conditional OR, not `if errorlevel`: cd leaves
REM  ERRORLEVEL unchanged when it succeeds.
cd /d "%~2" || goto :noroot
if exist "RealMe_Update.zip" goto :have_zip
echo  RealMe_Update.zip is not in this folder.
echo  Expected: "%CD%\RealMe_Update.zip"
pause
exit /b 1

:noroot
echo  Could not enter the RealMe folder:
echo      "%~2"
pause
exit /b 1

:have_zip
REM  Refuse to extract an update that would write into tools\.
REM
REM  tools\ holds ~3 GB that was downloaded or copied once: ffmpeg, the voice
REM  engine's virtual environment, and the model weights. Expand-Archive only
REM  overwrites paths present in the archive, so this check stops any update
REM  zip that contains tools/ entries before anything is extracted.
echo  Checking the update ...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop';" ^
  "Add-Type -AssemblyName System.IO.Compression.FileSystem;" ^
  "$z = [System.IO.Compression.ZipFile]::OpenRead((Resolve-Path -LiteralPath 'RealMe_Update.zip').ProviderPath);" ^
  "$bad = $z.Entries | Where-Object { $_.FullName -match '(^|/)tools/' };" ^
  "$z.Dispose();" ^
  "if ($bad) { Write-Host ''; Write-Host '  [X] This update contains tools/ entries:';" ^
  "  $bad | Select-Object -First 5 | ForEach-Object { Write-Host ('      ' + $_.FullName) };" ^
  "  Write-Host '  Refusing to extract - it would overwrite your installed engine.';" ^
  "  exit 1 }"
if errorlevel 1 ( echo. & echo  Nothing was changed. & pause & exit /b 1 )

echo  Updating RealMe in "%CD%" ...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Expand-Archive -LiteralPath 'RealMe_Update.zip' -DestinationPath '.' -Force"
if errorlevel 1 ( echo  Extract failed. & pause & exit /b 1 )
:extracted
REM  Superseded launchers are removed by `realme setup` at the end of this
REM  script, NOT here.
REM
REM  Two reasons. First, the .bat running during an update is the OLD copy,
REM  already on disk - it cannot know about a rename that arrives in the
REM  update it is applying. Code shipped inside the zip can. Second, the list
REM  belongs in one place, and a cmd `for (...)` list cannot span lines.
echo.
echo  Updated. 'pip install -e' points at this folder, so new code is live
echo  without reinstalling.
echo.
REM  The freshly extracted _env.bat, from the project folder, not from %TEMP%.
call "%CD%\00_Windows\_env.bat"
if not defined REALME_ENV goto :noenv
"%REALME_ENV%\Scripts\realme.exe" setup
pause
exit /b 0

:noenv
echo   Environment not found yet - run 1_Install.bat
pause
exit /b 0
