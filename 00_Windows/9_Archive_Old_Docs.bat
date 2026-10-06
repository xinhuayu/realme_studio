@echo off
REM  Move the development history out of the way, once.
REM
REM  The superseded architecture drafts, the design reviews and the original
REM  handoff package are worth keeping and are not worth sending to anyone.
REM  This moves them into `_Archive`, which is not part of a release: the
REM  packager walks 00_Windows, 02_Research and 03_App, so anything in
REM  `_Archive` is kept on this machine and travels nowhere.
REM
REM  Nothing is deleted. Run it once; running it again does nothing.

setlocal
set "ROOT=%~dp0.."
pushd "%ROOT%" || exit /b 1

if not exist "_Archive" mkdir "_Archive"

echo.
echo  Moving development history into _Archive
echo.

call :move_dir "00_Review"
call :move_dir "01_Architecture"
call :move_dir "04_Original_Package_v1"

echo.
echo  Done. The project folder now holds only what a user needs:
echo.
dir /b /ad
echo.
echo  _Archive is not included in an update or a migration package.
echo.
popd
endlocal
pause
exit /b 0

:move_dir
if not exist "%~1" (
  echo    %~1 - already moved, or never existed
  exit /b 0
)
if exist "_Archive\%~1" (
  echo    %~1 - _Archive already has one; leaving both alone
  exit /b 0
)
move "%~1" "_Archive\" >nul
if errorlevel 1 (
  echo    %~1 - COULD NOT MOVE. Close anything that has it open and re-run.
) else (
  echo    %~1 - moved
)
exit /b 0
