@echo off
REM  RealMe - one-time install. Double-click this.
REM
REM  Safe to run again. Nothing that is already installed is fetched twice:
REM  an existing "realme" environment is kept (only the RealMe package in it
REM  is refreshed with pip install -e), and ffmpeg and the draft voices are
REM  skipped by their own helpers when present.
REM
REM  No %VAR% or %~dp0 inside ( ) blocks: cmd expands those when it reads the
REM  whole block, and a folder like "Documents (work)" ends the block in the
REM  middle of the path. Branches that print a path use goto instead.
setlocal
cd /d "%~dp0.."
call "%~dp0_find_conda.bat"

echo.
echo  RealMe installer
echo  ================
echo.

if defined CONDA_BAT goto :have_conda
echo  [X] Could not find conda.
echo.
echo      Looked on PATH and in the usual install folders. If Anaconda is
echo      somewhere else, tell this script where, then run it again:
echo.
echo         set REALME_CONDA=C:\path\to\anaconda3
echo         "%~dp01_Install.bat"
echo.
pause
exit /b 1

:have_conda
echo  Using conda: "%CONDA_BAT%"
echo.

REM  Check BEFORE creating. Asked to create an environment that already
REM  exists, conda with -y deletes it and builds it again - Python and every
REM  package downloaded a second time, on every re-run of this file. The
REM  standing rule is to look for an installed component before fetching
REM  it, so an environment that is listed AND whose Python starts is kept.
REM  One that is listed but broken is rebuilt, which is the only case where
REM  recreating it is the fix.
call "%CONDA_BAT%" env list | findstr /R /C:"\\realme$" /C:"\\realme " >nul
if errorlevel 1 goto :env_create
call "%CONDA_BAT%" run -n realme python -c "import sys" >nul 2>nul
if errorlevel 1 goto :env_broken
echo  [1/4] The "realme" environment already exists - kept, not recreated.
goto :env_ok

:env_broken
echo  [1/4] The "realme" environment exists but its Python does not start.
echo        Recreating it...
goto :env_make

:env_create
echo  [1/4] Creating the "realme" environment ^(python 3.11^)...
:env_make
REM  Silenced, but then VERIFIED, so a genuine failure cannot slip through
REM  as a confusing error two steps later.
call "%CONDA_BAT%" create -y -n realme python=3.11 >nul 2>nul
call "%CONDA_BAT%" env list | findstr /R /C:"\\realme$" /C:"\\realme " >nul
if not errorlevel 1 goto :env_ok
echo  [!] The "realme" environment could not be created.
echo      Showing the real error:
echo.
call "%CONDA_BAT%" create -y -n realme python=3.11
echo.
echo  ------------------------------------------------------------------
echo  If that failed with a missing DLL, conda's environment machinery is
echo  the problem, not RealMe. You can install into your base Anaconda
echo  instead - it works identically:
echo.
echo      "%CONDA_ROOT%\python.exe" -m pip install -e "%CD%\03_App"
echo.
echo  Then run 4_Start_Studio.bat as normal.
echo  ------------------------------------------------------------------
echo.
pause
exit /b 1

:env_ok

echo  [2/4] Getting ffmpeg...
REM  Deliberately NOT `conda install -c conda-forge ffmpeg`. Mixing conda-forge
REM  into a defaults-based Anaconda environment is a well-known source of
REM  Windows DLL errors ("gdk_pixbuf-2.0-0.dll not found" and similar). A
REM  static build has no dependency tree at all.
REM  No "already present" check here on purpose. _get_ffmpeg.bat owns that
REM  decision (it checks ffmpeg.exe AND ffprobe.exe) and reports it, so the
REM  test lives in one place.
call "%~dp0_get_ffmpeg.bat" quiet
REM  Verify, don't assume. Both binaries: RealMe needs ffprobe for every
REM  duration measurement, and a folder with only ffmpeg.exe is not working.
set "FFOK=1"
if not exist "tools\ffmpeg\ffmpeg.exe"  set "FFOK="
if not exist "tools\ffmpeg\ffprobe.exe" set "FFOK="
if not defined FFOK (
  where ffmpeg >nul 2>nul
  if errorlevel 1 (
    echo  [!] ffmpeg is still missing. Run _get_ffmpeg.bat, then run this again.
    echo      Everything else will be installed, so you can continue afterwards.
  ) else (
    echo        using the ffmpeg already on your PATH
  )
)

echo  [3/4] Installing RealMe...
call "%CONDA_BAT%" run -n realme python -m pip install -e "03_App" --quiet
if errorlevel 1 (
  echo  [X] Install failed. Scroll up for the reason.
  pause & exit /b 1
)

echo  [3b/4] Installing the draft voice...
REM  espeak-ng is a Linux package and simply is not present on Windows, so the
REM  "draft voice" would silently have nothing to run. Piper is pip-installable,
REM  sounds far better, and is what makes the read-it-before-you-render loop
REM  usable here. ~60 MB, downloaded once.
REM
REM  The logic lives in realme.engines.install so there is ONE implementation of
REM  it: this launcher and `realme engine install` do exactly the same thing,
REM  including the skip-if-already-present checks.
call "%CONDA_BAT%" run -n realme python -c "from realme.engines.install import install_draft_voices, engine_home; install_draft_voices(engine_home())"
if errorlevel 1 echo        ^(draft voice unavailable - RealMe still works^)

echo  [4/4] Checking the environment...
echo.
REM  From here on, no conda. Everything runs straight out of the environment
REM  folder, which is why the other launchers start instantly.
call "%~dp0_env.bat"
if not defined REALME_ENV (
  echo  [X] Installed, but the environment folder was not found afterwards.
  echo      Tell the launchers where it is:
  echo         set REALME_ENV=C:\path\to\anaconda3\envs\realme
  pause & exit /b 1
)
"%REALME_ENV%\Scripts\realme.exe" setup

REM  Voice cloning is optional. A new installation is asked once: Gemini, the
REM  free local clone (in the package, or 3_Get_Voice_Engine.bat), or the
REM  draft voices only.
REM  An existing installation keeps what it has and is not asked again.
"%REALME_ENV%\Scripts\realme.exe" voice choose --if-new

echo.
echo  Done - and conda is not needed again.
echo.
echo  Next:
echo     2_Set_Key.bat        paste your Gemini key
echo     4_Start_Studio.bat   open the Studio - every time after this
echo.
echo  Change the voice any time: Studio - Twin Setup - Voice for lectures.
echo.
echo  From any Command Prompt you can also just do:
echo      cd /d "%~dp0.."
echo      realme setup
echo.
pause
