@echo off
REM ===================================================================
REM  Runs the expressive-dialogue experiment with RealMe's OWN Python.
REM
REM  Why this file exists: `python 03_App\explore_expressive.py` from an
REM  ordinary prompt picks whatever `python` is first on PATH, which may have
REM  none of RealMe's packages. The cloned voice is a DLL loaded through
REM  ctypes and needs no Python package, so such a run gets as far as the
REM  guest voice before it fails.
REM
REM  _env.bat puts the right environment first on PATH, which is all it takes.
REM
REM  Anything after the file name is passed through:
REM     8_Expressive_Experiment.bat --repeat 5
REM     8_Expressive_Experiment.bat --dry-run
REM     8_Expressive_Experiment.bat --list-voices
REM     8_Expressive_Experiment.bat --audition
REM     8_Expressive_Experiment.bat --audition female
REM ===================================================================
cd /d "%~dp0.."
call "%~dp0_env.bat"
if not defined REALME_ENV (
  echo  The "realme" environment was not found. Run 1_Install.bat first.
  pause & exit /b 1
)
REM  goto, not an if/else block: cmd expands %REALME_ENV% when it reads a
REM  whole ( ) block, and a folder like "Documents (work)" ends it mid-path.
if not "%~1"=="" goto :with_args
"%REALME_ENV%\python.exe" "03_App\explore_expressive.py" --out "runs\exp1" --repeat 3
goto :done
:with_args
"%REALME_ENV%\python.exe" "03_App\explore_expressive.py" --out "runs\exp1" %*
:done
echo.
pause
