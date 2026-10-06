@echo off
REM ===================================================================
REM  Runs the expressive-dialogue experiment with RealMe's OWN Python.
REM
REM  Why this file exists: `python 03_App\explore_expressive.py` from an
REM  ordinary prompt picks whatever `python` is first on PATH. On the machine
REM  this was first tried, that was a bare pythoncore 3.14 with none of
REM  RealMe's packages in it. The cloned voice still worked -- it is a DLL
REM  loaded through ctypes and needs no Python package -- so the run got all
REM  the way to the guest voice before anything complained.
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
if "%~1"=="" (
  "%REALME_ENV%\python.exe" "03_App\explore_expressive.py" --out "runs\exp1" --repeat 3
) else (
  "%REALME_ENV%\python.exe" "03_App\explore_expressive.py" --out "runs\exp1" %*
)
echo.
pause
