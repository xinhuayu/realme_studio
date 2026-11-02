@echo off
REM ===================================================================
REM  Find the fastest settings of the local voice on THIS computer.
REM
REM  For a new computer (a Core Ultra laptop, say): after
REM  _Rebuild_Voice_Engine.bat, this records the same short lecture text
REM  with each combination of the engine's switches and pass sizes, checks
REM  that every combination gives exactly the same audio, and reports the
REM  fastest with the settings to use. Nothing in the Studio is changed.
REM  About 20-40 minutes; close the Studio first.
REM
REM    _Tune_Voice_Engine.bat                    this computer's defaults
REM    _Tune_Voice_Engine.bat --slots 4,8        a 16 GB computer
REM    _Tune_Voice_Engine.bat --plan-only        show the cases, run none
REM
REM  The listening comparison of fewer predictor steps is
REM  03_App\run_codebook_listening.py (run with 5_Command_Prompt.bat).
REM ===================================================================
setlocal
cd /d "%~dp0.."
call "%~dp0_env.bat"
if not defined REALME_ENV (
  echo  The "realme" environment was not found. Run 1_Install.bat first.
  pause & exit /b 1
)
"%REALME_ENV%\python.exe" "03_App\run_cpu_combinations.py" %*
set "RC=%ERRORLEVEL%"
echo.
pause
exit /b %RC%
