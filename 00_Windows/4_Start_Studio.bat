@echo off
REM  Starts RealMe Studio. No conda, no activation, no waiting.
cd /d "%~dp0.."
call "%~dp0_env.bat"
if not defined REALME_ENV (
  echo  The "realme" environment was not found. Run 1_Install.bat first.
  pause & exit /b 1
)
echo Starting RealMe Studio...
start "" http://127.0.0.1:8000
"%REALME_ENV%\Scripts\realme.exe" studio
pause
