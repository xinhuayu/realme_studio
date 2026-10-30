@echo off
REM  Run RealMe from a plain Command Prompt with no setup:
REM      cd C:\path\to\RealMe
REM      realme setup
REM  This finds the environment and forwards everything to it directly.
call "%~dp000_Windows\_env.bat"
if not defined REALME_ENV (
  echo  The "realme" environment was not found. Run 00_Windows\1_Install.bat first.
  exit /b 1
)
"%REALME_ENV%\Scripts\realme.exe" %*
