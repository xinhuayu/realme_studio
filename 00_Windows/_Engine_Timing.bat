@echo off
setlocal
REM ===================================================================
REM  Where does the free local clone (qwen3cpp) spend its time?
REM
REM  Builds a second copy of the engine with its built-in stopwatch
REM  switched on, in tools\qwen3cpp_timing. The engine the Studio uses is
REM  not touched. The copy also applies the thread setting, which the
REM  upstream engine stores but does not use (it runs on 4 threads), so
REM  the measurement shows whether more threads help.
REM
REM  Then it records a few sentences at 4, 6, 8 and 12 threads and prints
REM  where the time went: turning codes into sound, the main model, the
REM  small model, and rebuilding the same calculation plans. Takes about
REM  10-20 minutes, most of it the first build. Needs the C++ compiler
REM  that built the engine. Leave the computer alone while it runs.
REM
REM  Options pass through, for example:
REM      _Engine_Timing.bat --threads 4,8,16 --sentences 5
REM
REM  Exit codes are compared against 0, never with `if errorlevel 1`,
REM  which is ">= 1" and so reads a crash as success.
REM ===================================================================
cd /d "%~dp0.."
call "%~dp0_env.bat"
if not defined REALME_ENV (
  echo  The "realme" environment was not found. Run 1_Install.bat first.
  pause & exit /b 1
)
set "PY=%REALME_ENV%\python.exe"
REM  Delayed expansion only from here, after every %~dp0 above, so a `!` in
REM  the folder name is not eaten; paths are read as !PY! from here on.
setlocal EnableDelayedExpansion

echo.
"!PY!" -m realme engine-timing %*
set "RC=!ERRORLEVEL!"
echo.
echo      exit=!RC!
if not "!RC!"=="0" (
  echo  The measurement did not finish. Nothing was changed.
  pause & exit /b 1
)
echo.
echo  Send the "results:" file named above to the RealMe developer.
echo.
pause
