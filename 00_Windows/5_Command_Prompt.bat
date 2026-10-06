@echo off
REM  A terminal where `realme` just works. No conda activation.
cd /d "%~dp0.."
call "%~dp0_env.bat"
if not defined REALME_ENV (
  echo  The "realme" environment was not found. Run 1_Install.bat first.
  pause & exit /b 1
)
echo.
echo  RealMe ready.  %REALME_ENV%
echo.
echo    realme setup            what is installed
echo    realme studio           the web app
echo    realme voice check my.wav
echo    realme lecture deck.pdf -o .\out --tts espeak
echo.
cmd /k
