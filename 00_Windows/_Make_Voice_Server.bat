@echo off
setlocal
REM ===================================================================
REM  Make the voice server: your cloned voice for other tools (Anim Studio,
REM  or anything that can use OpenAI's text-to-speech), in a folder of its
REM  own that runs without RealMe.
REM
REM  Writes the folder BESIDE the RealMe folder:
REM      ..\realme-qwen3-tts-server\
REM  with the voice engine as built here, its 8-bit model (about 1 GB), the
REM  server and its launcher. Needs the local clone built first
REM  (3_Get_Voice_Engine.bat). Run it again after rebuilding the engine:
REM  the folder is brought up to date and its voices\ are kept.
REM  Or say where:  _Make_Voice_Server.bat D:\voice-server
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
for %%I in ("%CD%\..") do set "OUT=%%~fI\realme-qwen3-tts-server"
if not "%~1"=="" set "OUT=%~f1"
setlocal EnableDelayedExpansion

echo.
echo  Your RealMe voice recording can go in as the voice "realme". The folder
echo  then holds your voice: keep it private, and do not publish it.
set "MINE="
set /p MINE="  Include your voice recording? (y/N): "
set "WITH="
if /i "!MINE!"=="y" set "WITH=--with-my-voice"

"!PY!" -m realme engine cpp --cpp-action export-server --export-to "!OUT!" !WITH!
set "RC=!ERRORLEVEL!"
echo.
echo      exit=!RC!
if not "!RC!"=="0" (
  echo  Nothing was written.
  pause & exit /b 1
)
echo.
echo  The voice server is in:  !OUT!
echo.
echo  Port: 8800 unless you change it to what your tool expects, in
echo      !OUT!\voice_server.json   ^(or start it with --port 5050^)
echo  Start it:   !OUT!\server\start_server.bat   ^(needs Python 3.9+^)
echo  Then, in Anim Studio: Setup - Other voices - A voice server:
echo      address  http://127.0.0.1:8800   model  realme-qwen3-tts   voice  realme   no key
echo  Any other tool: an OpenAI-compatible speech endpoint at
echo      http://127.0.0.1:8800/v1/audio/speech   ^(with your port^)
echo  More voices: put name.wav recordings in !OUT!\voices and restart it.
echo.
pause
