@echo off
REM Start the voice server: an OpenAI-compatible text-to-speech address on
REM this computer, speaking in the voices in the voices\ folder. The port,
REM default voice (realme) and key are in voice_server.json; options given
REM here override it, for example:
REM     server\start_server.bat --port 5050 --voice realme
REM Needs Python 3.9 or newer (nothing else to install) and the engine:
REM build.bat (from source) or a ready-made bundle's bin\ folder.
setlocal
cd /d "%~dp0.."
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY (
  echo  Python 3.9 or newer is needed: https://www.python.org/downloads/
  pause & exit /b 1
)
%PY% server\qwen3_tts_server.py %*
pause
