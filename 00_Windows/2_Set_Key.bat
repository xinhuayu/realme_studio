@echo off
REM  Stores API keys in %USERPROFILE%\RealMeStudio\.env  (takes effect immediately).
setlocal
call "%~dp0_env.bat"
cd /d "%~dp0..\03_App"
if not defined REALME_ENV (
  echo  The "realme" environment was not found. Run 1_Install.bat first.
  pause & exit /b 1
)
set "R=%REALME_ENV%\Scripts\realme.exe"
REM  Delayed expansion only from here, after every %~dp0 above: a `!` in the
REM  folder name would otherwise be eaten.
setlocal EnableDelayedExpansion
echo.
echo  Paste your Gemini API key (free from aistudio.google.com).
echo  Stored in a local .env file; it never leaves this machine.
echo.
set "KEY="
set /p KEY="  GEMINI_API_KEY: "
if not "!KEY!"=="" "!R!" key GEMINI_API_KEY "!KEY!"
echo.
echo  Optional, to draft narration with Claude or OpenAI instead of Gemini.
echo  Each is a separate API key, not a Claude.ai or ChatGPT plan. Press Enter to skip.
set "AK="
set /p AK="  ANTHROPIC_API_KEY (console.anthropic.com): "
if not "!AK!"=="" "!R!" key ANTHROPIC_API_KEY "!AK!"
set "OK="
set /p OK="  OPENAI_API_KEY (platform.openai.com): "
if not "!OK!"=="" "!R!" key OPENAI_API_KEY "!OK!"
echo.
echo  Optional - press Enter to skip.
set "EL="
set /p EL="  ELEVENLABS_API_KEY: "
if not "!EL!"=="" "!R!" key ELEVENLABS_API_KEY "!EL!"
echo.
"!R!" key --where
pause
