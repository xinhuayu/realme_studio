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
REM  folder name would otherwise be eaten. It is needed to strip quotes from
REM  a pasted path safely - see the QW prompt below.
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
REM  LEAVE THIS BLANK if you ran step 3. Deliberately not pre-filled with the
REM  legacy path: REALME_QWEN3_ROOT *overrides* the engine copied into
REM  tools\qwen3, so setting it would pin RealMe to the old folder - and then
REM  step 6 would delete that folder out from under it. It exists only for
REM  someone who would rather point at an external engine than copy one in.
set "QW="
set /p QW="  Path to an external qwen3 folder (leave blank if you ran step 3): "
REM  Explorer's "Copy as path" wraps the path in quotes. Removed with delayed
REM  expansion, which happens after cmd has parsed the line, so a `)` or `&`
REM  in the path is never seen by the parser. The old for /f "%%~A" idiom put
REM  the quoted value inside a second pair of quotes, which UN-quoted the path
REM  and let "C:\x (work)\qwen3" end the for-list at its `)`.
if defined QW set "QW=!QW:"=!"
if defined QW if "!QW:~-1!"=="\" set "QW=!QW!."
if not "!QW!"=="" "!R!" key REALME_QWEN3_ROOT "!QW!"
set "EL="
set /p EL="  ELEVENLABS_API_KEY: "
if not "!EL!"=="" "!R!" key ELEVENLABS_API_KEY "!EL!"
echo.
"!R!" key --where
pause
