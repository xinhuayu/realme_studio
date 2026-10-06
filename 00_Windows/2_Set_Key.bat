@echo off
REM  Stores API keys in %USERPROFILE%\RealMeStudio\.env  (takes effect immediately).
call "%~dp0_env.bat"
cd /d "%~dp0..\03_App"
if not defined REALME_ENV (
  echo  The "realme" environment was not found. Run 1_Install.bat first.
  pause & exit /b 1
)
set "R=%REALME_ENV%\Scripts\realme.exe"
echo.
echo  Paste your Gemini API key (free from aistudio.google.com).
echo  Stored in a local .env file; it never leaves this machine.
echo.
set /p KEY="  GEMINI_API_KEY: "
if not "%KEY%"=="" "%R%" key GEMINI_API_KEY "%KEY%"
echo.
echo  Optional - press Enter to skip.
REM  LEAVE THIS BLANK if you ran step 3. Deliberately not pre-filled with the
REM  legacy path: REALME_QWEN3_ROOT *overrides* the engine copied into
REM  tools\qwen3, so setting it would pin RealMe to the old folder - and then
REM  step 6 would delete that folder out from under it. It exists only for
REM  someone who would rather point at an external engine than copy one in.
set /p QW="  Path to an external qwen3 folder (leave blank if you ran step 3): "
if defined QW for /f "tokens=* delims=" %%A in ("%QW%") do set "QW=%%~A"
if not "%QW%"=="" "%R%" key REALME_QWEN3_ROOT "%QW%"
set /p EL="  ELEVENLABS_API_KEY: "
if not "%EL%"=="" "%R%" key ELEVENLABS_API_KEY "%EL%"
echo.
"%R%" key --where
pause
