@echo off
REM ===================================================================
REM  Points at the realme environment DIRECTLY, with no conda involved.
REM
REM  Why: `conda run -n realme ...` re-initialises conda's machinery on every
REM  invocation, which costs seconds before your command even starts. But an
REM  environment is just a folder. Putting its bin directories on PATH is
REM  exactly what `conda activate` does, and it is instant.
REM
REM  Conda is therefore needed ONCE, to create the environment. Never again.
REM
REM  NOTE: deliberately no `setlocal` - this script exists to modify the
REM  CALLER's PATH, and setlocal would discard everything it does.
REM
REM  Sets REALME_ENV. Override it to keep the environment elsewhere.
REM
REM  Also sets REALME_TOOLS, the one folder holding the installed payload:
REM  ffmpeg, the voice engine venv, the model weights, the draft voice. Pinning
REM  it here rather than letting Python infer it from where the package lives is
REM  what stops ffmpeg and the engine ending up in two different tools\ folders,
REM  which is exactly what happened before. Every launcher calls this file, so
REM  they all agree. An existing REALME_TOOLS wins, and an engine already
REM  installed somewhere else is still found - RealMe searches, and only writes
REM  to this one.
REM ===================================================================

REM  %~dp0 is 00_Windows\, so its parent is the project root.
if not defined REALME_TOOLS for %%D in ("%~dp0..") do set "REALME_TOOLS=%%~fD\tools"

if defined REALME_ENV if exist "%REALME_ENV%\python.exe" goto :apply
set "REALME_ENV="

for %%R in (
  "%USERPROFILE%\anaconda3"
  "%USERPROFILE%\Anaconda3"
  "%USERPROFILE%\miniconda3"
  "%USERPROFILE%\Miniconda3"
  "%LOCALAPPDATA%\anaconda3"
  "%LOCALAPPDATA%\Continuum\anaconda3"
  "C:\ProgramData\anaconda3"
  "C:\ProgramData\Anaconda3"
  "C:\ProgramData\Miniconda3"
  "C:\anaconda3"
) do (
  if exist "%%~R\envs\realme\python.exe" (
    set "REALME_ENV=%%~R\envs\realme"
    goto :apply
  )
)

REM  Fallback: if the dedicated environment could not be created (conda DLL
REM  problems are common on Windows) but RealMe was installed into the base
REM  Anaconda instead, use that rather than reporting nothing found.
for %%R in (
  "%USERPROFILE%\anaconda3"
  "%USERPROFILE%\Anaconda3"
  "%USERPROFILE%\miniconda3"
  "C:\ProgramData\anaconda3"
) do (
  if exist "%%~R\Scripts\realme.exe" (
    set "REALME_ENV=%%~R"
    goto :apply
  )
)
goto :eof

:apply
REM  The same directories `conda activate` prepends. Library\bin is where
REM  conda-forge's ffmpeg.exe lands, which is what makes ffmpeg work here
REM  without a separate system-wide install.
REM  Prepend ONCE per window. The realme wrapper in the project root calls this on every command, and an
REM  unconditional prepend grew PATH by ~300 characters each time until cmd's
REM  8191-character line limit cut it off a few dozen commands in.
if defined REALME_PATH_SET if "%REALME_PATH_SET%"=="%REALME_ENV%" goto :setexe
set "PATH=%REALME_ENV%;%REALME_ENV%\Library\mingw-w64\bin;%REALME_ENV%\Library\usr\bin;%REALME_ENV%\Library\bin;%REALME_ENV%\Scripts;%PATH%"
set "REALME_PATH_SET=%REALME_ENV%"
:setexe
set "REALME_EXE=%REALME_ENV%\Scripts\realme.exe"
goto :eof
