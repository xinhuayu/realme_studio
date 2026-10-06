@echo off
REM ===================================================================
REM  Locates conda. Needed ONCE, by 1_Install.bat. Daily use goes through
REM  _env.bat instead, which does not touch conda at all.
REM
REM  Anaconda on Windows does not add itself to PATH - you are meant to use
REM  "Anaconda Prompt" - so relying on PATH makes a double-click fail even
REM  though Anaconda is installed.
REM
REM  Three things in an Anaconda folder can run conda commands:
REM      condabin\conda.bat   the shell wrapper (preferred; handles activation)
REM      Scripts\conda.exe    the CLI executable
REM      _conda.exe           the standalone binary at the root - in current
REM                           versions conda.bat sets CONDA_EXE to this, so it
REM                           does work, but it is an internal entry point
REM  We prefer them in that order.
REM
REM  Sets CONDA_BAT and CONDA_ROOT. Override with REALME_CONDA.
REM ===================================================================
set "CONDA_BAT="
set "CONDA_ROOT="

if defined REALME_CONDA call :probe "%REALME_CONDA%"
if defined CONDA_BAT goto :eof

where conda >nul 2>nul
if not errorlevel 1 (
  for /f "delims=" %%i in ('where conda 2^>nul') do (
    set "CONDA_BAT=%%i"
    goto :eof
  )
)

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
  call :probe "%%~R"
  if defined CONDA_BAT goto :eof
)
goto :eof

:probe
if exist "%~1\condabin\conda.bat" ( set "CONDA_BAT=%~1\condabin\conda.bat" & set "CONDA_ROOT=%~1" & goto :eof )
if exist "%~1\Scripts\conda.exe"  ( set "CONDA_BAT=%~1\Scripts\conda.exe"  & set "CONDA_ROOT=%~1" & goto :eof )
if exist "%~1\_conda.exe"         ( set "CONDA_BAT=%~1\_conda.exe"         & set "CONDA_ROOT=%~1" & goto :eof )
goto :eof
