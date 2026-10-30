@echo off
setlocal
REM ===================================================================
REM  Make realme-qwen3-tts.cpp: RealMe's revised voice engine as its own
REM  source release, in its own folder, ready for GitHub Desktop.
REM
REM  Writes the folder BESIDE the RealMe folder (not inside it, so the two
REM  repositories never nest):
REM      ..\realme-qwen3-tts.cpp\                 the release, to publish
REM      ..\realme-qwen3-tts.cpp-<version>-source.zip  to attach to the release
REM  or to the folder you give:  _Make_qwen3cpp_Release.bat D:\code\realme-qwen3-tts.cpp
REM
REM  Run it again after a RealMe update: a folder that is already a git
REM  repository is brought up to date in place, its history kept, and
REM  GitHub Desktop shows exactly what changed. Nothing in RealMe changes.
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
for %%I in ("%CD%\..") do set "OUT=%%~fI\realme-qwen3-tts.cpp"
if not "%~1"=="" set "OUT=%~f1"
setlocal EnableDelayedExpansion

"!PY!" -m realme engine cpp --cpp-action export-release --export-to "!OUT!"
set "RC=!ERRORLEVEL!"
echo.
echo      exit=!RC!
if not "!RC!"=="0" (
  echo  Nothing was written.
  pause & exit /b 1
)
echo.
echo  The release is in:  !OUT!
echo.
echo  To publish it with GitHub Desktop, the first time:
echo    1. File - Add local repository - choose the folder above. It says
echo       this is not a git repository: click "create a repository". Name
echo       realme-qwen3-tts.cpp; leave README, .gitignore and licence at None
echo       ^(the folder has its own^). Create repository.
echo    2. Publish repository. Untick "Keep this code private" for a public one.
echo    3. On github.com, in the new repository: Releases - Draft a new
echo       release - tag v^<version^> as at the top of RELEASE_NOTES.md - paste
echo       that version's notes - attach the source zip named above - Publish.
echo.
echo  Later versions: run this again, then in GitHub Desktop commit the
echo  changes, Push origin, and draft the next release on github.com.
echo.
pause
