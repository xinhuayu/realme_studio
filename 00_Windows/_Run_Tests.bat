@echo off
setlocal
REM ===================================================================
REM  Run every check, from anywhere.
REM
REM  For a script or an AI agent, the same run with one exit code and a
REM  JSON summary:  python 03_App\run_tests.py --json
REM  The suite list lives there too; verify_tree checks the two agree.
REM
REM  What this launcher takes care of:
REM
REM    1. The Python. `python t_render.py` finds whichever one is first on
REM       PATH, which may not be the one RealMe is installed into.
REM       _env.bat points at the right interpreter.
REM
REM    2. The folder. The suites live in 03_App and anchor their fixtures
REM       on their own location; this file also cd's to the root.
REM
REM  How the result is read:
REM
REM    `if errorlevel 1` is NOT used. It means "errorlevel >= 1", so a
REM    process that dies with a negative code -- an access violation is
REM    -1073741819 -- passes it. The code is captured and compared
REM    against 0, and printed either way, so a failed check (1) and a
REM    crash (anything else) are told apart on sight.
REM
REM    Nothing is piped. `cmd | more` leaves %ERRORLEVEL% holding the exit
REM    code of `more`, not of the program.
REM
REM    The failure list is `name=code`, with no parentheses in it. A close
REM    paren inside a FOR block ends the block -- even inside a quoted value
REM    and even inside a REM comment. A caret in a quoted `set` is a
REM    literal caret, not an escape, so it cannot escape the paren either.
REM
REM    Output stays live: a suite can run for a minute, and buffered output
REM    looks hung. A suite that FAILS is re-run into tests.log, so there is
REM    one file to paste without losing the live view.
REM ===================================================================
cd /d "%~dp0.."
call "%~dp0_env.bat"
if not defined REALME_ENV (
  echo  The "realme" environment was not found. Run 1_Install.bat first.
  pause & exit /b 1
)
set "PY=%REALME_ENV%\python.exe"
set "LOG=%CD%\tests.log"
REM  Delayed expansion is turned on only here, after every %~dp0 above, so a
REM  `!` in the folder name is not eaten. Inside the ( ) blocks below, paths
REM  are read as !PY! and !LOG!: cmd expands %VAR% when it reads the whole
REM  block, and a folder like "Documents (work)" ends it mid-path.
setlocal EnableDelayedExpansion
echo RealMe test run > "!LOG!"
echo python: !PY! >> "!LOG!"
echo cwd   : !CD! >> "!LOG!"

echo.
echo  Using !PY!
echo  Failures are written to !LOG!
echo.
echo  === verify_tree ===
"!PY!" 03_App\verify_tree.py
set "RC=!ERRORLEVEL!"
echo      exit=!RC!
if not "!RC!"=="0" (
  echo === verify_tree === >> "!LOG!"
  "!PY!" 03_App\verify_tree.py >> "!LOG!" 2>&1
  echo.
  echo  verify_tree FAILED ^(exit !RC!^). Fix that before running anything else.
  echo  Details in !LOG!
  pause & exit /b 1
)

set "FAILED="
for %%T in (t_tau t_pace t_migrate t_setup t_locate t_encoding t_voices t_lexicon t_render t_engines t_gemini t_agent t_tuning t_text t_inserts t_lean t_parbench t_prerecord t_engine_timing t_gemparallel t_writers_more t_engine_batch t_voice_server t_acoustic_judge t_codebook_listening t_cpu_combinations t_engine_attention t_engine_cpu_compute t_engine_cpu_runtime t_engine_decode_async t_engine_embedding t_engine_embedding_validation t_engine_fewer_steps t_engine_text_mmap t_engine_vocoder_storage t_engine_weight_mmap t_lecture_cpu t_native_warmup t_rev_engine t_runaway t_weight_mmap_quality) do (
  echo.
  echo  === %%T ===
  "!PY!" 03_App\%%T.py
  set "RC=!ERRORLEVEL!"
  echo      exit=!RC!
  if not "!RC!"=="0" (
    set "FAILED=!FAILED! %%T=!RC!"
    echo. >> "!LOG!"
    echo === %%T === exit=!RC! >> "!LOG!"
    "!PY!" 03_App\%%T.py >> "!LOG!" 2>&1
  )
)

echo.
if defined FAILED (
  echo  FAILED:!FAILED!
  echo  The number after each name is its exit code: 1 means a check
  echo  failed, any other code means it crashed.
  echo  Full output of the failures: !LOG!
  pause & exit /b 1
)
echo  All checks and all suites passed.
echo  Safe to build the update zip with _Make_Update_Zip.bat
echo.
pause
