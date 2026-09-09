@echo off
REM Diagnose the install: torch build, GPU visibility, backends, self-test.
setlocal
cd /d "%~dp0"
REM Settings scripts/repair_rocm.py found this GPU needs, if any.
if exist "%~dp0pixal3d_env.bat" call "%~dp0pixal3d_env.bat"
if not exist ".venv\Scripts\python.exe" (
    echo No environment found. Run setup_rocm_windows.bat first.
    pause
    exit /b 1
)
REM Saved to a file as well as shown: the interesting part is near the top
REM and a console window scrolls it away.
".venv\Scripts\python.exe" -u scripts\check_env.py
set RESULT=%ERRORLEVEL%
echo.
echo ---------------------------------------------------------------
echo   Full report saved to:
echo     %~dp0check_env_report.txt
echo   Paste that file if you need help - it has the whole output,
echo   not just what fits on screen.
echo.
echo   If the GPU section looks wrong or missing, run:
echo     gpu_probe.bat
echo ---------------------------------------------------------------
pause
exit /b %RESULT%
