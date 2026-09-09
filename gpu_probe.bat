@echo off
REM Find out exactly where the GPU stack dies. Each step runs in its own
REM process, so a crash is reported instead of silently killing the probe.
setlocal
cd /d "%~dp0"
REM Settings scripts/repair_rocm.py found this GPU needs, if any.
if exist "%~dp0pixal3d_env.bat" call "%~dp0pixal3d_env.bat"
if not exist ".venv\Scripts\python.exe" (
    echo No environment found. Run setup_rocm_windows.bat first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" scripts\gpu_probe.py %* > "%~dp0gpu_probe_report.txt" 2>&1
set RESULT=%ERRORLEVEL%
type "%~dp0gpu_probe_report.txt"
echo.
echo   Saved to: %~dp0gpu_probe_report.txt
echo   For the HIP runtime's own log:  gpu_probe.bat --verbose
pause
exit /b %RESULT%
