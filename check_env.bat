@echo off
REM Diagnose the install: torch build, GPU visibility, backends, self-test.
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo No environment found. Run setup_rocm_windows.bat first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" scripts\check_env.py
pause
