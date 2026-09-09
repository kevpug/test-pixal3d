@echo off
REM Launch the local web UI in the environment setup_rocm_windows created.
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo No environment found. Run setup_rocm_windows.bat first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" app_local.py %*
pause
