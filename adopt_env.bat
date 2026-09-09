@echo off
REM Find the setup on this machine that already drives the GPU (ComfyUI or
REM similar) and report what makes it work, so it can be reused or copied.
REM
REM   adopt_env.bat
REM   adopt_env.bat --python "C:\ComfyUI\venv\Scripts\python.exe"
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo No environment found. Run setup_rocm_windows.bat first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -u scripts\adopt_env.py %*
set RESULT=%ERRORLEVEL%
echo.
echo   Saved to: %~dp0adopt_report.txt
pause
exit /b %RESULT%
