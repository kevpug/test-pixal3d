@echo off
REM Double-click entry point for the ROCm/Windows setup.
REM PowerShell blocks unsigned scripts by default, so invoke it with the
REM policy relaxed for this one process rather than changing it machine-wide.
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_rocm_windows.ps1" %*
set RESULT=%ERRORLEVEL%
echo.
pause
exit /b %RESULT%
