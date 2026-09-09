@echo off
REM ===================================================================
REM  Repair a ROCm GPU stack that will not enumerate.
REM
REM  Runs the whole remediation sequence unattended, re-testing after
REM  each step and stopping at the first thing that works:
REM    1. test what is installed
REM    2. sweep the HIP environment variables
REM    3. check the Visual C++ runtime
REM    4. reinstall a ROCm build known to work for this GPU
REM    5. fall back through older builds
REM
REM  Step 4 onward downloads several GB per attempt.
REM
REM    fix_gpu.bat                 default, up to 2 builds
REM    fix_gpu.bat --deep 4        try more builds
REM    fix_gpu.bat --dry-run       say what it would do, install nothing
REM ===================================================================
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo No environment found. Run setup_rocm_windows.bat first.
    pause
    exit /b 1
)
REM No redirect: the script tees itself, so a long download stays visible.
".venv\Scripts\python.exe" -u scripts\repair_rocm.py %*
set RESULT=%ERRORLEVEL%
echo.
echo ---------------------------------------------------------------
echo   Saved to: %~dp0repair_report.txt
if "%RESULT%"=="0" (
    echo   The GPU works. Next:  run_image.bat assets\images\0_img.png out.glb
) else (
    echo   Still broken. Paste repair_report.txt - it has every step tried.
    echo   To try more ROCm builds:  fix_gpu.bat --deep 4
)
echo ---------------------------------------------------------------
pause
exit /b %RESULT%
