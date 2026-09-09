@echo off
REM Generate a GLB from one image: run_image.bat <image> [output.glb]
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo No environment found. Run setup_rocm_windows.bat first.
    pause
    exit /b 1
)
if "%~1"=="" (
    echo Usage: run_image.bat ^<image^> [output.glb]
    pause
    exit /b 1
)
set OUT=%~2
if "%OUT%"=="" set OUT=output.glb
".venv\Scripts\python.exe" inference.py --image "%~1" --output "%OUT%"
pause
