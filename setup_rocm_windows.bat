@echo off
REM ===================================================================
REM  One-shot setup for Pixal3D on Windows with an AMD GPU (ROCm).
REM
REM  Deliberately plain cmd, with no PowerShell step. PowerShell adds
REM  an execution policy, a script encoding that Windows PowerShell 5.1
REM  reads as cp1252 unless there is a BOM, and a rule that turns any
REM  native command's stderr into a terminating error -- and pip and
REM  py.exe both write to stderr routinely. None of that is worth it
REM  for "make a venv and run pip".
REM
REM  Usage:
REM    setup_rocm_windows.bat
REM    setup_rocm_windows.bat --family gfx110X-dgpu
REM    setup_rocm_windows.bat --rocm-version 7.13.0a20260421
REM    setup_rocm_windows.bat --python "C:\Path\To\python.exe"
REM    setup_rocm_windows.bat --skip-torch
REM ===================================================================
setlocal
cd /d "%~dp0"

set "FAMILY="
set "ROCMVER="
set "SKIPTORCH="
set "PYCMD="
set "VENV=.venv"
set "VERFILE=%TEMP%\pixal3d_pyver.txt"

REM ------------------------------------------------------------ args ----
:parse
if "%~1"=="" goto parsed
if /i "%~1"=="--family"       goto opt_family
if /i "%~1"=="--rocm-version" goto opt_rocm
if /i "%~1"=="--python"       goto opt_python
if /i "%~1"=="--venv"         goto opt_venv
if /i "%~1"=="--skip-torch"   goto opt_skip
if /i "%~1"=="--help"         goto usage
if /i "%~1"=="-h"             goto usage
echo Unknown option: %~1
goto usage

:opt_family
set "FAMILY=%~2"
shift
shift
goto parse
:opt_rocm
set "ROCMVER=%~2"
shift
shift
goto parse
:opt_python
set "PYCMD=%~2"
shift
shift
goto parse
:opt_venv
set "VENV=%~2"
shift
shift
goto parse
:opt_skip
set "SKIPTORCH=1"
shift
goto parse

:usage
echo.
echo   setup_rocm_windows.bat [--family NAME] [--rocm-version ID]
echo                          [--python PATH] [--venv DIR] [--skip-torch]
echo.
echo   --family        gfx103X-dgpu  RX 6000 / RDNA2 (RX 6800M, 6700 XT, ...)
echo                   gfx110X-dgpu  RX 7000 / RDNA3
echo                   gfx120X-all   RX 9000 / RDNA4
echo                   gfx1151       Ryzen AI Max / Strix Halo
echo                   Default: detected from the installed adapter.
echo   --rocm-version  Pin a ROCm nightly build id.
echo   --python        Full path to a python.exe (3.12 or 3.13).
echo   --skip-torch    Reuse an existing torch install.
echo.
exit /b 1

:parsed

REM ---------------------------------------------------- prerequisites ----
echo.
echo ==^> Checking prerequisites

where git >nul 2>nul
if errorlevel 1 (
    echo.
    echo ERROR: git is not on PATH. MoGe-2 ^(camera estimation^) installs from a
    echo        git URL. Install it from https://git-scm.com/download/win
    echo        and re-run this script.
    goto fail
)
echo     git found

REM No adapter query here on purpose: wmic is gone from Windows 11 24H2 and
REM the alternatives all need PowerShell. check_env.py reports the GPU at the
REM end of this script, through torch, which is the answer that actually
REM matters -- Windows seeing the card does not mean ROCm can use it.

REM ------------------------------------------------------- interpreter ----
echo.
echo ==^> Looking for Python 3.12 or 3.13
if defined PYCMD goto check_given

call :probe "py -3.12"
if defined PYCMD goto have_python
call :probe "py -3.13"
if defined PYCMD goto have_python
call :probe "python"
if defined PYCMD goto have_python
call :probe "python3"
if defined PYCMD goto have_python

echo.
echo ERROR: No suitable Python found. This needs Python 3.12 or 3.13.
echo        AMD publishes ROCm wheels for cp312/cp313/cp314 only, and the
echo        mesh dependencies ship cp312/cp313, so other versions cannot work.
echo.
echo        Install 3.12 from https://www.python.org/downloads/
echo        and tick "Add python.exe to PATH", then re-run this script.
echo        Have it elsewhere?  setup_rocm_windows.bat --python "C:\Path\python.exe"
echo.
where py >nul 2>nul
if not errorlevel 1 (
    echo        py.exe reports these installed:
    py --list 2>&1
)
goto fail

:check_given
call :probe "%PYCMD%"
if defined PYCMD goto have_python
echo.
echo ERROR: --python "%PYCMD%" did not run, or is not Python 3.12/3.13.
goto fail

:have_python
echo     using: %PYCMD%

REM ------------------------------------------------------------ venv -----
echo.
echo ==^> Creating the virtual environment at %VENV%
if exist "%VENV%\Scripts\python.exe" (
    echo     reusing the existing environment
) else (
    %PYCMD% -m venv "%VENV%"
    if errorlevel 1 (
        echo ERROR: venv creation failed.
        goto fail
    )
)
set "VPY=%VENV%\Scripts\python.exe"
if not exist "%VPY%" (
    echo ERROR: %VPY% not found - is %VENV% a virtual environment?
    goto fail
)

"%VPY%" -m pip install --quiet --upgrade pip setuptools wheel
if errorlevel 1 (
    echo ERROR: could not upgrade pip.
    goto fail
)

REM ---------------------------------------------------------- pytorch ----
if defined SKIPTORCH goto deps
echo.
echo ==^> Installing PyTorch for ROCm ^(several GB, this takes a while^)
set "TORCHARGS="
if defined FAMILY  set "TORCHARGS=%TORCHARGS% --family %FAMILY%"
if defined ROCMVER set "TORCHARGS=%TORCHARGS% --rocm-version %ROCMVER%"
"%VPY%" scripts\install_rocm_torch.py%TORCHARGS%
if errorlevel 1 (
    echo.
    echo ERROR: PyTorch installation failed. To see the builds on offer:
    echo          "%VPY%" scripts\install_rocm_torch.py --list
    echo        then retry with:
    echo          setup_rocm_windows.bat --rocm-version THE_BUILD_ID
    goto fail
)

:deps
echo.
echo ==^> Installing the remaining dependencies
"%VPY%" -m pip install -r requirements-rocm.txt
if errorlevel 1 (
    echo ERROR: dependency installation failed.
    goto fail
)

REM ------------------------------------------------------------ check ----
echo.
echo ==^> Checking the installation
"%VPY%" scripts\check_env.py
set "CHECK=%ERRORLEVEL%"

echo.
if "%CHECK%"=="0" (
    echo Setup complete.
) else (
    echo Setup finished, but some checks failed - see above.
)
echo.
echo   Activate:      %VENV%\Scripts\activate
echo   Single image:  python inference.py --image assets\images\0_img.png --output out.glb
echo   Multi-view:    python inference_mv.py --views_dir assets\mv_images\example --output out_mv.glb
echo   Web UI:        python app_local.py
echo   Measure GPU:   python scripts\benchmark.py
echo.
echo   Weights ^(~20 GB^) download from Hugging Face on the first run.
echo.
pause
exit /b %CHECK%

:fail
echo.
pause
exit /b 1

REM ------------------------------------------------------- subroutine ----
REM Probe one candidate. Sets PYCMD when it is Python 3.12 or 3.13.
REM No percent signs in the -c string: cmd would eat them.
:probe
set "PYCMD="
set "CAND=%~1"
%CAND% -c "import sys;print(str(sys.version_info[0])+'.'+str(sys.version_info[1]))" >"%VERFILE%" 2>nul
if errorlevel 1 goto probe_done
if not exist "%VERFILE%" goto probe_done
set "VER="
set /p VER=<"%VERFILE%"
del "%VERFILE%" >nul 2>nul
if "%VER%"=="3.12" set "PYCMD=%CAND%"
if "%VER%"=="3.13" set "PYCMD=%CAND%"
if defined PYCMD (
    echo     found Python %VER%  ^(%CAND%^)
) else (
    if defined VER echo     skipping %CAND% - Python %VER%, need 3.12 or 3.13
)
:probe_done
goto :eof
