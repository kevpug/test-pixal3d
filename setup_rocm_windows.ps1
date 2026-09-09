<#
.SYNOPSIS
    One-shot setup for Pixal3D on Windows with an AMD GPU (ROCm).

.DESCRIPTION
    Creates a virtual environment, installs AMD's ROCm build of PyTorch for
    your GPU family, installs the rest of the dependencies, and runs the
    environment check.

    Nothing is compiled. Pixal3D's CUDA-only extensions (flash_attn, flex_gemm,
    cumesh, o_voxel, nvdiffrast) are replaced by the pure-PyTorch fallbacks in
    pixal3d/compat, because Triton - which flex_gemm needs - has no Windows
    ROCm build and the rest are CUDA-only.

.PARAMETER Family
    GPU family. Default: detected from the installed adapter.
      gfx103X-dgpu   RX 6000 series (RDNA2) - RX 6800M, 6800, 6900 XT, 6600 ...
      gfx110X-dgpu   RX 7000 series (RDNA3)
      gfx120X-all    RX 9000 series (RDNA4)
      gfx1151        Ryzen AI Max / Strix Halo

.PARAMETER Python
    Full path to a python.exe to build the venv from (spaces are fine).
    Default: py -3.12, then py -3.13, then python, then python3.

.PARAMETER VenvPath
    Where to create the environment. Default: .venv

.PARAMETER RocmVersion
    Pin a specific ROCm nightly build, e.g. 7.13.0a20260421.

.PARAMETER SkipTorch
    Reuse an existing torch install and only do the rest.

.EXAMPLE
    .\setup_rocm_windows.ps1
.EXAMPLE
    .\setup_rocm_windows.ps1 -Family gfx110X-dgpu
#>
[CmdletBinding()]
param(
    [string]$Family = "",
    [string]$Python = "",
    [string]$VenvPath = ".venv",
    [string]$RocmVersion = "",
    [switch]$SkipTorch
)

# NOT "Stop". In Windows PowerShell 5.1, anything a native command writes to
# stderr becomes a terminating NativeCommandError when this is "Stop" -- and
# py.exe (when a requested version is missing) and pip (ordinary warnings)
# both write to stderr routinely. Every native call below checks $LASTEXITCODE
# explicitly instead, which is what actually indicates failure.
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

function Write-Step($text) {
    Write-Host ""
    Write-Host "==> $text" -ForegroundColor Cyan
}

function Write-Warn($text) {
    Write-Host "    ! $text" -ForegroundColor Yellow
}

function Fail($text) {
    Write-Host ""
    Write-Host "ERROR: $text" -ForegroundColor Red
    exit 1
}

# ---------------------------------------------------------------- checks ----
Write-Step "Checking prerequisites"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Fail "git is not on PATH. MoGe-2 (camera estimation) installs from a git URL.
       Install it from https://git-scm.com/download/win and re-run."
}
Write-Host "    git found"

# The AMD display driver carries the HIP runtime the wheels load at import.
try {
    $gpu = (Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match "AMD|Radeon" })
    if ($gpu) {
        foreach ($g in $gpu) {
            Write-Host "    GPU: $($g.Name)  driver $($g.DriverVersion)"
        }
    } else {
        Write-Warn "No AMD adapter reported by Windows. ROCm will not find a device."
    }
} catch {
    Write-Warn "Could not query the display adapter: $_"
}

# ------------------------------------------------------------ interpreter ---
# 3.12 first: AMD publishes cp312/cp313/cp314 ROCm wheels, and the mesh
# dependencies (xatlas, fast-simplification) ship cp312/cp313.
$SupportedPython = @("3.12", "3.13")

function Get-PythonVersion($exe, $exeArgs) {
    # "3.12", or $null if this interpreter does not exist or does not run.
    # Merges stderr into the output stream so py.exe's "requested version is
    # not installed" chatter cannot derail the script; the exit code decides.
    $probe = "import sys;print('%d.%d' % sys.version_info[:2])"
    try {
        if ($exeArgs.Count -gt 0) {
            $out = & $exe @exeArgs -c $probe 2>&1
        } else {
            $out = & $exe -c $probe 2>&1
        }
    } catch {
        return $null
    }
    if ($LASTEXITCODE -ne 0) { return $null }
    return ($out | Where-Object { $_ -is [string] -and $_ -match '^\d+\.\d+$' } |
            Select-Object -First 1)
}

$pyExe = ""
$pyArgs = @()

if ($Python) {
    # An explicit -Python is taken whole, so a path with spaces still works.
    $pyExe = $Python
    $found = Get-PythonVersion $pyExe @()
    if (-not $found) { Fail "-Python '$Python' does not run." }
    if ($SupportedPython -notcontains $found) {
        Write-Warn "-Python '$Python' is Python $found; 3.12 or 3.13 is expected."
    }
    Write-Host "    Python $found ($Python)"
} else {
    foreach ($candidate in @(
        @{ exe = "py";      args = @("-3.12") },
        @{ exe = "py";      args = @("-3.13") },
        @{ exe = "python";  args = @() },
        @{ exe = "python3"; args = @() }
    )) {
        if (-not (Get-Command $candidate.exe -ErrorAction SilentlyContinue)) { continue }
        $found = Get-PythonVersion $candidate.exe $candidate.args
        if (-not $found) { continue }
        $shown = (@($candidate.exe) + $candidate.args) -join " "
        if ($SupportedPython -notcontains $found) {
            Write-Warn "$shown is Python $found - need 3.12 or 3.13, skipping"
            continue
        }
        $pyExe = $candidate.exe
        $pyArgs = $candidate.args
        Write-Host "    Python $found ($shown)"
        break
    }
}

if (-not $pyExe) {
    $installed = ""
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $list = (& py --list 2>&1 | Out-String).Trim()
        if ($list) { $installed = "`n`n       py.exe reports:`n" + $list }
    }
    Fail @"
No suitable Python found. This needs Python 3.12 or 3.13.
       AMD publishes ROCm wheels for cp312/cp313/cp314 only, and the mesh
       dependencies ship cp312/cp313, so older or newer versions cannot work.

       Install 3.12 from https://www.python.org/downloads/
       and tick "Add python.exe to PATH", then re-run this script.
       Already have it elsewhere?  .\setup_rocm_windows.ps1 -Python "C:\Path\To\python.exe"$installed
"@
}

# -------------------------------------------------------------- the venv ----
Write-Step "Creating the virtual environment at $VenvPath"
if (-not (Test-Path $VenvPath)) {
    & $pyExe @pyArgs -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) { Fail "venv creation failed" }
} else {
    Write-Host "    reusing the existing environment"
}

$venvPy = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $venvPy)) { Fail "$venvPy not found - is $VenvPath a virtual environment?" }

& $venvPy -m pip install --quiet --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { Fail "could not upgrade pip" }

# --------------------------------------------------------------- pytorch ----
if (-not $SkipTorch) {
    Write-Step "Installing PyTorch for ROCm (several GB, this takes a while)"
    $installArgs = @("scripts\install_rocm_torch.py")
    if ($Family) { $installArgs += @("--family", $Family) }
    if ($RocmVersion) { $installArgs += @("--rocm-version", $RocmVersion) }
    & $venvPy @installArgs
    if ($LASTEXITCODE -ne 0) {
        Fail @"
PyTorch installation failed.
       Run '$venvPy scripts\install_rocm_torch.py --list' to see the builds on offer,
       then retry with:  -RocmVersion THE_BUILD_ID
"@
    }
} else {
    Write-Step "Skipping PyTorch (-SkipTorch)"
}

# ---------------------------------------------------------- dependencies ----
Write-Step "Installing the remaining dependencies"
& $venvPy -m pip install -r requirements-rocm.txt
if ($LASTEXITCODE -ne 0) { Fail "dependency installation failed" }

# ----------------------------------------------------------------- check ----
Write-Step "Checking the installation"
& $venvPy scripts\check_env.py
$checkResult = $LASTEXITCODE

Write-Host ""
if ($checkResult -eq 0) {
    Write-Host "Setup complete." -ForegroundColor Green
} else {
    Write-Host "Setup finished, but some checks failed - see above." -ForegroundColor Yellow
}
Write-Host ""
Write-Host "  Activate:      .\$VenvPath\Scripts\Activate.ps1"
Write-Host "  Single image:  python inference.py --image assets\images\0_img.png --output out.glb"
Write-Host "  Multi-view:    python inference_mv.py --views_dir assets\mv_images\example --output out_mv.glb"
Write-Host "  Web UI:        python app_local.py"
Write-Host ""
Write-Host "  Weights (~20 GB) download from Hugging Face on the first run."
Write-Host ""
exit $checkResult
