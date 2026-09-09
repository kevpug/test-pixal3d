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
    Python executable to build the venv from. Default: py -3.12, then python.

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

$ErrorActionPreference = "Stop"
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
if (-not $Python) {
    # 3.12 first: AMD publishes cp312/cp313/cp314 Windows wheels, and every
    # CPU dependency here (xatlas, fast-simplification, opencv) ships cp312.
    foreach ($candidate in @("py -3.12", "py -3.13", "python")) {
        $parts = $candidate.Split(" ")
        if (Get-Command $parts[0] -ErrorAction SilentlyContinue) {
            $version = & $parts[0] $parts[1..($parts.Length - 1)] -c "import sys;print('%d.%d'%sys.version_info[:2])" 2>$null
            if ($LASTEXITCODE -eq 0 -and $version) {
                $Python = $candidate
                Write-Host "    Python $version ($candidate)"
                break
            }
        }
    }
}
if (-not $Python) {
    Fail "No Python found. Install Python 3.12 from python.org (tick 'Add to PATH')."
}

$pyParts = $Python.Split(" ")
$pyExe = $pyParts[0]
$pyArgs = if ($pyParts.Length -gt 1) { $pyParts[1..($pyParts.Length - 1)] } else { @() }

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
