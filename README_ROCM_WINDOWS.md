# Pixal3D on Windows with an AMD GPU (ROCm)

This branch runs Pixal3D on AMD hardware without compiling anything — including
RDNA2 laptop cards like the **RX 6800M (gfx1031, 12 GB)**.

Upstream Pixal3D inherits TRELLIS.2's native stack. None of it is available
here, so all of it has been replaced:

| Upstream dependency | Why it cannot be used | Replacement |
|---|---|---|
| `flash_attn` | No ROCm build for RDNA, none at all for Windows | PyTorch SDPA, chunked over queries |
| `flex_gemm` | Triton-based; AMD ships no Triton for Windows | Pure-torch submanifold sparse conv |
| `o_voxel` | CUDA C++ | Pure-torch dual-grid extraction and hash lookup |
| `cumesh` | CUDA C++ | `trimesh` + `fast-simplification` + `xatlas` |
| `nvdiffrast` | CUDA-only; the ROCm port miscomputes coverage on wave32 | Pure-torch UV rasteriser |

Everything is selected automatically: install a native extension and it is used,
leave it out and the fallback runs. On a normal NVIDIA machine every native
extension is picked up and the model path is upstream's; the only additions that
still apply there are the speed and VRAM controls (`--vram`, `--dtype`,
`--cond_dtype`, `--cfg_batch`, `--timing`), and `--cond_dtype float32` turns the
one numerics change — bfloat16 autocast on the frozen conditioning encoders —
back off.

---

## Install

**Requirements**

- Windows 10/11, 64-bit
- An AMD RDNA2 or newer GPU (RX 6000 / 7000 / 9000, or Ryzen AI Max)
- **A recent AMD Adrenalin driver**, which carries the HIP runtime the wheels
  load at import. Note that 26.1.1 specifically is named in several
  access-violation reports
  ([ROCm/ROCm#5871](https://github.com/ROCm/ROCm/issues/5871)); if the GPU
  stack crashes, changing the driver is a real lever in both directions
- **The latest [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)**
  — the HIP DLLs need it, and a missing one faults rather than erroring
- [Python 3.12 or 3.13](https://www.python.org/downloads/) with "Add to PATH"
  ticked. AMD publishes ROCm wheels for cp312/cp313/cp314 only, so 3.11 and
  earlier cannot work
- [Git for Windows](https://git-scm.com/download/win) - MoGe-2 and its
  utils3d fork install from git URLs
- ~40 GB free: ~20 GB of weights, plus the ROCm SDK

**Then**

```bat
git clone <your-fork-url> Pixal3D
cd Pixal3D
setup_rocm_windows.bat
```

That creates `.venv`, installs AMD's ROCm build of PyTorch for your GPU family,
installs the remaining dependencies, and runs the environment check. It takes a
while — the ROCm wheels are several gigabytes.

It is plain `cmd`, with no PowerShell step: PowerShell adds an execution policy,
a script encoding that Windows PowerShell 5.1 misreads without a BOM, and a rule
that turns any native command's stderr into a terminating error — and both
`pip` and `py.exe` write to stderr routinely. `setup_rocm_windows.ps1` is kept
as an alternative for people who prefer it, but the `.bat` is the tested path.

Options:

```bat
setup_rocm_windows.bat --family gfx110X-dgpu      REM force the GPU family
setup_rocm_windows.bat --rocm-version 7.13.0a20260421
setup_rocm_windows.bat --python "C:\Path\To\python.exe"
setup_rocm_windows.bat --skip-torch               REM keep an existing torch
```

**If you already run image models on this GPU**

Windows AMD setups for Stable Diffusion / ComfyUI usually use one of three
stacks, and only one of them works here. `check_env.bat` names yours:

| Your stack | Works? |
|---|---|
| **ROCm** (`torch.version.hip` set) | Yes — this is the target |
| **ZLUDA** (a CUDA torch build on an AMD card) | Probably, untested. The gfx target is inferred from the device name; native ROCm is the safer path |
| **DirectML** (`torch-directml`) | **No.** It is a separate backend from `torch.cuda`, and the sparse ops here need HIP |

DirectML and ROCm can coexist — put this in its own venv, which
`setup_rocm_windows.bat` does anyway. Your existing setup is untouched.

**Verify**

```bat
check_env.bat      REM torch build, GPU visibility, backends, device self-test
gpu_probe.bat      REM where the GPU stack dies, if it does
fix_gpu.bat        REM tries every known fix for a GPU that will not enumerate
adopt_env.bat      REM finds a setup that already works here and reuses it
run_tests.bat      REM checks the fallbacks against independent references
run_benchmark.bat  REM measures this GPU and says which speed flags to use
```

`check_env.bat` and `gpu_probe.bat` each save their full output next to the
script (`check_env_report.txt`, `gpu_probe_report.txt`), because the
interesting part is near the top and a console window scrolls it away.

`run_tests.bat` is worth running once. It compares the sparse convolution
against a dense `F.conv3d`, the volume sampler against a Python reference, and
the dual-grid extractor against a brute-force lookup — so it tells you the
fallbacks are *correct on your GPU*, not merely that they run.

---

## Use

```bat
run_image.bat assets\images\0_img.png output.glb
run_gui.bat
```

or, with the environment activated (`.venv\Scripts\activate`):

```bat
python inference.py --image assets\images\0_img.png --output output.glb
python inference_mv.py --views_dir assets\mv_images\example --output output_mv.glb
python app_local.py
```

Weights (~20 GB) download from Hugging Face on the first run and are cached in
`%USERPROFILE%\.cache\huggingface`.

### Multi-view

`inference_mv.py` conditions on several posed views at once. Point it at a
directory holding `transforms.json` and the images:

```
my_views/
├── transforms.json
├── view00_azim000.png
├── view01_azim090.png
├── view02_azim180.png
└── view03_azim270.png
```

Views without an alpha channel are matted automatically. If your shots are the
usual four-way orbit at eye level, copy
`assets/mv_images/example/transforms.json` and just change the `file_path`
fields — it already encodes that rig. **Frame 0 is the main view** and should be
the canonical front view; `inference_mv.py` warns if it is not.

---

## VRAM

`--vram` sets resolution, token cap, texture size, decimation target and
offloading together. It defaults to `auto`, which reads your card's capacity.

| Preset | Card | Resolution | Texture | Faces | Notes |
|---|---|---|---|---|---|
| `max` | 24 GB+ | 1536 | 4096 | 1 M | everything resident, batched CFG |
| `16gb` | 16 GB | 1536 | 4096 | 1 M | per-stage CPU offload, batched CFG |
| `12gb` | 12 GB | 1024 | 2048 | 300 k | **RX 6800M** |
| `8gb` | 8 GB | 1024 | 2048 | 250 k | streams DiT blocks |
| `6gb` | 6 GB | 1024 | 1024 | 150 k | streams DiT blocks |

```bat
python inference.py --image in.png --output out.glb --vram 12gb
python inference.py --image in.png --output out.glb --vram 8gb
```

Any single knob can be overridden:

```bat
python inference.py --image in.png --output out.glb ^
    --vram 12gb --texture_size 4096 --decimate 800000
```

**If a run dies with an out-of-memory error**, in order of impact:

1. Drop a preset (`--vram 8gb`, then `6gb`).
2. `--block_offload` — streams each DiT block onto the GPU only for its forward
   pass. Weight residency falls from ~2.6 GB to about one block; it costs time,
   not quality.
3. `--max_num_tokens 8192` — the strongest lever on peak activation memory.
4. `--texture_size 1024` and a lower `--decimate`, which affect only the export.
5. `set PIXAL3D_ATTN_BUDGET=16777216` — halves the attention working set.

Windows reserves part of the card for the desktop, so a 12 GB GPU has roughly
10.5–11 GB usable. Closing browsers and other GPU applications genuinely helps.

---

## Speed

**Measure before you tune.** The two things that decide how long a run takes are
whether your GPU has bfloat16 hardware and whether PyTorch has a fused attention
kernel for it, and both vary by architecture:

```bat
.venv\Scripts\python.exe scripts\benchmark.py
```

It prints fp16 vs bf16 matmul throughput, which SDPA backend torch will pick,
and sparse-convolution throughput, then says which flags to set. Add `--timing`
to a real run to see where the wall clock actually went:

```bat
python inference.py --image in.png --output out.glb --vram 12gb --timing
```

### Expectations

On an RX 6800M at `--vram 12gb`, budget **10-25 minutes** per model with default
settings, and roughly half that with the flags below. Two structural facts set
that floor, and neither is a bug in this port:

- **RDNA2 has no bfloat16 instructions.** A bf16 matmul is emulated through
  fp32, at about half the fp16 rate. `--dtype auto` therefore converts the DiT
  torsos to float16 on RDNA1/2. On RDNA3, RDNA4 and CUDA it changes nothing.
- **AOTriton ships no attention kernels for gfx103x.** There is no flash or
  memory-efficient SDPA on RDNA2 at all, so torch uses the math path, which
  materialises the full score matrix. `check_env.py` says which case you are in.
  RDNA3/RDNA4 cards do get fused attention, and are correspondingly faster.

### The levers, in order of payoff

| Flag | Effect | Cost |
|---|---|---|
| `--dtype float16` | ~2x on the DiTs where the GPU has no bf16 hardware | slightly less exponent range |
| `--steps 8` | ~33% off all sampling (default is 12 per stage) | marginally softer detail |
| `--resolution 1024` | ~2x versus 1536 | less geometric detail |
| `--decimate 200000` | export decimation and UV unwrap are CPU-bound and superlinear | fewer faces |
| `--texture_size 1024` | quarter of the bake and inpaint work | coarser texture |
| `--max_num_tokens 8192` | fewer sparse tokens in the HR stage | less detail |
| `--cfg_batch` | one batched forward per step instead of two | ~2x peak activation memory |
| `--tex_steps 6` | texture converges faster than shape | slightly flatter PBR |

A good starting point for a 12 GB laptop:

```bat
python inference.py --image in.png --output out.glb ^
    --vram 12gb --dtype float16 --steps 8 --texture_size 1024 --timing
```

`--cfg_batch` is on by default only for the `max` and `16gb` presets; it halves
the number of kernel launches per denoising step but roughly doubles activation
memory, which is the wrong trade below 16 GB. Try it with `--vram 12gb` if you
have headroom.

### Things that are already on

You do not need to ask for these; they are defaults:

- The four DINOv3 conditioning extractors share one backbone and one NAF
  upsampler instead of loading four copies (`PIXAL3D_SHARE_IMAGE_BACKBONE=0`
  restores the old behaviour).
- Weight initialisation is skipped at load time, since every tensor is
  immediately overwritten by the checkpoint (`PIXAL3D_FAST_INIT=0` restores it).
- The conditioning encoders run under autocast at the same precision as the
  DiTs (`--cond_dtype float32` opts out).
- Sparse convolution gathers all 27 kernel taps into a single GEMM per block
  rather than 27 small ones.
- Attention is query-chunked, which on the math path cuts the score matrix from
  O(N^2) resident to a bounded working set.

---

## ComfyUI

Clone this repository into `ComfyUI\custom_nodes\` and install the dependencies
into ComfyUI's own Python — see [`comfyui/README.md`](comfyui/README.md). Four
nodes appear under **Pixal3D**: pipeline loader, image-to-3D, multi-view-to-3D,
and GLB export. They are split so that changing a seed re-runs only sampling and
changing the texture size re-runs only the export.

ComfyUI must already be on a ROCm build of PyTorch. `scripts/check_env.py`, run
with ComfyUI's interpreter, will say whether it is.

---

## Environment variables

| Variable | Effect |
|---|---|
| `ATTN_BACKEND` | `sdpa`, `flash_attn`, `xformers`, `naive`. Default: fastest installed. |
| `SPARSE_CONV_BACKEND` | `torch`, `flex_gemm`, `spconv`, `torchsparse`. Default: fastest installed. |
| `PIXAL3D_ATTN_BUDGET` | Attention score elements held at once. Lower for less VRAM. |
| `PIXAL3D_ATTN_CHUNK` | Fixed query-chunk size; `0` disables chunking. |
| `PIXAL3D_FORCE_FALLBACK` | `1` ignores native extensions — for A/B testing. |
| `PIXAL3D_SHARE_IMAGE_BACKBONE` | `0` loads four separate DINOv3 copies instead of sharing one. |
| `PIXAL3D_FAST_INIT` | `0` restores random weight init at load (slower, no effect on output). |
| `PIXAL3D_CFG_BATCH` | `1` runs both guidance branches in one batched forward. |
| `PIXAL3D_TIMING` | `1` prints a per-stage wall-clock breakdown. |
| `HSA_OVERRIDE_GFX_VERSION` | Reports the GPU as another target, e.g. `10.3.0` to run gfx1031/gfx1032 on gfx1030 kernels. Only if the card is otherwise invisible. |

---

## Troubleshooting

**`no GPU visible to torch`** — the driver is older than 26.1.1, or the ROCm SDK
wheels do not match the torch wheel's build date. Reinstall with a pinned build:

```bat
.venv\Scripts\python.exe scripts\install_rocm_torch.py --list
.venv\Scripts\python.exe scripts\install_rocm_torch.py --rocm-version 7.13.0a20260421
```

**Installed before September 2026, or `ACCESS_VIOLATION` during device
enumeration** — the installer used to pull from
`rocm.nightlies.amd.com/v2-staging/<family>/`, whose `gfx103X-dgpu` bundle
crashes on gfx1031. AMD's current channel is
`nightly.repo.amd.com/rocm/whl-next/`, which ships a package per gfx
architecture (`amd-torch-device-gfx1031` and so on) rather than one bundle per
family. Just reinstall:

```bat
.venv\Scripts\python.exe scripts\install_rocm_torch.py
```

`--arch gfx1031` forces a target if detection gets it wrong; `--legacy` goes
back to the old index.

**`ACCESS_VIOLATION (0xC0000005)` from `gpu_probe.bat`** — the HIP runtime is
crashing, which happens before any GPU kernel is compiled or launched, so this
is not a missing-code-object problem. It is a version disagreement between the
wheels, the driver, and the C++ runtime. Run:

```bat
fix_gpu.bat
```

It works through the whole sequence unattended, re-testing after each step and
stopping at the first thing that works: test what is installed, sweep the HIP
environment variables, check the Visual C++ runtime, then reinstall a ROCm
build known to work for your GPU family, falling back through older ones.
Anything it finds is written to `pixal3d_env.bat`, which every other `.bat`
here picks up automatically, so the fix sticks.

From step 4 it downloads several GB per attempt. `fix_gpu.bat --dry-run` shows
what it would do first; `fix_gpu.bat --deep 4` tries more builds. The
individual pieces are still there if you want them by hand:

```bat
gpu_probe.bat --sweep        REM tries 13 env configurations, one process each
python scripts\install_rocm_torch.py --known-good
```

`--known-good` pins the build the community reports as working on RDNA2
(`7.12.0a20260204`) rather than the newest nightly. Newest is not safest here:
HIP SDK 7.1.1 stopped detecting gfx1030 altogether
([ROCm/hip#3899](https://github.com/ROCm/hip/issues/3899)), and 7.13 nightlies
have broken whole families
([ROCm/TheRock#5543](https://github.com/ROCm/TheRock/issues/5543)). Then
install the Visual C++ Redistributable, and try a different Adrenalin driver.

**Two AMD adapters (a laptop with an iGPU, or a Ryzen APU plus a card)** — this
is the most common cause of both enumeration crashes and
`hipErrorInvalidImage` on this stack. The runtime enumerates every AMD agent,
the `gfx103X-dgpu` wheels carry no kernels for an integrated part, and it dies
before your discrete card is used. `HIP_VISIBLE_DEVICES` does not reliably
help, because the damage is done during enumeration.

Disable the integrated GPU and retry:

> Device Manager > Display adapters > right-click the integrated Radeon >
> **Disable device**

Reversible, no reboot needed. (Disabling it in the BIOS works too, and is what
the [ROCm ComfyUI fork](https://github.com/patientx-cfz/comfyui-rocm)
tells people to do before installing.) `gpu_probe.bat` and `fix_gpu.bat` now
say when they see more than one AMD adapter.

**If ComfyUI or another app already drives this GPU but Pixal3D cannot**, the
hardware and driver are fine and the difference is the stack. On Windows AMD
there are three, and they are not interchangeable:

| Stack | How to tell | Notes |
|---|---|---|
| **ROCm wheels** | `torch.version.hip` set | what Pixal3D installs |
| **ZLUDA** | `torch.version.cuda` set, device is a Radeon | a CUDA torch plus a translation layer over the system HIP SDK |
| **DirectML** | neither; `torch_directml` present | cannot run this pipeline |

ZLUDA is common on RDNA2 precisely because the official ROCm nightlies have
been unreliable there — which is what an access violation across several
builds looks like. Run:

```bat
adopt_env.bat
```

It finds the interpreter that works, says which of the three it is, lists the
exact packages behind it, and prints how to reuse it. Pixal3D supports ZLUDA;
the usual answer is to install Pixal3D's dependencies into that environment
and run from there, adding `--dtype float16` since ZLUDA reports no gfx target
for `auto` to read.

If nothing on the machine drives the GPU either, Windows ROCm on RDNA2 is
simply not reliable right now, and the honest alternatives are dual-booting
Linux (where gfx1031 works with `HSA_OVERRIDE_GFX_VERSION=10.3.0`) or setting
up ZLUDA.

**No output at all about the GPU, and `python -c "import torch;
print(torch.cuda.device_count())"` prints nothing either** — the process is
being killed, not failing. If the HIP runtime hits an access violation or a
missing DLL while enumerating devices, Windows terminates the interpreter:
there is no exception to catch and nothing on stdout, so any diagnostic that
touches `torch.cuda` in its own process dies at the same line. Run:

```bat
gpu_probe.bat
gpu_probe.bat --verbose      REM adds the HIP runtime's own log
```

Each step runs in a separate process, so the crash is reported with its
Windows exit code (`0xC0000005` access violation, `0xC0000135` missing DLL,
`0xC0000139` wrong DLL version) instead of taking the probe down with it. That
code is the diagnosis: a missing or mismatched DLL means the ROCm SDK wheels
and the torch wheel are from different build dates, and an access violation
usually means the driver does not match the wheels or the GPU's gfx target has
no code objects in them.

**`no GPU visible to torch` but your other AI apps work** — check which stack
those apps use. `check_env.bat` prints it. DirectML and ZLUDA installs do not
give you a ROCm torch; they need a separate venv with AMD's wheels.

**The card is a 6700/6750/6800M/6600 and still is not visible** — those are
gfx1031/gfx1032, and a wheel may only carry gfx1030 code objects. Try:

```bat
set HSA_OVERRIDE_GFX_VERSION=10.3.0
python inference.py --image in.png --output out.glb
```

RDNA2 is binary-compatible enough for this to work in practice. If you get
wrong output rather than an error, unset it — that means it was not compatible.

**The first run seems frozen for several minutes** — that is usually MIOpen
compiling convolution kernels for your GPU, which happens once and is then
cached (`%USERPROFILE%\.miopen`). The conditioning encoders, background removal
and MoGe all use ordinary convolutions, so they hit it. Later runs skip it.
`--timing` will show the cost landing in whichever stage ran first.

**`No torch wheels for cpXXX/win_amd64`** — AMD publishes cp312, cp313 and cp314
Windows builds. Use Python 3.12.

**`ResolutionImpossible` mentioning `moge` and `trimesh`** — you are on an old
`requirements-rocm.txt`. MoGe v3 declares `flex-gemm` (CUDA/Triton, which is
exactly what cannot build here), `opencv-python` (the same `cv2` package as our
headless build) and `gradio>=6`, so it is no longer resolved with everything
else. Setup installs it on its own:

```bat
.venv\Scripts\python.exe -m pip install --no-deps ^
    git+https://github.com/microsoft/MoGe.git@74fbce054ebed49800de42d0ad0e83495065719a
```

Only one class is used from it, `MoGeModel` from `moge.model.v2`, whose whole
import graph is torch, numpy, huggingface_hub and `utils3d_moge` — and that
fork is installed normally, under its own distribution name so it cannot
collide with the `utils3d` Pixal3D pins. MoGe is optional: without it, pass the
camera FOV yourself with `--fov 0.2`.

**`git is not on PATH`** — MoGe-2 (camera estimation) installs from a git URL.
Install Git for Windows and re-run setup. To skip MoGe entirely, pass an
explicit field of view: `--fov 0.2`.

**Wrong GPU family detected** — override it:

```bat
setup_rocm_windows.bat -Family gfx110X-dgpu
```

`gfx103X-dgpu` = RX 6000, `gfx110X-dgpu` = RX 7000, `gfx120X-all` = RX 9000,
`gfx1151` = Ryzen AI Max.

**Black patches on the texture** — raise `--texture_size`, or `--decimate` less
aggressively. The bake samples the attribute volume at the decimated surface;
without `cumesh` there is no BVH to project back onto the original one, so heavy
decimation eventually shows.

**Everything runs but is very slow** — check `check_env.bat` says
`attention: sdpa` and `sparse conv: torch`, not `naive`. Then confirm the GPU
is actually being used: `[ok] device 0: ...` in the same output.

---

## What is different from upstream

Changes are additive; the CUDA path is untouched.

- **`pixal3d/compat/`** — the fallbacks, with a `dispatch` layer that picks
  native or portable per capability and prints which it chose.
- **`pixal3d/modules/sparse/conv/conv_torch.py`** — submanifold sparse
  convolution in the `flex_gemm` checkpoint layout, so weights load either way.
- **`pixal3d/runtime.py`** — VRAM presets, allocator setup, block offloading.
- **Attention** — every SDPA path is chunked over queries. Without this, a ROCm
  build with no fused kernel materialises the whole score matrix, and the shape
  stages run tens of thousands of tokens.
- **Backend selection** — `ATTN_BACKEND` and `SPARSE_CONV_BACKEND` now fall back
  to what is installed instead of hard-failing on `flash_attn`.
- **Shared DINOv3** — the four conditioning extractors share one frozen backbone
  (~3.5 GB saved).
- **Precision control** — `--dtype` converts the DiT torsos, `--cond_dtype`
  autocasts the conditioning encoders. `auto` picks float16 on GPUs with no
  bfloat16 hardware, where a bf16 matmul is emulated through fp32.
- **Batched classifier-free guidance** — `--cfg_batch` runs the conditional and
  unconditional branches as one batch-2 forward, reusing a cached batched
  skeleton so the windowed-attention serialisation is not recomputed per step.
- **`pixal3d/profiling.py`** — `--timing` prints a per-stage breakdown, and
  `scripts/benchmark.py` measures the GPU so tuning is not guesswork.
- **`app_local.py`**, **`comfyui/`**, **`tests/`**, **Windows setup scripts**.

### Known limitations

- **Narrow-band remeshing is skipped.** `remesh_narrow_band_dc` lives entirely
  in `cumesh`'s CUDA kernels. The export uses the decimation branch, which is
  what upstream itself runs with `remesh=False`.
- **No back-projection onto the pre-decimation surface.** That needs `cumesh`'s
  BVH. Colours are sampled at the decimated surface instead — a sub-voxel shift
  at sane decimation targets.
- **`app.py` (the Hugging Face Spaces demo) does not run here.** It imports
  `spaces` and renders previews with nvdiffrast. Use `app_local.py`.
- **Training is untested on this path.** The fallbacks are differentiable and
  should work, but they were built and validated for inference.

---

## Linux

If you dual-boot, ROCm on Linux is faster: Triton exists there, so `flex_gemm`
builds, and `cumesh` and `o_voxel` hipify cleanly. For gfx1031 set
`HSA_OVERRIDE_GFX_VERSION=10.3.0` so the runtime treats it as gfx1030.
[Lamothe/TRELLIS.2_rocm](https://github.com/Lamothe/TRELLIS.2_rocm) carries a
HIP port of `o_voxel` for the same upstream code, and
[yuripourre/Pixal3D-ROCm](https://github.com/yuripourre/Pixal3D-ROCm) is a
Linux-only ROCm fork (gfx1201) that keeps the native extensions.

This branch runs there too and picks up any native extension you manage to
build, so the two approaches compose.
