# Dual-System VLA for LIBERO

A dual-system vision-language-action architecture evaluated on the LIBERO manipulation
benchmark: a slow semantic **System 2** conditions a fast reactive **System 1** through a
single continuous latent vector, trained end-to-end.

## Setup

### Requirements

- **Linux** — LIBERO/MuJoCo simulation is Linux-only (`sys_platform == 'linux'`)
- **Python ≥ 3.12** — required by `lerobot` 0.6.x
- NVIDIA GPU with working driver (EGL headless rendering)

### Install

This project uses [`uv`](https://docs.astral.sh/uv/). It fetches the correct Python
version automatically, so no system Python upgrade or conda is needed.

```bash
# from the repo root
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install "lerobot[libero]"
```

### GPU / CUDA

Check that the installed torch matches your driver:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`lerobot` pulls a CUDA 13 build of torch by default. That requires a driver from the
580 series or newer — on an older driver `torch.cuda.is_available()` silently returns
`False` and everything falls back to CPU. Install a matching build instead:

```bash
# CUDA 12.8 build (works on 12.x drivers; required for Blackwell / RTX 50-series)
uv pip install --index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.11.0+cu128" "torchvision==0.26.0+cu128"
```

RTX 50-series (Blackwell, `sm_120`) needs CUDA 12.8 or newer — a `cu126` build will
not run on it.

### Rendering backend

MuJoCo needs an explicit rendering backend, chosen per use:

```bash
export MUJOCO_GL=egl    # headless: servers, training, evaluation
export MUJOCO_GL=glfw   # on-screen: the interactive viewer
```

Every headless script below assumes `egl`. The GUI script sets `glfw` itself.

## Verifying the install

One script checks everything later stages depend on — dependency versions and the
torch/driver CUDA match, the lerobot API surface, MuJoCo headless rendering, and LIBERO
env construction with simulator state injection. A failure here is much cheaper to
diagnose than the same failure inside a training run.

```bash
python scripts/smoke_test.py                    # everything
python scripts/smoke_test.py --quick            # skip the slow LIBERO env build
python scripts/smoke_test.py --only mujoco      # a single check
```

It exits non-zero if any check fails, so it can gate a setup script or CI.