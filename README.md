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
uv sync --extra dev
source .venv/bin/activate
```

### GPU / CUDA

Check that the installed torch matches your driver:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

This should print `2.11.0+cu128` and `True`. If it prints a bare `2.11.0` and `False`,
torch was resolved from PyPI rather than the CUDA 12.8 index.

PyPI serves a **cu130** build, which needs an NVIDIA driver from the 580 series. On
anything older, `torch.cuda.is_available()` returns `False` and everything falls back
to CPU — no exception, no failure, just a training run roughly 50x slower than it
should be. `uv sync` avoids this by installing the pinned `+cu128` build. To repair an
environment that already has the wrong one:

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.11.0+cu128" "torchvision==0.26.0+cu128"
```

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

## Dataset

This study uses **`lerobot/libero_10`** — the LIBERO-10 (long-horizon) suite, and only
that suite. 0.65 GB, 379 episodes, 101,469 frames, 10 tasks at 10 fps.

```bash
python scripts/download_dataset.py
```

The script fetches the slice, then prints the feature shapes, the task list with
per-task episode counts, and the dataset→benchmark id mapping, so the download can be confirmed before anything depends on it. Everything else auto-downloads on first use, but running this explicitly makes the step visible and verifiable.

Files land in `data/lerobot/` inside the repo rather than `~/.cache`, because
`src/env_setup.py` points `HF_LEROBOT_HOME` there. That variable is read once when
`lerobot` is imported, which is why every entry point calls `setup_env()` before
importing it — fetching the dataset any other way puts it in your home cache instead.

### Handling Dataset vs. Benchmark ID mismatch

Unit tests cover the dataset/simulator task mapping:

```bash
python -m pytest tests/ -q
python -m src.utils          # print the dataset-index -> benchmark-id table
```

The dataset's `task_index` and the LIBERO benchmark's task id are two different
orderings over the same tasks, sharing no fixed points — passing one where the other
is expected silently evaluates a different task. `src/utils.py` holds the adapter that
reconciles them, joining on the instruction string.

### Initial simulator states

The LeRobot dataset drops the MuJoCo state each demonstration started from; LIBERO's
original HDF5 demos keep it (`states[0]`), and `scripts/extract_init_states.py`
recovers it via exact action-sequence matching into `data/init_states.npz`.

`data/init_states.npz` and `data/unmatched_episodes_per_task.json` ship pre-extracted
and committed, so **it is not necessary to run the extraction script again** — only
re-run it if you switch to a different dataset or LIBERO suite:

```bash
python scripts/extract_init_states.py                    # all ten tasks (default)
python scripts/extract_init_states.py --task-index 5     # or just one
```

Not every episode has a match (61 of 379, listed in
`data/unmatched_episodes_per_task.json`) — a gap in the dataset's own conversion that
doesn't affect training, only which episodes can be replayed exactly.

### Replaying a recorded demonstration

`scripts/replay_episode.py` steps a dataset episode's recorded actions through the
simulator and reports whether the task is solved and how closely the end-effector
tracks the recording — the ground-truth check that the dataset's action convention
and control rate are right, since a demonstration that can't be replayed means no
policy trained on it could succeed either.

```bash
python scripts/replay_episode.py --episode 8
python scripts/replay_episode.py --episode 8 --viewer            # watch it live
python scripts/replay_episode.py --episode 13 --video out/ep13.mp4
```

A correct replay reaches task success with roughly 0.007 m mean tracking error. An
episode whose initial state was never matched still runs, but starts from the
environment's own reset and will almost certainly fail — the output says so
explicitly.
## Training

Two checkpoints are trained, identical apart from how the latent `z` is produced.
That is the point: any difference between them is attributable to the conditioning
and nothing else.

```bash
# CKPT-DUAL — System 2 conditions System 1 with a live latent
CUDA_VISIBLE_DEVICES=0 nohup python -m src.train \
  --conditioning live \
  --steps 100000 --batch-size 16 --num-workers 8 \
  --output-dir outputs/train > outputs/dual.log 2>&1 &

# CKPT-STATIC — the naive baseline: a static embedding of the instruction
CUDA_VISIBLE_DEVICES=1 nohup python -m src.train \
  --conditioning static \
  --steps 100000 --batch-size 16 --num-workers 8 \
  --output-dir outputs/train > outputs/static.log 2>&1 &
```

Checkpoints, the resolved config and the metrics log are written per run to
`outputs/train/{live,static}/`.

### Before a long run

The first invocation downloads the ~7.5 GB System 2 backbone
(`Qwen/Qwen2.5-VL-3B-Instruct`) into `data/huggingface/`. That is separate from the
dataset and happens once. A short overfitting run gets it out of the way and confirms
the model trains at all:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.train \
  --conditioning live --overfit-episodes 5 --steps 300 --batch-size 8 \
  --output-dir outputs/gate
```

Loss should fall steeply — five episodes are few enough that a working model overfits
hard. If it plateaus, something upstream is wrong and a full run will not fix it.

### Batch size

`--batch-size 16` peaks at ~23.5 GB on a 24 GB card, which fits but leaves little
headroom for fragmentation over a long run. 32 does not fit. If a run OOMs partway,
this usually reclaims enough without changing the optimisation:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 python -m src.train ...
```

Every sample costs a full Qwen-3B forward *and* backward through the LoRA adapters, so
memory scales with batch size far more steeply than the 71.7M System 1 alone suggests.

## Evaluation


```bash
python -m eval.run_eval --checkpoint outputs/train/live/best.pt 
  --conditioning live --init-source benchmark --output-dir outputs/eval/live

# same checkpoint, zero conditioning
python -m eval.run_eval --checkpoint outputs/train/live/best.pt 
  --conditioning zero --init-source benchmark --output-dir outputs/eval/ablation/zero
```

Defaults follow LIBERO's published protocol: 10 trials per task from the benchmark's
own `.pruned_init` states. `--init-source demo` starts from a demonstration's recovered
state instead — training has seen those, so it is for debugging only, never a reported
number. Pass `best.pt` explicitly: a bare run directory resolves to the *latest*
`step_*.pt`, not the best-validation one.

| Modality | Checkpoint | `--conditioning` |
|---|---|---|
| Full dual-system | `train/live` | *(omit — as trained)* |
| Frozen-S2 | `train/live` | `frozen` |
| Naive baseline | `train/static` | *(omit — as trained)* |
| Zero-latent | `train/live` | `zero` |