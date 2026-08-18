"""Environment smoke tests.

Verifies the pieces every later stage depends on, cheapest first, so a failure is
diagnosed here rather than inside a training run:

  deps     installed versions, and whether torch's CUDA build matches the driver
  api      the lerobot import paths this project builds against
  mujoco   MuJoCo headless rendering through EGL
  libero   LIBERO env construction, stepping, and MuJoCo state get/set
           (state injection is what the perturbation experiments require)

Usage::

    python scripts/smoke_test.py            # everything
    python scripts/smoke_test.py --quick    # skip the slow LIBERO env build
    python scripts/smoke_test.py --only mujoco libero
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env_setup import setup_env  # noqa: E402

# Both must precede any lerobot/mujoco/libero import: EGL selects headless rendering,
# and without a LIBERO config the `libero` import blocks on an interactive prompt.
os.environ.setdefault("MUJOCO_GL", "egl")
setup_env()


def _try_import(path: str):
    try:
        return importlib.import_module(path)
    except Exception as exc:  # noqa: BLE001 - this is a reporting tool
        return exc


# --------------------------------------------------------------------------- checks


def check_deps() -> bool:
    """Installed versions, plus the torch/driver CUDA match."""
    ok = True
    for name in ("lerobot", "torch", "torchvision", "torchcodec", "mujoco", "robosuite", "gymnasium", "libero"):
        mod = _try_import(name)
        if isinstance(mod, Exception):
            print(f"  {name:12s} MISSING ({type(mod).__name__})")
            ok = False
        else:
            print(f"  {name:12s} {getattr(mod, '__version__', 'installed')}")

    import torch

    if torch.cuda.is_available():
        print(f"\n  CUDA available   yes — {torch.cuda.get_device_name(0)} (sm_%d%d)" % torch.cuda.get_device_capability(0))
        free, total = torch.cuda.mem_get_info()
        print(f"  GPU memory       {free / 1e9:.2f}GB free / {total / 1e9:.2f}GB total")
    else:
        # Not fatal: MuJoCo renders through EGL/OpenGL, not CUDA, so the sim-side
        # checks below still pass. But training would silently run on CPU.
        print(
            f"\n  CUDA available   NO — torch built for CUDA {torch.version.cuda}, "
            "which does not match the installed driver.\n"
            "                   Simulation still works; training would fall back to CPU.\n"
            "                   See the GPU / CUDA section of README.md."
        )
    return ok


def check_api() -> bool:
    """The lerobot import paths this project depends on."""
    ok = True
    for path, wanted in (
        ("lerobot.datasets.lerobot_dataset", ("LeRobotDataset", "LeRobotDatasetMetadata")),
        ("lerobot.envs", ("make_env", "make_env_config")),
        ("lerobot.envs.configs", ("LiberoEnv",)),
    ):
        mod = _try_import(path)
        if isinstance(mod, Exception):
            print(f"  {path}: FAILED ({type(mod).__name__}: {mod})")
            ok = False
            continue
        missing = [n for n in wanted if not hasattr(mod, n)]
        if missing:
            print(f"  {path}: missing {missing}")
            ok = False
        else:
            print(f"  {path}: {', '.join(wanted)}")
    return ok


def check_mujoco() -> bool:
    """Headless rendering: a non-uniform frame of the expected shape."""
    import mujoco

    print(f"  MUJOCO_GL={os.environ['MUJOCO_GL']}")
    xml = """
    <mujoco>
      <worldbody>
        <light pos="0 0 3"/>
        <geom name="floor" type="plane" size="2 2 0.1" rgba="0.3 0.4 0.5 1"/>
        <body pos="0 0 1">
          <joint type="free"/>
          <geom type="box" size="0.2 0.2 0.2" rgba="0.9 0.3 0.2 1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=256, width=256)
    for _ in range(50):
        mujoco.mj_step(model, data)
    renderer.update_scene(data)
    frame = renderer.render()

    print(f"  frame {frame.shape} {frame.dtype}, pixel range {frame.min()}..{frame.max()}")
    if frame.shape != (256, 256, 3):
        print("  unexpected frame shape")
        return False
    if frame.max() == frame.min():
        print("  frame is uniform — nothing was rendered")
        return False
    return True


def check_libero() -> bool:
    """Build a task env, step it, and confirm simulator state can be read and written.

    The state get/set path is what the perturbation experiments rely on: without it,
    perturbations would have to be approximated with action noise.
    """
    import numpy as np
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    task = suite.get_task(0)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    print(f"  task: {task.language}")

    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    try:
        env.seed(0)
        obs = env.reset()

        for key in ("agentview_image", "robot0_eye_in_hand_image"):
            if key not in obs:
                print(f"  missing observation key: {key}")
                return False
            print(f"  {key}: {obs[key].shape} {obs[key].dtype}")

        if env.env.action_dim != 7:
            print(f"  unexpected action dim: {env.env.action_dim}")
            return False
        print(f"  action_dim: {env.env.action_dim}, control_timestep: {env.env.control_timestep}")

        for _ in range(5):
            obs, reward, done, _info = env.step(np.zeros(7))
        print(f"  stepped 5x — reward={reward} done={done}")

        state = env.get_sim_state()
        env.set_init_state(state.copy())
        print(f"  sim state: {state.shape} — get/set OK, perturbation injection supported")
    finally:
        env.close()
    return True


CHECKS = {
    "deps": ("Dependencies and CUDA", check_deps),
    "api": ("lerobot API surface", check_api),
    "mujoco": ("MuJoCo headless rendering", check_mujoco),
    "libero": ("LIBERO env and state injection", check_libero),
}
SLOW = {"libero"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help=f"skip slow checks ({', '.join(sorted(SLOW))})")
    parser.add_argument("--only", nargs="+", choices=list(CHECKS), metavar="CHECK",
                        help=f"run only these checks: {', '.join(CHECKS)}")
    args = parser.parse_args()

    selected = args.only or [name for name in CHECKS if not (args.quick and name in SLOW)]

    results: dict[str, bool] = {}
    for name in selected:
        title, fn = CHECKS[name]
        print(f"\n=== {title} ===")
        try:
            results[name] = fn()
        except Exception:
            traceback.print_exc()
            results[name] = False

    print("\n" + "=" * 48)
    for name in selected:
        print(f"  {'PASS' if results[name] else 'FAIL'}  {CHECKS[name][0]}")
    skipped = [n for n in CHECKS if n not in selected]
    if skipped:
        print(f"  ....  skipped: {', '.join(skipped)}")

    failed = [n for n in selected if not results[n]]
    print("=" * 48)
    print("FAILED" if failed else "ALL PASSED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
