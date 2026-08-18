"""Replay a recorded demonstration from the LeRobot dataset in the LIBERO simulator.

Answers the question everything downstream depends on: do the dataset's recorded
actions actually drive the simulator to task success? If replay fails, the action
convention or control rate is wrong, and no policy trained on this data could
succeed either.

Requires the episode's initial simulator state, which the dataset does not carry.
Run ``scripts/extract_init_states.py`` once per task to recover it.

Replay approach: reset, optionally force a MuJoCo state, then step the recorded
actions open-loop while an optional passive viewer syncs each step.

No action conversion is applied — LIBERO demonstrations are already robosuite
OSC_POSE-normalised 7-vectors, ready to step directly.

Examples::

    python scripts/replay_episode.py --episode 8
    python scripts/replay_episode.py --episode 8 --viewer
    python scripts/replay_episode.py --episode 8 --video out/replay.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env_setup import setup_env  # noqa: E402

setup_env()

from src.utils import (  # noqa: E402
    DUMMY_ACTION,
    TaskMapping,
    agentview_upright,
    load_episode,
    load_exact_init_state,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episode", type=int, required=True, help="Dataset episode index to replay")
    p.add_argument("--repo-id", default="lerobot/libero_10")
    p.add_argument("--viewer", action="store_true", help="Open a passive MuJoCo viewer (needs a display)")
    p.add_argument("--video", type=Path, help="Write an mp4 of the agentview camera")
    p.add_argument("--action-repeat", type=int, default=1,
                   help="Apply each recorded action N times (default: 1)")
    p.add_argument("--init-state", default="exact", choices=["exact", "none"],
                   help="'exact' uses the episode's true initial simulator state, recovered "
                        "by scripts/extract_init_states.py; 'none' uses the env's own reset "
                        "(the object layout will differ from the recording)")
    p.add_argument("--settle-steps", type=int, default=10,
                   help="No-op steps after reset to let objects settle, matching LiberoEnv's "
                        "num_steps_wait (default: 10)")
    p.add_argument("--realtime", action="store_true", help="Throttle the viewer to wall-clock speed")
    return p.parse_args()


def replay(env, actions, init_state, recorded_state, args) -> tuple[bool, int, float]:
    """Step recorded actions open-loop. Returns (success, steps_taken, mean_tracking_error)."""
    from contextlib import nullcontext

    import numpy as np

    env.reset()
    if init_state is not None:
        env.set_init_state(init_state)
    # Objects can be left slightly unstable by a reset; settle them before replaying,
    # matching what LiberoEnv does at the start of every evaluation episode.
    for _ in range(args.settle_steps):
        env.step(DUMMY_ACTION)

    trace = []
    frames = []
    viewer_ctx = nullcontext(None)
    if args.viewer:
        import mujoco.viewer as mjviewer

        # Reach the raw MjModel/MjData through robosuite's wrapper.
        viewer_ctx = mjviewer.launch_passive(env.env.sim.model._model, env.env.sim.data._data)

    success = False
    steps = 0
    with viewer_ctx as viewer:
        start = time.time()
        for action in actions:
            for _ in range(args.action_repeat):
                obs, _reward, done, _info = env.step(np.asarray(action, dtype=np.float64))
                steps += 1
                trace.append(obs["robot0_eef_pos"].copy())
                if args.video is not None:
                    frames.append(agentview_upright(obs["agentview_image"]))
                if viewer is not None:
                    viewer.sync()
                    if args.realtime:
                        lag = start + steps * env.env.control_timestep - time.time()
                        if lag > 0:
                            time.sleep(lag)
                # check_success latches: a task can be solved mid-episode and then disturbed.
                if env.check_success():
                    success = True
                if done:
                    break
            if done or (viewer is not None and not viewer.is_running()):
                break

    if frames:
        write_video(frames, args.video)
    trace = np.array(trace)
    error = float(np.linalg.norm(trace - recorded_state[:len(trace), :3], axis=1).mean()) if len(trace) else float("nan")
    return success, steps, error


def write_video(frames, path: Path) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), frames, fps=20)
    print(f"  wrote {path} ({len(frames)} frames)")


def main() -> int:
    import numpy as np
    from libero.libero.envs import OffScreenRenderEnv

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    args = parse_args()
    if args.viewer:
        # A passive viewer needs a window; EGL is headless-only.
        os.environ["MUJOCO_GL"] = "glfw"
    else:
        os.environ.setdefault("MUJOCO_GL", "egl")

    dataset = LeRobotDataset(args.repo_id)
    mapping = TaskMapping.from_dataset(dataset)

    episode_index = args.episode
    instruction, actions, recorded_state = load_episode(dataset, episode_index)
    task = mapping.by_episode(dataset, episode_index)

    print(f"\nepisode {episode_index}: {instruction}")
    print(f"  dataset_index={task.dataset_index} -> benchmark_id={task.benchmark_id} "
          f"(suite={task.suite}) steps={len(actions)}")

    init_state = None
    if args.init_state != "none":
        init_state = load_exact_init_state(episode_index)
        if init_state is not None:
            print("  init_state: exact (from the original LIBERO demonstration)")
        else:
            print(f"  init_state: no match for episode {episode_index} (see "
                  "data/unmatched_episodes_per_task.json). Falling back to the env's own "
                  "reset — the object layout will differ from the recording, so the "
                  "replay will very likely fail.")

    env = OffScreenRenderEnv(
        bddl_file_name=task.bddl,
        camera_heights=256,
        camera_widths=256,
        # Demonstrations can exceed the default horizon; replay should not be cut short.
        horizon=len(actions) * args.action_repeat + args.settle_steps + 100,
    )
    try:
        env.seed(0)
        success, steps, error = replay(env, actions, init_state, recorded_state, args)
    finally:
        env.close()

    print(f"  -> {'SUCCESS' if success else 'FAILURE'} after {steps} steps, "
          f"mean eef tracking error {error:.4f} m")
    return 0 if success else 2


if __name__ == "__main__":
    sys.exit(main())
