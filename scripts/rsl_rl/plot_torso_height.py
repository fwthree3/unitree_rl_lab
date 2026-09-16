# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run A progress check: roll out a checkpoint (read-only, doesn't touch a concurrent
training process) and plot torso_link's world-frame z height over time, overlaid across
episodes. Quick visual/quantitative proxy for gait quality -- a healthy periodic bob at
a sane height suggests a normal walking/running gait; a flat, sagging, or erratic trace
suggests the policy hasn't converged (or is faking speed without a real gait).
"""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Plot a body's world z-height over N policy rollout episodes.")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-Velocity", help="Name of the task.")
parser.add_argument("--body_name", type=str, default="torso_link", help="Body to read the z-height at.")
parser.add_argument("--target_height", type=float, default=0.78, help="Reference height line (m), from the base_height reward's target_height param.")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to roll out (single env).")
parser.add_argument("--out", type=str, default="torso_height.png", help="Output plot path.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from rsl_rl.runners import OnPolicyRunner

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
import gymnasium as gym


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, entry_point_key="play_env_cfg_entry_point")
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading checkpoint from: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    robot = env.unwrapped.scene["robot"]
    body_idx = robot.body_names.index(args_cli.body_name)
    dt = env.unwrapped.step_dt

    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]

    episodes = []  # list of (times: list[float], heights: list[float]), one entry per completed episode
    cur_times, cur_heights = [], []
    episode_count = 0
    all_heights = []
    t = 0.0

    with torch.inference_mode():
        while episode_count < args_cli.num_episodes:
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

            height = robot.data.body_pos_w[0, body_idx, 2].item()

            cur_times.append(t)
            cur_heights.append(height)
            all_heights.append(height)
            t += dt

            if bool(dones[0]):
                episode_count += 1
                print(f"[INFO] episode {episode_count}/{args_cli.num_episodes} done, length={t:.2f}s")
                episodes.append((cur_times, cur_heights))
                cur_times, cur_heights = [], []
                t = 0.0

    cmap = matplotlib.colormaps["viridis"]
    colors = [cmap(i / max(len(episodes) - 1, 1)) for i in range(len(episodes))]

    fig, ax = plt.subplots(figsize=(12, 5))
    for ep_idx, (ep_times, ep_heights) in enumerate(episodes):
        ax.plot(ep_times, ep_heights, color=colors[ep_idx], alpha=0.7, linewidth=1)
    ax.axhline(args_cli.target_height, color="red", linestyle="--", linewidth=1, label=f"target ({args_cli.target_height}m)")
    ax.set_title(f"{args_cli.body_name} world z-height, {len(episodes)} episodes overlaid (aligned to episode start)")
    ax.set_xlabel("time since episode start (s)")
    ax.set_ylabel("height (m)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(vmin=1, vmax=len(episodes)))
    fig.colorbar(sm, ax=ax, label="episode #", ticks=range(1, len(episodes) + 1))
    fig.tight_layout()
    fig.savefig(args_cli.out, dpi=150)
    print(f"[INFO] plot saved to: {os.path.abspath(args_cli.out)}")

    heights_t = torch.tensor(all_heights)
    print(f"\n[RESULT] height over {len(all_heights)} steps / {episode_count} episodes: "
          f"mean={heights_t.mean():.3f}m, std={heights_t.std():.3f}m, min={heights_t.min():.3f}m, max={heights_t.max():.3f}m "
          f"(target {args_cli.target_height}m)")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
