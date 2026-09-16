# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""M1.11 mechanism check (issue #202), walking variant: roll out a trained checkpoint
(not zero actions -- so the robot actually walks) for a fixed number of episodes and
plot the torso_link incoming joint reaction wrench over time. Same non-NaN/sane check
as check_joint_reaction_force.py, but under real gait loading instead of static stance.
"""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Plot joint reaction force/torque at a body over N policy rollout episodes.")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-Velocity", help="Name of the task.")
parser.add_argument("--body_name", type=str, default="torso_link", help="Body to read the incoming joint wrench at.")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to roll out (single env).")
parser.add_argument("--out", type=str, default="joint_reaction_force.png", help="Output plot path.")
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

    episodes = []  # list of (times: list[float], wrench: list[list[float]]), one entry per completed episode
    cur_times, cur_wrench = [], []
    episode_count, saw_nan, saw_nonzero = 0, False, False
    t = 0.0

    with torch.inference_mode():
        while episode_count < args_cli.num_episodes:
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

            wrench = robot.data.body_incoming_joint_wrench_b[0, body_idx, :].clone()
            if torch.isnan(wrench).any():
                saw_nan = True
            elif torch.count_nonzero(wrench) > 0:
                saw_nonzero = True

            cur_times.append(t)
            cur_wrench.append(wrench.cpu().tolist())
            t += dt

            if bool(dones[0]):
                episode_count += 1
                print(f"[INFO] episode {episode_count}/{args_cli.num_episodes} done, length={t:.2f}s")
                episodes.append((cur_times, cur_wrench))
                cur_times, cur_wrench = [], []
                t = 0.0

    labels = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
    cmap = matplotlib.colormaps["viridis"]
    colors = [cmap(i / max(len(episodes) - 1, 1)) for i in range(len(episodes))]

    fig, axes = plt.subplots(2, 3, figsize=(16, 7), sharex=True)
    for ep_idx, (ep_times, ep_wrench) in enumerate(episodes):
        ep_wrench_t = torch.tensor(ep_wrench)  # (T_ep, 6)
        for ch in range(6):
            ax = axes[ch // 3, ch % 3]
            ax.plot(ep_times, ep_wrench_t[:, ch], color=colors[ep_idx], alpha=0.7, linewidth=1)

    for ch, label in enumerate(labels):
        ax = axes[ch // 3, ch % 3]
        ax.set_title(label)
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("force (N)")
    axes[1, 0].set_ylabel("torque (Nm)")
    for ax in axes[1, :]:
        ax.set_xlabel("time since episode start (s)")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(vmin=1, vmax=len(episodes)))
    fig.colorbar(sm, ax=axes, label="episode #", shrink=0.8, ticks=range(1, len(episodes) + 1))
    fig.suptitle(f"{args_cli.body_name} incoming joint wrench, {len(episodes)} episodes overlaid (aligned to episode start)")
    fig.savefig(args_cli.out, dpi=150)
    print(f"[INFO] plot saved to: {os.path.abspath(args_cli.out)}")

    total_steps = sum(len(ep_times) for ep_times, _ in episodes)
    print(f"\n[RESULT] NaN seen over {total_steps} steps / {episode_count} episodes: "
          f"{'YES -- mechanism broken, same failure mode as the ZMP contact-API bug' if saw_nan else 'no'}")
    print(f"[RESULT] saw nonzero wrench at least once: {saw_nonzero} "
          f"({'looks alive' if saw_nonzero else 'ALWAYS ZERO -- suspicious, investigate before trusting this reading'})")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
