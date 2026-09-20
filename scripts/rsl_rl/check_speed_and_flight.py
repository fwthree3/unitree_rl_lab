# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""M1.2/#11 check: force a sustained max-speed command (not the play config's
randomly-resampled one) and measure actual achieved base velocity + flight-phase
fraction together, so "is it actually running yet" and "how fast does it really go"
come from the same rollout instead of separate ad-hoc checks.
"""

import argparse
import re

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Force a max-speed command and measure actual speed + flight-phase fraction.")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-Velocity", help="Name of the task.")
parser.add_argument("--command_lin_vel_x", type=float, default=3.0, help="Forced forward velocity command (m/s).")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to roll out (single env).")
parser.add_argument("--out", type=str, default="speed_and_flight.png", help="Output plot path.")
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
    contact_sensor = env.unwrapped.scene.sensors["contact_forces"]
    foot_ids = [i for i, name in enumerate(contact_sensor.body_names) if re.search("ankle_roll", name)]
    assert len(foot_ids) == 2, f"expected 2 feet, found {len(foot_ids)}: {foot_ids}"

    vel_cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    dt = env.unwrapped.step_dt

    def force_command():
        vel_cmd_term.vel_command_b[:, 0] = args_cli.command_lin_vel_x
        vel_cmd_term.vel_command_b[:, 1] = 0.0
        vel_cmd_term.vel_command_b[:, 2] = 0.0

    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]
    force_command()

    episodes = []
    cur_times, cur_speeds, cur_flight = [], [], []
    episode_count = 0
    t = 0.0
    all_speeds = []

    with torch.inference_mode():
        while episode_count < args_cli.num_episodes:
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            force_command()  # re-assert every step -- overrides the periodic resampler

            speed = robot.data.root_lin_vel_b[0, 0].item()  # forward (body-frame x) speed
            is_contact = (contact_sensor.data.current_contact_time[0, foot_ids] > 0).cpu().tolist()
            flight = not any(is_contact)

            cur_times.append(t)
            cur_speeds.append(speed)
            cur_flight.append(flight)
            all_speeds.append(speed)
            t += dt

            if bool(dones[0]):
                episode_count += 1
                print(f"[INFO] episode {episode_count}/{args_cli.num_episodes} done, length={t:.2f}s, "
                      f"mean speed this ep={sum(cur_speeds)/len(cur_speeds):.3f} m/s")
                episodes.append((cur_times, cur_speeds, cur_flight))
                cur_times, cur_speeds, cur_flight = [], [], []
                t = 0.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    cmap = matplotlib.colormaps["viridis"]
    colors = [cmap(i / max(len(episodes) - 1, 1)) for i in range(len(episodes))]
    for ep_idx, (ep_times, ep_speeds, ep_flight) in enumerate(episodes):
        ax1.plot(ep_times, ep_speeds, color=colors[ep_idx], alpha=0.7, linewidth=1)
        flight_t = [tt for tt, f in zip(ep_times, ep_flight) if f]
        ax2.scatter(flight_t, [ep_idx] * len(flight_t), color=colors[ep_idx], s=4)
    ax1.axhline(args_cli.command_lin_vel_x, color="red", linestyle="--", linewidth=1, label=f"commanded ({args_cli.command_lin_vel_x} m/s)")
    ax1.set_ylabel("forward speed (m/s)")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)
    ax2.set_ylabel("flight-phase steps\n(by episode)")
    ax2.set_xlabel("time since episode start (s)")
    fig.tight_layout()
    fig.savefig(args_cli.out, dpi=150)
    print(f"[INFO] plot saved to: {os.path.abspath(args_cli.out)}")

    speeds_t = torch.tensor(all_speeds)
    total_flight = sum(f for _, _, ep_flight in episodes for f in ep_flight)
    total_steps = len(all_speeds)
    print(f"\n[RESULT] commanded speed: {args_cli.command_lin_vel_x} m/s")
    print(f"[RESULT] actual speed -- mean: {speeds_t.mean():.3f} m/s, max: {speeds_t.max():.3f} m/s, "
          f"p95: {speeds_t.quantile(0.95):.3f} m/s")
    print(f"[RESULT] flight-phase steps: {total_flight}/{total_steps} ({100*total_flight/total_steps:.2f}%)")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
