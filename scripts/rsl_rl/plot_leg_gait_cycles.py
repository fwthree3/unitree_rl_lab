# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""M1.2/#11 close-out, visual companion to measure_torque.py: force a sustained
speed command and plot per-leg-joint torque/velocity/power as a function of
gait-cycle phase (0-1, normalized to one stride), overlaying every stride from
every rollout episode as a band -- so the recurring walk/run pattern shows up
as a shaded range instead of a flat time series. Cycle phase is referenced to
the left foot's contact rising edge (touchdown), so only left-leg joints are
plotted (right-leg joints would be ~180 degrees out of phase on the same axis).
"""

import argparse
import re

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Plot per-leg-joint torque/velocity/power vs gait-cycle phase.")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-Velocity", help="Name of the task.")
parser.add_argument("--command_lin_vel_x", type=float, default=4.0, help="Forced forward velocity command (m/s).")
parser.add_argument("--num_episodes", type=int, default=5, help="Total episodes to collect (across all envs).")
parser.add_argument("--num_envs", type=int, default=8,
                     help="Parallel envs -- all step in lockstep, multiplying strides collected per wall-clock "
                          "second (background execution here has an observed wall-clock time limit).")
parser.add_argument("--phase_points", type=int, default=100, help="Points to resample each stride to.")
parser.add_argument("--out", type=str, default="leg_gait_cycles.png", help="Output plot path.")
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
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from rsl_rl.runners import OnPolicyRunner

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
import gymnasium as gym

LEG_JOINTS = ["left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint"]


def resample_to_phase(values: np.ndarray, n_points: int) -> np.ndarray:
    """Linearly resample a 1D array of arbitrary length onto n_points evenly spaced in [0, 1]."""
    x_src = np.linspace(0.0, 1.0, num=len(values))
    x_dst = np.linspace(0.0, 1.0, num=n_points)
    return np.interp(x_dst, x_src, values)


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, entry_point_key="play_env_cfg_entry_point"
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
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
    left_foot_ids = [i for i, name in enumerate(contact_sensor.body_names) if re.search("left_ankle_roll", name)]
    assert len(left_foot_ids) == 1, f"expected exactly 1 left foot body, found {len(left_foot_ids)}: {left_foot_ids}"
    left_foot_id = left_foot_ids[0]

    joint_idx = {name: robot.data.joint_names.index(name) for name in LEG_JOINTS}

    vel_cmd_term = env.unwrapped.command_manager.get_term("base_velocity")

    def force_command():
        vel_cmd_term.vel_command_b[:, 0] = args_cli.command_lin_vel_x
        vel_cmd_term.vel_command_b[:, 1] = 0.0
        vel_cmd_term.vel_command_b[:, 2] = 0.0

    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]
    force_command()

    num_envs = args_cli.num_envs

    # per-env, per-episode step buffers, reset whenever that env's episode ends
    # (a stride never spans a reset)
    cur_torque = [{name: [] for name in LEG_JOINTS} for _ in range(num_envs)]
    cur_vel = [{name: [] for name in LEG_JOINTS} for _ in range(num_envs)]
    cur_is_contact = [[] for _ in range(num_envs)]

    # collected strides, resampled to a common phase axis: {joint: {"torque": [...], "vel": [...], "power": [...]}}
    strides = {name: {"torque": [], "vel": [], "power": []} for name in LEG_JOINTS}

    episode_count = 0
    total_strides = 0

    def flush_episode_strides(env_idx):
        nonlocal total_strides
        is_contact = np.array(cur_is_contact[env_idx])
        if len(is_contact) < 4:
            return
        # rising edges: False -> True transitions = touchdown = stride boundary
        touchdowns = np.where((~is_contact[:-1]) & is_contact[1:])[0] + 1
        for start, end in zip(touchdowns[:-1], touchdowns[1:]):
            if end - start < 3:
                continue  # degenerate/too-short segment, skip
            for name in LEG_JOINTS:
                torque_seg = np.array(cur_torque[env_idx][name][start:end])
                vel_seg = np.array(cur_vel[env_idx][name][start:end])
                power_seg = torque_seg * vel_seg
                strides[name]["torque"].append(resample_to_phase(torque_seg, args_cli.phase_points))
                strides[name]["vel"].append(resample_to_phase(vel_seg, args_cli.phase_points))
                strides[name]["power"].append(resample_to_phase(power_seg, args_cli.phase_points))
            total_strides += 1

    def save_plot(final):
        if total_strides == 0:
            return
        phase = np.linspace(0.0, 1.0, num=args_cli.phase_points)
        quantities = [("torque", "torque (Nm)"), ("vel", "joint velocity (rad/s)"), ("power", "power (W)")]

        fig, axes = plt.subplots(len(LEG_JOINTS), len(quantities), figsize=(15, 3.2 * len(LEG_JOINTS)), sharex=True)
        for row, joint_name in enumerate(LEG_JOINTS):
            for col, (key, ylabel) in enumerate(quantities):
                ax = axes[row, col]
                data = np.stack(strides[joint_name][key], axis=0)  # (n_strides, phase_points)
                for stride in data:
                    ax.plot(phase, stride, color="tab:blue", alpha=0.15, linewidth=0.8)
                mean = data.mean(axis=0)
                p10 = np.percentile(data, 10, axis=0)
                p90 = np.percentile(data, 90, axis=0)
                ax.fill_between(phase, p10, p90, color="tab:blue", alpha=0.25, label="10-90th pct band")
                ax.plot(phase, mean, color="tab:red", linewidth=2, label="mean")
                if row == 0:
                    ax.set_title(ylabel)
                if col == 0:
                    ax.set_ylabel(joint_name)
                ax.grid(alpha=0.3)
        axes[0, 0].legend(loc="upper right", fontsize=8)
        for ax in axes[-1, :]:
            ax.set_xlabel("gait-cycle phase (0=left touchdown, 1=next left touchdown)")

        fig.suptitle(
            f"Left-leg joint torque/velocity/power over gait cycle -- {total_strides} strides, "
            f"{episode_count} episodes, commanded {args_cli.command_lin_vel_x} m/s"
        )
        fig.tight_layout()
        fig.savefig(args_cli.out, dpi=150)
        plt.close(fig)
        tag = "[RESULT]" if final else "[INFO] checkpointed"
        print(f"{tag} {total_strides} strides / {episode_count} episodes -> {os.path.abspath(args_cli.out)}")

    with torch.inference_mode():
        while episode_count < args_cli.num_episodes:
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            force_command()  # re-assert every step -- overrides the periodic resampler

            applied_torque = robot.data.applied_torque.cpu()  # (num_envs, num_joints)
            joint_vel = robot.data.joint_vel.cpu()
            is_contact_now = (contact_sensor.data.current_contact_time[:, left_foot_id] > 0).cpu()

            for i in range(num_envs):
                for name in LEG_JOINTS:
                    j = joint_idx[name]
                    cur_torque[i][name].append(applied_torque[i, j].item())
                    cur_vel[i][name].append(joint_vel[i, j].item())
                cur_is_contact[i].append(bool(is_contact_now[i]))

                if bool(dones[i]) and episode_count < args_cli.num_episodes:
                    episode_count += 1
                    flush_episode_strides(i)
                    print(f"[INFO] episode {episode_count}/{args_cli.num_episodes} done (env {i}), "
                          f"{len(cur_is_contact[i])} steps, {total_strides} strides collected so far")
                    cur_torque[i] = {name: [] for name in LEG_JOINTS}
                    cur_vel[i] = {name: [] for name in LEG_JOINTS}
                    cur_is_contact[i] = []
                    save_plot(final=False)

    env.close()

    if total_strides == 0:
        print("[WARN] no complete strides detected -- can't plot. Check the contact sensor / foot naming.")
        simulation_app.close()
        return

    save_plot(final=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
