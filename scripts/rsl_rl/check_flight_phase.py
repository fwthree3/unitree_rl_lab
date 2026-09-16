# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct flight-phase check for M1.2/#11: roll out a checkpoint and read the same
`contact_forces` sensor the `gait` reward uses (body_names=".*ankle_roll.*", i.e. both
feet) to see whether there's ever a moment both feet are simultaneously off the ground.
This is ground truth, unlike inferring it from aggregate reward magnitudes.
"""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Check for a real flight phase (both feet off ground) over N rollout episodes.")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-Velocity", help="Name of the task.")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to roll out (single env).")
parser.add_argument("--out", type=str, default="flight_phase.png", help="Output plot path.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import re

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

    contact_sensor = env.unwrapped.scene.sensors["contact_forces"]
    foot_ids = [i for i, name in enumerate(contact_sensor.body_names) if re.search("ankle_roll", name)]
    print(f"[INFO] feet body ids: {foot_ids} ({[contact_sensor.body_names[i] for i in foot_ids]})")
    assert len(foot_ids) == 2, f"expected 2 feet, found {len(foot_ids)}: {foot_ids}"

    dt = env.unwrapped.step_dt

    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]

    episodes = []
    cur_times, cur_contacts = [], []
    episode_count = 0
    t = 0.0
    total_steps, flight_steps = 0, 0

    with torch.inference_mode():
        while episode_count < args_cli.num_episodes:
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

            is_contact = (contact_sensor.data.current_contact_time[0, foot_ids] > 0).cpu().tolist()
            both_off_ground = not any(is_contact)

            total_steps += 1
            if both_off_ground:
                flight_steps += 1

            cur_times.append(t)
            cur_contacts.append([float(c) for c in is_contact])
            t += dt

            if bool(dones[0]):
                episode_count += 1
                print(f"[INFO] episode {episode_count}/{args_cli.num_episodes} done, length={t:.2f}s")
                episodes.append((cur_times, cur_contacts))
                cur_times, cur_contacts = [], []
                t = 0.0

    fig, ax = plt.subplots(figsize=(14, 3))
    y_offset = 0
    for ep_times, ep_contacts in episodes:
        contacts_t = torch.tensor(ep_contacts)  # (T, 2)
        both_off = (contacts_t.sum(dim=1) == 0).float()
        # stance/flight strip: 0=neither foot down (flight), 1=one foot down, 2=both down
        strip = contacts_t.sum(dim=1).numpy()
        ax.scatter(ep_times, [y_offset] * len(ep_times), c=strip, cmap="RdYlGn", vmin=0, vmax=2, s=4)
        y_offset += 1
    ax.set_yticks(range(len(episodes)))
    ax.set_yticklabels([f"ep{i+1}" for i in range(len(episodes))])
    ax.set_xlabel("time since episode start (s)")
    ax.set_title("feet contact count per step (red=0 feet down/flight, yellow=1 foot down, green=2 feet down)")
    fig.tight_layout()
    fig.savefig(args_cli.out, dpi=150)
    print(f"[INFO] plot saved to: {os.path.abspath(args_cli.out)}")

    print(f"\n[RESULT] flight-phase (both feet off ground) steps: {flight_steps}/{total_steps} "
          f"({100*flight_steps/total_steps:.2f}%)")
    print(f"[RESULT] {'FLIGHT PHASE PRESENT' if flight_steps > 0 else 'NO FLIGHT PHASE -- still a walk gait'}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
