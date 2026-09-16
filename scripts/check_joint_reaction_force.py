# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""M1.11 mechanism check (issue #202): confirm Articulation.data.body_incoming_joint_wrench_b
returns non-NaN, sane values for torso_link on the existing G1 model. Not a policy, no
checkpoint involved -- just steps the env with zero actions and prints the reading.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Check joint reaction force/torque readback at a given body.")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-Velocity", help="Name of the task.")
parser.add_argument("--body_name", type=str, default="torso_link", help="Body to read the incoming joint wrench at.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument("--steps", type=int, default=200, help="Number of simulation steps to run.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
import gymnasium as gym


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    robot = env.unwrapped.scene["robot"]
    body_idx = robot.body_names.index(args_cli.body_name)
    print(f"[INFO] '{args_cli.body_name}' body index: {body_idx} (of {len(robot.body_names)} bodies)")

    action_dim = env.unwrapped.action_manager.total_action_dim
    zero_actions = torch.zeros(args_cli.num_envs, action_dim, device=env.unwrapped.device)

    saw_nonzero = False
    saw_nan = False
    for step in range(args_cli.steps):
        env.step(zero_actions)
        wrench = robot.data.body_incoming_joint_wrench_b[:, body_idx, :]  # (num_envs, 6): Fx,Fy,Fz,Mx,My,Mz

        if torch.isnan(wrench).any():
            saw_nan = True
            print(f"[step {step}] NaN detected in wrench: {wrench}")
            continue
        if torch.count_nonzero(wrench) > 0:
            saw_nonzero = True
        if step % 20 == 0:
            print(f"[step {step}] wrench (env0) Fz={wrench[0, 2]:.3f} Mx={wrench[0, 3]:.3f} My={wrench[0, 4]:.3f}")

    print(f"\n[RESULT] NaN seen over {args_cli.steps} steps: "
          f"{'YES -- mechanism broken, same failure mode as the ZMP contact-API bug' if saw_nan else 'no'}")
    print(f"[RESULT] saw nonzero wrench at least once: {saw_nonzero} "
          f"({'looks alive' if saw_nonzero else 'ALWAYS ZERO -- suspicious, investigate before trusting this reading'})")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
