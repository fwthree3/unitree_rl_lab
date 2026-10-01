# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""M1.2/#11 close-out: force a sustained speed command and measure per-joint
torque, velocity, and instantaneous power across several episodes, for use
as an actuator-sizing (and motor power) reference. Logs both
`applied_torque` (post actuator-model clipping -- what's actually sent into
the sim) and `computed_torque` (pre-clip -- what the actuator model
actually wanted). A joint whose applied torque sits at its effort_limit is
saturating: the true demand is hidden by the clip, so computed_torque (and
power computed from it) is the only honest number for sizing in that case.
Peak power is tracked at the timestep where |torque * velocity| peaks, not
from the separate per-joint torque-max and velocity-max (which usually
happen at different instants) -- that's the number that actually matters
for motor selection.
Mirrors check_speed_and_flight.py's "force the command every step" approach
so results are directly comparable to that check's speed/flight-phase data.
"""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Force a speed command and measure per-joint applied torque.")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-Velocity", help="Name of the task.")
parser.add_argument("--command_lin_vel_x", type=float, default=4.0, help="Forced forward velocity command (m/s).")
parser.add_argument("--num_episodes", type=int, default=20, help="Total number of episodes to collect (across all envs).")
parser.add_argument("--num_envs", type=int, default=16,
                     help="Parallel envs -- all step in lockstep, so this multiplies episodes collected per "
                          "wall-clock second (useful since this script's background execution has an observed "
                          "wall-clock time limit well under what num_episodes=20 needs at num_envs=1).")
parser.add_argument("--out", type=str, default="torque_reference.csv", help="Output CSV path (per-joint stats).")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import os

import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from rsl_rl.runners import OnPolicyRunner

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
import gymnasium as gym


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
    joint_names = robot.data.joint_names
    num_joints = len(joint_names)
    joint_effort_limits = robot.data.joint_effort_limits[0].cpu().tolist()
    joint_vel_limits = robot.data.joint_vel_limits[0].cpu().tolist()

    vel_cmd_term = env.unwrapped.command_manager.get_term("base_velocity")

    def force_command():
        vel_cmd_term.vel_command_b[:, 0] = args_cli.command_lin_vel_x
        vel_cmd_term.vel_command_b[:, 1] = 0.0
        vel_cmd_term.vel_command_b[:, 2] = 0.0

    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]
    force_command()

    def write_results(episodes, final=False):
        # Called after every episode (not just at the end) so results are never
        # lost if this process is killed externally before num_episodes is reached
        # -- background execution here has an observed wall-clock limit well under
        # what num_episodes=20 needs, so partial progress must be checkpointed.
        if not episodes:
            return
        success_eps = [e for e in episodes if e["success"]]
        n_success = len(success_eps)
        n_fell = len(episodes) - n_success
        use_eps = success_eps if n_success > 0 else episodes

        per_ep_max = torch.stack([e["max"] for e in use_eps], dim=0)  # (n_eps, num_joints), applied
        per_ep_mean = torch.stack([e["mean"] for e in use_eps], dim=0)  # (n_eps, num_joints), applied
        per_ep_computed_max = torch.stack([e["computed_max"] for e in use_eps], dim=0)
        per_ep_sat_frac = torch.stack([e["sat_frac"] for e in use_eps], dim=0)
        per_ep_vel_max = torch.stack([e["vel_max"] for e in use_eps], dim=0)
        per_ep_power_max = torch.stack([e["power_max"] for e in use_eps], dim=0)
        per_ep_computed_power_max = torch.stack([e["computed_power_max"] for e in use_eps], dim=0)
        mean_of_max = per_ep_max.mean(dim=0)
        worst_max = per_ep_max.max(dim=0).values
        mean_of_mean = per_ep_mean.mean(dim=0)
        worst_computed_max = per_ep_computed_max.max(dim=0).values
        worst_sat_frac = per_ep_sat_frac.max(dim=0).values
        worst_vel_max = per_ep_vel_max.max(dim=0).values
        worst_power_max = per_ep_power_max.max(dim=0).values
        worst_computed_power_max = per_ep_computed_power_max.max(dim=0).values

        rows = []
        for j, name in enumerate(joint_names):
            rows.append(
                {
                    "joint": name,
                    "mean_of_max_Nm": round(mean_of_max[j].item(), 3),
                    "worst_max_Nm": round(worst_max[j].item(), 3),
                    "mean_of_mean_Nm": round(mean_of_mean[j].item(), 3),
                    "worst_computed_max_Nm": round(worst_computed_max[j].item(), 3),
                    "saturation_frac": round(worst_sat_frac[j].item(), 4),
                    "effort_limit_Nm": round(joint_effort_limits[j], 3),
                    "worst_vel_max_rad_s": round(worst_vel_max[j].item(), 3),
                    "vel_limit_rad_s": round(joint_vel_limits[j], 3),
                    "worst_power_max_W": round(worst_power_max[j].item(), 3),
                    "worst_computed_power_max_W": round(worst_computed_power_max[j].item(), 3),
                }
            )

        out_path = os.path.abspath(args_cli.out)
        fieldnames = [
            "joint", "mean_of_max_Nm", "worst_max_Nm", "mean_of_mean_Nm",
            "worst_computed_max_Nm", "saturation_frac", "effort_limit_Nm",
            "worst_vel_max_rad_s", "vel_limit_rad_s",
            "worst_power_max_W", "worst_computed_power_max_W",
        ]
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        if final:
            print(f"\n[RESULT] commanded forward speed: {args_cli.command_lin_vel_x} m/s")
            print(f"[RESULT] {n_success}/{len(episodes)} episodes completed full length without falling")
            if n_success == 0:
                print(
                    "[WARN] no clean (non-fall) episodes at this command -- reporting over ALL episodes "
                    "instead, including ones that fell early. Torque near a fall may not reflect steady-state running."
                )
            saturating = [r for r in rows if r["saturation_frac"] > 0]
            if saturating:
                print(f"\n[WARN] {len(saturating)} joint(s) saturating at their effort_limit -- "
                      "worst_max_Nm is clipped, worst_computed_max_Nm is the true (pre-clip) demand:")
                for r in saturating:
                    print(f"  {r['joint']}: applied capped at {r['worst_max_Nm']} Nm (limit {r['effort_limit_Nm']}), "
                          f"true demand up to {r['worst_computed_max_Nm']} Nm, "
                          f"saturated {100*r['saturation_frac']:.1f}% of steps (worst episode)")
            print(f"\n{'joint':<28}{'mean_of_max[Nm]':<18}{'worst_max[Nm]':<16}{'mean_of_mean[Nm]':<18}"
                  f"{'worst_computed[Nm]':<20}{'sat_frac':<10}{'effort_limit[Nm]':<16}"
                  f"{'worst_vel[rad/s]':<18}{'vel_limit[rad/s]':<18}"
                  f"{'worst_power[W]':<16}{'worst_computed_power[W]'}")
            for row in rows:
                print(
                    f"{row['joint']:<28}{row['mean_of_max_Nm']:<18}{row['worst_max_Nm']:<16}"
                    f"{row['mean_of_mean_Nm']:<18}{row['worst_computed_max_Nm']:<20}"
                    f"{row['saturation_frac']:<10}{row['effort_limit_Nm']:<16}"
                    f"{row['worst_vel_max_rad_s']:<18}{row['vel_limit_rad_s']:<18}"
                    f"{row['worst_power_max_W']:<16}{row['worst_computed_power_max_W']}"
                )
            print(f"\n[INFO] per-joint torque/velocity/power reference saved to: {out_path}")
            print(
                f"[INFO] source: checkpoint={resume_path}, commanded speed={args_cli.command_lin_vel_x} m/s, "
                f"n_episodes={len(episodes)} ({n_success} success / {n_fell} fell), aggregated over "
                f"{'successful' if n_success > 0 else 'ALL (no successes)'} episodes"
            )
        else:
            print(f"[INFO] checkpointed {len(episodes)}-episode results to: {out_path}")

    effort_limit_t = torch.tensor(joint_effort_limits)
    num_envs = args_cli.num_envs

    # per-env accumulators -- all envs step in lockstep (same forced command),
    # so N envs collect N episodes' worth of data per wall-clock episode length.
    cur_abs_torque = [[] for _ in range(num_envs)]  # each: list of (num_joints,) tensors, one per step
    cur_computed_torque = [[] for _ in range(num_envs)]
    cur_abs_vel = [[] for _ in range(num_envs)]
    episode_count = 0
    episodes = []  # list of dicts: {"success", "max", "mean", "computed_max", "sat_frac", "vel_max",
    #                                "power_max", "computed_power_max", "length"}

    with torch.inference_mode():
        while episode_count < args_cli.num_episodes:
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)
            force_command()  # re-assert every step -- overrides the periodic resampler

            abs_torque = robot.data.applied_torque.abs().cpu()  # (num_envs, num_joints)
            abs_computed = robot.data.computed_torque.abs().cpu()
            abs_vel = robot.data.joint_vel.abs().cpu()
            time_outs = extras.get("time_outs", torch.zeros(num_envs, dtype=torch.bool)).cpu()

            for i in range(num_envs):
                cur_abs_torque[i].append(abs_torque[i])
                cur_computed_torque[i].append(abs_computed[i])
                cur_abs_vel[i].append(abs_vel[i])
                if bool(dones[i]) and episode_count < args_cli.num_episodes:
                    episode_count += 1
                    stacked = torch.stack(cur_abs_torque[i], dim=0)  # (steps, num_joints)
                    stacked_computed = torch.stack(cur_computed_torque[i], dim=0)
                    stacked_vel = torch.stack(cur_abs_vel[i], dim=0)
                    # a step saturates a joint if applied torque is within 1% of its effort limit
                    is_saturated = stacked >= (0.99 * effort_limit_t)
                    # instantaneous power at each step -- peaks where torque*vel peaks together,
                    # not necessarily where either individually maxes out
                    power = stacked * stacked_vel
                    computed_power = stacked_computed * stacked_vel
                    success = bool(time_outs[i])
                    episodes.append(
                        {
                            "success": success,
                            "max": stacked.max(dim=0).values,
                            "mean": stacked.mean(dim=0),
                            "computed_max": stacked_computed.max(dim=0).values,
                            "sat_frac": is_saturated.float().mean(dim=0),
                            "vel_max": stacked_vel.max(dim=0).values,
                            "power_max": power.max(dim=0).values,
                            "computed_power_max": computed_power.max(dim=0).values,
                            "length": stacked.shape[0],
                        }
                    )
                    tag = "SUCCESS (full length)" if success else "FELL early"
                    print(
                        f"[INFO] episode {episode_count}/{args_cli.num_episodes} done (env {i}, {tag}), "
                        f"length={stacked.shape[0]} steps, ep max |torque|={stacked.max():.2f} Nm, "
                        f"ep max power={power.max():.2f} W"
                    )
                    cur_abs_torque[i] = []
                    cur_computed_torque[i] = []
                    cur_abs_vel[i] = []
                    write_results(episodes, final=False)

    env.close()
    write_results(episodes, final=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
