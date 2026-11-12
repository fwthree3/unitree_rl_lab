"""M1b.0/#252 evaluation: Run success rate against the >=95% criterion.

Criterion (docs/phase1-plan.md, M1b.0): at the Run target speed (2.5-3.0 m/s,
skill-targets.md), with the *training* domain randomization on (friction/mass
at startup, interval pushes, observation noise), over 1,000 episodes; success =
the episode ends without a fall termination (i.e. it reaches time_out).

Method: N parallel envs, each runs exactly one episode. The forward command is
sampled once per env from U(cmd_min, cmd_max) and held for the whole episode
(lin_vel_y = ang_vel_z = 0). The velocity curriculum, standing envs and
terrain curriculum are disabled so the command distribution is exactly the one
being evaluated. Everything else is the training env config unchanged.

Outputs a per-episode CSV and a summary line (success rate with a 95% Wilson
interval, termination breakdown, mean tracked speed, flight-phase fraction).
"""

import argparse
import re

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate Run success rate (M1b.0).")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-Velocity", help="Name of the task.")
parser.add_argument("--num_episodes", type=int, default=1000, help="Episodes = parallel envs, one episode each.")
parser.add_argument("--cmd_min", type=float, default=2.5, help="Min forward command (m/s).")
parser.add_argument("--cmd_max", type=float, default=3.0, help="Max forward command (m/s).")
parser.add_argument("--no_push", action="store_true", help="Disable interval pushes (diagnostic only, not the criterion).")
parser.add_argument("--difficulty", type=float, default=0.5,
                    help="--terrain ascend only: every tile at this difficulty (0-1; stairs 5->20 cm, slopes 0->0.4 rad).")
parser.add_argument("--terrain", choices=["plane", "generator", "ascend", "flatgrid"], default="plane",
                    help="plane (default): infinite flat ground plane, so a 20 s run at 3 m/s cannot leave the map. "
                         "generator: the training terrain grid (all-flat tiles, ~112x208 m) -- running off its edge "
                         "shows up as a fall termination, which is a map-size artifact, not a policy failure. "
                         "flatgrid: the task's generator with every tile flat and a 100 m border -- for tasks whose "
                         "observations need the height scanner (perceptive), where plane cannot be used.")
parser.add_argument("--ramp_s", type=float, default=0.0,
                    help="Ramp the forward command linearly from 0 to its sampled value over this many seconds "
                         "(0 = step command from rest, the original criterion). Mimics the deploy-side keyboard ramp.")
parser.add_argument("--steady_after_s", type=float, default=3.0,
                    help="Steady-state speed excludes the first N s (acceleration from rest).")
parser.add_argument("--out_csv", type=str, default=None, help="Per-episode CSV path.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import math
import os

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from rsl_rl.runners import OnPolicyRunner

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
import isaaclab.terrains as terrain_gen


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)


def main():
    n = args_cli.num_episodes
    # Training env config (not the play config) so domain randomization matches training.
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=n, entry_point_key="env_cfg_entry_point")

    cmd = env_cfg.commands.base_velocity
    cmd.ranges.lin_vel_x = (args_cli.cmd_min, args_cli.cmd_max)
    cmd.ranges.lin_vel_y = (0.0, 0.0)
    cmd.ranges.ang_vel_z = (0.0, 0.0)
    cmd.rel_standing_envs = 0.0
    if hasattr(cmd, "uncapped_fast_prob"):
        cmd.uncapped_fast_prob = 0.0  # training-only fast band would override the pinned range
    # resample once per episode only (episode is 20 s; reset always resamples)
    cmd.resampling_time_range = (1.0e6, 1.0e6)
    cmd.debug_vis = False
    env_cfg.curriculum.lin_vel_cmd_levels = None
    env_cfg.curriculum.terrain_levels = None
    if args_cli.terrain == "ascend":
        # M1b.1/#21: the task's own generator (use an *-Ascend task), every row at one difficulty,
        # envs spread over all rows/columns; per-sub-terrain results are reported below.
        tg = env_cfg.scene.terrain.terrain_generator
        tg.difficulty_range = (args_cli.difficulty, args_cli.difficulty)
        env_cfg.scene.terrain.max_init_terrain_level = tg.num_rows - 1
    if args_cli.terrain == "plane":
        # The training grid is all-flat tiles, so the ground is the same surface; only its extent changes.
        env_cfg.scene.terrain.terrain_type = "plane"
        env_cfg.scene.terrain.terrain_generator = None
        env_cfg.scene.height_scanner = None  # unused by obs/rewards; ray caster needs a mesh, not a ground plane
    if args_cli.terrain == "flatgrid":
        # Perceptive tasks: the ray caster needs a mesh, so keep the generator but make every tile flat.
        tg = env_cfg.scene.terrain.terrain_generator
        tg.sub_terrains = {"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)}
        tg.border_width = 100.0
        env_cfg.scene.terrain.max_init_terrain_level = tg.num_rows - 1
    if args_cli.no_push:
        env_cfg.events.push_robot = None

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading checkpoint from: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    uenv = env.unwrapped
    device = uenv.device
    robot = uenv.scene["robot"]
    contact_sensor = uenv.scene.sensors["contact_forces"]
    foot_ids = [i for i, name in enumerate(contact_sensor.body_names) if re.search("ankle_roll", name)]
    assert len(foot_ids) == 2, f"expected 2 feet, found {foot_ids}"
    tm = uenv.termination_manager
    fall_terms = [t for t in tm.active_terms if t != "time_out"]
    vel_cmd_term = uenv.command_manager.get_term("base_velocity")

    # The env resets once at creation with the commands sampled from the overridden range.
    commanded = vel_cmd_term.vel_command_b[:, 0].clone()

    finished = torch.zeros(n, dtype=torch.bool, device=device)
    success = torch.zeros(n, dtype=torch.bool, device=device)
    reason = ["" for _ in range(n)]
    ep_len_s = torch.zeros(n, device=device)
    speed_sum = torch.zeros(n, device=device)
    flight_steps = torch.zeros(n, device=device)
    steps = torch.zeros(n, device=device)
    last_xy = torch.zeros(n, 2, device=device)
    start_xy = robot.data.root_pos_w[:, :2].clone()
    act_abs_max = torch.zeros(n, device=device)  # per-env max |policy action| (pre-clip), for picking clip_actions
    ss_speed_sum = torch.zeros(n, device=device)
    ss_steps = torch.zeros(n, device=device)
    dt = uenv.step_dt
    # half-extent of the walkable terrain (generated grid + flat border), centered at the origin
    tg = env_cfg.scene.terrain.terrain_generator
    if tg is not None:
        half_x = tg.num_rows * tg.size[0] / 2 + tg.border_width
        half_y = tg.num_cols * tg.size[1] / 2 + tg.border_width
    else:
        half_x = half_y = float("inf")

    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]

    def apply_ramp():
        # overrides the held command; the policy sees it from the next observation on
        if args_cli.ramp_s > 0:
            frac = torch.clamp((steps + 1) * dt / args_cli.ramp_s, max=1.0)
            vel_cmd_term.vel_command_b[:, 0] = commanded * frac

    apply_ramp()
    with torch.inference_mode():
        while not bool(finished.all()):
            actions = policy(obs)
            act_abs_max = torch.where(~finished, torch.maximum(act_abs_max, actions.abs().max(dim=1).values), act_abs_max)
            # stats for the state *before* this step's possible reset
            active = ~finished
            speed_sum += torch.where(active, robot.data.root_lin_vel_b[:, 0], 0.0)
            in_contact = contact_sensor.data.current_contact_time[:, foot_ids] > 0
            flight_steps += (active & ~in_contact.any(dim=1)).float()
            steps += active.float()
            ss = active & (steps * dt > args_cli.steady_after_s)
            ss_speed_sum += torch.where(ss, robot.data.root_lin_vel_b[:, 0], 0.0)
            ss_steps += ss.float()
            last_xy = torch.where(active.unsqueeze(-1), robot.data.root_pos_w[:, :2], last_xy)

            obs, _, dones, _ = env.step(actions)
            apply_ramp()
            dones = dones.bool()
            newly = dones & ~finished
            if bool(newly.any()):
                ids = newly.nonzero(as_tuple=False).squeeze(-1)
                ok = tm.time_outs[ids] & ~tm.terminated[ids]
                success[ids] = ok
                ep_len_s[ids] = steps[ids] * dt
                for name in fall_terms:
                    hit = tm.get_term(name)[ids]
                    for j in ids[hit].tolist():
                        reason[j] = reason[j] + ("+" if reason[j] else "") + name
                for j in ids[ok].tolist():
                    reason[j] = "time_out"
                finished |= newly

    k = int(success.sum())
    lo, hi = wilson_interval(k, n)
    mean_speed = speed_sum / steps.clamp(min=1)
    ss_speed = ss_speed_sum / ss_steps.clamp(min=1)
    flight_frac = flight_steps / steps.clamp(min=1)
    edge_dist = torch.minimum(half_x - last_xy[:, 0].abs(), half_y - last_xy[:, 1].abs())
    counts: dict[str, int] = {}
    for r in reason:
        counts[r] = counts.get(r, 0) + 1

    succ_mask = success
    print("\n[RESULT] checkpoint:", resume_path)
    print(f"[RESULT] command U({args_cli.cmd_min}, {args_cli.cmd_max}) m/s, ramp {args_cli.ramp_s:.1f} s, pushes={'off' if args_cli.no_push else 'on'}, terrain={args_cli.terrain}")
    print(f"[RESULT] success: {k}/{n} = {100*k/n:.1f}%  (95% Wilson CI {100*lo:.1f}-{100*hi:.1f}%)  "
          f"criterion >=95%: {'PASS' if k / n >= 0.95 else 'FAIL'}")
    print(f"[RESULT] termination breakdown: {counts}")
    if bool(succ_mask.any()):
        print(f"[RESULT] successful eps -- mean forward speed {mean_speed[succ_mask].mean():.3f} m/s "
              f"(commanded mean {commanded[succ_mask].mean():.3f}), flight-phase {100*flight_frac[succ_mask].mean():.1f}% of steps")
    if bool((~succ_mask).any()):
        print(f"[RESULT] failed eps -- mean fall time {ep_len_s[~succ_mask].mean():.2f} s, "
              f"commanded mean {commanded[~succ_mask].mean():.3f} m/s")
    fail = ~succ_mask
    if bool(fail.any()):
        off = fail & (edge_dist < 1.0)
        print(f"[RESULT] failed eps within 1 m of / beyond the terrain edge (|x|<={half_x:.0f}, |y|<={half_y:.0f}): "
              f"{int(off.sum())}/{int(fail.sum())}")
    q = torch.quantile(act_abs_max, torch.tensor([0.5, 0.99, 1.0], device=device))
    print(f"[RESULT] per-episode max |action| (pre-clip): median {q[0]:.2f}, p99 {q[1]:.2f}, max {q[2]:.2f}")
    if bool(succ_mask.any()):
        ratio = ss_speed[succ_mask] / commanded[succ_mask]
        stalled = ratio < 0.5
        print(f"[RESULT] steady-state (t > {args_cli.steady_after_s:.0f} s), successful eps: actual/commanded "
              f"mean {ratio[~stalled].mean():.3f} over eps that took off; did not take off (<50% of cmd): "
              f"{int(stalled.sum())}/{int(succ_mask.sum())}")
    if args_cli.terrain == "ascend":
        import numpy as np
        tg = env_cfg.scene.terrain.terrain_generator
        names = list(tg.sub_terrains.keys())
        props = np.array([c.proportion for c in tg.sub_terrains.values()]); props /= props.sum()
        col_to_sub = [int(np.min(np.where(c / tg.num_cols + 0.001 < np.cumsum(props))[0])) for c in range(tg.num_cols)]
        cols = uenv.scene.terrain.terrain_types.tolist()
        dist = (last_xy - start_xy).norm(dim=1)
        print(f"[RESULT] ascend terrain, difficulty {args_cli.difficulty:.2f} (by starting tile type):")
        for k, nm in enumerate(names):
            m = torch.tensor([col_to_sub[c] == k for c in cols], device=device)
            cnt = int(m.sum())
            if cnt:
                print(f"[RESULT]   {nm:<22} {int(success[m].sum())}/{cnt} = {100*success[m].float().mean():5.1f}%  "
                      f"mean distance {dist[m].mean():5.1f} m, steady speed {ss_speed[m & success].mean():.2f} m/s")
    # success rate vs commanded speed (5 bins)
    edges = torch.linspace(args_cli.cmd_min, args_cli.cmd_max, 6, device=device)
    for b in range(5):
        m = (commanded >= edges[b]) & (commanded <= edges[b + 1]) if b == 4 else (commanded >= edges[b]) & (commanded < edges[b + 1])
        cnt = int(m.sum())
        if cnt:
            print(f"[RESULT]   cmd {edges[b]:.2f}-{edges[b+1]:.2f}: {int(success[m].sum())}/{cnt} = {100*success[m].float().mean():.1f}%, "
                  f"steady-state actual {ss_speed[m & success].mean():.3f} m/s")

    if args_cli.out_csv:
        with open(args_cli.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["env", "commanded_vx", "success", "reason", "episode_s", "mean_vx", "flight_frac", "steady_vx", "last_x", "last_y", "edge_dist"])
            for i in range(n):
                w.writerow([i, f"{commanded[i]:.3f}", int(success[i]), reason[i], f"{ep_len_s[i]:.2f}",
                            f"{mean_speed[i]:.3f}", f"{flight_frac[i]:.4f}", f"{ss_speed[i]:.3f}",
                            f"{last_xy[i,0]:.2f}", f"{last_xy[i,1]:.2f}", f"{edge_dist[i]:.2f}"])
        print(f"[RESULT] per-episode CSV: {os.path.abspath(args_cli.out_csv)}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
