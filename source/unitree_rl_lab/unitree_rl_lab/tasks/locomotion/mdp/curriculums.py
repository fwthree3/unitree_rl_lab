from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)


def terrain_levels_by_sub_terrain(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> dict:
    """M1b.1/#21 (2026-11-11): report-only term -- mean terrain level per sub-terrain type.

    The overall Curriculum/terrain_levels mean is dominated by solved columns (flat/slopes),
    and Isaac Lab sends envs that solve the top row back to a random row, so a solved column
    hovers around (num_rows-1)/2 rather than at the top. Per-type means show where the
    curriculum is actually stuck. Does not change any level; run it after terrain_levels.
    """
    terrain = env.scene.terrain
    tg = terrain.cfg.terrain_generator
    if tg is None or getattr(terrain, "terrain_types", None) is None:
        return {}
    names = list(tg.sub_terrains.keys())
    props = np.array([c.proportion for c in tg.sub_terrains.values()], dtype=float)
    props /= props.sum()
    # same column -> sub-terrain assignment as TerrainGenerator's curriculum mode
    col_to_sub = [int(np.min(np.where(c / tg.num_cols + 0.001 < np.cumsum(props))[0])) for c in range(tg.num_cols)]
    sub_of_env = torch.tensor(col_to_sub, device=terrain.terrain_types.device)[terrain.terrain_types.long()]
    levels = terrain.terrain_levels.float()
    out = {}
    for i, name in enumerate(names):
        m = sub_of_env == i
        if bool(m.any()):
            out[name] = levels[m].mean()
    return out
