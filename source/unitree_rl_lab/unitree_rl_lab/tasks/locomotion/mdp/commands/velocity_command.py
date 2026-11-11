from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import numpy as np
import torch

from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING


class TerrainCappedVelocityCommand(UniformVelocityCommand):
    """M1b.1/#20 (2026-11-11): cap lin_vel_x on non-flat sub-terrains only.

    The first Ascend run resumed the Run policy with the full Run command range
    (-1..3.5 m/s) on every tile. On stairs/boxes that is unreachable, so within ~1000
    iterations the policy learned to stand still instead of falling (time_out 0.9,
    error_vel_xy 2.4 m/s, terrain level 4.4 -> 2.6). Capping the command only on
    non-flat tiles keeps the Run speed range trained on flat tiles (unified policy)
    while giving stairs/rough a reachable target.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._capped_mask = None

    def _get_capped_mask(self) -> torch.Tensor:
        if self._capped_mask is None:
            terrain = self._env.scene.terrain
            tg = terrain.cfg.terrain_generator
            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            if tg is not None and getattr(terrain, "terrain_types", None) is not None:
                names = list(tg.sub_terrains.keys())
                props = np.array([c.proportion for c in tg.sub_terrains.values()], dtype=float)
                props /= props.sum()
                # same column -> sub-terrain assignment as TerrainGenerator's curriculum mode
                col_to_sub = [
                    int(np.min(np.where(c / tg.num_cols + 0.001 < np.cumsum(props))[0])) for c in range(tg.num_cols)
                ]
                capped_cols = torch.tensor(
                    [names[s] not in getattr(self.cfg, "uncapped_sub_terrains", ("flat",)) for s in col_to_sub], device=self.device
                )
                mask = capped_cols[terrain.terrain_types.long()]
            self._capped_mask = mask
        return self._capped_mask

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        capped = self._get_capped_mask()[env_ids]
        lo, hi = getattr(self.cfg, "capped_lin_vel_x", (-0.5, 1.5))
        vx = self.vel_command_b[env_ids, 0]
        self.vel_command_b[env_ids, 0] = torch.where(capped, vx.clamp(lo, hi), vx)


@configclass
class TerrainCappedLevelVelocityCommandCfg(UniformLevelVelocityCommandCfg):
    class_type: type = TerrainCappedVelocityCommand
    capped_lin_vel_x: tuple[float, float] = (-0.5, 1.5)
    uncapped_sub_terrains: tuple[str, ...] = ("flat",)
