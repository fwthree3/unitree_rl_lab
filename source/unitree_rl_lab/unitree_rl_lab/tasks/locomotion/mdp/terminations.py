from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def root_height_below_minimum_terrain(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the root is less than ``minimum_height`` above the local terrain (M1b.1 Ascend).

    Isaac Lab's root_height_below_minimum uses world z, which misfires inside inverted
    stairs/slopes whose floor sits below z = 0. Ground = mean height-scanner hit.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    sensor = env.scene[sensor_cfg.name]
    ground = torch.nan_to_num(sensor.data.ray_hits_w[..., 2], nan=0.0, posinf=0.0, neginf=0.0).mean(dim=1)
    return (asset.data.root_pos_w[:, 2] - ground) < minimum_height
