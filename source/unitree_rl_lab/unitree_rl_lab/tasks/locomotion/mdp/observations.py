from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def waist_force_torque(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Fz/Mx/My of the incoming joint wrench at `asset_cfg`'s body (M1.11/#206 toy check:
    proxy for the real waist F/T sensor -- masked to Fz/Mx/My per the interface-spec plan,
    not the full 6-DoF wrench `mdp.body_incoming_wrench` would give)."""
    asset = env.scene[asset_cfg.name]
    wrench = asset.data.body_incoming_joint_wrench_b[:, asset_cfg.body_ids[0], :]  # (num_envs, 6)
    return wrench[:, [2, 3, 4]]  # Fz, Mx, My


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase
