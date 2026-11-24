"""Whole-body dynamic state for ZMP analysis (analysis only, not a reward).

Ported from the ZMP-reward research fork (`unitree_rl_lab_zmp`, mdp/zmp.py, 2026-08) to answer one question on the
current policies: does the ZMP support-polygon margin predict a fall earlier or better than simple signals such as
base tilt? Everything here reads simulator state; the policy and the training setup are untouched.
Design background: kairn_docs docs/notes/zmp-reward-design-260822.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import matrix_from_quat, quat_apply

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

# Sole contact corners in the ankle_roll_link frame (G1 MJCF sole spheres): back-left, front-left, front-right,
# back-right. Same offsets for both feet; each foot's own frame mirrors left/right.
FOOT_CORNERS_B = torch.tensor([[-0.05, 0.025, -0.03], [0.12, 0.03, -0.03], [0.12, -0.03, -0.03], [-0.05, -0.025, -0.03]])


def whole_body_state(asset: Articulation) -> dict[str, torch.Tensor]:
    """CoM position/velocity/acceleration, total mass and the rate of change of angular momentum about the CoM.

    Live masses/inertias from PhysX (not `default_mass`, which ignores mass randomization).
    L_dot = sum_i (r_i - r_com) x (m_i a_i) + I_i,w alpha_i + omega_i x (I_i,w omega_i).
    """
    mass = asset.root_physx_view.get_masses().to(asset.data.body_com_pos_w.device).unsqueeze(-1)  # (N,B,1)
    m_tot = mass.sum(dim=1)  # (N,1)
    r, v, a = asset.data.body_com_pos_w, asset.data.body_com_lin_vel_w, asset.data.body_com_lin_acc_w
    com = (r * mass).sum(1) / m_tot
    com_vel = (v * mass).sum(1) / m_tot
    com_acc = (a * mass).sum(1) / m_tot
    omega, alpha, quat = asset.data.body_com_ang_vel_w, asset.data.body_com_ang_acc_w, asset.data.body_com_quat_w
    n, b = quat.shape[:2]
    inertia_b = asset.root_physx_view.get_inertias().to(r.device).view(n, b, 3, 3)
    rot = matrix_from_quat(quat)
    inertia_w = rot @ inertia_b @ rot.transpose(-1, -2)
    i_omega = torch.einsum("nbij,nbj->nbi", inertia_w, omega)
    i_alpha = torch.einsum("nbij,nbj->nbi", inertia_w, alpha)
    l_dot = (torch.cross(r - com.unsqueeze(1), mass * a, dim=-1) + i_alpha + torch.cross(omega, i_omega, dim=-1)).sum(1)
    return dict(com=com, com_vel=com_vel, com_acc=com_acc, mass=m_tot.squeeze(-1), l_dot=l_dot)


def foot_corners_world(asset: Articulation, body_ids: list[int]) -> torch.Tensor:
    """World sole corners of the given foot bodies, shape (N, F, 4, 3)."""
    pos = asset.data.body_link_pos_w[:, body_ids]  # (N,F,3)
    quat = asset.data.body_link_quat_w[:, body_ids]  # (N,F,4)
    n, f = pos.shape[:2]
    c = FOOT_CORNERS_B.to(pos.device).view(1, 1, 4, 3).expand(n, f, 4, 3)
    return pos.unsqueeze(2) + quat_apply(quat.unsqueeze(2).expand(n, f, 4, 4), c)
