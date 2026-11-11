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


from isaaclab.managers import ManagerTermBase  # noqa: E402


class noisy_height_scan(ManagerTermBase):
    """Height scan with a perception-error model (M1b.1 perceptive Ascend, 2026-11-11).

    On the real robot the scan comes from a depth camera / LiDAR + elevation mapping, so
    it is never exact. Errors modeled (all re-sampled per episode unless noted):
      - vertical map drift: constant height bias, U(-bias_z, bias_z)
      - localization error: constant xy shift of the map, U(-shift_xy, shift_xy) m,
        applied by bilinear resampling of the grid (edge cells clamp)
      - per-step Gaussian noise per cell (std noise_std)
      - per-step random missing cells (prob p_missing) -> filled with fill_value
      - per-step outliers (prob p_outlier) -> +U(-outlier_mag, outlier_mag)
      - latency: the scan is delayed by 0..max_delay policy steps
      - perception blackout: with prob p_blackout an episode gets no map at all
        (every cell = fill_value)
    fill_value = 0.0 means "nominal flat ground at the stance height", i.e. the policy's
    fallback when it has no information is to assume flat.
    The critic gets the clean scan (asymmetric actor-critic).
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        p = cfg.params
        self.sensor = env.scene.sensors[p["sensor_cfg"].name]
        pat = self.sensor.cfg.pattern_cfg
        self.nx = int(round(pat.size[0] / pat.resolution)) + 1
        self.ny = int(round(pat.size[1] / pat.resolution)) + 1
        self.res = pat.resolution
        self.max_delay = int(p.get("max_delay", 2))
        n, dev = env.num_envs, env.device
        self.bias = torch.zeros(n, 1, device=dev)
        self.shift = torch.zeros(n, 2, device=dev)  # in cells (x, y)
        self.delay = torch.zeros(n, dtype=torch.long, device=dev)
        self.blackout = torch.zeros(n, dtype=torch.bool, device=dev)
        self.buf = None  # (max_delay+1, n, cells), index 0 = newest

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        p = self.cfg.params
        n = self.bias[env_ids].shape[0]
        dev = self.bias.device
        self.bias[env_ids] = (torch.rand(n, 1, device=dev) * 2 - 1) * p.get("bias_z", 0.05)
        self.shift[env_ids] = (torch.rand(n, 2, device=dev) * 2 - 1) * p.get("shift_xy", 0.05) / self.res
        self.delay[env_ids] = torch.randint(0, self.max_delay + 1, (n,), device=dev)
        self.blackout[env_ids] = torch.rand(n, device=dev) < p.get("p_blackout", 0.05)
        if self.buf is not None:
            self.buf[:, env_ids] = 0.0

    def _shift(self, h):
        # h: (n, ny*nx), x fastest (GridPattern ordering "xy")
        n = h.shape[0]
        g = h.view(n, 1, self.ny, self.nx)
        # normalized grid_sample coordinates; shift in cells -> 2/(N-1) per cell
        ys = torch.linspace(-1, 1, self.ny, device=h.device)
        xs = torch.linspace(-1, 1, self.nx, device=h.device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        base = torch.stack((gx, gy), dim=-1).unsqueeze(0)  # (1, ny, nx, 2)
        d = torch.stack(
            (self.shift[:, 0] * 2 / (self.nx - 1), self.shift[:, 1] * 2 / (self.ny - 1)), dim=-1
        ).view(n, 1, 1, 2)
        out = torch.nn.functional.grid_sample(g, base + d, mode="bilinear", padding_mode="border", align_corners=True)
        return out.view(n, -1)

    def __call__(
        self,
        env,
        sensor_cfg: SceneEntityCfg,
        offset: float = 0.5,
        bias_z: float = 0.05,
        shift_xy: float = 0.05,
        noise_std: float = 0.02,
        p_missing: float = 0.05,
        p_outlier: float = 0.01,
        outlier_mag: float = 0.3,
        max_delay: int = 2,
        p_blackout: float = 0.05,
        fill_value: float = 0.0,
        clip: tuple[float, float] = (-1.0, 1.0),
    ) -> torch.Tensor:
        s = self.sensor
        hits = torch.nan_to_num(s.data.ray_hits_w[..., 2], nan=0.0, posinf=0.0, neginf=0.0)
        h = s.data.pos_w[:, 2].unsqueeze(1) - hits - offset
        h = h.clamp(*clip)
        # latency buffer
        if self.buf is None:
            self.buf = h.unsqueeze(0).repeat(self.max_delay + 1, 1, 1)
        self.buf = torch.roll(self.buf, 1, dims=0)
        self.buf[0] = h
        h = self.buf[self.delay, torch.arange(h.shape[0], device=h.device)]
        # map errors
        h = self._shift(h) + self.bias
        h = h + torch.randn_like(h) * noise_std
        out_mask = torch.rand_like(h) < p_outlier
        h = torch.where(out_mask, h + (torch.rand_like(h) * 2 - 1) * outlier_mag, h)
        miss = torch.rand_like(h) < p_missing
        h = torch.where(miss | self.blackout.unsqueeze(1), torch.full_like(h, fill_value), h)
        return h.clamp(*clip)
