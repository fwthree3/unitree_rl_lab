"""Coupled parallel-ankle actuator (2-RSU, two motors per ankle) for Kairn.

The policy and the simulated articulation keep serial ankle pitch/roll joints. This actuator
computes the joint-space PD torque, maps it to the two motors of each ankle through the
linearised linkage, clips each motor to its own torque-speed envelope, and maps the
result back to joint torques:

    motor speed   w_m = kp * w_pitch  +/- kr * w_roll
    motor torque  t_m = t_pitch / (2 kp) +/- t_roll / (2 kr)
    joint torque  t_pitch = kp * (t_m1 + t_m2),   t_roll = kr * (t_m1 - t_m2)

The envelope (per motor) is: peak torque up to the rated speed, then linear to zero at the
no-load speed, applied only when driving (torque and speed in the same direction); braking
gets the full peak. Past the no-load speed the driving limit turns negative, i.e. the motor
is forced to brake (back-EMF), so external forces cannot spin it far beyond no-load for free. Analysis: kairn_docs docs/notes/parallel-ankle-2xrs03-261117.md.
"""

from __future__ import annotations

import torch

from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class CoupledAnkleActuator(IdealPDActuator):
    cfg: CoupledAnkleActuatorCfg

    def __init__(self, cfg: CoupledAnkleActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        names = self.joint_names
        self._pitch, self._roll = [], []
        for side in ("left", "right"):
            self._pitch.append(names.index(f"{side}_ankle_pitch_joint"))
            self._roll.append(names.index(f"{side}_ankle_roll_joint"))
        self._joint_vel = torch.zeros_like(self.computed_effort)

    def _limits(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.cfg
        aw = w.abs()
        lim = torch.where(aw <= c.motor_rated_speed, torch.full_like(aw, c.motor_peak_torque),
                          c.motor_peak_torque * (c.motor_noload_speed - aw) / (c.motor_noload_speed - c.motor_rated_speed))
        # beyond the no-load speed the line goes negative: back-EMF forces braking (as in Isaac Lab's DCMotor)
        lim = lim.clamp(-c.motor_peak_torque, c.motor_peak_torque)
        peak = torch.full_like(aw, c.motor_peak_torque)
        hi = torch.where(w > 0, lim, peak)
        lo = -torch.where(w < 0, lim, peak)
        return lo, hi

    def compute(self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
                ) -> ArticulationActions:
        self._joint_vel[:] = joint_vel
        return super().compute(control_action, joint_pos, joint_vel)

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        kp, kr = self.cfg.pitch_ratio, self.cfg.roll_ratio
        out = effort.clone()
        tp, tr = effort[:, self._pitch], effort[:, self._roll]
        wp, wr = self._joint_vel[:, self._pitch], self._joint_vel[:, self._roll]
        t1 = tp / (2 * kp) + tr / (2 * kr)
        t2 = tp / (2 * kp) - tr / (2 * kr)
        lo1, hi1 = self._limits(kp * wp + kr * wr)
        lo2, hi2 = self._limits(kp * wp - kr * wr)
        t1 = torch.maximum(torch.minimum(t1, hi1), lo1)
        t2 = torch.maximum(torch.minimum(t2, hi2), lo2)
        out[:, self._pitch] = kp * (t1 + t2)
        out[:, self._roll] = kr * (t1 - t2)
        return out


@configclass
class CoupledAnkleActuatorCfg(IdealPDActuatorCfg):
    class_type: type = CoupledAnkleActuator

    pitch_ratio: float = 0.84
    """k_p = r/c: rod offset behind the ankle over crank length (motor rad per pitch rad)."""
    roll_ratio: float = 1.0
    """k_r = b/c: half rod spacing over crank length (motor rad per roll rad)."""
    motor_peak_torque: float = 60.0
    """RS03 peak torque (N·m)."""
    motor_rated_speed: float = 10.47
    """RS03 rated speed (rad/s); the envelope keeps the peak torque up to here."""
    motor_noload_speed: float = 20.4
    """RS03 no-load speed (rad/s)."""
