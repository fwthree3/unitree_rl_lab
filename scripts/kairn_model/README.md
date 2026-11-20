# KAIRN parametric leg model (M1.5 bridge)

`gen_legs.py` builds a KAIRN-legs robot from the G1 29-DoF model, so leg dimensions from the M3.2 CAD can go into
simulation (Isaac Lab training/eval, MuJoCo sim2sim) before the full CAD → URDF export (M1.5) exists. It changes:

- **thigh length** (hip pitch axis → knee axis). The knee joint moves along the hip-yaw → knee segment, and the hip
  pitch/roll/yaw cluster is kept as is.
- **shank length** (knee axis → ankle pitch axis).
- **link masses / COMs** (optional), e.g. from SOLIDWORKS *Mass Properties*.

Joint names, axes, limits, order and actuators are G1/KairnLegs, so existing policies load and fine-tune as is.

## Usage

```bash
cd ~/workspace/kairn_rl/unitree_rl_lab
python3 scripts/kairn_model/gen_legs.py scripts/kairn_model/params/<name>.yaml   # → ~/workspace/kairn_rl/kairn_model/<name>/

# Isaac Lab (zero-shot eval / fine-tune of the current policies on the new model)
KAIRN_MODEL=<name> ../.venv-isaac/bin/python scripts/rsl_rl/eval_run_success.py --headless \
    --task Unitree-G1-29dof-Velocity-KairnLegs-Param --load_run from_P003 --checkpoint model_105300.pt \
    --terrain plane --ramp_s 2.0 --cmd_min 3.0 --cmd_max 4.5 --num_episodes 400 --legs_npz /tmp/<name>_run.npz
KAIRN_MODEL=<name> ... --task Unitree-G1-29dof-Velocity-KairnLegs-AscendOnly-Param --load_run from_P011 \
    --checkpoint model_161300.pt --terrain ascend --difficulty 1.0 --cmd_min 0.8 --cmd_max 1.2

# MuJoCo (unitree_mujoco accepts an absolute scene path)
cd ../unitree_mujoco/simulate/build && ./unitree_mujoco -s ~/workspace/kairn_rl/kairn_model/<name>/scene.xml
```

`logs/rsl_rl/unitree_g1_29dof_velocity_kairnlegs_param/from_P003/` and
`.../unitree_g1_29dof_velocity_kairnlegs_ascendonly_param/from_P011/` hold copies of P003 and P011 for warm starts.
`--legs_npz` traces give the per-joint torque/speed/power for the M1.6 margin check.

## Parameters (`params/*.yaml`)

| key | meaning | G1 value |
|---|---|---|
| `name` | output folder | |
| `thigh_length` | hip pitch axis → knee axis, m | 0.341 |
| `shank_length` | knee axis → ankle pitch axis, m | 0.300 |
| `link_mass.<link>` | kg per side, links `hip_pitch_link`, `hip_roll_link`, `hip_yaw_link` (thigh), `knee_link` (shank), `ankle_pitch_link`, `ankle_roll_link` (foot) | 1.35 / 1.52 / 1.702 / 1.932 / 0.074 / 0.608 |
| `link_com.<link>` | [x, y, z] m in the left link frame (right is mirrored) | |
| `pelvis_mass` | kg | 3.813 |

`params/g1_identity.yaml` is the sanity check: unchanged G1. `params/example_longer_legs.yaml` is a worked example.

## What is approximated

- Stretched links (thigh = `hip_yaw_link`, shank = `knee_link`) get their meshes scaled along z and inertia scaled
  by m·s². Use real CAD inertia when available. Feet and the hip cluster are unchanged.
- Collisions are the stock G1 USD's: shank = 2 cylinders, foot = sole box + 5 spheres. Shank cylinders stretch with
  the shank.
- The generated URDF is imported by Isaac Lab's URDF converter. It is *not* the stock USD the policies were trained
  on. Sanity check with `g1_identity` (2026-11-19):

  | | Stock USD | `g1_identity` URDF |
  |---|---|---|
  | Run 3-4.5 m/s, P003 | 100%, 4.03-4.06 m/s top band | 99.5%, 4.03 m/s |
  | Ascend d1.0, P011 | 87.8-90.8% | 85.0% (CI overlaps) |

  Always compare a new model against `g1_identity` (same pipeline), not against the stock-USD numbers.
- MJCF: same lengths/masses/mesh scaling applied to `unitree_mujoco`'s `g1_29dof_kairnlegs.xml`. Its collision set is
  that file's own.

## Next step when the real CAD exists (M1.5)

Export the full URDF from SOLIDWORKS (or write link frames + mass properties by hand). Then point
`kairn_legs_param_cfg` at it (same `robot.urdf` + `meta.json` folder layout, `dz` = pelvis height change vs G1), and
keep the primitive foot/shank collisions. The Isaac tasks, eval and `--legs_npz` analysis stay the same.
