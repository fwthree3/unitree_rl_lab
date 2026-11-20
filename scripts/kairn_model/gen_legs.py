"""Generate a parametric KAIRN-legs robot model (URDF for Isaac Lab, MJCF for MuJoCo) from the G1 29-DoF model.

Bridge until M1.5's real CAD export exists. It keeps G1's topology, joint names/axes/limits and hip actuator cluster,
and changes what the KAIRN leg CAD is expected to change first:

  - thigh length (hip pitch axis -> knee axis): the knee joint origin moves along the hip-yaw -> knee segment, so
    the hip pitch/roll/yaw cluster stays as is and the thigh tube gets longer/shorter;
  - shank length (knee axis -> ankle pitch axis): the ankle pitch joint origin is scaled;
  - link masses and COMs (optional overrides, e.g. from SOLIDWORKS mass properties).

Visual/collision meshes of the stretched links are scaled along their long axis and their COMs/inertias scaled
with them (inertia ~ m * s^2, a placeholder until CAD gives real values). Feet are unchanged.

Usage (from unitree_rl_lab/, any Python with PyYAML + numpy):
    python scripts/kairn_model/gen_legs.py scripts/kairn_model/params/<name>.yaml
Output: <KAIRN_MODEL_DIR>/<name>/ with
    robot.urdf (+ meshes symlink) -> Isaac Lab task Unitree-G1-29dof-Velocity-KairnLegs-Param (KAIRN_MODEL=<name>)
    g1_29dof_kairnlegs_param.xml, scene.xml -> MuJoCo / unitree_mujoco (robot_scene: absolute path to scene.xml)
    meta.json (lengths, pelvis height shift dz, total mass)
"""

from __future__ import annotations

import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import yaml

KAIRN_RL = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC_URDF = f"{KAIRN_RL}/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf"
SRC_URDF_MESHES = f"{KAIRN_RL}/unitree_ros/robots/g1_description/meshes"
SRC_MJCF_DIR = f"{KAIRN_RL}/unitree_mujoco/unitree_robots/g1"
SRC_MJCF = f"{SRC_MJCF_DIR}/g1_29dof_kairnlegs.xml"
SRC_SCENE = f"{SRC_MJCF_DIR}/scene_29dof_kairnlegs.xml"
OUT_ROOT = os.environ.get("KAIRN_MODEL_DIR", f"{KAIRN_RL}/kairn_model")

SIDES = ("left", "right")
LEG_LINKS = ("hip_pitch_link", "hip_roll_link", "hip_yaw_link", "knee_link", "ankle_pitch_link", "ankle_roll_link")


def rpy_R(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def vec(s):
    return np.array([float(x) for x in s.split()])


def fmt(v):
    return " ".join(f"{x:.6g}" for x in v)


def urdf_joint(root, name):
    return next(j for j in root.iter("joint") if j.get("name") == name)


def urdf_link(root, name):
    return next(l for l in root.iter("link") if l.get("name") == name)


def leg_points(root, side, k_knee=1.0, k_ankle=1.0):
    """Zero-pose positions (pelvis frame) of hip pitch, knee and ankle pitch axes for given scale factors."""
    chain = [f"{side}_hip_pitch_joint", f"{side}_hip_roll_joint", f"{side}_hip_yaw_joint", f"{side}_knee_joint",
             f"{side}_ankle_pitch_joint", f"{side}_ankle_roll_joint"]
    R, p, pts = np.eye(3), np.zeros(3), {}
    for jn in chain:
        o = urdf_joint(root, jn).find("origin")
        xyz = vec(o.get("xyz"))
        if jn.endswith("knee_joint"):
            xyz = xyz * k_knee
        if jn.endswith("ankle_pitch_joint"):
            xyz = xyz * k_ankle
        p = p + R @ xyz
        R = R @ rpy_R(*vec(o.get("rpy", "0 0 0")))
        pts[jn] = p.copy()
    return pts


def solve_scales(root, thigh, shank):
    p0 = leg_points(root, "left")
    thigh0 = np.linalg.norm(p0["left_knee_joint"] - p0["left_hip_pitch_joint"])
    shank0 = np.linalg.norm(p0["left_ankle_pitch_joint"] - p0["left_knee_joint"])
    k = 1.0
    for _ in range(50):  # secant on the knee-origin scale
        f = lambda kk: np.linalg.norm(leg_points(root, "left", kk)["left_knee_joint"]
                                      - leg_points(root, "left", kk)["left_hip_pitch_joint"]) - thigh
        f0, f1 = f(k), f(k + 1e-4)
        if abs(f0) < 1e-7:
            break
        k -= f0 * 1e-4 / (f1 - f0)
    k_ankle = shank / shank0
    p1 = leg_points(root, "left", k, k_ankle)
    dz = p0["left_ankle_roll_joint"][2] - p1["left_ankle_roll_joint"][2]  # sole drop = pelvis rise
    return k, k_ankle, thigh0, shank0, dz


def scale_inertia_urdf(inertial, m_ratio, s):
    it = inertial.find("inertia")
    for a in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz"):
        it.set(a, f"{float(it.get(a)) * m_ratio * s * s:.6g}")


def main(param_file):
    P = yaml.safe_load(open(param_file))
    name = P["name"]
    out = f"{OUT_ROOT}/{name}"
    os.makedirs(out, exist_ok=True)

    # ---------------- URDF ----------------
    tree = ET.parse(SRC_URDF)
    root = tree.getroot()
    p0 = leg_points(root, "left")
    g1_thigh = np.linalg.norm(p0["left_knee_joint"] - p0["left_hip_pitch_joint"])
    g1_shank = np.linalg.norm(p0["left_ankle_pitch_joint"] - p0["left_knee_joint"])
    k_knee, k_ankle, thigh0, shank0, dz = solve_scales(
        root, float(P.get("thigh_length", g1_thigh)), float(P.get("shank_length", g1_shank)))
    stretch = {"hip_yaw_link": k_knee, "knee_link": k_ankle}
    mass_over = P.get("link_mass", {}) or {}
    com_over = P.get("link_com", {}) or {}

    for side in SIDES:
        for jn, k in ((f"{side}_knee_joint", k_knee), (f"{side}_ankle_pitch_joint", k_ankle)):
            o = urdf_joint(root, jn).find("origin")
            o.set("xyz", fmt(vec(o.get("xyz")) * k))
        for ln in LEG_LINKS:
            link = urdf_link(root, f"{side}_{ln}")
            inert = link.find("inertial")
            m_el, o_el = inert.find("mass"), inert.find("origin")
            m0 = float(m_el.get("value"))
            s = stretch.get(ln, 1.0)
            com = vec(o_el.get("xyz")) * s
            if ln in com_over:
                c = np.array(com_over[ln], dtype=float)
                com = c if side == "left" else c * np.array([1, -1, 1])
            o_el.set("xyz", fmt(com))
            m1 = float(mass_over.get(ln, m0))
            m_el.set("value", f"{m1:.6g}")
            scale_inertia_urdf(inert, m1 / m0, s)
            if s != 1.0:
                for g in link.iter("mesh"):
                    g.set("scale", f"1 1 {s:.6g}")
    if "pelvis_mass" in P:
        urdf_link(root, "pelvis").find("inertial/mass").set("value", f"{float(P['pelvis_mass']):.6g}")
    # Match the stock G1 USD the policies were trained on: sensor links without <inertial> (d435, mid360, IMUs)
    # would otherwise get a mass from their collision volume when Isaac's URDF importer merges them into
    # torso_link (~+2.4 kg), and the hands are collapsed into the wrist like in the USD.
    for link in root.iter("link"):
        if link.find("inertial") is None:
            for c in link.findall("collision"):
                link.remove(c)
    for j in root.iter("joint"):
        j.attrib.pop("dont_collapse", None)
    # The stock USD's shank and foot collisions are primitives, not the URDF's (shank convex hull, 4 sole points):
    # shank = 2 cylinders, foot = sole box + 5 contact spheres. Copy them (from the USD, 2026-11-19), shank
    # positions/lengths stretched with the shank.
    def prim(link, kind, xyz, rpy=(0, 0, 0), **dims):
        c = ET.SubElement(link, "collision")
        ET.SubElement(c, "origin", xyz=fmt(xyz), rpy=fmt(rpy))
        ET.SubElement(ET.SubElement(c, "geometry"), kind, **{k: (fmt(v) if isinstance(v, (list, tuple, np.ndarray)) else f"{v:.6g}") for k, v in dims.items()})
    for side in SIDES:
        sy = 1 if side == "left" else -1
        knee = urdf_link(root, f"{side}_knee_link")
        for c in knee.findall("collision"):
            knee.remove(c)
        prim(knee, "cylinder", (0, 0, -0.2 * k_ankle), radius=0.023, length=0.15 * k_ankle)
        prim(knee, "cylinder", (0.055, 0, -0.13 * k_ankle), (0, 0.1, 0), radius=0.01, length=0.2 * k_ankle)
        foot = urdf_link(root, f"{side}_ankle_roll_link")
        for c in foot.findall("collision"):
            foot.remove(c)
        prim(foot, "box", (0.035, 0, -0.016), size=(0.15, 0.06, 0.015))
        for xyz in ((-0.05, 0.025 * sy, -0.03), (-0.05, -0.025 * sy, -0.03), (0.12, 0.03 * sy, -0.03),
                    (0.12, -0.03 * sy, -0.03), (0.14, 0, -0.028)):
            prim(foot, "sphere", xyz, radius=0.005)
    tree.write(f"{out}/robot.urdf")
    if not os.path.islink(f"{out}/meshes"):
        os.symlink(SRC_URDF_MESHES, f"{out}/meshes")
    total_mass = sum(float(l.find("inertial/mass").get("value")) for l in root.iter("link") if l.find("inertial") is not None)

    # ---------------- MJCF ----------------
    mtree = ET.parse(SRC_MJCF)
    mroot = mtree.getroot()
    mroot.find("compiler").set("meshdir", f"{SRC_MJCF_DIR}/meshes")
    bodies = {b.get("name"): b for b in mroot.iter("body")}
    bodies["pelvis"].set("pos", fmt(vec(bodies["pelvis"].get("pos")) + np.array([0, 0, dz])))
    for side in SIDES:
        for bn, k in ((f"{side}_knee_link", k_knee), (f"{side}_ankle_pitch_link", k_ankle)):
            bodies[bn].set("pos", fmt(vec(bodies[bn].get("pos")) * k))
        for ln in LEG_LINKS:
            b = bodies[f"{side}_{ln}"]
            inert = b.find("inertial")
            m0 = float(inert.get("mass"))
            s = stretch.get(ln, 1.0)
            com = vec(inert.get("pos")) * s
            if ln in com_over:
                c = np.array(com_over[ln], dtype=float)
                com = c if side == "left" else c * np.array([1, -1, 1])
            inert.set("pos", fmt(com))
            m1 = float(mass_over.get(ln, m0))
            inert.set("mass", f"{m1:.6g}")
            inert.set("diaginertia", fmt(vec(inert.get("diaginertia")) * (m1 / m0) * s * s))
    if "pelvis_mass" in P:
        bodies["pelvis"].find("inertial").set("mass", f"{float(P['pelvis_mass']):.6g}")
    for mesh in mroot.iter("mesh"):
        ln = mesh.get("name", "")
        for st, s in stretch.items():
            if ln.endswith(st) and s != 1.0:
                mesh.set("scale", f"1 1 {s:.6g}")
    for key in mroot.iter("key"):  # keep any keyframe base height consistent
        qpos = key.get("qpos")
        if qpos:
            q = vec(qpos)
            q[2] += dz
            key.set("qpos", fmt(q))
    mtree.write(f"{out}/g1_29dof_kairnlegs_param.xml")
    scene = open(SRC_SCENE).read().replace('file="g1_29dof_kairnlegs.xml"', f'file="{out}/g1_29dof_kairnlegs_param.xml"')
    open(f"{out}/scene.xml", "w").write(scene)

    meta = dict(name=name, thigh_length=float(np.linalg.norm(
        leg_points(root, "left")["left_knee_joint"] - leg_points(root, "left")["left_hip_pitch_joint"])),
        shank_length=float(np.linalg.norm(leg_points(root, "left")["left_ankle_pitch_joint"]
                                          - leg_points(root, "left")["left_knee_joint"])),
        g1_thigh_length=float(thigh0), g1_shank_length=float(shank0), k_knee=float(k_knee), k_ankle=float(k_ankle),
        dz=float(dz), total_mass=float(total_mass), params=P)
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in meta.items() if k != "params"}, indent=2))
    print(f"written: {out}")


if __name__ == "__main__":
    main(sys.argv[1])
