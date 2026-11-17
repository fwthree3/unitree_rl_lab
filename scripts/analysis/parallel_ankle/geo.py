"""Concrete 2-RSU ankle geometry (G1-style: two rods behind the ankle, centre U-joint pitch->roll),
vectorised over sim samples. Shank frame: x fwd, y left, z up, origin = U-joint centre.
geo = (r, b, c, H, beta): rods attach on the foot at (-r, +/-b, 0); crank pivot at (-r + c*cos(beta), +/-b, H + c*sin(beta))
so that at neutral the crank tip sits at (-r, +/-b, H) with the crank at angle beta to the horizontal (beta=0: crank
horizontal pointing back, perpendicular to the vertical rod of length H)."""
import numpy as np
from analyze import RS03, tmax

def Rmat(qp, qr):
    cp, sp, cr, sr = np.cos(qp), np.sin(qp), np.cos(qr), np.sin(qr)
    z, o = np.zeros_like(qp), np.ones_like(qp)
    Ry = np.stack([np.stack([cp, z, sp], -1), np.stack([z, o, z], -1), np.stack([-sp, z, cp], -1)], -2)
    Rx = np.stack([np.stack([o, z, z], -1), np.stack([z, cr, -sr], -1), np.stack([z, sr, cr], -1)], -2)
    return Ry @ Rx

Q0 = [0.0]   # linkage neutral pitch (rad): crank perpendicular to a vertical rod when q_pitch = Q0, roll = 0

def foot_pt(s, geo):
    r, b, c, H, beta = geo
    c0, s0 = np.cos(Q0[0]), np.sin(Q0[0])
    p = np.array([-r, s * b, 0.0])
    return np.array([c0 * p[0] - s0 * p[2], p[1], s0 * p[0] + c0 * p[2]])   # Ry(-Q0) p

def pivot_pt(s, geo):
    r, b, c, H, beta = geo
    return np.array([-r + c * np.cos(beta), s * b, H + c * np.sin(beta)])

def tip(alpha, s, geo):
    r, b, c, H, beta = geo
    a = pivot_pt(s, geo)
    # crank arm direction at angle (beta + alpha): tip = a + c*(-cos, 0, -sin)
    ang = beta + alpha
    return a[None, :] + c * np.stack([-np.cos(ang), np.zeros_like(ang), -np.sin(ang)], -1)

def solve_alpha(qp, qr, s, geo):
    r, b, c, H, beta = geo
    B = (Rmat(qp, qr) @ foot_pt(s, geo))
    a = pivot_pt(s, geo)
    d = a[None, :] - B
    # |a + c*u(ang) - B|^2 = H^2, u = (-cos ang, 0, -sin ang): d.u = K
    K = (H ** 2 - (d * d).sum(1) - c ** 2) / (2 * c)
    A_, B_ = -d[:, 0], -d[:, 2]                 # A cos + B sin = K
    Rr = np.hypot(A_, B_); ph = np.arctan2(B_, A_)
    x = K / Rr
    bad = np.abs(x) > 1
    x = np.clip(x, -1, 1)
    c1 = ph + np.arccos(x); c2 = ph - np.arccos(x)
    w = lambda v: (v - beta + np.pi) % (2 * np.pi) - np.pi     # alpha relative to neutral
    a1, a2 = w(c1), w(c2)
    return np.where(np.abs(a1) < np.abs(a2), a1, a2), B, bad

def analyse(q, qd, tau, fw, geo, mode="A"):
    """q,qd,tau: [N,2] (pitch, roll); fw: [N,3] serial-model joint force on the foot (foot frame)."""
    qp, qr = q[:, 0], q[:, 1]
    R = Rmat(qp, qr)
    yf = R[:, :, 1]; xr = R[:, :, 0]                        # foot y axis (rod-end bore), roll axis (shank frame)
    yhat = np.array([0, 1.0, 0])
    out = {"bad": np.zeros(len(qp), bool)}
    h = 1e-5
    rows, Ts, us, tm, wm = [], [], [], [], []
    for s in (+1, -1):
        al, B, bad = solve_alpha(qp, qr, s, geo); out["bad"] |= bad
        p = tip(al, s, geo)
        u = (p - B) / np.linalg.norm(p - B, axis=1, keepdims=True)    # foot -> crank (tension direction on foot)
        m = np.cross(B, u)                                            # moment per unit tension, about the U-joint centre
        rows.append(np.stack([(m * yhat).sum(1), (m * xr).sum(1)], 1))
        us.append(u)
        # crank-end and foot-end misalignment (bore axes parallel to motor axis y / foot y)
        out[f"mis_crank_{s}"] = np.degrees(np.arcsin(np.abs((u * yhat).sum(1))))
        out[f"mis_foot_{s}"] = np.degrees(np.arcsin(np.abs((u * yf).sum(1))))
        out[f"alpha_{s}"] = np.degrees(al)
        arm = p - pivot_pt(s, geo)[None, :]
        out[f"crank_arm_{s}"] = np.cross(arm, u)[:, 1]               # motor torque per unit rod tension (about y)
        # motor speed by finite difference of alpha along the joint velocity
        al2, _, _ = solve_alpha(qp + h * qd[:, 0], qr + h * qd[:, 1], s, geo)
        wm.append((al2 - al) / h)
    A = np.stack(rows, 2)                                             # [N, 2(gen), 2(rod)]
    T = np.linalg.solve(A, tau[..., None])[..., 0]                    # rod tensions
    out["T"] = T
    tmot = np.stack([-T[:, i] * out[f"crank_arm_{s}"] for i, s in enumerate((+1, -1))], 1)
    out["tau_m"] = tmot; out["w_m"] = np.stack(wm, 1)
    out["util"] = np.abs(tmot) / np.maximum(tmax(out["w_m"], tmot, RS03, mode), 1e-3)
    # pivot (U-joint) force on the foot, foot frame: serial joint force minus the rod forces
    Frod_shank = T[:, 0:1] * us[0] + T[:, 1:2] * us[1]
    Frod_foot = np.einsum("nji,nj->ni", R, Frod_shank)
    out["F_pivot"] = fw - Frod_foot
    out["F_rod_sum"] = Frod_foot
    return out
