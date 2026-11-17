# Usage: python analyze.py <dir with eval_run_success.py --ankle_npz outputs>   (linear kp x kr sweep, 2x RS03)
"""Parallel (2-RSU) ankle feasibility with 2x RS03, from sim ankle traces (serial pitch/roll joints).

Linkage (G1-style, both motors on the shank near the knee, cranks parallel to the pitch axis):
  rod i lower end on the foot at (-r, +/-b, 0) from the ankle centre (behind the pitch axis, +/-b lateral),
  crank length c, crank perpendicular to a vertical rod at neutral.
  Linearised at neutral:  motor speed  w_i = kp*wp + s_i*kr*wr,     kp = r/c, kr = b/c, s = +1/-1
                          motor torque t_i = tp/(2kp) + s_i*tr/(2kr)
  Nonlinear check: full rod-length kinematics with numeric Jacobian, for one concrete geometry.
"""
import sys, glob, numpy as np

RS03 = dict(peak=60.0, rated_t=20.0, rated_w=10.47, noload_w=20.4)
RS04 = dict(peak=120.0, rated_t=40.0, rated_w=17.5, noload_w=20.9)


def tmax(w, tau, m, mode):
    """Torque available at speed w for a motoring (tau*w>0) demand; braking gets the peak."""
    aw = np.abs(w)
    if mode == "A":   # peak flat to rated speed, then linear to no-load
        lim = np.where(aw <= m["rated_w"], m["peak"], m["peak"] * (m["noload_w"] - aw) / (m["noload_w"] - m["rated_w"]))
    else:             # B: straight line peak -> no-load (voltage-limited DC motor), conservative
        lim = m["peak"] * (1 - aw / m["noload_w"])
    lim = np.clip(lim, 0, m["peak"])
    return np.where(tau * w > 0, lim, m["peak"])


def load(paths):
    out = {}
    for p in paths:
        d = np.load(p)
        act = d["active"] & d["success"][None, :]          # successful episodes only
        q, qd, tau = d["q"], d["qd"], d["tau"]             # [T, N, 4] = L pitch, L roll, R pitch, R roll
        samples = []
        for leg in (0, 2):
            sel = act
            samples.append(np.stack([q[..., leg][sel], q[..., leg + 1][sel], qd[..., leg][sel], qd[..., leg + 1][sel],
                                     tau[..., leg][sel], tau[..., leg + 1][sel]], 1))
        out[p.split("/")[-1].replace(".npz", "")] = (np.concatenate(samples), float(d["dt"]))
    return out


def motor_demand(S, kp, kr):
    qp, qr, wp, wr, tp, tr = S.T
    res = []
    for s in (+1, -1):
        res.append((kp * wp + s * kr * wr, tp / (2 * kp) + s * tr / (2 * kr)))
    return res


def eval_pair(S, kp, kr, mode, pct=99.9):
    """Return (worst torque utilisation p-quantile, RMS torque, p99.9 |w|) over both motors."""
    util, rms, wmax = [], [], []
    for w, t in motor_demand(S, kp, kr):
        u = np.abs(t) / np.maximum(tmax(w, t, RS03, mode), 1e-3)
        util.append(np.percentile(u, pct)); rms.append(np.sqrt(np.mean(t ** 2))); wmax.append(np.percentile(np.abs(w), pct))
    return max(util), max(rms), max(wmax)


def serial_baseline(S, mode, pct=99.9):
    qp, qr, wp, wr, tp, tr = S.T
    up = np.percentile(np.abs(tp) / np.maximum(tmax(wp, tp, RS04, mode), 1e-3), pct)
    ur = np.percentile(np.abs(tr) / np.maximum(tmax(wr, tr, RS03, mode), 1e-3), pct)
    return up, ur, np.sqrt(np.mean(tp ** 2)), np.sqrt(np.mean(tr ** 2))


def stats(S):
    qp, qr, wp, wr, tp, tr = S.T
    P = lambda x: np.percentile(np.abs(x), 99.9)
    return (f"|tau_p| p99.9 {P(tp):5.1f} Nm  |tau_r| {P(tr):5.1f} Nm  |w_p| {P(wp):5.1f}  |w_r| {P(wr):5.1f} rad/s  "
            f"q_p {np.percentile(qp,0.1):+.2f}..{np.percentile(qp,99.9):+.2f}  q_r {np.percentile(qr,0.1):+.2f}..{np.percentile(qr,99.9):+.2f} rad  "
            f"rms tau_p {np.sqrt(np.mean(tp**2)):.1f} tau_r {np.sqrt(np.mean(tr**2)):.1f}")


# ---------- nonlinear 2-RSU kinematics ----------
def rot(qp, qr):
    cp, sp, cr, sr = np.cos(qp), np.sin(qp), np.cos(qr), np.sin(qr)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Ry @ Rx                                  # pitch (shank side) then roll (foot side)


def crank_angles(qp, qr, r, b, c, H):
    """Crank angle of each motor so that both rods (length H) close the loop."""
    al = []
    for s in (+1, -1):
        B = rot(qp, qr) @ np.array([-r, s * b, 0.0])
        a = np.array([-r + c, s * b, H])           # crank pivot; tip at (-r, s*b, H) when alpha=0
        d = a - B
        K = (H ** 2 - d @ d - c ** 2) / (2 * c)     # d . u(alpha) = K,  u = (-cos a, 0, sin a)
        # -dx cos a + dz sin a = K  ->  R cos(a + phi) form
        A_, B_ = -d[0], d[2]
        Rr = np.hypot(A_, B_); ph = np.arctan2(B_, A_)
        x = np.clip(K / Rr, -1, 1)
        cands = [ph + np.arccos(x), ph - np.arccos(x)]
        cands = [(cc + np.pi) % (2 * np.pi) - np.pi for cc in cands]
        al.append(min(cands, key=abs))
    return np.array(al)


def jac(qp, qr, geo, h=1e-5):
    J = np.zeros((2, 2))
    for k, (dp, dr) in enumerate(((h, 0), (0, h))):
        J[:, k] = (crank_angles(qp + dp, qr + dr, *geo) - crank_angles(qp - dp, qr - dr, *geo)) / (2 * h)
    return J                                        # d alpha / d q


def nonlinear_check(S, geo, mode, n=200000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(S), size=min(n, len(S)), replace=False)
    sub = S[idx]
    # Jacobian on a grid over the joint range, interpolated (fast)
    gp = np.linspace(sub[:, 0].min(), sub[:, 0].max(), 25); gr = np.linspace(sub[:, 1].min(), sub[:, 1].max(), 15)
    JJ = np.array([[jac(p, r, geo) for r in gr] for p in gp])
    ip = np.clip(np.searchsorted(gp, sub[:, 0]), 0, len(gp) - 1); ir = np.clip(np.searchsorted(gr, sub[:, 1]), 0, len(gr) - 1)
    J = JJ[ip, ir]                                    # [n,2,2]
    w = np.einsum("nij,nj->ni", J, sub[:, 2:4])
    tm = np.linalg.solve(np.transpose(J, (0, 2, 1)), sub[:, 4:6][..., None])[..., 0]   # tau_q = J^T tau_m
    util = np.abs(tm) / np.maximum(tmax(w, tm, RS03, mode), 1e-3)
    al = np.array([[crank_angles(p, r, *geo) for r in gr] for p in gp])
    return np.percentile(util, 99.9), np.sqrt(np.mean(tm ** 2, 0)).max(), np.percentile(np.abs(w), 99.9), np.degrees(al.min()), np.degrees(al.max()), np.abs(np.linalg.det(JJ)).min() / np.abs(np.linalg.det(JJ)).max()


if __name__ == "__main__":
    data = load(sorted(glob.glob(sys.argv[1] + "/*.npz")))
    allS = np.concatenate([v[0] for v in data.values()])
    for k, (S, _) in data.items():
        print(f"{k:10s} n={len(S):8d}  {stats(S)}")
    print()
    for mode in ("A", "B"):
        print(f"=== torque-speed envelope {mode} ===")
        for k, (S, _) in list(data.items()) + [("ALL", (allS, 0))]:
            up, ur, rp, rr = serial_baseline(S, mode)
            print(f"serial baseline {k:10s}: RS04 pitch util p99.9 {up:.2f}, RS03 roll util {ur:.2f}, rms {rp:.1f}/{rr:.1f} Nm")
        kps = np.round(np.arange(0.6, 2.01, 0.1), 2); krs = np.round(np.arange(0.4, 2.01, 0.1), 2)
        U = np.array([[eval_pair(allS, kp, kr, mode)[0] for kr in krs] for kp in kps])
        print("worst-motor torque utilisation p99.9 (ALL scenarios; <1 = feasible), rows kp, cols kr:")
        print("kp\\kr " + " ".join(f"{x:4.1f}" for x in krs))
        for i, kp in enumerate(kps):
            print(f"{kp:4.1f}  " + " ".join(f"{u:4.2f}" for u in U[i]))
        i, j = np.unravel_index(U.argmin(), U.shape)
        print(f"best kp={kps[i]} kr={krs[j]} util {U[i,j]:.2f}")
        for k, (S, _) in data.items():
            u, rms, wm = eval_pair(S, kps[i], krs[j], mode)
            print(f"   {k:10s} util {u:.2f}  rms {rms:.1f} Nm  |w_m| p99.9 {wm:.1f} rad/s")
        print()
