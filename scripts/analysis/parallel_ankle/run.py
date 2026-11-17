# Usage: python run.py <npz dir> r,b,c,H,beta [neutral_pitch_rad]   (nonlinear check of one geometry)
import sys, glob, numpy as np, geo as g
from analyze import load, eval_pair, serial_baseline, stats

def load_full(p):
    d = np.load(p)
    act = d["active"] & d["success"][None, :]
    legs = []
    for li, leg in enumerate((0, 2)):
        legs.append(dict(q=np.stack([d["q"][..., leg][act], d["q"][..., leg + 1][act]], 1),
                         qd=np.stack([d["qd"][..., leg][act], d["qd"][..., leg + 1][act]], 1),
                         tau=np.stack([d["tau"][..., leg][act], d["tau"][..., leg + 1][act]], 1),
                         fw=d["foot_wrench"][:, :, li, :3][act]))
    return {k: np.concatenate([l[k] for l in legs]) for k in legs[0]}

P = lambda x, p=99.9: np.percentile(x, p)
if __name__ == "__main__":
    D = {p.split("/")[-1][:-4]: load_full(p) for p in sorted(glob.glob(sys.argv[1] + "/*.npz"))}
    geo = tuple(float(x) for x in sys.argv[2].split(","))     # r,b,c,H,beta
    g.Q0[0] = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    print(f"geometry r,b,c,H,beta = {geo}, kp={geo[0]/geo[2]:.2f} kr={geo[1]/geo[2]:.2f}, neutral pitch {np.degrees(g.Q0[0]):.1f} deg")
    for k, d in D.items():
        for mode in ("A", "B"):
            o = g.analyse(d["q"], d["qd"], d["tau"], d["fw"], geo, mode)
            tm, wm, T = o["tau_m"], o["w_m"], o["T"]
            if mode == "A":
                print(f"\n[{k}] n={len(T)}  unreachable {o['bad'].mean()*100:.3f}%")
                print(f"  motor |tau| p99.9 {P(np.abs(tm)):.1f} Nm (max {np.abs(tm).max():.0f}), rms {np.sqrt((tm**2).mean(0)).max():.1f} Nm, "
                      f"|w| p99.9 {P(np.abs(wm)):.1f} rad/s")
                al = np.concatenate([o['alpha_1'], o['alpha_-1']])
                print(f"  crank angle range {P(al,0.1):.0f}..{P(al):.0f} deg (from neutral)")
                print(f"  rod tension p0.1/p99.9 {P(T,0.1):.0f} / {P(T):.0f} N (max |T| {np.abs(T).max():.0f})")
                mf = np.concatenate([o['mis_foot_1'], o['mis_foot_-1']]); mc = np.concatenate([o['mis_crank_1'], o['mis_crank_-1']])
                print(f"  rod-end misalignment p99.9 / max: foot end {P(mf):.1f} / {mf.max():.1f} deg, crank end {P(mc):.1f} / {mc.max():.1f} deg")
                Fp = np.linalg.norm(o["F_pivot"], axis=1); Fs = np.linalg.norm(d["fw"], axis=1)
                print(f"  U-joint force p99.9: {P(Fp):.0f} N (serial-ankle joint force {P(Fs):.0f} N); p50 {P(Fp,50):.0f} vs {P(Fs,50):.0f}")
                print(f"  serial-model fw mean (foot frame) {np.round(d['fw'].mean(0),0)}  rod sum mean {np.round(o['F_rod_sum'].mean(0),0)}")
            print(f"  RS03 envelope {mode}: utilisation p99.9 {P(o['util']):.2f}, p100 {o['util'].max():.2f}, steps >1: {100*(o['util']>1).mean():.3f}%")
