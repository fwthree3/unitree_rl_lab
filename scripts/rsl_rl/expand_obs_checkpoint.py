"""Warm-start a checkpoint for a larger observation by zero-padding the first layer.

M1b.1/#18 (2026-11-11): the perceptive Ascend task appends a height scan after the
existing proprioceptive inputs. New input columns of actor.0 / critic.0 (and their Adam
moments) are zero, so the expanded network computes exactly the same outputs as the
source checkpoint until training moves the new weights.

Usage:
  python scripts/rsl_rl/expand_obs_checkpoint.py SRC.pt DST.pt --actor_add 187 --critic_add 187
"""

import argparse

import torch

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--actor_add", type=int, required=True)
ap.add_argument("--critic_add", type=int, required=True)
a = ap.parse_args()

ck = torch.load(a.src, map_location="cpu", weights_only=False)
sd = ck["model_state_dict"]
for k in sd:
    assert "normalizer" not in k, f"obs normalizer present ({k}); padding it is not implemented"

old_shapes = {}
for name, add in (("actor.0.weight", a.actor_add), ("critic.0.weight", a.critic_add)):
    w = sd[name]
    old_shapes[name] = tuple(w.shape)
    sd[name] = torch.cat([w, torch.zeros(w.shape[0], add, dtype=w.dtype)], dim=1)
    print(f"{name}: {tuple(w.shape)} -> {tuple(sd[name].shape)}")

# Adam state is keyed by parameter index; pad the moments of the two expanded weights
st = ck["optimizer_state_dict"]["state"]
padded = 0
for idx, s in st.items():
    for name, add in (("actor.0.weight", a.actor_add), ("critic.0.weight", a.critic_add)):
        if tuple(s["exp_avg"].shape) == old_shapes[name]:
            for m in ("exp_avg", "exp_avg_sq"):
                s[m] = torch.cat([s[m], torch.zeros(s[m].shape[0], add, dtype=s[m].dtype)], dim=1)
            padded += 1
            break
assert padded == 2, f"expected 2 Adam states to pad, padded {padded}"
torch.save(ck, a.dst)
print(f"saved {a.dst} (iter {ck['iter']})")
