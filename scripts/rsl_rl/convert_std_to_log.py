"""Convert a checkpoint's scalar 'std' actor-critic parameter to log-parameterized 'log_std'.

Needed after switching RslRlPpoActorCriticCfg.noise_std_type from the rsl_rl default
"scalar" to "log" (see rsl_rl_ppo_cfg.py) -- old checkpoints store a raw 'std' tensor
and the "log" model expects a 'log_std' tensor instead, so a direct --resume otherwise
fails with a state_dict key mismatch. This produces a converted copy so training can
resume from an existing checkpoint instead of restarting from scratch.

Usage:
    python scripts/rsl_rl/convert_std_to_log.py <path/to/model_XXXXX.pt>

Writes <path/to/model_XXXXX_logstd.pt> next to the input by default.
"""

import argparse

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    if "log_std" in sd:
        raise SystemExit("checkpoint already has 'log_std' -- nothing to convert")
    if "std" not in sd:
        raise SystemExit(f"no 'std' key found in model_state_dict; keys: {list(sd.keys())}")

    std = sd.pop("std")
    if (std <= 0).any():
        raise SystemExit(f"found non-positive std value(s), cannot take log: {std}")
    sd["log_std"] = torch.log(std)

    # The std/log_std parameter's old Adam moments were computed in a different
    # gradient space and aren't meaningful after this swap. Clear optimizer 'state'
    # (Adam re-initializes it lazily on the next step) but keep 'param_groups' as-is
    # so the adaptive schedule's current learning rate carries over.
    if "optimizer_state_dict" in ckpt:
        ckpt["optimizer_state_dict"]["state"] = {}

    out_path = args.output or args.checkpoint.rsplit(".pt", 1)[0] + "_logstd.pt"
    torch.save(ckpt, out_path)
    print(f"iter={ckpt.get('iter')}  log_std range [{sd['log_std'].min():.4f}, {sd['log_std'].max():.4f}]")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
