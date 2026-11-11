# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = ""  # same as task name
    empirical_normalization = False
    # M1b.0/#252 (2026-11-10): bound env-side actions. With clip_actions=None, a few envs
    # occasionally ran away (Episode_Reward/action_rate -150 .. -4e4 within ~5 iterations of a
    # resume, mean reward -1.6e6, value loss 1.5e12), then a NaN std tripped the
    # "normal expects std >= 0" CUDA assert. last_action is fed back into the observation
    # unclipped, so one large action feeds the next. Normal operation (model_95300, 1,000
    # episodes, cmd -1..3.5 m/s) peaks at |a| = 6.9 deterministic; 10.0 leaves room for
    # exploration noise (std ~0.42) and only cuts off the runaway tail.
    clip_actions = 10.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        # "scalar" (rsl_rl's default) optimizes the action-distribution std as a raw,
        # unconstrained parameter -- an ordinary Adam step can push it negative, which
        # trips torch.distributions.Normal's "std >= 0.0" CUDA assert. "log" optimizes
        # log(std) instead (exp'd before use), which is positive for any finite value.
        noise_std_type="log",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
