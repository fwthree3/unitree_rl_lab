import gymnasium as gym

gym.register(
    id="Unitree-G1-29dof-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# M1.11/#206 toy check -- separate task id, does not affect M1.2's Run A/B/C
gym.register(
    id="Unitree-G1-29dof-Velocity-WaistFT-Toy",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:WaistFTToyEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# M1b.0/#252 -- G1 with Kairn's provisional RS04/RS03 leg actuators + torque terms
gym.register(
    id="Unitree-G1-29dof-Velocity-KairnLegs",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:KairnLegsEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:KairnLegsPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# M1b.1/#18 -- KairnLegs + ascend terrain curriculum (walk/run/ascend unified, blind)
gym.register(
    id="Unitree-G1-29dof-Velocity-KairnLegs-Ascend",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:KairnLegsAscendEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:KairnLegsAscendPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# M1b.1/#18 -- perceptive Ascend: noisy height scan to the actor, clean to the critic
gym.register(
    id="Unitree-G1-29dof-Velocity-KairnLegs-AscendPerceptive",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:KairnLegsAscendPerceptiveEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:KairnLegsAscendPerceptivePlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:PerceptivePPORunnerCfg",
    },
)

# M1b.1/#18 (2026-11-16) -- plan C: Ascend-only perceptive policy (Run stays with P003)
gym.register(
    id="Unitree-G1-29dof-Velocity-KairnLegs-AscendOnly",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:KairnLegsAscendOnlyEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:KairnLegsAscendOnlyPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:PerceptivePPORunnerCfg",
    },
)

# Parallel ankle (2x RS03, coupled) variants of Run (KairnLegs, P003) and Ascend-only (P011)
gym.register(
    id="Unitree-G1-29dof-Velocity-KairnLegs-PAnkle",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:KairnLegsPAnkleEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:KairnLegsPAnklePlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Velocity-KairnLegs-AscendOnly-PAnkle",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:KairnLegsAscendOnlyPAnkleEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:KairnLegsAscendOnlyPAnklePlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:PerceptivePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Velocity-KairnLegs-AscendOnly-Tall",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:KairnLegsAscendOnlyTallEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:KairnLegsAscendOnlyTallPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:PerceptivePPORunnerCfg",
    },
)
