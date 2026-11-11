# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""


import gymnasium as gym
import pathlib
import sys

sys.path.insert(0, f"{pathlib.Path(__file__).parent.parent}")
from list_envs import import_packages  # noqa: F401

sys.path.pop(0)

tasks = []
for task_spec in gym.registry.values():
    if "Unitree" in task_spec.id and "Isaac" not in task_spec.id:
        tasks.append(task_spec.id)

import argparse

import argcomplete

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, choices=tasks, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
argcomplete.autocomplete(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# for distributed training, check minimum supported rsl-rl version
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import inspect
import json
import os
import shutil
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner  # TODO: Consider printing the experiment name in the terminal.

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# The velocity-command curriculum (curriculums.py's lin_vel_cmd_levels/
# ang_vel_cmd_levels) mutates env.command_manager's live range config
# in-place; RSL-RL's checkpoint only saves network weights, not this, so
# every single --resume (crash-triggered or not -- a brand-new run folder
# and environment object are created on every launch) silently restarts
# the velocity range from scratch. Measured impact: ~22min of training
# only climbs the commanded range from 0.1 to 0.9 m/s, so frequent
# resumes were preventing training from ever spending sustained time in
# the higher-speed regime where running/falling behavior actually lives
# -- the real fix is here, not in reducing crash frequency. Saved as a
# sidecar JSON next to each checkpoint rather than patching rsl_rl itself
# (a vendored third-party package) or the checkpoint format.
CURRICULUM_STATE_FILENAME = "curriculum_state.json"
_CURRICULUM_RANGE_FIELDS = ("lin_vel_x", "lin_vel_y", "ang_vel_z")


def _get_curriculum_state(env) -> dict:
    ranges = env.unwrapped.command_manager.get_term("base_velocity").cfg.ranges
    return {field: list(getattr(ranges, field)) for field in _CURRICULUM_RANGE_FIELDS}


def _set_curriculum_state(env, state: dict) -> None:
    ranges = env.unwrapped.command_manager.get_term("base_velocity").cfg.ranges
    for field in _CURRICULUM_RANGE_FIELDS:
        if field in state:
            setattr(ranges, field, state[field])


class NonFiniteGuardWrapper(RslRlVecEnvWrapper):
    """M1b.1/#20 (2026-11-11): keep a NaN/Inf physics state from poisoning PPO.

    The "normal expects std >= 0" CUDA assert came back on rough terrain even with
    clip_actions=10 and no action/value-loss spike beforehand -- consistent with a rare
    PhysX glitch producing a non-finite robot state. Observation clipping does not remove
    NaN, and a NaN state also escapes termination (NaN comparisons are False), so one bad
    env can push NaN through the batch into the gradient and the weights. Here, any env
    whose observations, reward or root state are non-finite after a step is zeroed,
    marked done (non-timeout) and reset immediately. Occurrences are appended to
    <log_dir>/nonfinite_guard.log so the rate is visible.
    """

    def __init__(self, env, clip_actions=None, log_dir=None):
        super().__init__(env, clip_actions=clip_actions)
        self._guard_log = os.path.join(log_dir, "nonfinite_guard.log") if log_dir else None
        self._guard_count = 0

    def step(self, actions):
        obs, rew, dones, extras = super().step(actions)
        bad = ~torch.isfinite(rew)
        for key in obs.keys():
            bad |= ~torch.isfinite(obs[key]).reshape(self.num_envs, -1).all(dim=1)
        root = self.unwrapped.scene["robot"].data.root_state_w
        bad |= ~torch.isfinite(root).all(dim=1)
        if bool(bad.any()):
            ids = bad.nonzero(as_tuple=False).squeeze(-1)
            for key in obs.keys():
                obs[key][ids] = 0.0
            rew[ids] = 0.0
            dones[ids] = 1
            if "time_outs" in extras:
                extras["time_outs"][ids] = False
            self.unwrapped._reset_idx(ids)
            self.unwrapped.scene.write_data_to_sim()
            self.unwrapped.sim.forward()
            self._guard_count += len(ids)
            if self._guard_log:
                with open(self._guard_log, "a") as f:
                    f.write(f"step={self.unwrapped.common_step_counter} envs={ids.tolist()} total={self._guard_count}\n")
        return obs, rew, dones, extras


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # This way, the Ray Tune workflow can extract experiment name.
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = NonFiniteGuardWrapper(env, clip_actions=agent_cfg.clip_actions, log_dir=log_dir)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)

    # Piggyback the curriculum-state sidecar write on every checkpoint save
    # (runner.save() already runs at well-defined iteration intervals, see
    # on_policy_runner.py) rather than adding a separate per-step hook.
    _original_save = runner.save

    def _save_with_curriculum_state(path, infos=None):
        _original_save(path, infos)
        state = _get_curriculum_state(env)
        with open(os.path.join(os.path.dirname(path), CURRICULUM_STATE_FILENAME), "w") as f:
            json.dump(state, f)

    runner.save = _save_with_curriculum_state

    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)
        curriculum_state_path = os.path.join(os.path.dirname(resume_path), CURRICULUM_STATE_FILENAME)
        if os.path.exists(curriculum_state_path):
            with open(curriculum_state_path) as f:
                curriculum_state = json.load(f)
            _set_curriculum_state(env, curriculum_state)
            print(f"[INFO]: Restored velocity-command curriculum state from {curriculum_state_path}: {curriculum_state}")
        else:
            print(
                f"[INFO]: No curriculum-state sidecar at {curriculum_state_path} -- velocity-command range starts "
                "from velocity_env_cfg.py's configured initial ranges, not a restored value."
            )

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    export_deploy_cfg(env.unwrapped, log_dir)
    # copy the environment configuration file to the log directory
    shutil.copy(
        inspect.getfile(env_cfg.__class__),
        os.path.join(log_dir, "params", os.path.basename(inspect.getfile(env_cfg.__class__))),
    )

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
