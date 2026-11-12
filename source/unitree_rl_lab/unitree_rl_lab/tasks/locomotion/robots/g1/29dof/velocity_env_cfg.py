import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG
from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_KAIRN_LEGS_CFG
from unitree_rl_lab.tasks.locomotion import mdp

COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    # M1b.0/#252 (2026-11-10): was 20.0. With a 20 m border, an env spawned on an edge
    # tile has only ~24 m of ground in the worst direction, i.e. <10 s at Run speed
    # (2.5 m/s) in a 20 s episode. A 1,000-episode eval of model_74000 at 2.5-3.0 m/s
    # showed 262/262 "falls" were the robot running off the terrain edge (0 falls in
    # the first 9 s; 100% success on an infinite plane). 80 m covers 20 s at 4 m/s.
    border_width=80.0,
    num_rows=9,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
    },
)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # "plane", "generator"
        terrain_generator=COBBLESTONE_ROAD_CFG,  # None, ROUGH_TERRAINS_CFG
        max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.1, 0.1)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            # Run D update (M1.2/#11, 2026-09-17): raised from (-0.5, 3.0) to push past
            # the ambiguous "fast mincing walk vs real run" boundary near 3.0 m/s and force
            # an unambiguous flight-phase gait. Curriculum (lin_vel_cmd_levels) only expands
            # into this range while tracking reward stays high, so an unreachable ceiling
            # just caps out naturally rather than forcing bad behavior. Actuator sizing
            # (M1.4) should still pull torque/velocity data from the ~2.5-3.0 m/s operating
            # point, not this full range -- this ceiling is a training aid, not the target.
            # M1b.0/#252 (2026-11-10): upper bound 10.0 -> 3.5. The 10 m/s ceiling was a
            # training aid to force a flight-phase gait (now established); uniformly
            # sampled, it left only ~4.5% of commands in the 2.5-3.0 m/s Run band, and
            # actual speed undershoots that band by 0.1-0.4 m/s. 3.5 gives margin above
            # the band while concentrating samples on it (~11%).
            lin_vel_x=(-1.0, 3.5), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.2, 0.2)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        # clip bounds (2026-09-19, M1.2/#11 crash investigation): a rare PhysX contact
        # glitch can produce one extreme velocity value in one of 4096 envs; unclipped,
        # it feeds straight into this forward pass and can produce a NaN/invalid action
        # std (the recurring CUDA assert crash). Bounds are generous -- well past any
        # legitimate value even at the widened 10 m/s command range -- so normal
        # operation is untouched; only the outlier tail gets clamped.
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2), clip=(-20.0, 20.0)
        )
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5), clip=(-30.0, 30.0)
        )
        last_action = ObsTerm(func=mdp.last_action)
        # gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-15.0, 15.0))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-20.0, 20.0))
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-30.0, 30.0))
        last_action = ObsTerm(func=mdp.last_action)
        # gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})
        # height_scanner = ObsTerm(func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     clip=(-1.0, 5.0),
        # )

        def __post_init__(self):
            self.history_length = 5

    # privileged observations
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    alive = RewTerm(func=mdp.is_alive, weight=0.15)

    # -- base
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*",
                ],
            )
        },
    )
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist.*",
                ],
            )
        },
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )

    # -- robot
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-10, params={"target_height": 0.78})

    # -- feet
    gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.5,
        params={
            "period": 0.8,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            # Run B (M1.2/#11): relax flight-phase suppression above the human
            # walk-run transition speed (~2.0 m/s, Froude ~0.5 -- standard
            # biomechanics literature value; not in
            # docs/notes/human-joint-kinetics-reference-260829.md, which covers
            # joint torque/power, not gait-transition speed). threshold_run=0.4
            # opens a real double-swing (both-feet-off-ground) window; ramped
            # over run_transition_width so the reward doesn't jump discontinuously.
            "threshold_run": 0.4,
            "run_transition_speed": 2.0,
            "run_transition_width": 0.5,
            # Run D (M1.2/#11): Run B/C's threshold-only relaxation plateaued (~2% flight-phase
            # steps, flat across 25k-44k iterations) with a "mincing fast walk" instead of
            # running -- period was left fixed at the walking cadence (0.8s) regardless of
            # speed, which may have been forcing stride-length-only scaling instead of the
            # higher-cadence/shorter-ground-contact-time shift real running actually needs.
            # period_run=0.55 shortens the gait cycle in the same speed band the threshold
            # already relaxes in.
            "period_run": 0.55,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=1.0,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        },
    )

    # -- other
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # Reverted 2026-10-03: the enable_external_forces_every_iteration=True +
        # solver_velocity_iteration_count=6 PhysX re-layering (applied 2026-10-01 on top
        # of the noise_std_type="log" fix, aimed at the same suspected contact/velocity
        # NaN source) was followed by a run of 6 consecutive sub-20s instant-crash-on-resume
        # failures across multiple different checkpoints (94400, 94500), which is a
        # qualitatively different and much worse failure mode than anything seen before
        # or during the ~29-crash combo history. Reverting to isolate whether this
        # layering is itself the cause before re-attempting it.

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


# M1.11/#206 toy check: does adding the waist F/T signal to the policy's observations
# break anything in the training pipeline? Not a real policy, not used by M1.2's Run
# A/B/C -- separate task id so it can't collide with or accidentally affect them.
@configclass
class WaistFTToyObservationsCfg(ObservationsCfg):
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        waist_force_torque = ObsTerm(
            func=mdp.waist_force_torque,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class WaistFTToyEnvCfg(RobotEnvCfg):
    observations: WaistFTToyObservationsCfg = WaistFTToyObservationsCfg()


# M1b.0/#252 (2026-11-10): same task, but with Kairn's provisional leg actuators
# (RS04/RS03 torque + rated-speed limits, see UNITREE_G1_29DOF_KAIRN_LEGS_CFG) and two
# torque terms. Real-speed re-measurement showed hip pitch / knee peak demand ~119 Nm
# (RS04 peak 120 Nm) and hip pitch mean |tau| ~44 Nm (RS04 rated 40 Nm); this variant
# checks whether Run still holds >=95% with the policy shaped to stay inside margin.
_KAIRN_LEG_JOINTS = [".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_hip_yaw_joint", ".*_knee_joint", ".*_ankle_.*_joint"]


@configclass
class KairnLegsSceneCfg(RobotSceneCfg):
    robot: ArticulationCfg = UNITREE_G1_29DOF_KAIRN_LEGS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class KairnLegsRewardsCfg(RewardsCfg):
    # peaks: zero below 70% of each actuator's peak rating, quadratic above it
    # (pre-clip demand, so clipped demand is still penalized)
    leg_torque_soft_limit = RewTerm(
        func=mdp.torque_soft_limit,
        weight=-10.0,
        params={"soft_ratio": 0.7, "asset_cfg": SceneEntityCfg("robot", joint_names=_KAIRN_LEG_JOINTS)},
    )
    # continuous load: small L2 on leg torques (targets the ~40 Nm mean hip-pitch bias)
    leg_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_KAIRN_LEG_JOINTS)},
    )


@configclass
class KairnLegsEnvCfg(RobotEnvCfg):
    scene: KairnLegsSceneCfg = KairnLegsSceneCfg(num_envs=4096, env_spacing=2.5)
    rewards: KairnLegsRewardsCfg = KairnLegsRewardsCfg()


@configclass
class KairnLegsPlayEnvCfg(KairnLegsEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


# M1b.1/#18 Ascend (2026-11-11): walk/run/ascend as ONE policy, same reward, ascend
# added purely through terrain curriculum (decision recorded on #168 / the
# policy-unification survey). Blind (proprioception only -- obs unchanged, so it
# warm-starts from the KairnLegs Run policy). Flat stays ~25% of tiles so Run is retained.
# Step/box heights stay in the stair/curb range a blind policy can plausibly learn;
# skill-targets.md's ~40 cm obstacle reference likely needs perception (follow-up).
ASCEND_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=80.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.25),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.15, step_height_range=(0.05, 0.20), step_width=0.3,
            platform_width=3.0, border_width=1.0, holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15, step_height_range=(0.05, 0.20), step_width=0.3,
            platform_width=3.0, border_width=1.0, holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.1, grid_width=0.45, grid_height_range=(0.02, 0.15), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15, noise_range=(0.01, 0.08), noise_step=0.01, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
)


@configclass
class KairnLegsAscendSceneCfg(KairnLegsSceneCfg):
    def __post_init__(self):
        super().__post_init__()
        self.terrain.terrain_generator = ASCEND_TERRAINS_CFG
        self.terrain.max_init_terrain_level = 3  # start easy; terrain_levels curriculum climbs


@configclass
class KairnLegsAscendRewardsCfg(KairnLegsRewardsCfg):
    def __post_init__(self):
        # same terms/weights as Run; only the height references follow the terrain
        self.base_height.params["sensor_cfg"] = SceneEntityCfg("height_scanner")
        self.feet_clearance.func = mdp.foot_clearance_reward_terrain
        self.feet_clearance.params["sensor_cfg"] = SceneEntityCfg("height_scanner")
        # M1b.1/#20 (2026-11-11): gate style terms by tracking quality (see rewards._tracking_gate)
        self.feet_clearance.params["track_gate_std"] = 0.5
        self.gait.params["track_gate_std"] = 0.5


@configclass
class KairnLegsAscendTerminationsCfg(TerminationsCfg):
    base_height = DoneTerm(
        func=mdp.root_height_below_minimum_terrain,
        params={"minimum_height": 0.2, "sensor_cfg": SceneEntityCfg("height_scanner")},
    )


@configclass
class KairnLegsAscendEnvCfg(KairnLegsEnvCfg):
    scene: KairnLegsAscendSceneCfg = KairnLegsAscendSceneCfg(num_envs=4096, env_spacing=2.5)
    rewards: KairnLegsAscendRewardsCfg = KairnLegsAscendRewardsCfg()
    terminations: KairnLegsAscendTerminationsCfg = KairnLegsAscendTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # M1b.1/#20 (2026-11-11): full Run command range (-1..3.5 m/s) on stairs/boxes made
        # the first Ascend run collapse into standing still. Cap lin_vel_x on non-flat tiles
        # only; flat tiles keep the Run range (and the restored curriculum state).
        self.commands.base_velocity.class_type = mdp.TerrainCappedVelocityCommand
        self.commands.base_velocity.capped_lin_vel_x = (-0.5, 1.5)
        self.commands.base_velocity.uncapped_sub_terrains = ("flat",)


@configclass
class KairnLegsAscendPlayEnvCfg(KairnLegsAscendEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 10
        self.scene.terrain.max_init_terrain_level = None
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


# M1b.1/#18 (2026-11-11): perceptive Ascend. Blind stair climbing has a hard limit (the
# policy can only assume flat ground until a foot hits something), so the policy gets a
# height scan -- with a perception-error model, since on hardware the scan comes from an
# estimated elevation map, never ground truth. Separate obs groups (no history) so the
# existing policy/critic inputs keep their layout and P003 can be warm-started by
# zero-padding the first layer (scripts/rsl_rl/expand_obs_checkpoint.py).
@configclass
class PerceptiveObservationsCfg(ObservationsCfg):
    @configclass
    class HeightScanCfg(ObsGroup):
        height_scan = ObsTerm(
            func=mdp.noisy_height_scan,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "bias_z": 0.05,  # map drift, m
                "shift_xy": 0.05,  # localization error, m
                "noise_std": 0.02,  # per-cell, m
                "p_missing": 0.05,
                "p_outlier": 0.01,
                "outlier_mag": 0.3,
                "max_delay": 2,  # policy steps (0-40 ms)
                "p_blackout": 0.05,  # episodes with no map at all
            },
        )

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = True

    @configclass
    class HeightScanCleanCfg(ObsGroup):
        height_scan = ObsTerm(
            func=mdp.height_scan, params={"sensor_cfg": SceneEntityCfg("height_scanner")}, clip=(-1.0, 1.0)
        )

        def __post_init__(self):
            self.concatenate_terms = True

    height_scan: HeightScanCfg = HeightScanCfg()
    height_scan_clean: HeightScanCleanCfg = HeightScanCleanCfg()


@configclass
class PerceptiveCurriculumCfg(CurriculumCfg):
    # report-only: mean terrain level per sub-terrain (must come after terrain_levels)
    terrain_levels_by_type = CurrTerm(func=mdp.terrain_levels_by_sub_terrain)


@configclass
class KairnLegsAscendPerceptiveEnvCfg(KairnLegsAscendEnvCfg):
    observations: PerceptiveObservationsCfg = PerceptiveObservationsCfg()
    curriculum: PerceptiveCurriculumCfg = PerceptiveCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # M1b.1/#21 (2026-11-11), from P004 (perceptive 115300) evals:
        # (A) Run regression -- 3.6-4.5 m/s ramp eval 100% (P003) -> 0%. Half of the flat-tile
        #     commands are drawn from 2.5-4.5 m/s, and flat tiles go 25% -> 30%.
        # (B) Stairs/boxes fail at difficulty 1.0 (stairs 8%, inverted stairs 0%, boxes 15%)
        #     while slopes/rough are solved (92-98%). Shift the column budget to stairs/boxes.
        tg = self.scene.terrain.terrain_generator = self.scene.terrain.terrain_generator.copy()
        tg.sub_terrains = {k: v.copy() for k, v in tg.sub_terrains.items()}
        for name, prop in {
            "flat": 0.30,
            "pyramid_stairs": 0.20,
            "pyramid_stairs_inv": 0.20,
            "boxes": 0.12,
            "random_rough": 0.08,
            "hf_pyramid_slope": 0.05,
            "hf_pyramid_slope_inv": 0.05,
        }.items():
            tg.sub_terrains[name].proportion = prop
        self.commands.base_velocity.uncapped_fast_prob = 0.5
        # v2.1 (2026-11-12): 2.5-4.5 -> 2.0-3.5. With 4.5 m/s *step* commands (resampled every
        # 10 s, no ramp) the training fall rate rose to 0.65 and the 3.0-4.5 ramp eval went
        # 26% (P004) -> 41% (117300) -> 16% (119300). P003 never trained above 3.5 m/s either;
        # its 3.6-4.5 m/s eval success was generalization from a 3.5 m/s ceiling.
        self.commands.base_velocity.uncapped_fast_lin_vel_x = (2.0, 3.5)


@configclass
class KairnLegsAscendPerceptivePlayEnvCfg(KairnLegsAscendPerceptiveEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.curriculum.terrain_levels_by_type = None
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 10
        self.scene.terrain.max_init_terrain_level = None
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
