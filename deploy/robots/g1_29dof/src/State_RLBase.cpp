#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <unordered_map>

namespace isaaclab
{
// keyboard velocity commands example
// change "velocity_commands" observation name in policy deploy.yaml to "keyboard_velocity_commands"
REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    std::string key = FSMState::keyboard->key();
    static auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    // kaon fork addition (#192): each key used to send a hardcoded +-1.0 regardless of
    // what the policy was actually trained on. cfg here (deploy.yaml's own
    // commands.base_velocity.ranges) already reflects the full trained curriculum
    // envelope at export time -- e.g. lin_vel_y only ever reached +-0.3 -- so a raw
    // +-1.0 strafe/yaw command is 3-5x outside anything the policy has ever seen. That
    // was implementing the TODO below by *not* limiting the command, which turned every
    // sim2sim robustness probe into "command something never trained" rather than a
    // real robustness reading.
    //
    // Scale to a fraction of each axis's trained extreme rather than the extreme itself:
    // curriculum-based training (UniformLevelVelocityCommandCfg) spends most of its time
    // near the low end and only reaches limit_ranges as the curriculum expands, so the
    // ceiling is plausibly under-trained relative to a typical/representative speed --
    // exercising it should be a deliberate, separate probe, not the default "walk" test.
    // kaon fork addition (#192): 0.5 confirmed empirically -- v1_model40500 completes
    // all sim2sim probes cleanly at this fraction of the trained envelope, but reliably
    // falls partway through a sustained *forward* command at the full trained max
    // (kMidFrac=1.0, A/B tested this session) -- a real, reproducible speed-ceiling
    // finding, not a harness bug. Keep at 0.5 for the actual mass/friction/delay/noise
    // sweep so falls there are attributable to the perturbation being swept, not to
    // also pushing the baseline policy's own edge at the same time.
    static float kMidFrac = 0.5f;
    static float vx_max = kMidFrac * cfg["lin_vel_x"][1].as<float>();
    static float vx_min = kMidFrac * cfg["lin_vel_x"][0].as<float>();
    static float vy_max = kMidFrac * cfg["lin_vel_y"][1].as<float>();
    static float vy_min = kMidFrac * cfg["lin_vel_y"][0].as<float>();
    static float wz_max = kMidFrac * cfg["ang_vel_z"][1].as<float>();
    static float wz_min = kMidFrac * cfg["ang_vel_z"][0].as<float>();

    static std::unordered_map<std::string, std::vector<float>> key_commands = {
        {"w", {vx_max, 0.0f, 0.0f}},
        {"s", {vx_min, 0.0f, 0.0f}},
        {"a", {0.0f, vy_max, 0.0f}},
        {"d", {0.0f, vy_min, 0.0f}},
        {"q", {0.0f, 0.0f, wz_max}},
        {"e", {0.0f, 0.0f, wz_min}}
    };
    std::vector<float> cmd = {0.0f, 0.0f, 0.0f};
    if (key_commands.find(key) != key_commands.end())
    {
        // TODO: smooth the velocity commands (still an instant step, just within range now)
        cmd = key_commands[key];
    }
    return cmd;
}

}

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}