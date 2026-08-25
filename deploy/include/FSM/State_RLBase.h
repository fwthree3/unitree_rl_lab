// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <atomic>

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"

class State_RLBase : public FSMState
{
public:
    State_RLBase(int state_mode, std::string state_string);
    
    void enter()
    {
        // set gain
        for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
            lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
            lowcmd->msg_.motor_cmd()[i].dq() = 0;
            lowcmd->msg_.motor_cmd()[i].tau() = 0;
        }

        env->robot->update();

        // kaon fork addition (#192): CtrlFSM::run_() calls this state's run() on its
        // very next 1kHz tick right after enter() returns, and run() reads
        // env->action_manager->processed_actions() -- but that's only ever set inside
        // env->step(), which the policy thread spawned below doesn't reach until after
        // env->reset() plus a full observation-compute + ONNX-inference pass. Without
        // waiting for it, run() sends whatever processed_actions() holds pre-first-step
        // as a *position* target under the kp/kd gains just set above. Confirmed
        // empirically: 100% reproducible bad_orientation trip back to Passive within
        // ~1ms of every single FixStand->Velocity transition, identical timing
        // regardless of mass/friction -- a software race, not a balance failure. Block
        // enter() on the first real step so run()'s first call sees a genuine action.
        first_step_done_ = false;

        // Start policy thread
        policy_thread_running = true;
        policy_thread = std::thread([this]{
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desiredDuration(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

            // Initialize timing
            auto sleepTill = clock::now() + dt;
            env->reset();
            env->step();
            first_step_done_ = true;

            while (policy_thread_running)
            {
                // Sleep
                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;

                env->step();
            }
        });
        while (!first_step_done_) {
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
    }

    void run();
    
    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
    }

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;

    std::thread policy_thread;
    bool policy_thread_running = false;
    std::atomic<bool> first_step_done_{false};
};

REGISTER_FSM(State_RLBase)
