#pragma once

#include <chrono>
#include <mutex>
#include "FSM/FSMState.h"
#include "HumanPushPolicy.h"

// Runs in the existing FSM thread. No asynchronous action buffer or gain/index
// assumptions are shared with the walking controller.
class State_HumanPushRecovery : public FSMState {
public:
    State_HumanPushRecovery(int state, std::string name)
      : FSMState(state,name), policy_((param::parser_policy_dir(
            param::config["FSM"][name]["policy_dir"].as<std::string>()) / "exported/policy.onnx").string()) {
        registered_checks.insert(registered_checks.begin(), {
            [this]{ return fault_; }, FSMStringMap.right.at("Passive")});
    }

    void enter() override {
        fault_=false;
        previous_.fill(0);
        next_step_=Clock::now();
        if (lowstate->msg_.mode_machine()!=4) {
            fail("Human-push policy requires the 23-DoF G1 (mode_machine=4)");
            return;
        }
        lowcmd->msg_.mode_pr()=0;  // Ankle pitch/roll coordinates, matching training.
        run();  // Initialize valid targets before the first publish.
    }

    void run() override {
        if (fault_) return;
        if (lowstate->isTimeout()) { fail("Robot state timed out"); return; }
        auto now=Clock::now();
        if (now<next_step_) return;
        // Drop missed periods instead of running a burst of stale policy steps.
        const auto period=std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(human_push::kStepDt));
        next_step_+=period;
        if (next_step_<=now) next_step_=now+period;
        try {
            human_push::Sensors sensors{};
            {
                std::lock_guard<std::mutex> lock(lowstate->mutex_);
                for (size_t i=0;i<4;++i) sensors.quaternion[i]=lowstate->msg_.imu_state().quaternion()[i];
                for (size_t i=0;i<3;++i) sensors.gyro[i]=lowstate->msg_.imu_state().gyroscope()[i];
                for (int id:human_push::kMotorIds) {
                    sensors.position[id]=lowstate->msg_.motor_state()[id].q();
                    sensors.velocity[id]=lowstate->msg_.motor_state()[id].dq();
                }
            }
            const auto observation=human_push::observation(sensors,previous_);
            if (-observation[5]<std::cos(3.141592653589793/4)) {
                fail("Pelvis tilt exceeded the training fall threshold"); return;
            }
            const auto action=policy_.infer(observation);
            const auto target=human_push::targets(action);
            for (size_t i=0;i<23;++i) {
                auto& motor=lowcmd->msg_.motor_cmd()[human_push::kMotorIds[i]];
                motor.mode()=1;
                motor.q()=target[i];
                motor.kp()=human_push::kKp[i];
                motor.kd()=human_push::kKd[i];
                motor.dq()=0;
                motor.tau()=0;
            }
            previous_=action;  // Clipped normalized action, never joint targets.
        } catch (const std::exception& error) { fail(error.what()); }
    }

    void exit() override { previous_.fill(0); }
private:
    using Clock=std::chrono::steady_clock;
    human_push::Policy policy_;
    human_push::Action previous_{};
    Clock::time_point next_step_{};
    bool fault_=false;

    void fail(const std::string& reason) {
        fault_=true;
        // The FSM publishes once before evaluating transitions. Set damping
        // immediately so that tick cannot publish stale active targets.
        for (int id:human_push::kMotorIds) {
            auto& motor=lowcmd->msg_.motor_cmd()[id];
            motor.kp()=0; motor.kd()=3; motor.dq()=0; motor.tau()=0;
        }
        spdlog::error("HumanPushRecovery: {}",reason);
    }
};

REGISTER_FSM(State_HumanPushRecovery)
