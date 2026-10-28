#pragma once
#include "FSM/FSMState.h"
#include "FSM/State_Passive.h"
#include "BoxTransferTransport.h"
#include <mutex>

// Both FSM states share one receiver/core. The prepare state moves from measured
// joint positions to the selected reference start; it never advances the policy.
class BoxTransferSession {
public:
    box_transfer::Controller core;
    box_transfer::StateReceiver receiver;
    int selection;
    bool prepared=false;
    BoxTransferSession(const YAML::Node& cfg)
      :core(param::parser_policy_dir(cfg["policy_dir"].as<std::string>())),
       receiver(cfg["state_port"].as<int>(),cfg["state_timeout_s"].as<double>()),
       selection(select(cfg["policy"].as<std::string>())){core.reset(selection);}
    static std::shared_ptr<BoxTransferSession> get(){
        static std::weak_ptr<BoxTransferSession> shared;
        auto session=shared.lock();if(!session){session=std::make_shared<BoxTransferSession>(param::config["FSM"]["BoxTransfer"]);shared=session;}
        return session;
    }
    static int select(const std::string& name){
        if(name=="sequence")return -1;
        const std::array<std::string,5> names{"go-to","take","carry","put-down","return"};
        for(int i=0;i<5;++i)if(name==names[i])return i;
        throw std::runtime_error("unknown box-transfer policy: "+name);
    }
};

class State_BoxTransfer : public FSMState {
    using Clock=std::chrono::steady_clock;
    std::shared_ptr<BoxTransferSession> session_;
    bool prepare_,fault_=false,first_=true,announced_=false;
    Clock::time_point next_,started_,lastTickTime_;
    uint32_t lastTick_=0;
    box_transfer::Joints initial_{},target_{},gravity_{};
    bool targetsReady_=false;
    bool haveTick_=false;
public:
    State_BoxTransfer(int state,std::string name):FSMState(state,name),prepare_(param::config["FSM"][name]["prepare"].as<bool>(false)){
        session_=BoxTransferSession::get();
        registered_checks.insert(registered_checks.begin(),{[this]{return fault_;},FSMStringMap.right.at("Passive")});
    }
    void enter()override{
        fault_=false;first_=true;announced_=false;haveTick_=false;targetsReady_=false;next_=started_=Clock::now();
        try{
            if(lowstate->msg_.mode_machine()!=4)throw std::runtime_error("requires 23-DoF G1 mode_machine=4");
            lowcmd->msg_.mode_pr()=0;lowstate->set_timeout_ms(50);
            auto sensors=read();
            if(prepare_){
                session_->prepared=false;session_->core.reset(session_->selection);
                for(int j=0;j<23;++j)initial_[j]=sensors[box_transfer::JointPosition+j];
                // Verify fixed-layout position and object setup before any pose change.
                auto pose=sensors;auto target=session_->core.initialJoints();
                for(int j=0;j<23;++j)pose[box_transfer::JointPosition+j]=target[j];
                session_->core.requireStart(pose);
                spdlog::info("Box transfer: preparing reference start pose over 3 seconds");
            }else{
                if(!session_->prepared)throw std::runtime_error("prepare state has not completed");
                session_->core.reset(session_->selection);session_->core.requireStart(sensors);
                spdlog::info("Box transfer: starting selected policy");
            }
            run();
        }catch(const std::exception& e){fail(e.what());}
    }
    void run()override{
        if(fault_)return;
        try{
            auto now=Clock::now();
            auto sensors=read(); // Freshness is checked at the 1 kHz FSM rate.
            if(now>=next_){
            if(now-next_>std::chrono::milliseconds(20))throw std::runtime_error("policy scheduler missed its deadline");
            next_+=std::chrono::milliseconds(20);
            auto& target=target_;auto& gravity=gravity_;
            if(prepare_){
                double t=std::chrono::duration<double>(now-started_).count();
                double a=std::clamp(t/3.,0.,1.);a=a*a*a*(10+a*(-15+6*a));
                auto end=session_->core.initialJoints();for(int j=0;j<23;++j)target[j]=initial_[j]+a*(end[j]-initial_[j]);
                gravity=session_->core.gravity(sensors);
                if(t>=3&&!session_->prepared){session_->core.requireStart(sensors);session_->prepared=true;spdlog::info("Box transfer prepared. Press R2 + A to run.");}
            }else{
                auto output=session_->core.tick(sensors,!first_);first_=false;target=output.target;gravity=output.gravity;
                if(session_->core.completed()&&!announced_){announced_=true;spdlog::info("Box transfer complete; holding endpoint under policy control. L2 + B returns to Passive.");}
            }
            if(Clock::now()-now>=std::chrono::milliseconds(20))throw std::runtime_error("policy computation exceeded 20 ms");
            targetsReady_=true;
            }
            if(!targetsReady_)return;
            const auto& target=target_;const auto& gravity=gravity_;
            // Bound commanded PD + gravity torque to the training actuator limits.
            // Express any saturation through q_des, preserving motor-side damping.
            for(int j=0;j<23;++j){
                double q=sensors[box_transfer::JointPosition+j],dq=sensors[box_transfer::JointVelocity+j];
                double torque=box_transfer::kKp[j]*(target[j]-q)-box_transfer::kKd[j]*dq+gravity[j];
                double bounded=std::clamp(torque,-box_transfer::kEffort[j],box_transfer::kEffort[j]);
                auto& motor=lowcmd->msg_.motor_cmd()[box_transfer::kMotorIds[j]];
                motor.mode()=1;motor.kp()=box_transfer::kKp[j];motor.kd()=box_transfer::kKd[j];motor.dq()=0;motor.tau()=gravity[j];
                motor.q()=q+(bounded+box_transfer::kKd[j]*dq-gravity[j])/box_transfer::kKp[j];
            }
        }catch(const std::exception& e){fail(e.what());}
    }
    void exit()override{if(!prepare_)session_->prepared=false;}
private:
    box_transfer::Sensors read(){
        if(lowstate->isTimeout())throw std::runtime_error("DDS low-state timed out");
        auto s=session_->receiver.read();
        std::lock_guard<std::mutex> lock(lowstate->mutex_);
        if(lowstate->msg_.mode_machine()!=4)throw std::runtime_error("robot mode changed");
        auto tick=lowstate->msg_.tick();auto now=Clock::now();
        if(!haveTick_||tick!=lastTick_){lastTick_=tick;lastTickTime_=now;haveTick_=true;}
        if(now-lastTickTime_>std::chrono::milliseconds(50))throw std::runtime_error("DDS sensor tick stopped advancing");
        for(int j=0;j<23;++j){auto& motor=lowstate->msg_.motor_state()[box_transfer::kMotorIds[j]];s[box_transfer::JointPosition+j]=motor.q();s[box_transfer::JointVelocity+j]=motor.dq();}
        for(int i=0;i<3;++i)s[box_transfer::Gyro+i]=lowstate->msg_.imu_state().gyroscope()[i];
        // External heading is calibrated to the layout. Roll/pitch must agree
        // with the IMU; absolute IMU yaw need not share the tracking origin.
        auto imu=lowstate->msg_.imu_state().quaternion();
        box_transfer::Q4 q{imu[0],imu[1],imu[2],imu[3]};q=box_transfer::unit(q);
        auto measured=box_transfer::rotate(box_transfer::inverse(q),{0,0,-1});
        auto tracked=box_transfer::rotate(box_transfer::inverse(box_transfer::unit(box_transfer::quat(s.data()+3))),{0,0,-1});
        if(box_transfer::norm(box_transfer::sub(measured,tracked))>.15)throw std::runtime_error("tracker orientation disagrees with IMU tilt");
        return s;
    }
    void fail(const std::string& reason){
        fault_=true;session_->prepared=false;
        for(int id:box_transfer::kMotorIds){auto& motor=lowcmd->msg_.motor_cmd()[id];motor.kp()=0;motor.kd()=3;motor.dq()=0;motor.tau()=0;}
        spdlog::error("Box transfer stopped: {}",reason);
    }
};
REGISTER_FSM(State_BoxTransfer)

// Drain tracking packets while passive so preparation starts from current data.
class State_BoxTransferPassive : public State_Passive {
    std::shared_ptr<BoxTransferSession> session_;
public:
    State_BoxTransferPassive(int state,std::string name):State_Passive(state,name),session_(BoxTransferSession::get()){}
    void enter()override{session_->prepared=false;State_Passive::enter();}
    void run()override{State_Passive::run();try{session_->receiver.read();}catch(const std::exception&){} }
};
REGISTER_FSM(State_BoxTransferPassive)
