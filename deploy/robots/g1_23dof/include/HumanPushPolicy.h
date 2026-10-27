#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "onnxruntime_cxx_api.h"
#include "../config/policy/human_push/v1/params/PolicyConfig.h"

namespace human_push {

struct Sensors {
    std::array<float, 4> quaternion;  // Pelvis orientation in world, wxyz.
    std::array<float, 3> gyro;        // Pelvis angular velocity, local axes.
    std::array<float, 29> position;   // Unitree SDK motor IDs, including gaps.
    std::array<float, 29> velocity;
};

using Action = std::array<float, 23>;
using Observation = std::array<float, 75>;

inline Observation observation(const Sensors& sensors, const Action& previous) {
    Observation obs{};
    float norm2 = 0;
    for (float value : sensors.quaternion) norm2 += value * value;
    if (!std::isfinite(norm2) || norm2 < 1e-8f)
        throw std::runtime_error("Invalid pelvis IMU quaternion");
    const float inv_norm = 1.0f / std::sqrt(norm2);
    const float w=sensors.quaternion[0]*inv_norm, x=sensors.quaternion[1]*inv_norm;
    const float y=sensors.quaternion[2]*inv_norm, z=sensors.quaternion[3]*inv_norm;
    std::copy(sensors.gyro.begin(), sensors.gyro.end(), obs.begin());
    obs[3] = 2.0f * (w*y - x*z);
    obs[4] = -2.0f * (y*z + w*x);
    obs[5] = -1.0f + 2.0f * (x*x + y*y);
    for (size_t i=0; i<23; ++i) {
        obs[6+i] = sensors.position[kMotorIds[i]] - kDefault[i];
        obs[29+i] = sensors.velocity[kMotorIds[i]];
        obs[52+i] = previous[i];
    }
    for (float value : obs)
        if (!std::isfinite(value)) throw std::runtime_error("Non-finite policy observation");
    return obs;
}

inline Action targets(const Action& action) {
    Action q{};
    for (size_t i=0; i<23; ++i) {
        if (!std::isfinite(action[i])) throw std::runtime_error("Non-finite action");
        q[i] = kDefault[i] + kScale[i] * std::clamp(action[i], -1.0f, 1.0f);
    }
    return q;
}

class Policy {
public:
    explicit Policy(const std::string& path)
      : runtime_(ORT_LOGGING_LEVEL_WARNING, "g1_human_push") {
        options_.SetIntraOpNumThreads(1);
        options_.SetInterOpNumThreads(1);
        options_.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
        session_ = std::make_unique<Ort::Session>(runtime_, path.c_str(), options_);
        if (session_->GetInputCount()!=1 || session_->GetOutputCount()!=1)
            throw std::runtime_error("Policy must have one input and one output");
        const auto in_type = session_->GetInputTypeInfo(0);
        const auto out_type = session_->GetOutputTypeInfo(0);
        const auto in = in_type.GetTensorTypeAndShapeInfo();
        const auto out = out_type.GetTensorTypeAndShapeInfo();
        if (in.GetShape()!=std::vector<int64_t>{1,75} || out.GetShape()!=std::vector<int64_t>{1,23}
            || in.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
            || out.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            throw std::runtime_error("Policy must use float32 [1,75] -> [1,23]");
        Ort::AllocatorWithDefaultOptions allocator;
        if (std::string(session_->GetInputNameAllocated(0,allocator).get())!="obs"
            || std::string(session_->GetOutputNameAllocated(0,allocator).get())!="actions")
            throw std::runtime_error("Unexpected policy tensor names");
        const auto meta = session_->GetModelMetadata();
        for (const auto& item : std::array<std::pair<const char*,const char*>,3>{{
             {"task_id",kTask}, {"checkpoint_sha256",kCheckpointHash}, {"robot_model_sha256",kRobotHash}}}) {
            auto value=meta.LookupCustomMetadataMapAllocated(item.first,allocator);
            if (!value || std::string(value.get())!=item.second)
                throw std::runtime_error(std::string("Policy/config identity mismatch: ")+item.first);
        }
    }

    Action infer(const Observation& observation) {
        auto info=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
        const std::array<int64_t,2> shape{1,75};
        // ONNX Runtime accepts a mutable pointer but does not modify this input.
        auto input=Ort::Value::CreateTensor<float>(info,const_cast<float*>(observation.data()),75,shape.data(),2);
        const char* inputs[]{"obs"}; const char* outputs[]{"actions"};
        auto result=session_->Run(Ort::RunOptions{nullptr},inputs,&input,1,outputs,1);
        const float* data=result[0].GetTensorData<float>();
        Action action{};
        for (size_t i=0; i<23; ++i) {
            if (!std::isfinite(data[i])) throw std::runtime_error("Non-finite policy output");
            action[i]=std::clamp(data[i],-1.0f,1.0f);
        }
        return action;
    }
private:
    Ort::Env runtime_;
    Ort::SessionOptions options_;
    std::unique_ptr<Ort::Session> session_;
};
} // namespace human_push
