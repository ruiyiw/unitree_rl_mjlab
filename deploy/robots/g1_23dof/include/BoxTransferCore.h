#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <mujoco/mujoco.h>
#include <openssl/evp.h>
#include <onnxruntime_cxx_api.h>
#include "../config/policy/box_transfer/v1/params/BoxTransferConfig.h"

namespace box_transfer {
using V3 = std::array<double,3>;
using Q4 = std::array<double,4>;
using Sensors = std::array<float,kSensorWidth>;
using Joints = std::array<float,23>;
// Sensor layout is shared with the binary transport and Python adapter.
// Pose/linear velocity are in the fixed task-layout world; gyro is pelvis-local.
enum SensorIndex {BasePosition=0, BaseQuaternion=3, BaseVelocity=7, Gyro=10,
                  JointPosition=13, JointVelocity=36, BoxPosition=59,
                  BoxQuaternion=62, BoxVelocity=66, Feet=69, Hands=71,
                  BoxSupported=73, PropContact=74, RobotBoxContact=75};
inline V3 add(V3 a,V3 b){for(int i=0;i<3;++i)a[i]+=b[i];return a;}
inline V3 sub(V3 a,V3 b){for(int i=0;i<3;++i)a[i]-=b[i];return a;}
inline double dot(V3 a,V3 b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
inline double norm(V3 a){return std::sqrt(dot(a,a));}
inline Q4 inverse(Q4 q){return {q[0],-q[1],-q[2],-q[3]};}
inline Q4 multiply(Q4 a,Q4 b){Q4 q; mju_mulQuat(q.data(),a.data(),b.data());return q;}
inline V3 rotate(Q4 q,V3 v){V3 out;mju_rotVecQuat(out.data(),v.data(),q.data());return out;}
inline double yaw(Q4 q){return std::atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]*q[2]+q[3]*q[3]));}
inline double wrap(double x){return std::atan2(std::sin(x),std::cos(x));}
inline Q4 yawQuat(double y){return {std::cos(y/2),0,0,std::sin(y/2)};}
inline V3 yawFrame(V3 v,double y){return rotate(yawQuat(-y),v);}
inline double angle(double c){return std::acos(std::clamp(c,-1.,1.));}
inline Q4 unit(Q4 q){double n=0;for(auto v:q)n+=v*v;
    if(!std::isfinite(n)||n<1e-8)throw std::runtime_error("invalid quaternion");
    for(auto& v:q)v/=std::sqrt(n);return q;}
template<class T> inline V3 vec(const T* p){return {p[0],p[1],p[2]};}
template<class T> inline Q4 quat(const T* p){return {p[0],p[1],p[2],p[3]};}
inline std::string hashFile(const std::filesystem::path& path){
    std::ifstream file(path,std::ios::binary);if(!file)throw std::runtime_error("cannot read "+path.string());
    auto ctx=std::unique_ptr<EVP_MD_CTX,decltype(&EVP_MD_CTX_free)>(EVP_MD_CTX_new(),EVP_MD_CTX_free);
    if(!ctx||EVP_DigestInit_ex(ctx.get(),EVP_sha256(),nullptr)!=1)throw std::runtime_error("SHA256 init failed");
    std::array<char,65536> block{};
    while(file){file.read(block.data(),block.size());if(EVP_DigestUpdate(ctx.get(),block.data(),file.gcount())!=1)throw std::runtime_error("SHA256 update failed");}
    if(!file.eof())throw std::runtime_error("file read failed");
    unsigned char hash[EVP_MAX_MD_SIZE];unsigned int size;
    if(EVP_DigestFinal_ex(ctx.get(),hash,&size)!=1)throw std::runtime_error("SHA256 final failed");
    std::ostringstream out;for(unsigned i=0;i<size;++i)out<<std::hex<<std::setw(2)<<std::setfill('0')<<int(hash[i]);return out.str();
}
inline void checkHash(const std::filesystem::path& p,const char* expected){
    if(hashFile(p)!=expected)throw std::runtime_error("bundle identity mismatch: "+p.string());
}

class Actor {
    Ort::Session session_;
    int width_;
public:
    Actor(Ort::Env& runtime,const std::filesystem::path& path,int width,const Ort::SessionOptions& opts)
      :session_(runtime,path.c_str(),opts),width_(width){
        if(session_.GetInputCount()!=1||session_.GetOutputCount()!=1)throw std::runtime_error("actor tensor count");
        auto it=session_.GetInputTypeInfo(0),ot=session_.GetOutputTypeInfo(0);
        auto in=it.GetTensorTypeAndShapeInfo(),out=ot.GetTensorTypeAndShapeInfo();
        auto a=in.GetShape(),b=out.GetShape();
        if(a.size()!=2||b.size()!=2||a[1]!=width||b[1]!=23||(a[0]!=-1&&a[0]!=1)||(b[0]!=-1&&b[0]!=1)
          ||in.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT||out.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            throw std::runtime_error("actor shape/type mismatch");
        Ort::AllocatorWithDefaultOptions alloc;
        if(std::string(session_.GetInputNameAllocated(0,alloc).get())!="obs"||std::string(session_.GetOutputNameAllocated(0,alloc).get())!="actions")throw std::runtime_error("actor tensor names");
    }
    Joints infer(const std::vector<float>& obs){
        if(int(obs.size())!=width_)throw std::runtime_error("observation width mismatch");
        std::array<int64_t,2> shape{1,width_};auto info=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
        auto in=Ort::Value::CreateTensor<float>(info,const_cast<float*>(obs.data()),obs.size(),shape.data(),2);
        const char* inputs[]={"obs"};const char* outputs[]={"actions"};
        auto result=session_.Run(Ort::RunOptions{nullptr},inputs,&in,1,outputs,1);
        Joints action{};std::copy_n(result[0].GetTensorData<float>(),23,action.begin());
        for(auto x:action)if(!std::isfinite(x))throw std::runtime_error("non-finite actor output");
        return action; // Training uses raw residuals: never clamp to [-1,1].
    }
};

struct Output {std::vector<float> observation;Joints action{},target{},gravity{};};
class Controller {
    std::unique_ptr<mjModel,decltype(&mj_deleteModel)> model_{nullptr,mj_deleteModel};
    std::unique_ptr<mjData,decltype(&mj_deleteData)> data_{nullptr,mj_deleteData};
    Ort::Env runtime_{ORT_LOGGING_LEVEL_WARNING,"box_transfer"};
    std::array<std::unique_ptr<Actor>,4> actors_;
    std::vector<float> references_;
    std::array<int,23> qadr_{},vadr_{};
    std::array<int,14> bodies_{};
    std::array<int,2> palms_{};
    int imuLinear_=-1,imuAngular_=-1,anchor_=7;
    int phase_=0,clock_=0;
    bool sequence_=true,fresh_=true,release_=false,turn_=false,completed_=false;
    float elapsed_=0,stood_=0,releaseTime_=0,turnTime_=0;
    Joints previous_{};
    Sensors sensors_{};
    V3 base_{},box_{},boxVelocity_{};Q4 root_{},boxQuat_{};
    std::array<V3,2> palmError_{};std::array<double,2> gap_{};
    double boxTilt_=0;
    static constexpr int BodyOffset=82, BoxOffset=180, GripOffset=184;
    const float* frame(int index)const{
        index=std::clamp(index,0,kLengths[phase_]-1);
        return references_.data()+(kOffsets[phase_]+index)*kFrameWidth;
    }
    V3 bodyPosition(int b)const{return vec(data_->xpos+3*bodies_[b]);}
    Q4 bodyQuaternion(int b)const{return quat(data_->xquat+4*bodies_[b]);}
    V3 palmPosition(int side)const{return vec(data_->site_xpos+3*palms_[side]);}
    V3 refBody(int b,int offset=0)const{return vec(frame(clock_+offset)+BodyOffset+7*b);}
    Q4 refQuaternion(int b,int offset=0)const{return quat(frame(clock_+offset)+BodyOffset+7*b+3);}
    bool handsClear()const{return sensors_[Hands]==0&&sensors_[Hands+1]==0;}
    bool released()const{
        return sensors_[BoxSupported]>0&&norm(boxVelocity_)<.1&&boxTilt_<.15&&handsClear()&&std::min(gap_[0],gap_[1])>.05;
    }
    bool armsLowered()const{
        const float* ref=frame(kGates[3]);
        for(int j=13;j<23;++j)if(std::abs(sensors_[JointPosition+j]-ref[7+j])>=.3)return false;
        return palmPosition(0)[2]<kTables[5]-.15&&palmPosition(1)[2]<kTables[5]-.15;
    }
    void append(std::vector<float>& out,V3 v)const{for(auto x:v)out.push_back(float(x));}
    void kinematics(const Sensors& s){
        for(float v:s)if(!std::isfinite(v))throw std::runtime_error("non-finite sensors");
        for(int i=Feet;i<kSensorWidth;++i)if(s[i]!=0&&s[i]!=1)throw std::runtime_error("contact flags must be 0 or 1");
        sensors_=s;base_=vec(s.data()+BasePosition);root_=unit(quat(s.data()+BaseQuaternion));
        box_=vec(s.data()+BoxPosition);boxQuat_=unit(quat(s.data()+BoxQuaternion));boxVelocity_=vec(s.data()+BoxVelocity);
        if(norm(vec(s.data()+BaseVelocity))>10||norm(vec(s.data()+Gyro))>30)throw std::runtime_error("invalid base velocity");
        std::copy(base_.begin(),base_.end(),data_->qpos);std::copy(root_.begin(),root_.end(),data_->qpos+3);
        for(int i=0;i<3;++i){data_->qvel[i]=s[BaseVelocity+i];data_->qvel[3+i]=s[Gyro+i];}
        for(int j=0;j<23;++j){
            if(s[JointPosition+j]<kJointMin[j]-.2||s[JointPosition+j]>kJointMax[j]+.2||std::abs(s[JointVelocity+j])>100)throw std::runtime_error("invalid joint measurement");
            data_->qpos[qadr_[j]]=s[JointPosition+j];data_->qvel[vadr_[j]]=s[JointVelocity+j];
        }
        mj_forward(model_.get(),data_.get());
        double flip=dot(rotate(boxQuat_,{0,1,0}),rotate(yawQuat(yaw(root_)),{0,1,0}))<0?-1:1;
        for(int side=0;side<2;++side){
            V3 p=rotate(inverse(boxQuat_),sub(palmPosition(side),box_));p[0]*=flip;p[1]*=flip;
            palmError_[side]=sub(p,{0,(side==0?1.:-1.)*(kBoxHalf[1]+.035),0});
            V3 outside{};for(int i=0;i<3;++i)outside[i]=std::max(0.,std::abs(p[i])-kBoxHalf[i]);
            gap_[side]=norm(outside)-.035;
        }
        boxTilt_=angle(std::abs(rotate(boxQuat_,{0,0,1})[2]));
    }
    void checkFailure()const{
        if(rotate(root_,{0,0,1})[2]<std::cos(M_PI/3))throw std::runtime_error("fell");
        V3 actual=bodyPosition(anchor_),ref=refBody(anchor_,1);
        Q4 actualQ=bodyQuaternion(anchor_),refQ=refQuaternion(anchor_,1);
        if(std::abs(actual[2]-ref[2])>.25||std::hypot(actual[0]-ref[0],actual[1]-ref[1])>.5
           ||std::abs(rotate(inverse(actualQ),{0,0,-1})[2]-rotate(inverse(refQ),{0,0,-1})[2])>.8)
            throw std::runtime_error("tracking");
        // Yaw-only alignment leaves the reference end-effector heights unchanged.
        for(int b:{3,6,10,13})if(std::abs(bodyPosition(b)[2]-refBody(b,1)[2])>.25)throw std::runtime_error("tracking/end-effector");
        if(phase_>=1&&(box_[2]<.45||norm(sub(box_,vec(frame(clock_+1)+BoxOffset)))>.25))throw std::runtime_error("box_lost");
        if(phase_==3){
            if(std::abs(wrap(yaw(quat(frame(clock_+1)+3))-yaw(root_)))>.6)throw std::runtime_error("putdown_heading");
            if(clock_>=kGates[0]&&(sensors_[PropContact]||sensors_[RobotBoxContact]))throw std::runtime_error("postrelease_collision");
            if(clock_>=kGates[1]&&(base_[2]<.73||base_[2]>.81||rotate(actualQ,{0,0,1})[2]<std::cos(M_PI/30)))throw std::runtime_error("postrelease_posture");
        }
        if(!completed_&&elapsed_>(kLengths[phase_]-1)*kDt+12)throw std::runtime_error("phase_timeout");
    }
public:
    explicit Controller(const std::filesystem::path& bundle){
        if(std::string(mj_versionString())!="3.11.0")throw std::runtime_error("box transfer requires MuJoCo 3.11.0");
        checkHash(bundle/"deployment/robot.mjb",kModelHash);checkHash(bundle/"deployment/references.bin",kReferenceHash);
        model_.reset(mj_loadModel((bundle/"deployment/robot.mjb").c_str(),nullptr));
        if(!model_)throw std::runtime_error("failed to load robot model");data_.reset(mj_makeData(model_.get()));
        if(!data_||model_->nq!=30||model_->nv!=29||model_->nu!=23)throw std::runtime_error("unexpected robot model");
        for(int j=0;j<23;++j){int id=mj_name2id(model_.get(),mjOBJ_JOINT,kJointNames[j]);if(id<0)throw std::runtime_error("missing joint");qadr_[j]=model_->jnt_qposadr[id];vadr_[j]=model_->jnt_dofadr[id];}
        for(int b=0;b<14;++b){bodies_[b]=mj_name2id(model_.get(),mjOBJ_BODY,kBodyNames[b]);if(bodies_[b]<0)throw std::runtime_error("missing body");}
        for(int side=0;side<2;++side){palms_[side]=mj_name2id(model_.get(),mjOBJ_SITE,side==0?"left_palm":"right_palm");if(palms_[side]<0)throw std::runtime_error("missing palm");}
        int l=mj_name2id(model_.get(),mjOBJ_SENSOR,"imu_lin_vel"),a=mj_name2id(model_.get(),mjOBJ_SENSOR,"imu_ang_vel");
        if(l<0||a<0)throw std::runtime_error("missing IMU sensors");imuLinear_=model_->sensor_adr[l];imuAngular_=model_->sensor_adr[a];
        references_.resize((kOffsets[4]+kLengths[4])*kFrameWidth);
        if(std::filesystem::file_size(bundle/"deployment/references.bin")!=references_.size()*sizeof(float))throw std::runtime_error("reference dimensions");
        std::ifstream refs(bundle/"deployment/references.bin",std::ios::binary);refs.read(reinterpret_cast<char*>(references_.data()),references_.size()*sizeof(float));
        if(!refs)throw std::runtime_error("reference read failed");
        Ort::SessionOptions opts;opts.SetIntraOpNumThreads(1);opts.SetInterOpNumThreads(1);opts.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
        for(int i=0;i<4;++i){auto p=bundle/"exported"/kActorNames[i]/"policy.onnx";checkHash(p,kActorHashes[i]);actors_[i]=std::make_unique<Actor>(runtime_,p,kActorWidths[i],opts);}
        reset(-1);
    }
    void reset(int selected){if(selected < -1||selected>4)throw std::runtime_error("invalid phase selection");
        sequence_=selected==-1;phase_=std::max(0,selected);clock_=0;fresh_=true;completed_=release_=turn_=false;
        elapsed_=stood_=releaseTime_=turnTime_=0;previous_.fill(0);}
    int phase()const{return phase_;}int clock()const{return clock_;}bool completed()const{return completed_;}
    bool releaseAllowed()const{return release_;}bool turnAllowed()const{return turn_;}
    Joints initialJoints()const{Joints q{};std::copy_n(frame(0)+7,23,q.begin());return q;}
    void requireStart(const Sensors& s){
        kinematics(s);auto ref=frame(0);
        if(norm(sub(base_,vec(ref)))>.12||std::abs(wrap(yaw(root_)-yaw(quat(ref+3))))>.2)throw std::runtime_error("robot is not at selected phase start pose");
        if(norm(sub(box_,vec(ref+BoxOffset)))>.08||std::abs(wrap(yaw(boxQuat_)-ref[BoxOffset+3]))>.2)throw std::runtime_error("box is not at selected phase start pose");
        if(s[Feet]!=1||s[Feet+1]!=1||s[PropContact]!=0)throw std::runtime_error("start requires both feet supported and no robot/table contact");
        for(int j=0;j<23;++j)if(std::abs(s[JointPosition+j]-ref[7+j])>.3)throw std::runtime_error("joints are not at selected phase start pose");
        if(ref[GripOffset]>0&&!(s[Hands]==1&&s[Hands+1]==1))throw std::runtime_error("selected phase starts with the box held in both hands");
    }
    Joints gravity(const Sensors& s){kinematics(s);Joints g{};for(int j=0;j<23;++j)g[j]=float(data_->qfrc_gravcomp[vadr_[j]]);return g;}
    bool standing()const{
        const float* end=frame(kLengths[phase_]-1);
        if(std::hypot(base_[0]-end[0],base_[1]-end[1])>=.1||std::abs(wrap(yaw(quat(end+3))-yaw(root_)))>=.2)return false;
        Q4 torso=bodyQuaternion(anchor_);double tilt=angle(rotate(torso,{0,0,1})[2]);
        if(phase_==1||phase_==2){
            Q4 ref=quat(end+BodyOffset+anchor_*7+3);
            if(tilt>=M_PI/12||angle(dot(rotate(inverse(torso),{0,0,1}),rotate(inverse(ref),{0,0,1})))>=.1)return false;
        }else if(tilt>=.1)return false;
        if(sensors_[PropContact]||!sensors_[Feet]||!sensors_[Feet+1])return false;
        if(phase_==1||phase_==2){
            if(!sensors_[Hands]||!sensors_[Hands+1]||std::max(norm(palmError_[0]),norm(palmError_[1]))>=.06||boxTilt_>=.20||norm(sub(box_,vec(frame(clock_)+BoxOffset)))>=.08)return false;
        }else{
            if(phase_>=3&&(!released()||norm(sub(box_,vec(frame(clock_)+BoxOffset)))>=.08))return false;
            if(phase_==0&&!handsClear())return false;
            for(int j=13;j<23;++j)if(std::abs(sensors_[JointPosition+j]-end[7+j])>=.3)return false;
        }
        return true;
    }
    void update(){
        checkFailure();
        int next=clock_+!fresh_;
        if(phase_==3){
            bool clear=released()&&!sensors_[PropContact]&&!sensors_[RobotBoxContact];
            releaseTime_=(clear&&clock_>=kGates[0])?releaseTime_+.02f:0;
            turnTime_=(clear&&armsLowered()&&clock_>=kGates[3])?turnTime_+.02f:0;
            release_|=releaseTime_>=.2f;turn_|=turnTime_>=.4f;
            if(!release_)next=std::min(next,kGates[0]);if(!turn_)next=std::min(next,kGates[3]);
        }
        clock_=std::min(next,kLengths[phase_]-1);elapsed_+=fresh_?0:.02f;fresh_=false;
        stood_=standing()&&clock_==kLengths[phase_]-1?stood_+.02f:0;
        if(stood_>=1){
            if(sequence_&&phase_<4){++phase_;clock_=0;elapsed_=stood_=releaseTime_=turnTime_=0;release_=turn_=false;fresh_=true;}
            else completed_=true;
        }
    }
    std::vector<float> observation()const{
        std::vector<float> obs;obs.reserve(180);auto ref=frame(clock_);
        for(int j=0;j<23;++j)obs.push_back(ref[7+j]-float(kDefault[j]));
        for(int j=0;j<23;++j)obs.push_back(ref[30+6+j]);
        V3 torso=bodyPosition(anchor_);Q4 tq=bodyQuaternion(anchor_);
        append(obs,rotate(inverse(tq),sub(refBody(anchor_),torso)));
        double rot[9];mju_quat2Mat(rot,multiply(inverse(tq),refQuaternion(anchor_)).data());
        for(int r=0;r<3;++r)for(int c=0;c<2;++c)obs.push_back(float(rot[r*3+c]));
        for(int off:{10,25}){append(obs,yawFrame(sub(refBody(anchor_,off),torso),yaw(tq)));double a=wrap(yaw(refQuaternion(anchor_,off))-yaw(tq));obs.push_back(std::sin(a));obs.push_back(std::cos(a));}
        obs.push_back(float(clock_)/kLengths[phase_]);obs.push_back(clock_>=kLengths[phase_]-1?1:0);
        append(obs,vec(data_->sensordata+imuLinear_));append(obs,vec(data_->sensordata+imuAngular_));append(obs,rotate(inverse(root_),{0,0,-1}));
        for(int j=0;j<23;++j)obs.push_back(sensors_[JointPosition+j]-float(kDefault[j]));
        for(int j=0;j<23;++j)obs.push_back(sensors_[JointVelocity+j]);
        obs.insert(obs.end(),previous_.begin(),previous_.end());
        for(int t=0;t<2;++t)append(obs,yawFrame(sub(vec(kTables.data()+3*t),base_),yaw(root_)));
        const float* end=frame(kLengths[phase_]-1);append(obs,yawFrame(sub(vec(end),base_),yaw(root_)));
        double heading=wrap(yaw(quat(end+3))-yaw(root_));obs.push_back(std::sin(heading));obs.push_back(std::cos(heading));
        if(phase_>=1&&phase_<=3){
            append(obs,yawFrame(sub(box_,base_),yaw(root_)));append(obs,yawFrame(rotate(boxQuat_,{1,0,0}),yaw(root_)));append(obs,yawFrame(rotate(boxQuat_,{0,0,1}),yaw(root_)));append(obs,yawFrame(boxVelocity_,yaw(root_)));
            append(obs,palmError_[0]);append(obs,palmError_[1]);obs.push_back(sensors_[Hands]);obs.push_back(sensors_[Hands+1]);
            append(obs,yawFrame(sub(vec(ref+BoxOffset),base_),yaw(root_)));obs.push_back(ref[GripOffset]);
        }
        return obs;
    }
    Output tick(const Sensors& sensors,bool advance){
        kinematics(sensors);if(advance)update();else checkFailure();
        Output out;out.observation=observation();int actor=(phase_==4)?0:phase_;
        out.action=actors_[actor]->infer(out.observation);
        for(int j=0;j<23;++j){out.target[j]=float(frame(clock_+1)[59+j]+kScale[j]*out.action[j]);out.gravity[j]=float(data_->qfrc_gravcomp[vadr_[j]]);if(!std::isfinite(out.target[j])||!std::isfinite(out.gravity[j]))throw std::runtime_error("invalid motor target");}
        previous_=out.action;return out;
    }
    // Validation-only state injection; not exposed by the hardware transport.
    void validationState(int phase,int clock,const Joints& previous){reset(phase);clock_=std::clamp(clock,0,kLengths[phase]-1);previous_=previous;}
};
} // namespace box_transfer
