// Headless, localhost-only DDS integration simulator. Never connects a real robot.
#include "BoxTransferTransport.h"
#include "unitree/dds_wrapper/robots/g1/g1.h"
#include <boost/program_options.hpp>
#include <csignal>
#include <iostream>
#include <thread>
using namespace box_transfer;
namespace {volatile sig_atomic_t running=1;void stop(int){running=0;}}
bool below(const mjModel* m,int body,int root){for(;body>0;body=m->body_parentid[body])if(body==root)return true;return false;}
int main(int argc,char** argv){
    namespace po=boost::program_options;
    std::string bundle,selection;double duration,drop;int port;bool autoStart;
    po::options_description opts("Box-transfer localhost DDS simulator");
    opts.add_options()("help,h","show help")("bundle",po::value<std::string>(&bundle)->required(),"policy bundle directory")
      ("policy",po::value<std::string>(&selection)->default_value("sequence"),"sequence, go-to, take, carry, put-down, return")
      ("duration",po::value<double>(&duration)->default_value(120),"wall time limit in seconds")
      ("state-port",po::value<int>(&port)->default_value(17600),"localhost tracking receiver port")
      ("auto-start",po::bool_switch(&autoStart),"simulate L2+Up then R2+A after controller connects")
      ("drop-state-after",po::value<double>(&drop)->default_value(-1),"fault test: stop tracking packets after this wall time");
    po::variables_map vm;po::store(po::parse_command_line(argc,argv,opts),vm);if(vm.count("help")){std::cout<<opts<<'\n';return 0;}po::notify(vm);
    if(duration<=0||port<1024||port>65535)throw std::runtime_error("invalid simulator arguments");
    int phase=0;if(selection!="sequence"){
        std::array<std::string,5> names{"go-to","take","carry","put-down","return"};auto it=std::find(names.begin(),names.end(),selection);
        if(it==names.end())throw std::runtime_error("invalid policy selection");phase=it-names.begin();}
    checkHash(std::filesystem::path(bundle)/"deployment/scene.mjb",kSceneHash);
    Controller robot(bundle);robot.reset(phase);
    std::unique_ptr<mjModel,decltype(&mj_deleteModel)> m(mj_loadModel((std::filesystem::path(bundle)/"deployment/scene.mjb").c_str(),nullptr),mj_deleteModel);
    if(!m)throw std::runtime_error("cannot load simulator scene");
    if(std::abs(m->opt.timestep-.005)>1e-10)throw std::runtime_error("scene physics must run at 200 Hz");
    std::unique_ptr<mjData,decltype(&mj_deleteData)> d(mj_makeData(m.get()),mj_deleteData);mj_resetDataKeyframe(m.get(),d.get(),0);
    std::vector<float> refs((kOffsets[4]+kLengths[4])*185);
    std::ifstream file(std::filesystem::path(bundle)/"deployment/references.bin",std::ios::binary);file.read(reinterpret_cast<char*>(refs.data()),refs.size()*4);
    auto ref=refs.data()+185*kOffsets[phase];
    int root=mj_name2id(m.get(),mjOBJ_BODY,"robot/pelvis"),box=mj_name2id(m.get(),mjOBJ_BODY,"box/box");
    int shelf=mj_name2id(m.get(),mjOBJ_BODY,"shelf/shelf"),cart=mj_name2id(m.get(),mjOBJ_BODY,"cart/cart");
    int ground=mj_name2id(m.get(),mjOBJ_BODY,"terrain");
    if(root<0||box<0||shelf<0||cart<0||ground<0)throw std::runtime_error("scene body names");
    int rq=m->jnt_qposadr[m->body_jntadr[root]],rv=m->jnt_dofadr[m->body_jntadr[root]];
    int bq=m->jnt_qposadr[m->body_jntadr[box]],bv=m->jnt_dofadr[m->body_jntadr[box]];
    std::copy_n(ref,7,d->qpos+rq);std::copy_n(ref+30,6,d->qvel+rv);
    std::copy_n(ref+180,3,d->qpos+bq);auto bquat=yawQuat(ref[183]);std::copy(bquat.begin(),bquat.end(),d->qpos+bq+3);
    std::array<int,23> qadr,vadr,act;
    for(int j=0;j<23;++j){int id=mj_name2id(m.get(),mjOBJ_JOINT,(std::string("robot/")+kJointNames[j]).c_str());qadr[j]=m->jnt_qposadr[id];vadr[j]=m->jnt_dofadr[id];d->qpos[qadr[j]]=ref[7+j];d->qvel[vadr[j]]=ref[36+j];act[j]=-1;
        for(int a=0;a<m->nu;++a)if(m->actuator_trnid[2*a]==id)act[j]=a;if(act[j]<0)throw std::runtime_error("missing actuator");}
    std::array<int,2> feet,hands;for(int i=0;i<2;++i){std::string side=i?"right":"left";feet[i]=mj_name2id(m.get(),mjOBJ_BODY,("robot/"+side+"_ankle_roll_link").c_str());hands[i]=mj_name2id(m.get(),mjOBJ_BODY,("robot/"+side+"_wrist_roll_rubber_hand").c_str());}
    // Apply DDS PD+tau with affine actuator bias so implicitfast sees motor-side
    // damping. Disable model gravity compensation: DDS already supplies it.
    for(int b=0;b<m->nbody;++b)m->body_gravcomp[b]=0;
    for(int j=0;j<m->njnt;++j)m->jnt_actgravcomp[j]=0;
    for(int a=0;a<m->nu;++a){m->actuator_gaintype[a]=mjGAIN_FIXED;m->actuator_biastype[a]=mjBIAS_AFFINE;std::fill_n(m->actuator_gainprm+10*a,10,0);std::fill_n(m->actuator_biasprm+10*a,10,0);m->actuator_gainprm[10*a]=1;}
    mj_forward(m.get(),d.get());
    auto sensors=[&]{Sensors s{};std::copy_n(d->qpos+rq,7,s.begin());std::copy_n(d->qvel+rv,3,s.begin()+BaseVelocity);std::copy_n(d->qvel+rv+3,3,s.begin()+Gyro);
        for(int j=0;j<23;++j){s[JointPosition+j]=d->qpos[qadr[j]];s[JointVelocity+j]=d->qvel[vadr[j]];}
        std::copy_n(d->qpos+bq,7,s.begin()+BoxPosition);std::copy_n(d->qvel+bv,3,s.begin()+BoxVelocity);
        for(int c=0;c<d->ncon;++c){auto& contact=d->contact[c];if(contact.dist>0)continue;int a=m->geom_bodyid[contact.geom[0]],b=m->geom_bodyid[contact.geom[1]];
            for(int side=0;side<2;++side){if((below(m.get(),a,feet[side])&&b==ground)||(below(m.get(),b,feet[side])&&a==ground))s[Feet+side]=1;
                if((below(m.get(),a,hands[side])&&b==box)||(below(m.get(),b,hands[side])&&a==box))s[Hands+side]=1;}
            if((a==box&&b==shelf)||(b==box&&a==shelf))s[BoxSupported]=1;
            if((below(m.get(),a,root)&&(b==shelf||b==cart))||(below(m.get(),b,root)&&(a==shelf||a==cart)))s[PropContact]=1;
            if((below(m.get(),a,root)&&b==box)||(below(m.get(),b,root)&&a==box))s[RobotBoxContact]=1;}
        return s;};
    unitree::robot::ChannelFactory::Instance()->Init(0,"lo");
    unitree::robot::g1::subscription::LowCmd input;input.set_timeout_ms(50);
    unitree::robot::g1::publisher::LowState output;
    int sock=socket(AF_INET,SOCK_DGRAM|SOCK_CLOEXEC,0);if(sock<0)throw std::runtime_error("socket failed");
    sockaddr_in destination{};destination.sin_family=AF_INET;destination.sin_port=htons(port);destination.sin_addr.s_addr=htonl(INADDR_LOOPBACK);
    signal(SIGINT,stop);signal(SIGTERM,stop);
    auto start=std::chrono::steady_clock::now(),next=start;double connected=-1;bool controlled=false,released=false;uint64_t seq=0;
    std::cout<<"Simulation only: DDS on lo; tracking on localhost:"<<port<<"; policy "<<selection<<std::endl;
    while(running){
        auto now=std::chrono::steady_clock::now();double elapsed=std::chrono::duration<double>(now-start).count();if(elapsed>=duration)break;
        // Simulated setup support holds the reference until the start button.
        // No state is reset after release, including at phase transitions.
        if(!released){
            std::copy_n(ref,2,d->qpos+rq);std::copy_n(ref+3,4,d->qpos+rq+3);
            std::fill_n(d->qvel+rv,2,0);std::fill_n(d->qvel+rv+3,3,0);
            std::copy_n(ref+180,3,d->qpos+bq);std::copy(bquat.begin(),bquat.end(),d->qpos+bq+3);std::fill_n(d->qvel+bv,6,0);
            mj_forward(m.get(),d.get());
        }
        auto s=sensors();
        if(connected<0&&!input.isTimeout())connected=elapsed;
        unitree_hg::msg::dds_::LowCmd_ cmd;{std::lock_guard<std::mutex> lock(input.mutex_);cmd=input.msg_;}
        for(int j=0;j<23;++j)if(cmd.motor_cmd()[kMotorIds[j]].kp()>0&&!input.isTimeout())controlled=true;
        auto gravity=controlled?Joints{}:robot.gravity(s);
        for(int j=0;j<23;++j){auto& motor=cmd.motor_cmd()[kMotorIds[j]];
            double kp,kd,feed;
            if(!controlled){kp=kKp[j];kd=kKd[j];feed=kp*ref[59+j]+gravity[j];}
            else if(input.isTimeout()){kp=0;kd=3;feed=0;}
            else{kp=motor.kp();kd=motor.kd();feed=kp*motor.q()+kd*motor.dq()+motor.tau();}
            m->actuator_biasprm[10*act[j]+1]=-kp;m->actuator_biasprm[10*act[j]+2]=-kd;
            d->ctrl[act[j]]=feed;}
        mj_step(m.get(),d.get());mj_forward(m.get(),d.get());s=sensors();
        output.lock();auto& msg=output.msg_;msg.mode_machine()=4;msg.mode_pr()=0;msg.tick()=uint32_t(d->time*1000);
        for(int j=0;j<23;++j){msg.motor_state()[kMotorIds[j]].q()=s[JointPosition+j];msg.motor_state()[kMotorIds[j]].dq()=s[JointVelocity+j];}
        for(int i=0;i<4;++i)msg.imu_state().quaternion()[i]=s[BaseQuaternion+i];for(int i=0;i<3;++i)msg.imu_state().gyroscope()[i]=s[Gyro+i];
        unitree::common::REMOTE_DATA_RX remote{};remote.RF_RX.head[0]=0xfe;
        if(autoStart&&connected>=0){double t=elapsed-connected;
            // Triggers are filtered by the SDK: hold them before the button edge.
            if(t>=.5&&t<1.2)remote.RF_RX.btn.components.L2=1;
            if(t>=1&&t<1.2)remote.RF_RX.btn.components.up=1;
            if(t>=4&&t<4.7)remote.RF_RX.btn.components.R2=1;
            if(t>=4.5&&t<4.7){remote.RF_RX.btn.components.A=1;released=true;}}
        std::copy_n(remote.buff,40,msg.wireless_remote().begin());output.unlockAndPublish();
        if(drop<0||elapsed<drop){
            std::array<unsigned char,172> packet{};std::memcpy(packet.data(),"BTSTATE1",8);++seq;auto stamp=monotonicNs();uint32_t valid=7;
            std::memcpy(packet.data()+8,&seq,8);std::memcpy(packet.data()+16,&stamp,8);std::memcpy(packet.data()+56,&valid,4);
            for(int i=0;i<32;++i)packet[24+i]=std::stoul(std::string(kLayoutHash+2*i,2),nullptr,16);
            std::copy_n(reinterpret_cast<const unsigned char*>(s.data()),40,packet.data()+60);
            std::copy_n(reinterpret_cast<const unsigned char*>(s.data()+BoxPosition),68,packet.data()+100);
            uint32_t crc=crc32(0,packet.data(),168);std::memcpy(packet.data()+168,&crc,4);
            sendto(sock,packet.data(),packet.size(),0,reinterpret_cast<sockaddr*>(&destination),sizeof(destination));}
        next+=std::chrono::milliseconds(5);std::this_thread::sleep_until(next);
    }
    auto s=sensors();std::cout<<"SIM_END time="<<d->time<<" base="<<s[0]<<","<<s[1]<<","<<s[2]<<" box="<<s[59]<<","<<s[60]<<","<<s[61]<<std::endl;
    close(sock);return 0;
}
