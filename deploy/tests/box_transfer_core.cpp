#include "BoxTransferTransport.h"
namespace { thread_local std::string lastError; }
extern "C" {
const char* bt_error(){return lastError.c_str();}
void* bt_receiver_create(int port,double age){try{return new box_transfer::StateReceiver(port,age);}catch(const std::exception& e){lastError=e.what();return nullptr;}}
void bt_receiver_destroy(void* p){delete static_cast<box_transfer::StateReceiver*>(p);}
int bt_receiver_read(void* p,float* sensor){try{auto s=static_cast<box_transfer::StateReceiver*>(p)->read();std::copy(s.begin(),s.end(),sensor);return 0;}catch(const std::exception& e){lastError=e.what();return -1;}}
void* bt_create(const char* bundle){try{return new box_transfer::Controller(bundle);}catch(const std::exception& e){lastError=e.what();return nullptr;}}
void bt_destroy(void* p){delete static_cast<box_transfer::Controller*>(p);}
int bt_reset(void* p,int phase){try{static_cast<box_transfer::Controller*>(p)->reset(phase);return 0;}catch(const std::exception& e){lastError=e.what();return -1;}}
int bt_tick(void* p,const float* sensor,int advance,float* obs,float* action,float* target,float* gravity,int* status){
    try{
        auto& core=*static_cast<box_transfer::Controller*>(p);box_transfer::Sensors s;
        std::copy_n(sensor,s.size(),s.begin());auto out=core.tick(s,advance!=0);
        std::fill_n(obs,180,0);std::copy(out.observation.begin(),out.observation.end(),obs);
        std::copy(out.action.begin(),out.action.end(),action);std::copy(out.target.begin(),out.target.end(),target);std::copy(out.gravity.begin(),out.gravity.end(),gravity);
        status[0]=core.phase();status[1]=core.clock();status[2]=core.completed();status[3]=core.releaseAllowed();status[4]=core.turnAllowed();return 0;
    }catch(const std::exception& e){lastError=e.what();return -1;}
}
int bt_validation_state(void* p,int phase,int clock,const float* previous){try{box_transfer::Joints a;std::copy_n(previous,23,a.begin());static_cast<box_transfer::Controller*>(p)->validationState(phase,clock,a);return 0;}catch(const std::exception& e){lastError=e.what();return -1;}}
int bt_require_start(void* p,const float* sensor){try{box_transfer::Sensors s;std::copy_n(sensor,s.size(),s.begin());static_cast<box_transfer::Controller*>(p)->requireStart(s);return 0;}catch(const std::exception& e){lastError=e.what();return -1;}}
}
