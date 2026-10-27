// Offline C interface to the exact observation/inference/target code used on G1.
#include <cstring>
#include "HumanPushPolicy.h"

static thread_local std::string last_error;
extern "C" {
const char* hp_error() { return last_error.c_str(); }
void* hp_create(const char* path) {
    try { return new human_push::Policy(path); }
    catch(const std::exception& e) { last_error=e.what(); return nullptr; }
}
void hp_destroy(void* pointer) { delete static_cast<human_push::Policy*>(pointer); }
int hp_run(void* pointer,int count,const float* telemetry,float* previous,float* observations,float* actions,float* positions) {
    try {
        for(int n=0;n<count;++n) {
            human_push::Sensors sensors{};
            const float* row=telemetry+n*65;
            std::copy(row,row+4,sensors.quaternion.begin());
            std::copy(row+4,row+7,sensors.gyro.begin());
            std::copy(row+7,row+36,sensors.position.begin());
            std::copy(row+36,row+65,sensors.velocity.begin());
            human_push::Action prior{};
            std::copy(previous+n*23,previous+(n+1)*23,prior.begin());
            const auto obs=human_push::observation(sensors,prior);
            const auto action=static_cast<human_push::Policy*>(pointer)->infer(obs);
            const auto target=human_push::targets(action);
            std::copy(obs.begin(),obs.end(),observations+n*75);
            std::copy(action.begin(),action.end(),actions+n*23);
            std::copy(target.begin(),target.end(),positions+n*23);
            std::copy(action.begin(),action.end(),previous+n*23);
        }
        return 0;
    } catch(const std::exception& e) { last_error=e.what(); return 1; }
}
}
