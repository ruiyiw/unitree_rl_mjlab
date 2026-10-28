#pragma once
#include "BoxTransferCore.h"
#include <arpa/inet.h>
#include <chrono>
#include <cerrno>
#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>
#include <zlib.h>

namespace box_transfer {
inline uint64_t monotonicNs(){timespec t{};clock_gettime(CLOCK_MONOTONIC,&t);return uint64_t(t.tv_sec)*1000000000ull+t.tv_nsec;}
// Little-endian Linux protocol, 172 bytes. The publisher runs on this host;
// remote sensor timestamps must first be synchronized by the measurement adapter.
class StateReceiver {
    int socket_=-1;
    uint64_t sequence_=0,stamp_=0,received_=0;
    bool have_=false;
    Sensors latest_{};
    uint64_t maxAge_;
public:
    static constexpr size_t PacketBytes=172;
    StateReceiver(int port,double maxAgeSeconds):maxAge_(0){
        uint16_t endian=1;if(*reinterpret_cast<uint8_t*>(&endian)!=1)throw std::runtime_error("state protocol requires little endian");
        if(port<1024||port>65535||!std::isfinite(maxAgeSeconds)||maxAgeSeconds<=0||maxAgeSeconds>.1)throw std::runtime_error("invalid state receiver configuration");
        maxAge_=uint64_t(maxAgeSeconds*1e9);
        socket_=::socket(AF_INET,SOCK_DGRAM|SOCK_NONBLOCK|SOCK_CLOEXEC,0);
        if(socket_<0)throw std::runtime_error("cannot create state socket");
        sockaddr_in address{};address.sin_family=AF_INET;address.sin_port=htons(port);address.sin_addr.s_addr=htonl(INADDR_LOOPBACK);
        if(bind(socket_,reinterpret_cast<sockaddr*>(&address),sizeof(address))<0){close(socket_);socket_=-1;throw std::runtime_error("cannot bind box-transfer state port");}
    }
    ~StateReceiver(){if(socket_>=0)close(socket_);}
    StateReceiver(const StateReceiver&)=delete;StateReceiver& operator=(const StateReceiver&)=delete;
    Sensors read(){
        for(int count=0;count<256;++count){
            std::array<unsigned char,PacketBytes> p{};ssize_t size=recv(socket_,p.data(),p.size(),MSG_TRUNC);
            if(size<0){if(errno==EAGAIN||errno==EWOULDBLOCK)break;if(errno==EINTR)continue;throw std::runtime_error("state receive failed");}
            if(size!=ssize_t(p.size())||std::memcmp(p.data(),"BTSTATE1",8)!=0)throw std::runtime_error("malformed external state packet");
            uint32_t crc,valid;uint64_t seq,stamp;std::memcpy(&seq,p.data()+8,8);std::memcpy(&stamp,p.data()+16,8);std::memcpy(&valid,p.data()+56,4);std::memcpy(&crc,p.data()+168,4);
            if(crc32(0,p.data(),168)!=crc)throw std::runtime_error("external state CRC mismatch");
            if(valid!=7)throw std::runtime_error("tracking or contact estimate invalid");
            std::ostringstream layout;for(int i=0;i<32;++i)layout<<std::hex<<std::setw(2)<<std::setfill('0')<<int(p[24+i]);
            if(layout.str()!=kLayoutHash)throw std::runtime_error("external state layout identity mismatch");
            if(have_&&(seq<=sequence_||stamp<=stamp_))continue;
            auto now=monotonicNs();if(stamp>now+5000000ull)throw std::runtime_error("external state timestamp is in the future");
            if(now>stamp&&now-stamp>maxAge_)continue;
            std::array<float,27> values{};std::memcpy(values.data(),p.data()+60,108);
            for(float x:values)if(!std::isfinite(x))throw std::runtime_error("external state contains non-finite values");
            for(int i=20;i<27;++i)if(values[i]!=0&&values[i]!=1)throw std::runtime_error("invalid contact flags");
            for(int offset:{3,13}){double norm=0;for(int i=0;i<4;++i)norm+=double(values[offset+i])*values[offset+i];
                if(std::abs(std::sqrt(norm)-1)>.01)throw std::runtime_error("external state quaternion must be unit wxyz");}
            std::copy_n(values.data(),10,latest_.begin());
            std::copy_n(values.data()+10,17,latest_.begin()+BoxPosition);
            sequence_=seq;stamp_=stamp;received_=now;have_=true;
        }
        auto now=monotonicNs();
        if(!have_)throw std::runtime_error("no external tracking/contact state received");
        if(now-received_>maxAge_||(now>stamp_&&now-stamp_>maxAge_))throw std::runtime_error("external state timed out");
        return latest_; // Joint measurements and gyro are overwritten from DDS.
    }
};
} // namespace box_transfer
