#ifndef RACERLIB_INTERFACE_H
#define RACERLIB_INTERFACE_H

namespace RACERlib{

struct Engines{
    HostEngine* sge;
    DeviceEngine* sgw;

    //Add copy ctr
    //Add def ctr
    //Add ref ctr
}

static std::set<HostEngine*> sges;

void start(void * ref);
void stop(void * ref);
void flush(void * ref);
void put(void * ref, T & value, int PE, int offset);
void get(void * ref, int PE, int offset);


} // namespace RACERlib

#endif