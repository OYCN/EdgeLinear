#ifndef _INC_PIPER_H
#define _INC_PIPER_H

#include <queue>

class Piper
{
public:
    Piper(int len):pipstep(step), piplen(len) {init();}
    ~Piper() {uinit();}
    void setFeeder(void* (*feeder)()) {pipfeeder = feeder;}
    void setEnder(bool (*ender)()) {pipender = ender;}
    void run();

private:
    void init();
    void uinit();

private:
    int pipstep;
    int piplen;
    void* (*pipfeeder)();
    bool (*pipender)();
    // ED kernel -> smartconecting
    std::queue<uchar*> in0_blurd;
    std::queue<uchar*> in0_gMapd;
    std::queue<uchar*> out0_fMapd;
    // smartconecting -> linear kernel

}

#endif // _INC_PIPER_H