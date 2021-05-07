#ifndef _INC_BLOCKED_H
#define _INC_BLOCKED_H

#include "common.h"

class BlockConnect
{
public:
    BlockConnect() {init();}
    ~BlockConnect(){deinit();}
    void start();
    static void CUDART_CB Callback(cudaStream_t stream, cudaError_t status, void *data)
    {
        BlockConnect* thiz = (BlockConnect*) data;
        thiz->callbackFunc();
    }

private:
    void init();
    void deinit();
    void compute();
    void kernel();
    void callbackFunc();

private:

};


#endif // _INC_BLOCKED_H