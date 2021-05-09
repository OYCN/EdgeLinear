#ifndef _INC_BLOCKLINEAR_H
#define _INC_BLOCKLINEAR_H

#include <opencv2/core/cuda_stream_accessor.hpp>
#include "common.h"

class BlockLinear
{
public:
    BlockLinear(int _rows, int _cols, float _th)
        :rows(_rows), cols(_cols), th(_th) {init();}
    ~BlockLinear(){deinit();}
    void enqueue(_EDoutput fMaph, cv::cuda::Stream& cvstream);
    bool* getOutput(){return flags_h;}

private:
    void init();
    void deinit();

private:
    int rows;
    int cols;
    float th;

    POINT* edge_set_d;
    int* edge_offset_d;
    bool* flags_d;
    bool* flags_h;

};

#endif  // _INC_BLOCKLINEAR_H