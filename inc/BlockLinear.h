#ifndef _INC_BLOCKLINEAR_H
#define _INC_BLOCKLINEAR_H

#include <opencv2/core/cuda_stream_accessor.hpp>
#include "common.h"

class BlockLinear
{
public:
    BlockLinear(int _rows, int _cols, float _th, bool _returnH)
        :rows(_rows), cols(_cols), th(_th), returnH(_returnH) {init();}
    ~BlockLinear(){deinit();}
    void enqueue(_EDoutput fMaph, cv::cuda::Stream& cvstream);
    bool* getOutput(){ return returnH ? flags_h : flags_d;}

private:
    void init();
    void deinit();

private:
    int rows;
    int cols;
    float th;

    bool returnH;

    POINT* edge_set_d;
    int* edge_offset_d;
    bool* flags_d;
    bool* flags_h;

};

#endif  // _INC_BLOCKLINEAR_H