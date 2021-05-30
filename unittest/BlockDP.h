#ifndef _INC_BLOCKDP_H
#define _INC_BLOCKDP_H

#include <opencv2/core/cuda_stream_accessor.hpp>
#include "../common/common.h"

class BlockDP
{
public:
    BlockDP(int _rows, int _cols, float _th, bool _returnH)
        :rows(_rows), cols(_cols), th(_th), returnH(_returnH) {init();}
    ~BlockDP(){deinit();}
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
    POINT* stack_d;

};

#endif  // _INC_BLOCKDP_H