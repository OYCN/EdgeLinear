#ifndef _INC_BLOCKGETFLAG_H
#define _INC_BLOCKGETFLAG_H

#include "common.h"
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudafilters.hpp>

class BlockGetFlag
{
public:
    BlockGetFlag(int _rows, int _cols, float _th, int _k, int _GFSize, int _GFs1, int _GFs2)
        :rows(_rows), cols(_cols), th(_th), k(_k), GFSize(_GFSize), GFs1(_GFs1), GFs2(_GFs2) {init();}
    ~BlockGetFlag(){deinit();}
    void enqueue(cv::Mat& sMaph, cv::cuda::Stream& cvstream);
    uchar* getOutput(){return fMaph;}

private:
    void init();
    void deinit();

private:
    // 稀疏度
    int k;
    // 阈值
    float th;
    // 高斯模糊 卷积核大小
    int GFSize;
    int GFs1;
    int GFs2;
    // 行
    int rows;
    // 列
    int cols;

    uchar* srcd;
    cv::cuda::GpuMat *gmat_src;
    uchar* grayd;
    cv::cuda::GpuMat *gmat_gray;
    uchar* blurd;
	cv::cuda::GpuMat *gmat_blur;

    uchar* fMapd;
    uchar* fMaph;

    cv::Ptr<cv::cuda::Filter> gauss;


};

__global__ void kernelC(uchar *blur, uchar *fMap, int gcols, int grows, int ANCHOR_TH, int K);

#endif // _INC_BLOCKGETFLAG_H