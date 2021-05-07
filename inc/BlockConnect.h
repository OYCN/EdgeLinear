#ifndef _INC_BLOCKED_H
#define _INC_BLOCKED_H

#include "common.h"
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudafilters.hpp>

class BlockGetFlag
{
public:
    BlockGetFlag(int _rows, int _cols, float _th, int _k, int _GFSize, int _GFs1, int _GFs2)
        :rows(_rows), cols(_cols), th(_th), k(_k), GFSize(_GFSize), GFs1(_GFs1), GFs2(_GFs2) {init();}
    ~BlockGetFlag(){deinit();}
    void start();
    void setFeeder(std::function<void (cv::Mat*)> _feeder) {feeder = _feeder;}
    static void CUDART_CB Callback(cudaStream_t stream, cudaError_t status, void *data)
    {
        BlockGetFlag* thiz = (BlockGetFlag*) data;
        thiz->callbackFunc();
    }

private:
    void init();
    void deinit();
    void compute();
    void kernel();
    void callbackFunc();

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

    cv::cuda::Stream cvstream;
    cudaStream_t custream;
    std::function<void (cv::Mat*)> feeder = {};

    uchar* srch;
    cv::Mat* sMap;
    uchar* srcd;
    cv::cuda::GpuMat *gmat_src;
    uchar* grayd;
    cv::cuda::GpuMat *gmat_gray;
    uchar* blurd;
	cv::cuda::GpuMat *gmat_blur;

    uchar* fMapd;

    cv::Ptr<cv::cuda::Filter> gauss;


};

__global__ void kernelC(uchar *blur, uchar *fMap, int gcols, int grows, int ANCHOR_TH, int K);

#endif // _INC_BLOCKED_H