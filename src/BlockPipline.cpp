#include "BlockPipline.h"

void BlockPipline::init()
{
    HANDLE_ERROR(cudaMallocHost(&srch, sizeof(uchar)*rows*cols*3));
    sMaph = new cv::Mat(rows, cols, CV_8UC3, srch);
    fMaph = blockA.getOutput();
    result = blockC.getOutput();
    edges = blockB.getOutput();
}

void BlockPipline::deinit()
{
    HANDLE_ERROR(cudaFreeHost(srch));
    delete sMaph;
}

void BlockPipline::run()
{
    cv::cuda::Stream cvstream;
    cudaStream_t custream = cv::cuda::StreamAccessor::getStream(cvstream);
    blockA.enqueue(*sMaph, cvstream);
    HANDLE_ERROR(cudaStreamSynchronize(custream));
    blockB.execute(fMaph);
    blockC.enqueue(*edges, cvstream);
    HANDLE_ERROR(cudaStreamSynchronize(custream));
}
