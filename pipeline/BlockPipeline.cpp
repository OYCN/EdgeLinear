#include "BlockPipeline.h"

void BlockPipeline::init()
{
    custream = cv::cuda::StreamAccessor::getStream(cvstream);
    HANDLE_ERROR(cudaMallocHost(&srch, sizeof(uchar)*rows*cols*3));
    sMaph = new cv::Mat(rows, cols, CV_8UC3, srch);
    fMaph = blockA.getOutput();
    result = blockC.getOutput();
    edges = blockB.getOutput();
    run();
}

void BlockPipeline::deinit()
{
    HANDLE_ERROR(cudaFreeHost(srch));
    delete sMaph;
}

void BlockPipeline::run()
{
    blockA.enqueue(*sMaph, cvstream);
    HANDLE_ERROR(cudaStreamSynchronize(custream));
    blockB.execute(fMaph);
    blockC.enqueue(*edges, cvstream);
    // if(returnH)
    //     HANDLE_ERROR(cudaStreamSynchronize(custream));
}
