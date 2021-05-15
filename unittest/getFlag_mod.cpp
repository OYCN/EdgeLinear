#include "BlockGetFlag.h"

int main()
{
    TDEF(time);

    cv::Mat img = cv::imread("/home/opluss/Documents/EdgeLinear/img/1.jpg");

    auto runner = BlockGetFlag(img.rows, img.cols, 6, 2, 5, 1, 0);
    cv::cuda::Stream cvstream;
    cudaStream_t custream = cv::cuda::StreamAccessor::getStream(cvstream);
    TSTART(time);
    runner.enqueue(img, cvstream);
    HANDLE_ERROR(cudaStreamSynchronize(custream));
    TEND(time);
    TPRINTUS(time, "time is (ms): ");
    uchar* res = runner.getOutput();
    cv::Mat oMap = cv::Mat(img.rows, img.cols, CV_8UC1, res);
    cv::imwrite("true_map.jpg", oMap);

    return 0;
}