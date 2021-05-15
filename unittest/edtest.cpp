#include "BlockGetFlag.h"
#include "BlockConnect.h"

int main()
{
    TDEF(time);

    cv::Mat img = cv::imread("/home/opluss/Documents/EdgeLinear/img/1.jpg");

    auto runner1 = BlockGetFlag(img.rows, img.cols, 6, 2, 5, 1, 0);
    auto runner2 = BlockConnect(img.rows, img.cols);
    cv::cuda::Stream cvstream;
    cudaStream_t custream = cv::cuda::StreamAccessor::getStream(cvstream);
    uchar* res1 = runner1.getOutput();
    _EDoutput* res2 = runner2.getOutput();

    TSTART(time);
    runner1.enqueue(img, cvstream);
    HANDLE_ERROR(cudaStreamSynchronize(custream));
    runner2.execute(res1);
    TEND(time);
    TPRINTUS(time, "time is (ms): ");
    
    cv::Mat oMap = cv::Mat(img.rows, img.cols, CV_8UC1, res2->eMap);
    cv::imwrite("edge_map.jpg", oMap * 255);

    return 0;
}