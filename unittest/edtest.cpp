#include "../pipeline/BlockGetFlag.h"
#include "../pipeline/BlockConnect.h"

int main(int argc, char* argv[])
{
    TDEF(time);

    cv::Mat img;
    
	if(argc == 1)
		img = cv::imread("/home/opluss/Documents/EdgeLinear/img/11.jpg");
	else
		img = cv::imread(argv[1]);

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
    cv::Mat fMap = cv::Mat(img.rows, img.cols, CV_8UC1, res1);
    cv::Mat oMap = cv::Mat(img.rows, img.cols, CV_8UC1, res2->eMap);
    cv::imshow("edge_map.jpg", oMap * 255);
    cv::imwrite("fmap_err.jpg", fMap);
    cv::waitKey();

    return 0;
}