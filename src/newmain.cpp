#include "BlockPipline.h"

int main()
{
    cv::VideoCapture capture;
    capture.open(0);
    if(!capture.isOpened())
	{
		printf("[%s][%d]could not load video data...\n",__FUNCTION__,__LINE__);
		return -1;
	}
    int rows = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
    int cols = capture.get(CV_CAP_PROP_FRAME_WIDTH);

    BlockGetFlag blk(rows, cols, 6, 2, 5, 1, 0);
    blk.setFeeder([&](cv::Mat* src){capture.read(*src);});
    blk.start();
    std::cout << "end of main" << std::endl;
}