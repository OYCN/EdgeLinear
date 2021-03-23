#include "EdgeDrawing.h"

main(int argc, char *args[])
{
    cv::Mat src;
    _EDoutput* EDoutput;

    cv::VideoCapture capture;

    if(argc!=1)
        capture.open(args[1]);
	else
        capture.open(0);

	if(!capture.isOpened())
	{
		printf("[%s][%d]could not load video data...\n",__FUNCTION__,__LINE__);
		return -1;
	}
    
    int rows = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
    int cols = capture.get(CV_CAP_PROP_FRAME_WIDTH);

    EdgeDrawing ED(rows, cols);
    std::cout << rows << " * " << cols << std::endl;
    while(capture.read(src))
	{
        EDoutput = ED.run(src);
        cv::Mat eMap(rows ,cols, cv::CV_8UC1, (unsigned char*)EDoutput.eMap);
        cv::imshow("src", src);
        cv::imshow("ED out", eMap);
        if (key==27)	// esc退出
		{
			break;
		}
    }
}