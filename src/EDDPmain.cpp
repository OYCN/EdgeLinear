#include "EdgeDrawing.h"
#include "DouglasPeucker.h"

main(int argc, char *args[])
{
    cv::Mat src;
    _EDoutput* EDoutput;
    bool* flag;

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

    EdgeDrawing ED(rows, cols, 6, 2);
    DouglasPeucker DP(rows, cols, 5);
    std::cout << rows << " * " << cols << std::endl;
    while(capture.read(src))
	{
        EDoutput = ED.run(src);
        cv::Mat eMap(rows ,cols, CV_8UC1, (unsigned char*)(EDoutput->eMap));
        flag = DP.run(*EDoutput);

        // 绘制直线
        cv::Mat DPMap = cv::Mat::zeros(rows, cols, CV_8UC3);
        for(int i = 0; i < (EDoutput->edge_offset_len - 1); i++)
		{
            int old_idx = -1;
			for(int j = EDoutput->edge_offset[i]; j < EDoutput->edge_offset[i+1]; j++)
			{
				if(flag[j])
				{
                    if(old_idx > 0)
                    {
                        cv::line(DPMap, EDoutput->edge_set[old_idx], EDoutput->edge_set[j], cv::Scalar(0, 255, 0), 1, 4);
                    }
                    old_idx = j;
					
				}
			}
		}

        cv::imshow("src", src);
        cv::imshow("ED out", eMap * 255);
        cv::imshow("DP out", DPMap);
        char key = cv::waitKey(1);
        if (key==27)	// esc退出
		{
			break;
		}
    }
    cv::waitKey(0);
}