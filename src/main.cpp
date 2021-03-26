#include "EdgeDrawing.h"
#ifdef _DP
#include "DouglasPeucker.h"
#define _LINEAR DouglasPeucker
#elif defined _LS
#include "LinearSum.h"
#define _LINEAR LinearSum
#elif defined _LD
#include "LinearDis.h"
#define _LINEAR LinearDis
#else
#define _NLINEAR
#endif

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
    #ifndef _NLINEAR
    _LINEAR Linear(rows, cols, 5);
    #endif  // _NLINEAR
    std::cout << rows << " * " << cols << std::endl;
    while(capture.read(src))
	{
        EDoutput = ED.run(src);
        cv::Mat eMap(rows ,cols, CV_8UC1, (unsigned char*)(EDoutput->eMap));
        #ifndef _NLINEAR
        flag = Linear.run(*EDoutput);
        // 绘制直线
        cv::Mat outMap = cv::Mat::zeros(rows, cols, CV_8UC3);
        for(int i = 0; i < (EDoutput->edge_offset_len - 1); i++)
		{
            int old_idx = -1;
			for(int j = EDoutput->edge_offset[i]; j < EDoutput->edge_offset[i+1]; j++)
			{
				if(flag[j])
				{
                    if(old_idx > 0)
                    {
                        cv::line(outMap, EDoutput->edge_set[old_idx], EDoutput->edge_set[j], cv::Scalar(0, 255, 0), 1, 4);
                    }
                    old_idx = j;
					
				}
			}
		}
        #endif  // _NLINEAR

        cv::imshow("src", src);
        cv::imshow("ED out", eMap * 255);
        #ifndef _NLINEAR
        cv::imshow("Linear out", outMap);
        #endif // _NLINEAR
        char key = cv::waitKey(1);
        if (key==27)	// esc退出
		{
			break;
		}
    }
    cv::waitKey(0);
}