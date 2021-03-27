#include <string>
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
    double fps, fps_max = 0, fps_min = 999, fps_sum = 0;
    int fps_num = 0;

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
        fps = (double)cv::getTickCount();
        EDoutput = ED.run(src);
        cv::Mat eMap(rows ,cols, CV_8UC1, (unsigned char*)(EDoutput->eMap));
        #ifndef _NLINEAR
        flag = Linear.run(*EDoutput);
        #endif  // _NLINEAR
        fps = ((double)cv::getTickCount() - fps)/cv::getTickFrequency();
        if(fps > fps_max) fps_max = fps;
        if(fps < fps_min) fps_min = fps;
        fps_sum += fps;
        fps_num++;
        #ifndef _NLINEAR
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
        // std::string str = "FPS:" + std::to_string(fps);
        cv::putText(src, std::to_string(fps), cv::Point(5,50), cv::FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2);
        char key = cv::waitKey(1);
        if (key==27)	// esc退出
		{
			break;
		}
    }
    std::cout << "fps avg: " << fps_sum / fps_num << std::endl;
    std::cout << "fps max: " << fps_max << std::endl;
    std::cout << "fps min: " << fps_min << std::endl;
    cv::waitKey(0);
}