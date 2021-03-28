#include <string>
#include <fstream>

#include "Config.h"
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

    // 配置初始化
    std::string cwd(args[0]);
    {
    size_t a = cwd.find_last_of('/');
    cwd = cwd.substr(0, a+1);
    }
    cwd += "configure";
    std::ifstream f(cwd);
    if(!f.good())
    {
        std::ofstream of(cwd);
        of.close();
    }
    // else
    f.close();
    Config cfg(cwd);

    cv::VideoCapture capture;

    if(argc!=1)
    {
        capture.open(args[1]);
    }
	else
    {
        if(cfg.KeyExists("Dev"))
        {
            std::cout << "Using Dev:" << cfg.Read("Dev", 0) << std::endl;
            capture.open(cfg.Read("Dev", 0), cfg.Read("ApiPreference", (int)cv::CAP_V4L2));
        }
        else if(cfg.KeyExists("Pip"))
        {
            std::cout << "Using Pip:\n\t" << cfg.Read("Pip", std::string()) << std::endl;
            capture.open(cfg.Read("Pip", std::string()), cfg.Read("ApiPreference", (int)cv::CAP_GSTREAMER));
        }
    }

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
        fps = cv::getTickFrequency()/((double)cv::getTickCount() - fps);
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

        // std::string str = "FPS:" + std::to_string(fps);
        cv::putText(src, std::to_string(fps), cv::Point(5,50), cv::FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2);

        cv::namedWindow("Src", CV_WINDOW_NORMAL);
        cv::imshow("Src", src);
        cv::namedWindow("Edge", CV_WINDOW_NORMAL);
        cv::imshow("Edge", eMap * 255);
        #ifndef _NLINEAR
        cv::namedWindow("Linear", CV_WINDOW_NORMAL);
        cv::imshow("Linear", outMap);
        #endif // _NLINEAR

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