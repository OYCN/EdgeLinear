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

    bool isDisplay = cfg.Read("Display", true);
    bool isWriteFile = cfg.Read("WriteFile", false);

    cv::VideoCapture capture;
    // 根据配置和参数调用不同数据源
    if(argc!=1)
    {
        std::cout << "Using Img:" << args[1] << std::endl;
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

    EdgeDrawing ED(rows, cols, cfg.Read("EDth", 6), cfg.Read("EDk", 2), cfg.Read("GFSize", 5), cfg.Read("GFs1", 1), cfg.Read("GFs2", 0));
    #ifndef _NLINEAR
    _LINEAR Linear(rows, cols, cfg.Read("LNth", 5));
    #endif  // _NLINEAR
    std::cout << rows << " * " << cols << std::endl;
    std::cout << cfg.Read("LoopTime", 1) << " Times per img" << std::endl;
    std::cout << std::endl;
    while(capture.read(src))
	{
        for(int LoopTime = cfg.Read("LoopTime", 1); LoopTime != 0; LoopTime--)
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
            cv::Mat outMap = cv::Mat::ones(rows, cols, CV_8UC1);
            outMap *= 255;
            for(int i = 0; i < (EDoutput->edge_offset_len - 1); i++)
            {
                int old_idx = -1;
                for(int j = EDoutput->edge_offset[i]; j < EDoutput->edge_offset[i+1]; j++)
                {
                    if(flag[j])
                    {
                        if(old_idx > 0)
                        {
                            int s0 = rand() % 256, s1 = rand() % 256, s2 = rand() % 256;
                            cv::line(outMap, EDoutput->edge_set[old_idx], EDoutput->edge_set[j], cv::Scalar(0, 0, 0), 1, 4);
                        }
                        old_idx = j;
                        
                    }
                }
            }
            #endif  // _NLINEAR
            if(isDisplay)
            {
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
            if(isWriteFile)
            {
                cv::imwrite(cfg.Read("SrcFileName", std::string("out_src.jpg")), src);
                cv::imwrite(cfg.Read("EdgeFileName", std::string("out_edge.jpg")), (1 - eMap) * 255);
                #ifndef _NLINEAR
                cv::imwrite(cfg.Read("LinearFileName", std::string("out_linear.jpg")), outMap);
                #endif // _NLINEAR
            }
        }
    }
    std::cout << "fps avg: " << fps_sum / fps_num << std::endl;
    std::cout << "fps max: " << fps_max << std::endl;
    std::cout << "fps min: " << fps_min << std::endl;
    std::cout << "time avg: " << fps_num / fps_sum << std::endl;
    if(cfg.Read("Display", true))
        cv::waitKey(0);
}