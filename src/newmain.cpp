#include "BlockWarper.h"
#include <unistd.h>

int main(int argc, char* argv[])
{
    double fps, fps_max = 0, fps_min = 999, fps_sum = 0;
    int fps_num = 0;

    int delay = 0;
    int level = 1;
    bool display = false;

    if(argc > 2)
    {
        level = std::atoi(argv[1]);
        delay = std::atoi(argv[2]);
    }
    if(argc > 3) display = true;
    cv::VideoCapture capture;
    capture.open("img/Robotica_1080.wmv");
    if(!capture.isOpened())
	{
		printf("[%s][%d]could not load video data...\n",__FUNCTION__,__LINE__);
		return -1;
	}
    int rows = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
    int cols = capture.get(CV_CAP_PROP_FRAME_WIDTH);

    _Configure cfg;
    cfg.rows = rows;
    cfg.cols = cols;
    cfg.th1 = 6;
    cfg.k = 2;
    cfg.GFSize = 5;
    cfg.GFs1 = 1;
    cfg.GFs2 = 0;
    cfg.th2 = 5;
    cfg.returnH = false;

    BlockWarper warper(level, cfg);

    warper.setFeeder([&](cv::Mat& v){return capture.read(v) ? true : false;});

    warper.start();

    POINT* edge_set = new POINT[rows * cols];
    int* edge_offset = new int[rows * cols];
    int edge_offset_len = 0;
    bool* flags; // = new bool[rows * cols];
    HANDLE_ERROR(cudaMallocHost(&flags, sizeof(bool)*rows*cols));

    fps = (double)cv::getTickCount();
    while(warper.waitOne(edge_set, edge_offset, edge_offset_len, flags))
    {   
        usleep(1000 * delay);
        
        if(display)
        {
            cv::Mat outMap = cv::Mat::ones(rows, cols, CV_8UC3);
            for(int i = 0; i < (edge_offset_len - 1); i++)
            {
                int old_idx = -1;
                for(int j = edge_offset[i]; j < edge_offset[i+1]; j++)
                {
                    if(flags[j])
                    {
                        if(old_idx > 0)
                        {
                            int s0 = rand() % 256, s1 = rand() % 256, s2 = rand() % 256;
                            cv::line(outMap, edge_set[old_idx], edge_set[j], cv::Scalar(255, 255, 255), 1, 4);
                        }
                        old_idx = j;
                        
                    }
                }
            }

            cv::imshow("outMap", outMap);
            if(cv::waitKey(1) == ' ')
            {
                break;
            }
        }
        
        fps = cv::getTickFrequency()/((double)cv::getTickCount() - fps);
        if(fps > fps_max) fps_max = fps;
        if(fps < fps_min) fps_min = fps;
        fps_sum += fps;
        fps_num++;
        fps = (double)cv::getTickCount();
    }
    warper.join();

    delete[] edge_set;
    delete[] edge_offset;
    HANDLE_ERROR(cudaFreeHost(flags));

    std::cout << "fps avg: " << fps_sum / fps_num << std::endl;
    std::cout << "fps max: " << fps_max << std::endl;
    std::cout << "fps min: " << fps_min << std::endl;
    std::cout << "time avg: " << fps_num / fps_sum << std::endl;
    std::cout << "img num: " << fps_num << std::endl;
    
}