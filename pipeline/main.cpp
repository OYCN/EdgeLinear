#include "BlockWarper.h"
#include <unistd.h>

int main(int argc, char* argv[])
{
    int loop_time = 0;

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
    capture.open("/home/opluss/Documents/EdgeLinear/img/dataset.mp4");
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

    int64 tickcount = cv::getTickCount();
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
                        if(old_idx >= 0)
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
        loop_time++;
    }
    tickcount = cv::getTickCount() - tickcount;
    double fps = cv::getTickFrequency() * loop_time / tickcount;
    double time = tickcount / loop_time / cv::getTickFrequency();
    warper.join();

    delete[] edge_set;
    delete[] edge_offset;
    HANDLE_ERROR(cudaFreeHost(flags));

    std::cout << "fps avg: " << fps << std::endl;
    std::cout << "time avg: " << time << std::endl;
    std::cout << "loop_time: " << loop_time << std::endl;

    double delay_time = warper.feedtime_sum / warper.loop_time / cv::getTickFrequency();
    std::cout << "delay avg: " << delay_time << std::endl;
    std::cout << "loop_time: " << loop_time << std::endl;
    
}