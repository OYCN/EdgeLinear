#include "../pipeline/BlockGetFlag.h"
#include "../pipeline/BlockConnect.h"

int main(int argc, char* argv[])
{

    cv::Mat img;
    cv::VideoCapture cap;
    double fps, fps_max = 0, fps_min = 999, fps_sum = 0;
    int fps_num = 0;
    
	if(argc == 1)
		cap.open("/home/opluss/Documents/EdgeLinear/img/dataset.mp4");
	else
		cap.open(argv[1]);

    int rows = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    int cols = cap.get(CV_CAP_PROP_FRAME_WIDTH);

    auto runner1 = BlockGetFlag(rows, cols, 6, 2, 5, 1, 0);
    auto runner2 = BlockConnect(rows, cols);
    cv::cuda::Stream cvstream;
    cudaStream_t custream = cv::cuda::StreamAccessor::getStream(cvstream);
    uchar* res1 = runner1.getOutput();
    _EDoutput* res2 = runner2.getOutput();

    while(cap.read(img))
    {
        fps = (double)cv::getTickCount();

        runner1.enqueue(img, cvstream);
        HANDLE_ERROR(cudaStreamSynchronize(custream));
        runner2.execute(res1);

        fps = cv::getTickFrequency()/((double)cv::getTickCount() - fps);
		if(fps > fps_max) fps_max = fps;
		if(fps < fps_min) fps_min = fps;
		fps_sum += fps;
		fps_num++;
        // cv::Mat eMap = cv::Mat(img.rows, img.cols, CV_8UC1, res2->eMap);
        // cv::imshow("eMap", eMap);
        // cv::waitKey(1);
    }
    std::cout << "fps avg: " << fps_sum / fps_num << std::endl;
    std::cout << "fps max: " << fps_max << std::endl;
    std::cout << "fps min: " << fps_min << std::endl;
    std::cout << "time avg: " << fps_num / fps_sum << std::endl;
    
    cv::Mat eMap = cv::Mat(rows, cols, CV_8UC1, res2->eMap);
    cv::imshow("edge_map.jpg", eMap * 255);
    cv::waitKey();

    return 0;
}