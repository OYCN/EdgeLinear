#include "EDProcess.h"

#include <iostream>

using namespace cv;

int main(int argc, char* argv[])
{
	ED edgeDrawing;
	std::vector<std::vector<Point>> edge_seg;
	Mat src, grayImg, blurImg, edgeSegments, eMap;
	double fps, fps_max = 0, fps_min = 999, fps_sum = 0;
    int fps_num = 0;

	VideoCapture cap;
	if(argc == 1)
		cap.open("/home/opluss/Documents/EdgeLinear/img/dataset.mp4");
	else
		cap.open(argv[1]);

	std::cout << "start" << std::endl;
	while(cap.read(src))
	{
		fps = (double)cv::getTickCount();

		cvtColor(src, grayImg, CV_RGB2GRAY);
		GaussianBlur(grayImg, blurImg, Size(5,5), 1, 0);
		std::vector<std::vector<Point>>().swap(edge_seg);
		eMap = edgeDrawing.Process(blurImg, edge_seg);

		fps = cv::getTickFrequency()/((double)cv::getTickCount() - fps);
		if(fps > fps_max) fps_max = fps;
		if(fps < fps_min) fps_min = fps;
		fps_sum += fps;
		fps_num++;
		
		if(argc > 1)
		{
			cv::putText(src, std::to_string(fps), cv::Point(5,50), cv::FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2);
			cv::namedWindow("eMap", CV_WINDOW_NORMAL);
			imshow("eMap", eMap);
			cv::namedWindow("src", CV_WINDOW_NORMAL);
			imshow("src", src);
			if(waitKey(1)==27) break;
		}
		
		// cv::imwrite("edge_true.jpg", eMap);
	}
	std::cout << "fps avg: " << fps_sum / fps_num << std::endl;
    std::cout << "fps max: " << fps_max << std::endl;
    std::cout << "fps min: " << fps_min << std::endl;
    std::cout << "time avg: " << fps_num / fps_sum << std::endl;
	cv::waitKey();

	return 0;
}