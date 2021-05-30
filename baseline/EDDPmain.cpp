#include "EDProcess.h"

#include <iostream>

using namespace cv;

int main(int argc, char* argv[])
{
	ED edgeDrawing;
	std::vector<std::vector<Point>> edge_seg;
	std::vector<std::vector<Point>> linear_seg;
	Mat src, grayImg, blurImg, edgeSegments, eMap, lMap;
	double fps, fps_max = 0, fps_min = 999, fps_sum = 0;
    int fps_num = 0;

	VideoCapture cap;
	if(argc == 1)
		cap.open("/home/opluss/Documents/EdgeLinear/img/dataset.mp4");
	else
		cap.open(argv[1]);

	std::cout << "start" << std::endl;
	int64 start_time = cv::getTickCount();
	while(cap.read(src))
	{

		cvtColor(src, grayImg, CV_RGB2GRAY);
		GaussianBlur(grayImg, blurImg, Size(5,5), 1, 0);
		std::vector<std::vector<Point>>().swap(edge_seg);
		eMap = edgeDrawing.Process(blurImg, edge_seg);
		std::vector<std::vector<Point>>().swap(linear_seg);
		for(int e = 0; e < edge_seg.size(); e++)
		{
			// std::cout << edge_seg[e] << std::endl;
			std::vector<Point> line;
			approxPolyDP(edge_seg[e], line, 5, false);
			// std::cout << line.size() << std::endl;
			linear_seg.push_back(line);
		}
		fps_num++;
		// lMap = cv::Mat::zeros(src.size(), src.type());
		// std::cout << edge_seg.size() << std::endl;
		// std::cout << linear_seg.size() << std::endl;
		// for(int e = 0; e < linear_seg.size(); e++)
		// {
		// 	// std::cout << linear_seg[e].size() << std::endl;
		// 	for(int i = 0; i < linear_seg[e].size() - 1; i++)
		// 	{
		// 		line(lMap, linear_seg[e][i], linear_seg[e][i + 1], Scalar(0, 255, 0), 1, 4);
		// 		// std::cout << "(" << linear_seg[e][i].x << "," << linear_seg[e][i].y << ")" << " -> (" << linear_seg[e][i + 1].x << "," << linear_seg[e][i + 1].y << ")" << std::endl;
		// 	}
		// }
		
		// if(argc > 1)
		// {
		// 	cv::putText(src, std::to_string(fps), cv::Point(5,50), cv::FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2);
		// 	cv::namedWindow("eMap", CV_WINDOW_NORMAL);
		// 	imshow("eMap", eMap);
		// 	cv::namedWindow("src", CV_WINDOW_NORMAL);
		// 	imshow("src", src);
		// 	cv::namedWindow("lMap", CV_WINDOW_NORMAL);
		// 	imshow("lMap", lMap);
		// 	if(waitKey(1)==27) break;
		// }
	}
	int64 end_time = cv::getTickCount();
	std::cout << "fps avg: " << (double)cv::getTickFrequency() * fps_num / (end_time - start_time) << std::endl;
	std::cout << "time avg: " << (end_time - start_time) / cv::getTickFrequency() / fps_num << std::endl;
	std::cout << "loop time: " << fps_num << std::endl;
	cv::namedWindow("eMap", CV_WINDOW_NORMAL);
	imshow("eMap", eMap);
	cv::waitKey();

	return 0;
}