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
		cap.open("img/11.jpg");
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
		std::vector<std::vector<Point>>().swap(linear_seg);
		for(int e = 0; e < edge_seg.size(); e++)
		{
			std::vector<Point> line;
			Point A, B, T;
			float now_len = 0;
			A = edge_seg[e][0];
			line.push_back(A);
			for(int j = 1; j < edge_seg[e].size(); j++)
			{
				B = edge_seg[e][j];
				T = edge_seg[e][j-1];
				float dx = T.x - B.x;
				float dy = T.y - B.y;
				now_len += sqrt(dx * dx + dy * dy);
				dx = A.x - B.x;
				dy = A.y - B.y;
				float now_dis = sqrt(dx * dx + dy * dy);
				// 若本次超过阈值，上次的为最佳点
				if(fabs(now_len - now_dis) > 5)
				{
					line.push_back(T);
					A = T;
					now_len = 0;
					// 需要重新计算本点
					j--;
				}
			}
			// 结束点为最佳点
			if(A != T) line.push_back(A);
			linear_seg.push_back(line);
		}

		fps = cv::getTickFrequency()/((double)cv::getTickCount() - fps);
		if(fps > fps_max) fps_max = fps;
		if(fps < fps_min) fps_min = fps;
		fps_sum += fps;
		fps_num++;
		lMap = cv::Mat::zeros(src.size(), src.type());
		// std::cout << edge_seg.size() << std::endl;
		// std::cout << linear_seg.size() << std::endl;
		for(int e = 0; e < linear_seg.size(); e++)
		{
			// std::cout << linear_seg[e].size() << std::endl;
			for(int i = 0; i < linear_seg[e].size() - 1; i++)
			{
				line(lMap, linear_seg[e][i], linear_seg[e][i + 1], Scalar(0, 255, 0), 1, 4);
				// std::cout << "(" << linear_seg[e][i].x << "," << linear_seg[e][i].y << ")" << " -> (" << linear_seg[e][i + 1].x << "," << linear_seg[e][i + 1].y << ")" << std::endl;
			}
		}
		
		// cv::putText(src, std::to_string(fps), cv::Point(5,50), cv::FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2);
		// cv::namedWindow("eMap", CV_WINDOW_NORMAL);
     	// imshow("eMap", eMap);
		// cv::namedWindow("src", CV_WINDOW_NORMAL);
	    // imshow("src", src);
		// cv::namedWindow("lMap", CV_WINDOW_NORMAL);
	    // imshow("lMap", lMap);
		// if(waitKey(1)==27) break;
	}
	std::cout << "fps avg: " << fps_sum / fps_num << std::endl;
    std::cout << "fps max: " << fps_max << std::endl;
    std::cout << "fps min: " << fps_min << std::endl;
    std::cout << "time avg: " << fps_num / fps_sum << std::endl;
	// waitKey(0);

	return 0;
}