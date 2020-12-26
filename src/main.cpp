#include "EDProcess_par.h"
#include "Timer.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>

#define PI 3.14159265

TDEF(EdgeDrawing);
TDEF(EdInit);
TDEF(EdProc);
TDEF(SearchLines);
// TDEF(deal);
// TDEF(proc);

int main(int argc, char *args[])
{
	
	ED edgeDrawing;
	VECTOR_H<VECTOR_H<cv::Point>> edge_seg;
	#ifdef USE_DP
	VECTOR_H<VECTOR_H<cv::Point>> line_seg;
	#endif
	#ifdef USE_CUR
	VECTOR_H<VECTOR_H<VECTOR_H<cv::Point>>> line_segn;
	VECTOR_H<VECTOR_H<cv::Point>> line_sego;
	#endif
	cv::Mat src, grayImg, blurImg, eMap, result; 
	if(argc!=1) src = cv::imread(args[1]);
	else src = cv::imread(DEFIMG);

	//检查图片是否成功加载
	if (src.empty()) 
	{
		std::cout << "Can not load image!" << std::endl;
		return 0;
	}
	TSTART(EdgeDrawing);
	TSTART(EdInit);
	// get grayImg
	cv::cvtColor(src, grayImg, CV_RGB2GRAY);
	// gaussian filter
	cv::GaussianBlur(grayImg, blurImg, cv::Size(5, 5), 1, 0);
	TEND(EdInit);
	TSTART(EdProc);
	// ED edge detection 
	eMap = edgeDrawing.Process(blurImg, edge_seg);
	TEND(EdProc);
	TEND(EdgeDrawing);
	TPRINTMS(EdInit, "EdgeImgInit: ");
	TPRINTMS(EdProc, "EdgeImgProc: ");
	TPRINTMS(EdgeDrawing, "EdgeDrawing: ");

	std::cout << "Number of Edge Segment : " << edge_seg.size() << std::endl;

	#ifdef JUST_ED
	eMap *= 255;
	cv::imwrite("out.png", eMap);
	#ifdef SHOW_IMG
	cv::namedWindow("eMap",CV_WINDOW_NORMAL);
	cv::imshow("eMap", eMap);
	cv::waitKey();
	#endif
	#endif

	#ifndef JUST_ED
	#ifdef USE_DP
	TSTART(SearchLines);
	for(VECTOR_H<VECTOR_H<cv::Point>>::const_iterator e=edge_seg.begin(); e != edge_seg.end(); e++)
	{
		VECTOR_H<cv::Point> line;
		// cv::approxPolyDP(*e, line, 5, false);
		// mygpu::approxPolyDP(*e, line, 5, false);
		DouglasPeucker(*e, line, 5);
		line_seg.push_back(line);
	}
	TEND(SearchLines);TPRINTMS(SearchLines, "SearchLines: ");
	#endif
	#ifdef USE_CUR
	TSTART(SearchLines);
	line_sego = orgline(edge_seg);
	for(VECTOR_H<VECTOR_H<cv::Point>>::const_iterator e=edge_seg.begin(); e != edge_seg.end(); e++)
	{
		VECTOR_H<VECTOR_H<cv::Point>> line;
		getLine(*e, line);
		line_segn.push_back(line);
	}
	TEND(SearchLines);TPRINTMS(SearchLines, "SearchLines: ");
	#endif
	// imshow("src image", src);
	// imshow("eMap", eMap);
	// cv::imwrite("out.png", eMap);
	grayImg /= 2; 
	cv::cvtColor(grayImg, result, CV_GRAY2RGB);
	// draw edges
	for(VECTOR_H<VECTOR_H<cv::Point>>::const_iterator e=edge_seg.begin(); e != edge_seg.end(); e++)
	{
		for(VECTOR_H<cv::Point>::const_iterator p=(*e).begin(); p != (*e).end(); p++)
		{
			result.at<cv::Vec3b>(*p)[0] = 0;
			result.at<cv::Vec3b>(*p)[1] = 0;
			result.at<cv::Vec3b>(*p)[2] = 255;
		}
	}
	#ifdef USE_DP
	// draw lines
	for(VECTOR_H<VECTOR_H<cv::Point>>::const_iterator l=line_seg.begin(); l != line_seg.end(); l++)
	{
		for(int idx=0; idx < ((*l).size()-1); idx++)
		{
			cv::line(result, (*l)[idx], (*l)[idx+1], cv::Scalar(0, 255, 0), 1, 4);
		}
	}
	#endif
	#ifdef USE_CUR
	// draw lines
	for(VECTOR_H<VECTOR_H<cv::Point>>::const_iterator l=line_sego.begin(); l != line_sego.end(); l++)
	{
		for(VECTOR_H<cv::Point>::const_iterator p=(*l).begin(); p != (*l).end(); p++)
		{
			result.at<cv::Vec3b>(*p)[0] = 0;
			result.at<cv::Vec3b>(*p)[1] = 255;
			result.at<cv::Vec3b>(*p)[2] = 0;
		}
	}
	for(VECTOR_H<VECTOR_H<VECTOR_H<cv::Point>>>::const_iterator l=line_segn.begin(); l != line_segn.end(); l++)
	{
		for(VECTOR_H<VECTOR_H<cv::Point>>::const_iterator al=(*l).begin(); al != (*l).end(); al++)
		{
			cv::line(result, (*al)[0], (*al)[1], cv::Scalar(255, 0, 0), 1, 4);
		}
	}
	#endif
	cv::imwrite("out.png", result);
	#ifdef SHOW_IMG
	cv::namedWindow("result",CV_WINDOW_NORMAL);
	cv::imshow("result", result);
	cv::waitKey();
	#endif
	#endif	//JUST_ED
	VECTOR_H<VECTOR_H<cv::Point>>().swap(edge_seg);

	eMap.release();
	return 0;
}