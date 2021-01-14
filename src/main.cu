#include "EDProcess_par.h"
#include "Timer.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>

#define PI 3.14159265

TDEF(EdgeDrawing);
TDEF(SearchLines_cpu);
TDEF(SearchLines_gpu);
#ifdef TIM_GPUDP
TDEF(Sgpu_h2d);
TDEF(Sgpu_kernel);
TDEF(Sgpu_d2h);
TDEF(Sgpu_dealf);
#endif
// TDEF(deal);
// TDEF(proc);

int main(int argc, char *args[])
{
	cv::Mat src, grayImg, eMap, result; 
	POINT *edge_seg;
	int *edge_seg_offset;
	int edge_seg_len;
	VECTOR_H<VECTOR_H<POINT>> line_all_gpu, line_all_cpu;
	int point_sum_gpu = 0;
	int point_sum_cpu = 0;

	// 图像加载
	if(argc!=1) src = cv::imread(args[1]);
	else src = cv::imread(DEFIMG);
	if (src.empty()) 
	{
		std::cout << "Can not load image!" << std::endl;
		return 0;
	}

	// 主类初始化
	Main MainClass(src.rows, src.cols);

	TSTART(EdgeDrawing);

	// 边缘提取
	eMap = MainClass.Process(src, edge_seg, edge_seg_offset, edge_seg_len);

	TEND(EdgeDrawing);
	TPRINTMS(EdgeDrawing, "EdgeDrawing: ");
	std::cout << "Number of Edge Segment : " << edge_seg_len-1 << std::endl;
	std::cout << "Number of Line Point : " << edge_seg_offset[edge_seg_len-1] << std::endl;

	TSTART(SearchLines_gpu);
	point_sum_gpu = MainClass.runDP(line_all_gpu);
	TEND(SearchLines_gpu);TPRINTMS(SearchLines_gpu, "SearchLines_GPU: ");
	std::cout << "line points_gpu is " << point_sum_gpu << std::endl;

	#ifndef JUST_ED
	VECTOR_H<VECTOR_H<POINT>> edge_seg_vec;
	for(int i=0; i<(edge_seg_len-1); i++)
	{
		VECTOR_H<POINT> tmp;
		for(int j=edge_seg_offset[i]; j<edge_seg_offset[i+1]; j++)
		{
			tmp.push_back(edge_seg[j]);
		}
		edge_seg_vec.push_back(tmp);
		VECTOR_H<POINT>().swap(tmp);
	}
	TSTART(SearchLines_cpu);
	for(VECTOR_H<VECTOR_H<POINT>>::const_iterator e=edge_seg_vec.begin(); e != edge_seg_vec.end(); e++)
	{
		VECTOR_H<POINT> line;
		// cv::approxPolyDP(*e, line, 5, false);
		// mygpu::approxPolyDP(*e, line, 5, false);
		DouglasPeucker(*e, line, 5);
		line_all_cpu.push_back(line);
	}
	TEND(SearchLines_cpu);TPRINTMS(SearchLines_cpu, "SearchLines_CPU: ");
	point_sum_cpu = 0;
	for(VECTOR_H<VECTOR_H<POINT>>::const_iterator l=line_all_cpu.begin(); l != line_all_cpu.end(); l++)
	{
		point_sum_cpu += (*l).size();
	}
	std::cout << "line points_cpu is " << point_sum_cpu << std::endl;

	cv::cvtColor(src, grayImg, CV_RGB2GRAY);
	grayImg /= 2; 
	cv::cvtColor(grayImg, result, CV_GRAY2RGB);
	for(int i=0; i<(edge_seg_len-1); i++)
	{
		for(int j=edge_seg_offset[i]; j<edge_seg_offset[i+1]; j++)
		{
			result.at<cv::Vec3b>((cv::Point)(edge_seg[j]))[0] = 0;
			result.at<cv::Vec3b>((cv::Point)(edge_seg[j]))[1] = 0;
			result.at<cv::Vec3b>((cv::Point)(edge_seg[j]))[2] = 255;
		}
	}
	for(int i=0; i<line_all_gpu.size(); i++)
	{
		if(line_all_gpu[i].size()>1)
		{
			for(int idx=0; idx < (line_all_gpu[i].size()-1); idx++)
			{
				cv::line(result, line_all_gpu[i][idx], line_all_gpu[i][idx+1], cv::Scalar(0, 255, 0), 1, 4);
			}
		}
	}
	// draw lines
	for(VECTOR_H<VECTOR_H<POINT>>::const_iterator l=line_all_cpu.begin(); l != line_all_cpu.end(); l++)
	{
		for(int idx=0; idx < ((*l).size()-1); idx++)
		{
			cv::line(result, (*l)[idx], (*l)[idx+1], cv::Scalar(255, 0, 0), 1, 4);
		}
	}
	#ifdef USE_CHECK
	bool check_success = true;
	if(line_all_cpu.size() == line_all_gpu.size())
	{
		for(int i=0; i< line_all_cpu.size(); i++)
		{
			if(line_all_cpu[i].size() == line_all_gpu[i].size())
			{
				for(int j=0; j<line_all_cpu[i].size(); j++)
				{
					if(line_all_cpu[i][j] != line_all_gpu[i][j])
					{
						check_success = false;
						std::cout << "line_all_cpu[" << i << "][" << j << "] error" << std::endl;
						break;
					}
				}
			}
			else
			{
				check_success = false;
				std::cout << "line_all_cpu[" << i << "] error" << std::endl;
				break;
			}
			
		}
	} 
	else
	{
		check_success = false;
		std::cout << "line_all num error" << std::endl;
	}
	if(check_success) std::cout << "check success" << std::endl;
	else std::cout << "check failed" << std::endl;
	
	#endif
	cv::imwrite("out.png", result);
	#ifdef SHOW_IMG
	cv::namedWindow("result",CV_WINDOW_NORMAL);
	cv::imshow("result", result);
	cv::waitKey();
	#endif
	#endif	//JUST_ED

	eMap.release();
	return 0;
}