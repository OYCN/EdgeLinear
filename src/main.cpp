#include "EDProcess_par.h"
#include "Timer.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>

#define PI 3.14159265

TDEF(EdgeDrawing);
TDEF(SearchLines_cpu);
TDEF(SearchLines_gpu);
TDEF(EDToVector);
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

	cv::Mat imgED = cv::Mat::zeros(src.rows, src.cols, CV_8UC3);
	cv::Mat imgDP_C = cv::Mat::zeros(src.rows, src.cols, CV_8UC3);
	cv::Mat imgDP_G = cv::Mat::zeros(src.rows, src.cols, CV_8UC3);

	// 主类初始化
	Main MainClass(src.rows, src.cols);

	TSTART(EdgeDrawing);

	// 边缘提取
	eMap = MainClass.Process(src, edge_seg, edge_seg_offset, edge_seg_len);

	TEND(EdgeDrawing);
	TPRINTMS(EdgeDrawing, "EdgeDrawing: ");
	std::cout << "Number of Edge Segment : " << edge_seg_len-1 << std::endl;
	std::cout << "Number of Line Point : " << edge_seg_offset[edge_seg_len-1] << std::endl;
	int max = 0;
	for(int i=0; i<edge_seg_len; i++)
	{
		int l = edge_seg_offset[i+1]-edge_seg_offset[i];
		if(l > max) max = l;
	}
	std::cout << "Max len is : " << max << std::endl;

	// gpu 的 DP算法
	TSTART(SearchLines_gpu);
	point_sum_gpu = MainClass.runDP(line_all_gpu);
	TEND(SearchLines_gpu);TPRINTMS(SearchLines_gpu, "SearchLines_GPU: ");
	std::cout << "line points_gpu is " << point_sum_gpu << std::endl;

	#ifndef JUST_ED
	// 将ED化线段转为vector类型
	TSTART(EDToVector);
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
	TEND(EDToVector);TPRINTMS(EDToVector, "EDToVector: ");
	// cpu 的 DP 算法
	TSTART(SearchLines_cpu);
	cpuDP(edge_seg_vec, line_all_cpu);
	TEND(SearchLines_cpu);TPRINTMS(SearchLines_cpu, "SearchLines_CPU: ");
	// 统计点个数
	point_sum_cpu = 0;
	for(VECTOR_H<VECTOR_H<POINT>>::const_iterator l=line_all_cpu.begin(); l != line_all_cpu.end(); l++)
	{
		point_sum_cpu += (*l).size();
	}
	std::cout << "line points_cpu is " << point_sum_cpu << std::endl;

	// 绘制原图
	cv::cvtColor(src, grayImg, CV_RGB2GRAY);
	grayImg /= 2; 
	cv::cvtColor(grayImg, result, CV_GRAY2RGB);
	// 绘制边缘
	for(int i=0; i<(edge_seg_len-1); i++)
	{
		for(int j=edge_seg_offset[i]; j<edge_seg_offset[i+1]; j++)
		{
			result.at<cv::Vec3b>((cv::Point)(edge_seg[j]))[0] = 0;
			result.at<cv::Vec3b>((cv::Point)(edge_seg[j]))[1] = 0;
			result.at<cv::Vec3b>((cv::Point)(edge_seg[j]))[2] = 255;

			imgED.at<cv::Vec3b>((cv::Point)(edge_seg[j]))[0] = 0;
			imgED.at<cv::Vec3b>((cv::Point)(edge_seg[j]))[1] = 0;
			imgED.at<cv::Vec3b>((cv::Point)(edge_seg[j]))[2] = 255;
		}
	}
	// 绘制GPU的DP后的直线
	for(int i=0; i<line_all_gpu.size(); i++)
	{
		if(line_all_gpu[i].size()>1)
		{
			for(int idx=0; idx < (line_all_gpu[i].size()-1); idx++)
			{
				cv::line(result, line_all_gpu[i][idx], line_all_gpu[i][idx+1], cv::Scalar(0, 255, 0), 1, 4);
				cv::line(imgDP_C, line_all_gpu[i][idx], line_all_gpu[i][idx+1], cv::Scalar(0, 255, 0), 1, 4);
			}
		}
	}
	// 绘制CPU的DP后的直线
	for(VECTOR_H<VECTOR_H<POINT>>::const_iterator l=line_all_cpu.begin(); l != line_all_cpu.end(); l++)
	{
		for(int idx=0; idx < ((*l).size()-1); idx++)
		{
			cv::line(result, (*l)[idx], (*l)[idx+1], cv::Scalar(255, 0, 0), 1, 4);
			cv::line(imgDP_G, (*l)[idx], (*l)[idx+1], cv::Scalar(255, 0, 0), 1, 4);
		}
	}
	#ifdef USE_CHECK
	// 检验GPU与CPU的DP结果是否相同
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
	cv::namedWindow("org",CV_WINDOW_NORMAL);
	cv::namedWindow("gray",CV_WINDOW_NORMAL);
	cv::namedWindow("ED",CV_WINDOW_NORMAL);
	cv::namedWindow("DP_CPU",CV_WINDOW_NORMAL);
	cv::namedWindow("DP_GPU",CV_WINDOW_NORMAL);
	cv::imshow("result", result);
	cv::imshow("org", src);
	cv::imshow("gray", grayImg);
	cv::imshow("ED", imgED);
	cv::imshow("DP_CPU", imgDP_C);
	cv::imshow("DP_GPU", imgDP_G);
	cv::waitKey();
	#endif
	#endif	//JUST_ED

	eMap.release();
	return 0;
}