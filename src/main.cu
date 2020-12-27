#include "EDProcess_par.h"
#include "Timer.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>

#define PI 3.14159265

TDEF(EdgeDrawing);
TDEF(EdInit);
TDEF(EdProc);
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
	ED edgeDrawing;
	VECTOR_H<VECTOR_H<POINT>> edge_seg;
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
	#ifdef USE_GPUDP
	TSTART(SearchLines_gpu);
	// 每个边缘的堆栈内存和flag内存偏移数组
	int *edge_offset = new int[edge_seg.size()+1];
	int *edge_offset_d;
	// 全部边缘点集数组
	POINT **edge_seg_dd;
	int sum = 0;
	bool *flags_h;
	bool *flags_d;
	POINT *stack_d;
	edge_offset[0] = 0;
	#ifdef TIM_GPUDP
	TSTART(Sgpu_h2d);
	#endif
	POINT **edge_seg_hd = new POINT*[edge_seg.size()];
	for(int i=0; i<edge_seg.size(); i++)
	{
		sum += edge_seg[i].size();
		edge_offset[i+1] = sum;
		HANDLE_ERROR(cudaMalloc(&edge_seg_hd[i], sizeof(POINT)*edge_seg[i].size()));
		HANDLE_ERROR(cudaMemcpy(edge_seg_hd[i], edge_seg[i].data(), sizeof(POINT)*edge_seg[i].size(), cudaMemcpyHostToDevice));
	}
	std::cout << "sum is " << sum << std::endl;
	HANDLE_ERROR(cudaMalloc(&flags_d, sizeof(bool)*sum));
	HANDLE_ERROR(cudaMemset(flags_d, 0, sizeof(bool)*sum));
	HANDLE_ERROR(cudaMalloc(&stack_d, sizeof(POINT)*sum));
	HANDLE_ERROR(cudaMalloc(&edge_offset_d, sizeof(int)*(edge_seg.size()+1)));
	HANDLE_ERROR(cudaMemcpy(edge_offset_d, edge_offset, sizeof(int)*(edge_seg.size()+1), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMalloc(&edge_seg_dd, sizeof(POINT*)*edge_seg.size()));
	HANDLE_ERROR(cudaMemcpy(edge_seg_dd, edge_seg_hd, sizeof(POINT*)*edge_seg.size(), cudaMemcpyHostToDevice));
	#ifdef TIM_GPUDP
	TEND(Sgpu_h2d);TPRINTMS(Sgpu_h2d, "SearchLines_GPU H2D: ");
	TSTART(Sgpu_kernel);
	#endif
	DouglasPeucker<<<edge_seg.size(),1>>>(edge_seg_dd, edge_offset_d, edge_seg.size(), stack_d, flags_d, 5);
	HANDLE_ERROR(cudaDeviceSynchronize());
	#ifdef TIM_GPUDP
	TEND(Sgpu_kernel);TPRINTMS(Sgpu_kernel, "SearchLines_GPU Kernel: ");
	TSTART(Sgpu_d2h);
	#endif
	flags_h = new bool[sum];
	HANDLE_ERROR(cudaMemcpy(flags_h, flags_d, sizeof(bool)*sum, cudaMemcpyDeviceToHost));
	#ifdef TIM_GPUDP
	TEND(Sgpu_d2h);TPRINTMS(Sgpu_d2h, "SearchLines_GPU D2H: ");
	TSTART(Sgpu_dealf);
	#endif
	VECTOR_H<VECTOR_H<POINT>> line_all_gpu;
	for(int i=0; i<edge_seg.size(); i++)
	{
		VECTOR_H<POINT> oneline;
		for(int j=0; j<edge_seg[i].size(); j++)
		{
			if(flags_h[edge_offset[i]+j])
				oneline.push_back(edge_seg[i][j]);
		}
		line_all_gpu.push_back(oneline);
		VECTOR_H<POINT>().swap(oneline);
	}
	#ifdef TIM_GPUDP
	TEND(Sgpu_dealf);TPRINTMS(Sgpu_dealf, "SearchLines_GPU Deal Flags: ");
	#endif
	TEND(SearchLines_gpu);TPRINTMS(SearchLines_gpu, "SearchLines_GPU: ");
	int point_sum_gpu = 0;
	for(int i=0; i<sum; i++)
	{
		if(flags_h[i]) point_sum_gpu++;
	}
	std::cout << "line points_gpu is " << point_sum_gpu << std::endl;
	#endif
	#ifdef USE_CPUDP
	VECTOR_H<VECTOR_H<POINT>> line_all_cpu;
	TSTART(SearchLines_cpu);
	for(VECTOR_H<VECTOR_H<POINT>>::const_iterator e=edge_seg.begin(); e != edge_seg.end(); e++)
	{
		VECTOR_H<POINT> line;
		// cv::approxPolyDP(*e, line, 5, false);
		// mygpu::approxPolyDP(*e, line, 5, false);
		DouglasPeucker(*e, line, 5);
		line_all_cpu.push_back(line);
	}
	TEND(SearchLines_cpu);TPRINTMS(SearchLines_cpu, "SearchLines_CPU: ");
	int point_sum_cpu = 0;
	for(VECTOR_H<VECTOR_H<POINT>>::const_iterator l=line_all_cpu.begin(); l != line_all_cpu.end(); l++)
	{
		point_sum_cpu += (*l).size();
	}
	std::cout << "line points_cpu is " << point_sum_cpu << std::endl;
	#endif
	grayImg /= 2; 
	cv::cvtColor(grayImg, result, CV_GRAY2RGB);
	// draw edges
	for(VECTOR_H<VECTOR_H<POINT>>::const_iterator e=edge_seg.begin(); e != edge_seg.end(); e++)
	{
		for(VECTOR_H<POINT>::const_iterator p=(*e).begin(); p != (*e).end(); p++)
		{
			result.at<cv::Vec3b>((cv::Point)(*p))[0] = 0;
			result.at<cv::Vec3b>((cv::Point)(*p))[1] = 0;
			result.at<cv::Vec3b>((cv::Point)(*p))[2] = 255;
		}
	}
	#ifdef USE_GPUDP
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
	#endif
	#ifdef USE_CPUDP
	// draw lines
	for(VECTOR_H<VECTOR_H<POINT>>::const_iterator l=line_all_cpu.begin(); l != line_all_cpu.end(); l++)
	{
		for(int idx=0; idx < ((*l).size()-1); idx++)
		{
			cv::line(result, (*l)[idx], (*l)[idx+1], cv::Scalar(0, 255, 0), 1, 4);
		}
	}
	#endif
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
	VECTOR_H<VECTOR_H<POINT>>().swap(edge_seg);

	eMap.release();
	return 0;
}