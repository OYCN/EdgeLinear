#include "EDProcess_par.h"
#include "Timer.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <sstream>

int main(int argc, char *args[])
{
	cv::Mat src, grayImg, eMap, result; 
	POINT *edge_seg;
	int *edge_seg_offset;
	int edge_seg_len;
	bool *flag;
	VECTOR_H<VECTOR_H<POINT>> line_all_gpu, line_all_cpu;
	int point_sum_gpu = 0;
	int point_sum_cpu = 0;

	// 图像加载
	cv::VideoCapture capture;
	if(argc!=1) capture.open(args[1]);
	else capture.open("img/2.mp4");
	if(!capture.isOpened())
	{
		printf("[%s][%d]could not load video data...\n",__FUNCTION__,__LINE__);
		return -1;
	}
	Main MainClass(capture.get(CV_CAP_PROP_FRAME_HEIGHT), capture.get(CV_CAP_PROP_FRAME_WIDTH));
	std::cout << capture.get(CV_CAP_PROP_FRAME_HEIGHT) << " * " << capture.get(CV_CAP_PROP_FRAME_WIDTH) << std::endl;
	double fps_start;
	double fps_ED;
	double fps_DP;
	double cpu_times = 0, gpu_timesA = 0, gpu_timesB = 0;
	int loop_time = 0;
	double fps_ED_sum = 0;
	double fps_DP_sum = 0;
	double fps_DP_max = 0, fps_DP_min = 99999;
	double fps_ED_max = 0, fps_ED_min = 99999;
	int fps_ED_max_seg = 0, fps_ED_min_seg = 99999;
	int fps_DP_max_seg = 0, fps_DP_min_seg = 99999;
	int seg_sum = 0;
	while(capture.read(src))
	{
		++loop_time;
		fps_start = (double)cv::getTickCount();

		// 边缘提取
		eMap = MainClass.Process(src, edge_seg, edge_seg_offset, edge_seg_len);
		fps_ED = cv::getTickFrequency() / ((double)cv::getTickCount() - fps_start);
		// std::cout << "Number of Edge Segment : " << edge_seg_len << std::endl;
		// std::cout << "Number of Line Point : " << edge_seg_offset[edge_seg_len-1] << std::endl;
		// int max = 0;
		// for(int i=0; i<edge_seg_len-1; i++)
		// {
		// 	int l = edge_seg_offset[i+1]-edge_seg_offset[i];
		// 	if(l > max) max = l;
		// }
		// std::cout << "Max len is : " << max << std::endl;

		// gpu 的 DP算法
		// double gpu_a = (double)cv::getTickCount();
		MainClass.runDP(flag);
		// double gpu_b = (double)cv::getTickCount();
		// gpu_timesA += (gpu_b - gpu_a) / cv::getTickFrequency();
		// gpu_a = (double)cv::getTickCount();
		// point_sum_gpu = 0;
		for(int i=0; i<(edge_seg_len-1); i++)
		{
			VECTOR_H<POINT> oneline;
			for(int j=edge_seg_offset[i]; j<edge_seg_offset[i+1]; j++)
			{
				if(flag[j])
				{
					oneline.push_back(edge_seg[j]);
					// point_sum_gpu++;
				}
			}
			line_all_gpu.push_back(oneline);
			VECTOR_H<POINT>().swap(oneline);
		}
		// gpu_b = (double)cv::getTickCount();
		// gpu_timesB += (gpu_b - gpu_a) / cv::getTickFrequency();
		// std::cout << "line points_gpu is " << point_sum_gpu << std::endl;

		// 将ED线段转为vector类型
		// VECTOR_H<VECTOR_H<POINT>> edge_seg_vec;
		// for(int i=0; i<(edge_seg_len-1); i++)
		// {
		// 	VECTOR_H<POINT> tmp;
		// 	for(int j=edge_seg_offset[i]; j<edge_seg_offset[i+1]; j++)
		// 	{
		// 		tmp.push_back(edge_seg[j]);
		// 	}
		// 	edge_seg_vec.push_back(tmp);
		// 	VECTOR_H<POINT>().swap(tmp);
		// }

		// cpu 的 DP 算法
		// double cpu_a = (double)cv::getTickCount();
		// cpuDP(edge_seg_vec, line_all_cpu);
		// double cpu_b = (double)cv::getTickCount();
		// cpu_times += (cpu_b - cpu_a) / cv::getTickFrequency();

		// 统计点个数
		// point_sum_cpu = 0;
		// for(VECTOR_H<VECTOR_H<POINT>>::const_iterator l=line_all_cpu.begin(); l != line_all_cpu.end(); l++)
		// {
		// 	point_sum_cpu += (*l).size();
		// }
		// std::cout << "line points_cpu is " << point_sum_cpu << std::endl;

		// 检验GPU与CPU的DP结果是否相同
		// bool check_success = true;
		// if(line_all_cpu.size() == line_all_gpu.size())
		// {
		// 	for(int i=0; i< line_all_cpu.size(); i++)
		// 	{
		// 		if(line_all_cpu[i].size() == line_all_gpu[i].size())
		// 		{
		// 			for(int j=0; j<line_all_cpu[i].size(); j++)
		// 			{
		// 				if(line_all_cpu[i][j] != line_all_gpu[i][j])
		// 				{
		// 					check_success = false;
		// 					std::cout << "line_all_cpu[" << i << "][" << j << "] error" << std::endl;
		// 					break;
		// 				}
		// 			}
		// 		}
		// 		else
		// 		{
		// 			check_success = false;
		// 			std::cout << "line_all_cpu[" << i << "] error" << std::endl;
		// 			break;
		// 		}
				
		// 	}
		// } 
		// else
		// {
		// 	check_success = false;
		// 	std::cout << "line_all num error" << std::endl;
		// }
		// if(!check_success) { std::cout << "check failed" << std::endl; exit(-1);}

		// 绘制边缘
		// #ifdef SHOW
		src.copyTo(result);
		cv::Mat imgED = cv::Mat::zeros(capture.get(CV_CAP_PROP_FRAME_HEIGHT), capture.get(CV_CAP_PROP_FRAME_WIDTH), CV_8UC3);
		// cv::Mat imgDP_C = cv::Mat::zeros(capture.get(CV_CAP_PROP_FRAME_HEIGHT), capture.get(CV_CAP_PROP_FRAME_WIDTH), CV_8UC3);
		cv::Mat imgDP_G = cv::Mat::zeros(capture.get(CV_CAP_PROP_FRAME_HEIGHT), capture.get(CV_CAP_PROP_FRAME_WIDTH), CV_8UC3);
		for(int i=0; i<(edge_seg_len-1); i++)
		{
			for(int j=edge_seg_offset[i]; j<edge_seg_offset[i+1]; j++)
			{
				result.at<cv::Vec3b>((cv::Point)(edge_seg[j]))[0] = 0;
				result.at<cv::Vec3b>((cv::Point)(edge_seg[j]))[1] = 0;
				result.at<cv::Vec3b>((cv::Point)(edge_seg[j]))[2] = 255;

				imgED.at<cv::Vec3b>((cv::Point)(edge_seg[j]))[0] = 255;
				imgED.at<cv::Vec3b>((cv::Point)(edge_seg[j]))[1] = 255;
				imgED.at<cv::Vec3b>((cv::Point)(edge_seg[j]))[2] = 0;
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
					cv::line(imgDP_G, line_all_gpu[i][idx], line_all_gpu[i][idx+1], cv::Scalar(0, 255, 0), 1, 4);
				}
			}
		}
		
		// 绘制CPU的DP后的直线
		// for(VECTOR_H<VECTOR_H<POINT>>::const_iterator l=line_all_cpu.begin(); l != line_all_cpu.end(); l++)
		// {
		// 	for(int idx=0; idx < ((*l).size()-1); idx++)
		// 	{
		// 		cv::line(result, (*l)[idx], (*l)[idx+1], cv::Scalar(255, 0, 0), 1, 4);
		// 		cv::line(imgDP_C, (*l)[idx], (*l)[idx+1], cv::Scalar(255, 0, 0), 1, 4);
		// 	}
		// }
		
		fps_DP = cv::getTickFrequency() / ((double)cv::getTickCount() - fps_start);
		fps_ED_sum += fps_ED;
		fps_DP_sum += fps_DP;
		seg_sum += edge_seg_len;
		if(fps_ED > fps_ED_max) {fps_ED_max = fps_ED; fps_ED_max_seg = edge_seg_len;}
		if(fps_ED < fps_ED_min) {fps_ED_min = fps_ED; fps_ED_min_seg = edge_seg_len;}
		if(fps_DP > fps_DP_max) {fps_DP_max = fps_DP; fps_DP_max_seg = edge_seg_len;}
		if(fps_DP < fps_DP_min) {fps_DP_min = fps_DP; fps_DP_min_seg = edge_seg_len;}
		std::ostringstream buffer_ED;
		buffer_ED << "FPS ED " << fps_ED;
		std::ostringstream buffer_DP;
		buffer_DP << "FPS DP " << fps_DP;
		cv::putText(result,
				buffer_ED.str(),
				cv::Point(5,20),
				cv::FONT_HERSHEY_SIMPLEX,
				0.5,
				cv::Scalar(255,0,255));
		cv::putText(result,
				buffer_DP.str(),
				cv::Point(5,40),
				cv::FONT_HERSHEY_SIMPLEX,
				0.5,
				cv::Scalar(255,0,255));
		cv::imwrite("out.png", result);
		cv::namedWindow("org",CV_WINDOW_NORMAL);
		// cv::namedWindow("gray",CV_WINDOW_NORMAL);
		cv::namedWindow("ED",CV_WINDOW_NORMAL);
		// cv::namedWindow("DP_CPU",CV_WINDOW_NORMAL);
		cv::namedWindow("DP_GPU",CV_WINDOW_NORMAL);
		cv::namedWindow("result",CV_WINDOW_NORMAL);
		cv::imshow("org", src);
		// cv::imshow("gray", grayImg);
		cv::imshow("ED", imgED);
		// cv::imshow("DP_CPU", imgDP_C);
		cv::imshow("DP_GPU", imgDP_G);
		cv::imshow("result", result);
		char key = cv::waitKey(1);

		if (key==27)	// esc退出
		{
			break;
		}
		else if(key == 44)	// ',' 减小阈值
		{
			int th = MainClass.getTH()!=0?(MainClass.getTH()-1):0;
			MainClass.setTH(th);
			std::cout << "th change to " << th << std::endl;
		}
		else if(key == 46)	// '.' 增大阈值
		{
			int th = MainClass.getTH()+1;
			MainClass.setTH(th);
			std::cout << "th change to " << th << std::endl;
		}
		else if(key == ' ')
		{
			while(cv::waitKey(1) != ' ');
		}
		// #endif // SHOW
		line_all_gpu.clear();
		line_all_cpu.clear();
		// VECTOR_H<VECTOR_H<POINT>>().swap(line_all_gpu);
		// VECTOR_H<VECTOR_H<POINT>>().swap(line_all_cpu);
	}

	std::cout << "shard mem stack size " << sharedMemPerBlock / sizeof(POINT) / 16 << "\n" <<
				"loop time: " << loop_time << "\n" <<
				// "cpu_times:" << cpu_times << "\n" <<
				// "gpu_timesA:" << gpu_timesA << "\n" <<
				// "gpu_timesB:" << gpu_timesB << "\n" <<
				"seg_avg:" << seg_sum / loop_time << "\n" <<
				" ===== " << "\n"
				"[ED]fps avg:" << fps_ED_sum / loop_time << "\n" <<
				"[ED]fps max:" << fps_ED_max << "\n" <<
				"[ED]when_seg:" << fps_ED_max_seg << "\n" <<
				"[ED]fps_min:" << fps_ED_min << "\n" <<
				"[ED]when_seg:" << fps_ED_min_seg << "\n" <<
				" ===== " << "\n"
				"[DP]fps avg:" << fps_DP_sum / loop_time << "\n" <<
				"[DP]fps max:" << fps_DP_max << "\n" <<
				"[DP]when seg:" << fps_DP_max_seg << "\n" <<
				"[DP]fps min:" << fps_DP_min << "\n" <<
				"[DP]when seg:" << fps_DP_min_seg << "\n" <<
				std::endl;
	
	return 0;
}