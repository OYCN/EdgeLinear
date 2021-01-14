#include "EDProcess_par.h"
#include "Timer.h"
#include <assert.h>

/*
*Summary: 构造函数
*Parameters: 
*    _rows: 待处理图像行数
*    _cols: 待处理图像列数
*    _anchor_th: 锚点阈值
*    _k: 锚点稀疏程度
*Return: 无
*/
Main::Main(int _rows, int _cols, int _anchor_th, int _k)
:rows(_rows), cols(_cols), anchor_th(_anchor_th), k(_k)
{
	int count;
	HANDLE_ERROR(cudaGetDeviceCount(&count));
	if(count!=1) {printf( "zero or multiple gpu\n"); exit( EXIT_FAILURE );}
	cudaSetDevice(0);

	dimBlock = dim3(32,32);
	dimGrid = dim3((cols+27)/28, (rows+27)/28);
	dimGridOld = dim3(rows, cols);
	_InitED();
	_InitPD();
}

/*
*Summary: 析构函数
*Parameters: 无
*Return: 无
*/
Main::~Main()
{
	_FreeED();
	_FreePD();
}

/*
*Summary: 类所需内存的申请等初始化操作，一个实例仅需一次
*Parameters: 无
*Return: 无
*/
void Main::_InitED()
{
	HANDLE_ERROR(cudaMalloc(&gMapd, sizeof(uchar)*rows*cols));
	#ifndef USE_OPENCV_GPU
	HANDLE_ERROR(cudaMalloc(&blurd, sizeof(uchar)*rows*cols));
	#endif
	HANDLE_ERROR(cudaMalloc(&fMapd, sizeof(uchar)*rows*cols));
	HANDLE_ERROR(cudaMemset(gMapd, 0, sizeof(uchar)*rows*cols));
	HANDLE_ERROR(cudaMemset(fMapd, 0, sizeof(uchar)*rows*cols));
	gMaph = new uchar[rows*cols];
	fMaph = new uchar[rows*cols];
	eMaph = cv::Mat::zeros(rows, cols, CV_8UC1);
	edge_set = new POINT[rows*cols];
	edge_offset = new int[rows*cols+1];
	edge_smart = new POINT[rows*cols];
}
/*
*Summary: 释放内存
*Parameters: 无
*Return: 无
*/
void Main::_FreeED()
{
	cudaFree(gMapd);
	#ifndef USE_OPENCV_GPU
	cudaFree(blurd);
	#endif
	cudaFree(fMapd);
	delete[] gMaph;
	delete[] fMaph;
	delete[] edge_set;
	delete[] edge_offset;
	delete[] edge_smart;
}


/*
*Summary: 对原图像进行预处理
*Parameters: 
*    src: 原图像
*Return: 无
*/
void Main::PerProcED(cv::Mat &src)
{
	#ifndef USE_OPENCV_GPU
	cv::cvtColor(src, grayImg, CV_RGB2GRAY);
	cv::GaussianBlur(grayImg, blurImg, cv::Size(5, 5), 1, 0);
	#endif
	#ifdef USE_OPENCV_GPU
	src_d.upload(src);
	cv::cuda::cvtColor(src_d, grayImg, CV_RGB2GRAY);
	cv::cuda::GaussianBlur(grayImg, blurImg, cv::Size(5, 5), 1, 0);
	#endif
}

/*
*Summary: 对传入图像进行单次边缘提取
*Parameters:
*     blurImg: 传入图像
*     edge_seg: 用于保存边缘信息
*Return: 边缘信息的图像
*/
cv::Mat Main::Process(cv::Mat& src, POINT *&edge_seg, int *&edge_seg_offset, int &edge_seg_len)
{
	// 确保图像尺寸与内存相符
	assert(src.rows == rows);
	assert(src.cols == cols);

	PerProcED(src);

	#ifndef USE_OPENCV_GPU
	// 本次数据拷贝
	HANDLE_ERROR(cudaMemcpy(blurd, blurImg.data, sizeof(uchar)*rows*cols, cudaMemcpyHostToDevice));
	#endif
	#ifdef USE_OPENCV_GPU
	blurd = blurImg.data;
	#endif

	// 核函数启动
	kernelC<<< dimGrid, dimBlock >>>(blurd, gMapd, fMapd, cols, rows, anchor_th, k);
	
	// 核函数同步
	HANDLE_ERROR(cudaDeviceSynchronize());

	// 数据拷回主内存
	HANDLE_ERROR(cudaMemcpy(gMaph, gMapd, sizeof(uchar)*rows*cols, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(fMaph, fMapd, sizeof(uchar)*rows*cols, cudaMemcpyDeviceToHost));

	// 锚点连接，边缘提取
	cv::Mat eMap = smartConnecting();

	edge_seg = edge_set;
	edge_seg_offset = edge_offset;
	edge_seg_len = edge_offset_len;

	return eMap;
}
