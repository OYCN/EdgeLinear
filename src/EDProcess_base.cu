#include "EDProcess_par.h"
#include "Timer.h"

ED::~ED()
{
	#ifndef USE_UNIMEM
	cudaFree(gMapd);
	cudaFree(blurd);
	cudaFree(fMapd);
	free(gMaph);
	free(fMaph);
	#endif
	#ifdef USE_UNIMEM
	cudaFree(gMap);
	cudaFree(fMap);
	cudaFree(blurd);
	#endif
}

ED::ED()
{
	int count;
	HANDLE_ERROR(cudaGetDeviceCount(&count));
	if(count!=1) {printf( "zero or multiple gpu\n"); exit( EXIT_FAILURE );}
	// cudaDeviceProp prop;
	// HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
	cudaSetDevice(0);
	cudaFree(0);
}

void ED::initinal(cv::Mat& blurImg, const int anchor_th, const int k)
{
	K = k;
	ANCHOR_TH = anchor_th;
	rows = blurImg.rows;
	cols = blurImg.cols;
	#ifndef USE_UNIMEM
	HANDLE_ERROR(cudaMalloc(&gMapd, sizeof(uchar)*rows*cols));
	HANDLE_ERROR(cudaMalloc(&blurd, sizeof(uchar)*rows*cols));
	HANDLE_ERROR(cudaMalloc(&fMapd, sizeof(uchar)*rows*cols));
	HANDLE_ERROR(cudaMemcpy(blurd, blurImg.data, sizeof(uchar)*rows*cols, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemset(gMapd, 0, sizeof(uchar)*rows*cols));
	HANDLE_ERROR(cudaMemset(fMapd, 0, sizeof(uchar)*rows*cols));
	gMaph = (uchar *)malloc(sizeof(uchar)*rows*cols);
	fMaph = (uchar *)malloc(sizeof(uchar)*rows*cols);
	#endif
	#ifdef USE_UNIMEM
	HANDLE_ERROR(cudaMallocManaged(&gMap, sizeof(uchar)*rows*cols));
	HANDLE_ERROR(cudaMallocManaged(&fMap, sizeof(uchar)*rows*cols));
	HANDLE_ERROR(cudaMallocManaged(&blurd, sizeof(uchar)*rows*cols));
	HANDLE_ERROR(cudaMemcpy(blurd, blurImg.data, sizeof(uchar)*rows*cols, cudaMemcpyDefault));
	HANDLE_ERROR(cudaMemset(gMap, 0, sizeof(uchar)*rows*cols));
	HANDLE_ERROR(cudaMemset(fMap, 0, sizeof(uchar)*rows*cols));
	#endif
}

cv::Mat ED::Process(cv::Mat& blurImg, VECTOR_H<VECTOR_H<cv::Point>>& edge_seg, const int anchor_th,  const int k)
{
	cv::Mat debug = cv::Mat::zeros(blurImg.size(), blurImg.type());
	#ifdef TIM_PROC
	TDEF(init);
	TDEF(kernel);
	TDEF(sync);
	TDEF(smart);
	TDEF(all);
	TSTART(all);
	TSTART(init);
	#endif
	initinal(blurImg, anchor_th, k);
	dim3 dimBlock(32,32);
	dim3 dimGrid((cols+27)/28, (rows+27)/28);
	dim3 dimGridOld(rows, cols);
	#ifdef TIM_PROC
	TEND(init);TPRINTMS(init, "init: ");
	TSTART(kernel);
	#endif
	#ifndef USE_UNIMEM
	kernelC<<< dimGrid, dimBlock >>>(blurd, gMapd, fMapd, cols, rows, ANCHOR_TH, K);
	#endif
	#ifdef USE_UNIMEM
	kernelC<<< dimGrid, dimBlock >>>(blurd, gMap, fMap, cols, rows, ANCHOR_TH, K);
	#endif
	HANDLE_ERROR(cudaDeviceSynchronize());
	#ifdef TIM_PROC
	TEND(kernel);TPRINTMS(kernel, "kernel: ");
	TSTART(sync);
	#endif
	#ifndef USE_UNIMEM
	HANDLE_ERROR(cudaMemcpy(gMaph, gMapd, sizeof(uchar)*rows*cols, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(fMaph, fMapd, sizeof(uchar)*rows*cols, cudaMemcpyDeviceToHost));
	#endif
	#ifdef TIM_PROC
	TEND(sync);TPRINTMS(sync, "sync: ");
	TSTART(smart);
	#endif
	#ifndef USE_UNIMEM
	cv::Mat eMap = smartConnecting(gMaph, fMaph, rows, cols, edge_seg);
	#endif
	#ifdef USE_UNIMEM
	cv::Mat eMap = smartConnecting(gMap, fMap, rows, cols, edge_seg);
	#endif
	#ifdef TIM_PROC
	TEND(smart);TPRINTMS(smart, "smart: ");
	TEND(all);TPRINTMS(all, "all: ");
	#endif
	return eMap;
}
