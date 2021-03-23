#include "EdgeDrawing.h"

EdgeDrawing::EdgeDrawing(int _rows=0, int _cols=0, int _th, int _k)
{
    th = _th;
    k = _h;
    rows = _rows;
    cols = _cols;

    dimBlock_ED = dim3(32,32);
	dimGrid_ED = dim3((cols+27)/28, (rows+27)/28);
	HANDLE_ERROR(cudaMalloc(&gMapd, sizeof(uchar)*rows*cols));
	HANDLE_ERROR(cudaMalloc(&srcd, sizeof(uchar)*rows*cols));
	HANDLE_ERROR(cudaMalloc(&fMapd, sizeof(uchar)*rows*cols));
	HANDLE_ERROR(cudaMemset(gMapd, 0, sizeof(uchar)*rows*cols));
	HANDLE_ERROR(cudaMemset(fMapd, 0, sizeof(uchar)*rows*cols));
	gMaph = new uchar[rows*cols];
	fMaph = new uchar[rows*cols];
	eMaph = new uchar[rows*cols];
	EDoutput.edge_set = new POINT[rows*cols];
	EDoutput.edge_offset = new int[rows*cols+1];
	edge_smart = new POINT[rows*cols];
    EDoutput.eMap = eMaph;
}

EdgeDrawing::~EdgeDrawing()
{
    cudaFree(gMapd);
	cudaFree(srcd);
	cudaFree(fMapd);
	delete[] gMaph;
	delete[] fMaph;
    delete[] eMaph;
	delete[] EDoutput.edge_set;
	delete[] EDoutput.edge_offset;
	delete[] edge_smart;
}

void EdgeDrawing::initLoop()
{
    memset(eMaph, 0, rows*cols*sizeof(uchar));
}

EdgeDrawing::run(cv::Mat& _src)
{
    initLoop();
    
    cv::cvtColor(_src, srch, CV_RGB2GRAY);
	cv::GaussianBlur(srch, srch, cv::Size(5, 5), 1, 0);

    HANDLE_ERROR(cudaMemcpy(srcd, srch.data, sizeof(uchar)*rows*cols, cudaMemcpyHostToDevice));
    kernelC<<< dimGrid_ED, dimBlock_ED >>>(blurd, gMapd, fMapd, cols, rows, anchor_th, k);
    // HANDLE_ERROR(cudaDeviceSynchronize());
    HANDLE_ERROR(cudaMemcpy(gMaph, gMapd, sizeof(uchar)*rows*cols, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(fMaph, fMapd, sizeof(uchar)*rows*cols, cudaMemcpyDeviceToHost));
    smartConnecting();
}