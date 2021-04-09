#ifndef _INC_EDGEDRAWING_H
#define _INC_EDGEDRAWING_H

#include "common.h"

#ifdef USE_OPENCV_GPU
#include <opencv2/cudafilters.hpp>
#endif

class EdgeDrawing 
{
public:
    EdgeDrawing(int _rows, int _cols, float _anchor_th, int _k);
    ~EdgeDrawing();
    _EDoutput* run(cv::Mat &_src);

private:
    void initLoop();
    void smartConnecting();
    void goMove(int x, int y, uchar mydir, POINT *edge_s, int &idx);

public:
    // 稀疏度
    int k;
    // 阈值
    float th;

private:
    // 行
    int rows;
    // 列
    int cols;
    // 梯度图
    uchar* gMapd;
    uchar* gMaph;
    // 标记图
    uchar* fMapd;
    uchar* fMaph;
    // 模糊图
    uchar* blurd;
    // 边缘效果图
    uchar* eMaph;
    uchar* eMaph_bk;    // 每次拷贝入eMaph
    // 用于预处理的临时变量
    cv::Mat srch;
    // 锚点链接时临时变量
    POINT *edge_smart;
    // 边缘结果
    struct _EDoutput EDoutput;
    #ifdef USE_OPENCV_GPU
    // 三通道原图
    uchar* srcd;
    // 灰度图
    uchar* grayd;
    cv::cuda::GpuMat *gmat_src;
	cv::cuda::GpuMat *gmat_gray;
	cv::cuda::GpuMat *gmat_blur;
    cv::Ptr<cv::cuda::Filter> gauss;
    #endif

};

#endif // _INC_EDGEDRAWING_H