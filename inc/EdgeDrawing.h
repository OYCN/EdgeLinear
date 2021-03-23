#ifndef _INC_EDGEDRAWING_H
#define _INC_EDGEDRAWING_H

#include "common.h"

class EdgeDrawing 
{
public:
    EdgeDrawing(int _rows=0, int _cols=0, int _anchor_th=6, int _k=2);
    ~EdgeDrawing();
    _EDoutput* run(cv::Mat &_src);

private:
    void initLoop();
    void Kernel();
    void smartConnecting();
    void goMove(int x, int y, uchar mydir, POINT *edge_s, int &idx);

public:
    // 稀疏度
    int k;
    // 阈值
    int th;

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
    // 输入图
    uchar* srcd;
    // 边缘效果图
    uchar* eMaph;
    // 用于预处理的临时变量
    cv::Mat srch;
    // GPU Block 划分
    dim3 dimBlock;
    // GPU Grid 划分
    dim3 dimGrid;
    // 锚点链接时临时变量
    POINT *edge_smart;
    // 边缘结果
    struct _EDoutput EDoutput;

};

__global__ void kernelC(uchar *blur, uchar * gMap, uchar *fMap, int cols, int rows, int ANCHOR_TH, int K);

#endif // _INC_EDGEDRAWING_H