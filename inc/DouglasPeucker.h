#ifndef _INC_DOUGLASpEUCKER_H
#define _INC_DOUGLASpEUCKER_H

#include "Linear.h"

class DouglasPeucker
{
public:
    DouglasPeucker(int _rows, int _cols, float _th);
    ~DouglasPeucker();

private:
    void initLoop();
    void Kernel() { };

public:
    float th;

private:
    int rows;
    int cols;
    // GPU Block 划分
    dim3 dimBlock;
    // GPU Grid 划分
    dim3 dimGrid;
    // 关键点标志
    uchar* flagh;
}

#endif // _INC_DOUGLASpEUCKER_H