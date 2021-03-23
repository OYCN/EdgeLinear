#ifndef _INC_LINEAR_H
#define _INC_LINEAR_H

#include "common.h"

class Linear
{
public:
    Linear(int _rows, int _cols, float _th)
        :rows(_rows), cols(_cols), th(_th)
    {
        flagh = new bool[rows*cols];
    }
    virtual ~Linear() = 0;
    virtual bool* run(EDoutput input);

protected:
    virtual void initLoop();
    virtual void Kernel() { };

public:
    float th;

protected:
    int rows;
    int cols;
    // GPU Block 划分
    dim3 dimBlock;
    // GPU Grid 划分
    dim3 dimGrid;
    // 关键点标志
    uchar* flagh;

}

#endif // _INC_LINEAR_H