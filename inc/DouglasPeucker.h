#ifndef _INC_DOUGLASpEUCKER_H
#define _INC_DOUGLASpEUCKER_H

#include "common.h"

class DouglasPeucker
{
public:
    DouglasPeucker(int _rows, int _cols, float _th);
    ~DouglasPeucker();
    bool* run(_EDoutput input);

private:
    void initLoop();
    void Kernel() { };

public:
    float th;

private:
    int rows;
    int cols;
    POINT *edge_set_d;
    int *edge_offset_d;
    bool *flags_h;
    bool *flags_d;
    POINT *stack_d;
    POINT *stack_h;
    // GPU Block 划分
    dim3 dimBlock;
    // GPU Grid 划分
    dim3 dimGrid;
};

#endif // _INC_DOUGLASpEUCKER_H