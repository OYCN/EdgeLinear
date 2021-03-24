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
};

#endif // _INC_DOUGLASpEUCKER_H