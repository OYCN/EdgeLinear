#ifndef _INC_LINEARDIS_H
#define _INC_LINEARDIS_H

#include "common.h"

class LinearDis
{
public:
    LinearDis(int _rows, int _cols, float _th);
    ~LinearDis();
    bool* run(_EDoutput input);

private:
    void initLoop(_EDoutput input);

public:
    float th;

private:
    int rows;
    int cols;
    bool *flags_h;
};

#endif // _INC_LINEARDIS_H