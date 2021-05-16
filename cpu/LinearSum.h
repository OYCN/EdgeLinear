#ifndef _INC_LINEARSUM_H
#define _INC_LINEARSUM_H

#include "../common/common.h"

class LinearSum
{
public:
    LinearSum(int _rows, int _cols, float _th);
    ~LinearSum();
    bool* run(_EDoutput input);

private:
    void initLoop(_EDoutput input);

public:
    float th;

private:
    int rows;
    int cols;
    bool *flags_d;
    bool *flags_h;
    POINT *edge_set_d;
    int *edge_offset_d;
};

#endif // _INC_LINEARSUM_H