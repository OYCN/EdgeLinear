#ifndef _INC_BLOCKCONNECT_H
#define _INC_BLOCKCONNECT_H

#include "common.h"

class BlockConnect
{
public:
    BlockConnect(int _rows, int _cols)
        :rows(_rows), cols(_cols) {init();}
    ~BlockConnect(){deinit();}
    void execute(uchar* fMaph);
    _EDoutput* getOutput() {return &edges;}

private:
    void init();
    void deinit();

private:
    // 行
    int rows;
    // 列
    int cols;

    _EDoutput edges;
    uchar* eMaph_bk;
    POINT* edge_smart;

};

void smartConnecting(int rows, int cols, uchar* fMaph, uchar* eMaph, POINT* edge_smart, POINT* edge_set, int* edge_offset, int& edge_offset_len);
void goMove(int x, int y, uchar mydir, POINT *edge_s, int &idx, uchar* eMaph, uchar* fMaph, int rows, int cols);

#endif  // _INC_BLOCKCONNECT_H