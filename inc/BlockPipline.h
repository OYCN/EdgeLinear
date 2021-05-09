#ifndef _INC_BLOCKPIPLINE_H
#define _INC_BLOCKPIPLINE_H

#include "BlockGetFlag.h"
#include "BlockConnect.h"
#include "BlockLinear.h"

class BlockPipline
{
public:
    BlockPipline(int _rows, int _cols, float _th1, int _k, int _GFSize, int _GFs1, int _GFs2, float _th2, bool _returnH)
        : rows(_rows), cols(_cols), returnH(_returnH),
          blockA(_rows, _cols, _th1, _k, _GFSize, _GFs1, _GFs2), 
          blockB(_rows, _cols),
          blockC(_rows, _cols, _th2, _returnH) {init();}
    ~BlockPipline() {deinit();}
    void run();
    cv::Mat* getInput() {return sMaph;}
    _EDoutput* getEdges() {return edges;}
    // uchar* getfMap() {return fMaph;}
    bool* getResult() {return result;}
    cudaStream_t getcuStream(){return custream;}

private:
    void init();
    void deinit();

private:
    // 行
    int rows;
    // 列
    int cols;

    bool returnH;

    cv::cuda::Stream cvstream;
    cudaStream_t custream;

    BlockGetFlag blockA;
    BlockConnect blockB;
    BlockLinear blockC;

    uchar* srch;
    cv::Mat* sMaph;

    uchar* fMaph;
    bool* result;

    _EDoutput* edges;
};

#endif // _INC_BLOCKPIPLINE_H