#ifndef _INC_BLOCKWARPER_H
#define _INC_BLOCKWARPER_H

#include "BlockPipline.h"
#include <pthread.h> 
#include <mutex>
#include <atomic>
#include <condition_variable>

struct _LoopListNode
{
    BlockPipline* main;
    _LoopListNode* next;
    pthread_t tid;
    int index;
};

struct _LoopList
{
    std::vector<_LoopListNode*> list;
    _LoopListNode* geter;
};

struct _ThreadInput
{
    _LoopListNode* node;
    std::function<bool(cv::Mat&)>* feeder;
    std::mutex mutex;
    std::mutex* feeder_lock;
    std::condition_variable condition;
    std::atomic_bool pauseFlag;
    std::atomic_bool endFlag;
    std::condition_variable* worker_condition;
};

struct _Configure
{
    int rows;
    int cols;
    float th1;
    int k;
    int GFSize;
    int GFs1;
    int GFs2;
    float th2;
};

class BlockWarper
{
public:
    BlockWarper(int _level, _Configure _configure) : level(_level), configure(_configure) {init();}
    ~BlockWarper(){deinit();}
    void setFeeder(std::function<bool(cv::Mat&)> _feeder);
    void start();
    bool waitOne(POINT* edge_set, int* edge_offset, int& edge_offset_len, bool* flags);
    void join();

private:
    void init();
    void deinit();

    static void *perThread(void* data);

private:
    int level;
    std::function<bool(cv::Mat&)> feeder;
    _LoopList looplist;
    _Configure configure;
    std::vector<_ThreadInput*> thInputs;
    std::mutex worker_lock;
    std::mutex feeder_lock;
    std::condition_variable worker_condition;
};

#endif // _INC_BLOCKWARPER_H