#ifndef _INC_BLOCKWARPER_H
#define _INC_BLOCKWARPER_H

#include "BlockPipline.h"
#include <pthread.h> 
#include <mutex>
#include <atomic>
#include <condition_variable>

struct _Context
{
    BlockPipline* app;
    std::vector<BlockPipline*>* app_list;
    std::vector<_Context*>* context_list;
    std::function<bool(cv::Mat&)>* feeder;
    std::condition_variable* worker_condition;

    int app_index;
    int next_app_index;
    pthread_t tid;
    bool first_feed;

    std::mutex mutex;
    std::condition_variable pauseCondition;
    std::atomic_bool pauseFlag;
    std::atomic_bool endFlag;
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
    bool returnH;
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

    static bool runFeed(_Context* args, cv::Mat& v);
    static void *perThread(void* data);

private:
    int level;
    std::function<bool(cv::Mat&)> feeder;
    std::vector<_Context*> context_list;
    std::vector<BlockPipline*> app_list;
    _Configure configure;
    std::mutex worker_lock;
    std::condition_variable worker_condition;
};

#endif // _INC_BLOCKWARPER_H