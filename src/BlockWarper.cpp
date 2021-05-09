#include "BlockWarper.h"

void BlockWarper::init()
{
    std::cout << "rows: " << configure.rows << std::endl;
    std::cout << "cols: " << configure.cols << std::endl;
    std::cout << "th1: " << configure.th1 << std::endl;
    std::cout << "k: " << configure.k << std::endl;
    std::cout << "GFSize: " << configure.GFSize << std::endl;
    std::cout << "GFs1: " << configure.GFs1 << std::endl;
    std::cout << "GFs2: " << configure.GFs2 << std::endl;
    std::cout << "th2: " << configure.th2 << std::endl;
    if(level <= 0)
    {
        std::cerr << "level set error" << std::endl;
        std::abort();
    }
    for(int i = 0; i < level; i++)
    {
        BlockPipline* ptr = new BlockPipline(configure.rows,
                                             configure.cols,
                                             configure.th1,
                                             configure.k,
                                             configure.GFSize,
                                             configure.GFs1,
                                             configure.GFs2,
                                             configure.th2);
        _LoopListNode* node = new _LoopListNode({.main = ptr, .next = nullptr, .tid = 0, .index = i});
        _ThreadInput* thinput = new _ThreadInput();
        thInputs.push_back(thinput);
        thinput->node = node;
        thinput->feeder = &feeder;
        thinput->feeder_lock = &feeder_lock;
        thinput->pauseFlag = true;
        thinput->endFlag = false;
        thinput->worker_condition = &worker_condition;
        
        looplist.list.push_back(node);
    }
    looplist.geter = looplist.list[0];
    for(int i = 0; i < (level - 1); i++)
    {
        looplist.list[i]->next = looplist.list[i + 1];
    }
    looplist.list[level - 1]->next = looplist.list[0];

}

void BlockWarper::deinit()
{
    for(int i = 0; i < level; i++)
    {
        delete looplist.list[i]->main;
        delete thInputs[i];
    }
}

void BlockWarper::setFeeder(std::function<bool(cv::Mat&)> _feeder)
{
    feeder = _feeder;
    // std::cout << "feedre addr: " << &feeder << std::endl;
}

void BlockWarper::start()
{
    for(int i = 0; i < level; i++)
    {
        int ret = pthread_create(&thInputs[i]->node->tid, NULL, BlockWarper::perThread, thInputs[i]);
        if(ret != 0)
        {
            printf("create pthread error!\n");
            std::abort();
        }
        thInputs[i]->pauseFlag = false;
        thInputs[i]->condition.notify_all();
    }
}

void BlockWarper::join()
{
    for(int i =0 ; i < level; i++)
    {
        pthread_join(thInputs[i]->node->tid, nullptr);
    }
}

bool BlockWarper::waitOne(POINT* edge_set, int* edge_offset, int& edge_offset_len, bool* flags)
{
    // static int num = 0;
    _ThreadInput* thinput = thInputs[looplist.geter->index];
    looplist.geter = looplist.geter->next;
    BlockPipline* app = thinput->node->main;
    bool* res = app->getResult();
    _EDoutput* edges = app->getEdges();

    if(thinput->endFlag)
    {
        return false;
    }

    // 等待此帧处理完
    std::unique_lock<std::mutex> locker(worker_lock);
    while (!thinput->pauseFlag)
    {
        worker_condition.wait(locker);
    }

    memcpy(flags, res, sizeof(bool) * configure.rows * configure.cols);
    memcpy(edge_set, edges->edge_set, sizeof(POINT) * configure.rows * configure.cols);
    memcpy(edge_offset, edges->edge_offset, sizeof(int) * configure.rows * configure.cols);
    edge_offset_len = edges->edge_offset_len;
    // num++;
    // std::cout << num << std::endl;

    // 通知线程继续
    thinput->pauseFlag = false;
    thinput->condition.notify_all();
    return true;
}

bool runFeed(std::function<bool(cv::Mat&)> &feeder, cv::Mat& v)
{
    static std::mutex feeder_lock;
    feeder_lock.lock();
    bool ret = feeder(v);
    feeder_lock.unlock();

    return ret;
}

void *BlockWarper::perThread(void* data)
{
    _ThreadInput* args = reinterpret_cast<_ThreadInput*>(data);
    BlockPipline* app = args->node->main;
    std::function<bool(cv::Mat&)> &feeder = *args->feeder;
    cv::Mat* input = app->getInput();
    bool* result = app->getResult();

    // std::cout << "feedre addr: " << &feeder << std::endl;
    // cv::Mat out;
    // runFeed(feeder, out);
    // cv::imshow("test", out);
    // cv::waitKey();
    // exit(0);

    while (runFeed(feeder, *input))
    {
        std::unique_lock<std::mutex> locker(args->mutex);
        while (args->pauseFlag)
        {
            args->condition.wait(locker);
        }
        // std::cout << args->node->tid << std::endl;
    
        app->run();

        args->pauseFlag = true;
        args->worker_condition->notify_all();
    }
    args->pauseFlag = true;
    args->endFlag = true;
    args->worker_condition->notify_all();
    return nullptr;
}
