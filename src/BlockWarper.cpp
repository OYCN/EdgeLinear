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
                                             configure.th2,
                                             configure.returnH);
        _LoopListNode* node = new _LoopListNode({.main = ptr, .next = nullptr, .tid = 0, .index = i});
        _ThreadInput* thinput = new _ThreadInput();
        thInputs.push_back(thinput);
        thinput->node = node;
        thinput->feeder = &feeder;
        thinput->feeder_condition = &feeder_condition;
        thinput->this_read = false;
        thinput->pauseFlag = true;
        thinput->endFlag = false;
        thinput->worker_condition = &worker_condition;
        
        looplist.list.push_back(node);
    }
    looplist.geter = looplist.list[0];
    for(int i = 0; i < (level - 1); i++)
    {
        looplist.list[i]->next = looplist.list[i + 1];
        thInputs[i]->next_read = &thInputs[i+1]->this_read;
    }
    looplist.list[level - 1]->next = looplist.list[0];
    thInputs[level - 1]->next_read = &thInputs[0]->this_read;

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
    thInputs[0]->this_read = true;
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
        pthread_cancel(thInputs[i]->node->tid);
    }

    for(int i =0 ; i < level; i++)
    {
        pthread_join(thInputs[i]->node->tid, nullptr);
    }
}

bool BlockWarper::waitOne(POINT* edge_set, int* edge_offset, int& edge_offset_len, bool* flags)
{
    // static double fps_sum = 0;
    // static int fps_num = 0;
    // double fps = (double)cv::getTickCount();

    _ThreadInput* thinput = thInputs[looplist.geter->index];
    looplist.geter = looplist.geter->next;
    BlockPipline* app = thinput->node->main;
    bool* res = app->getResult();
    _EDoutput* edges = app->getEdges();
    cudaStream_t custream = app->getcuStream();

    // double fps = (double)cv::getTickCount();
    if(!thinput->endFlag)
    {
        // 等待此帧处理完
        std::unique_lock<std::mutex> locker(worker_lock);
        while (!thinput->pauseFlag)
        {
            worker_condition.wait(locker);
        }
    }

    // fps = cv::getTickFrequency()/((double)cv::getTickCount() - fps);
    // double fps = (double)cv::getTickCount();

    if(configure.returnH)
    {
        memcpy(flags, res, sizeof(bool) * configure.rows * configure.cols);
    }
    else
    {
        HANDLE_ERROR(cudaMemcpyAsync(flags, res, sizeof(bool)*edges->edge_offset[edges->edge_offset_len-1], cudaMemcpyDeviceToHost, custream));
    }
    edge_offset_len = edges->edge_offset_len;
    memcpy(edge_offset, edges->edge_offset, sizeof(int) * edge_offset_len);
    memcpy(edge_set, edges->edge_set, sizeof(POINT) * edge_offset[edge_offset_len-1]);
    
    
    // if(!configure.returnH)
    {
        HANDLE_ERROR(cudaStreamSynchronize(custream));
    }
    // num++;
    // std::cout << num << std::endl;

    // 通知线程继续
    thinput->pauseFlag = false;
    thinput->condition.notify_all();

    // fps = cv::getTickFrequency()/((double)cv::getTickCount() - fps);
    // fps_sum += fps;
    // fps_num++;
    // if(fps_num == 480)
    // std::cout << "th0 ms avg: " << fps_num * 1000 / fps_sum << " when " << fps_num << std::endl;
    // std::cout << "num: " << fps_num << std::endl;

    return !thinput->endFlag;
}

bool BlockWarper::runFeed(_ThreadInput* args, cv::Mat& v)
{
    // static std::mutex feeder_lock;
    // feeder_lock.lock();

    std::function<bool(cv::Mat&)> &feeder = *args->feeder;

    std::unique_lock<std::mutex> locker(args->mutex);
    // std::unique_lock<std::mutex> locker(feeder_lock);

    while (!args->this_read)
    {
        args->feeder_condition->wait(locker);
    }

    bool ret = feeder(v);

    args->this_read = false;
    *args->next_read = true;
    args->feeder_condition->notify_all();

    // feeder_lock.unlock();

    return ret;
}

void *BlockWarper::perThread(void* data)
{
    // double fps_sum = 0;
    // int fps_num = 0;
    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, nullptr);
    pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, nullptr);
    _ThreadInput* args = reinterpret_cast<_ThreadInput*>(data);
    BlockPipline* app = args->node->main;
    cv::Mat* input = app->getInput();
    bool* result = app->getResult();
    
    while (runFeed(args, *input))
    {
        // double fps = (double)cv::getTickCount();
        std::unique_lock<std::mutex> locker(args->mutex);
        while (args->pauseFlag)
        {
            args->condition.wait(locker);
        }
        // fps = cv::getTickFrequency()/((double)cv::getTickCount() - fps);
        // double fps = (double)cv::getTickCount();
        // std::cout << args->node->tid << std::endl;
    
        app->run();

        args->pauseFlag = true;
        args->worker_condition->notify_all();
        // fps = cv::getTickFrequency()/((double)cv::getTickCount() - fps);
        // fps_sum += fps;
        // fps_num++;
    }
    args->pauseFlag = true;
    args->endFlag = true;
    args->worker_condition->notify_all();
    // std::cout << "th ms avg: " << fps_num * 1000 / fps_sum << std::endl;
    // std::cout << "th ms num: " << fps_num << std::endl;
    return nullptr;
}
