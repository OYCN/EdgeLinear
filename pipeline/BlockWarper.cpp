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
    if(level < 0)
    {
        std::cerr << "level set error" << std::endl;
        std::abort();
    }
    if(level == 0)
    {
        sync = true;
    }
    else
    {
        sync = false;
    }
    for(int i = 0; i < level; i++)
    {
        BlockPipeline* ptr = new BlockPipeline(configure.rows,
                                             configure.cols,
                                             configure.th1,
                                             configure.k,
                                             configure.GFSize,
                                             configure.GFs1,
                                             configure.GFs2,
                                             configure.th2,
                                             configure.returnH);
        app_list.push_back(ptr);
        _Context* context = new _Context();
        context->app = ptr;
        context->app_list = &app_list;
        context->context_list = &context_list;
        context->feeder = &feeder;
        context->worker_condition = &worker_condition;
        context->app_index = i;
        context->next_app_index = -1;
        context->tid = 0;

        context->pauseFlag = true;
        context->endFlag = false;
        context->first_feed = false;
        
        context_list.push_back(context);
    }

}

void BlockWarper::deinit()
{
    for(int i = 0; i < level; i++)
    {
        delete context_list[i]->app;
        delete context_list[i];
    }
    if(sync)
    {
        for(int i = 0; i < app_list.size(); i++)
        {
            delete app_list[i];
        }
    }
}

void BlockWarper::setFeeder(std::function<bool(cv::Mat&)> _feeder)
{
    feeder = _feeder;
}

void BlockWarper::start()
{
    if(sync)
    {
        BlockPipeline* ptr = new BlockPipeline(configure.rows,
                                             configure.cols,
                                             configure.th1,
                                             configure.k,
                                             configure.GFSize,
                                             configure.GFs1,
                                             configure.GFs2,
                                             configure.th2,
                                             configure.returnH);
        app_list.push_back(ptr);
        return;
    }

    {
        int ret = pthread_create(&context_list[0]->tid, NULL, BlockWarper::perThread, context_list[0]);
        if(ret != 0)
        {
            printf("create pthread error!\n");
            std::abort();
        }
        context_list[0]->pauseFlag = false;
        context_list[0]->pauseCondition.notify_all();
    }
    while(context_list[0]->first_feed == false);
    for(int i = 1; i < level; i++)
    {
        int ret = pthread_create(&context_list[i]->tid, NULL, BlockWarper::perThread, context_list[i]);
        if(ret != 0)
        {
            printf("create pthread error!\n");
            std::abort();
        }
        context_list[i]->pauseFlag = false;
        context_list[i]->pauseCondition.notify_all();
    }
}

void BlockWarper::join()
{
    for(int i =0 ; i < level; i++)
    {
        pthread_cancel(context_list[i]->tid);
    }

    for(int i =0 ; i < level; i++)
    {
        pthread_join(context_list[i]->tid, nullptr);
    }
}

bool BlockWarper::waitOne(POINT* edge_set, int* edge_offset, int& edge_offset_len, bool* flags)
{
    if(sync)
    {
        return syncRun(edge_set, edge_offset, edge_offset_len, flags);
    }
    else 
    {
        return asyncRun(edge_set, edge_offset, edge_offset_len, flags);
    }
}

bool BlockWarper::syncRun(POINT* edge_set, int* edge_offset, int& edge_offset_len, bool* flags)
{
    BlockPipeline* app = app_list[0];
    cv::Mat* input = app->getInput();
    bool* res = app->getResult();
    _EDoutput* edges = app->getEdges();
    cudaStream_t custream = app->getcuStream();

    if(!feeder(*input))
    {
        return false;
    }

    feedtime_sum -= cv::getTickCount();

    app->run();

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
    
    HANDLE_ERROR(cudaStreamSynchronize(custream));

    loop_time++;
    feedtime_sum += cv::getTickCount();

    return true;
}

bool BlockWarper::asyncRun(POINT* edge_set, int* edge_offset, int& edge_offset_len, bool* flags)
{
    static int now_idx = 0;

    _Context* context = context_list[now_idx];
    bool* res = context->app->getResult();
    _EDoutput* edges = context->app->getEdges();
    cudaStream_t custream = context->app->getcuStream();

    // 等待此帧处理完
    std::unique_lock<std::mutex> locker(worker_lock);
    while (!context->pauseFlag)
    {
        worker_condition.wait(locker);
    }

    if(context->endFlag == true)
    {
        return false;
    }

    now_idx = context->next_app_index;

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
    
    HANDLE_ERROR(cudaStreamSynchronize(custream));

    loop_time++;
    feedtime_sum += cv::getTickCount() - context->feed_time;

    // 通知线程继续
    context->pauseFlag = false;
    context->pauseCondition.notify_all();

    return true;
}

bool BlockWarper::runFeed(_Context* context, cv::Mat& v)
{
    static std::mutex feeder_lock;
    static int now_idx = 0;
    feeder_lock.lock();

    std::function<bool(cv::Mat&)> &feeder = *context->feeder;

    (*context->context_list)[now_idx]->next_app_index = context->app_index;
    now_idx = context->app_index;
    bool ret = feeder(v);

    context->first_feed = true;
    
    context->feed_time = cv::getTickCount();

    feeder_lock.unlock();

    return ret;
}

void *BlockWarper::perThread(void* data)
{
    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, nullptr);
    pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, nullptr);
    _Context* context = reinterpret_cast<_Context*>(data);
    cv::Mat* input = context->app->getInput();
    bool* result = context->app->getResult();
    
    while (true)
    {
        bool ret = runFeed(context, *input);

        std::unique_lock<std::mutex> locker(context->mutex);
        while (context->pauseFlag)
        {
            context->pauseCondition.wait(locker);
        }

        if(!ret) break;
    
        context->app->run();

        context->pauseFlag = true;
        context->worker_condition->notify_all();
    }
    context->pauseFlag = true;
    context->endFlag = true;
    context->worker_condition->notify_all();
    
    return nullptr;
}
