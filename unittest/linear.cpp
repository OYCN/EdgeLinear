#include "../common/common.h"
#include "../pipeline/BlockGetFlag.h"
#include "../pipeline/BlockConnect.h"
#include "../pipeline/BlockLinear.h"

int main()
{
    cv::Mat src = cv::Mat::zeros(320, 320, CV_8UC3);
    int r = src.cols > src.rows ? src.rows / 3 : src.cols / 3;
    cv::circle(src, {src.cols/2, src.rows/2}, r, {255, 255, 255}, 1);
    cv::imshow("src.jpg", src);
    auto blockA = BlockGetFlag(src.rows, src.cols, 6, 2, 5, 1, 0);
    auto blockB = BlockConnect(src.rows, src.cols);
    auto blockC = BlockLinear(src.rows, src.cols, 5, true);

    cv::cuda::Stream cvstream;
    cudaStream_t custream = cv::cuda::StreamAccessor::getStream(cvstream);

    uchar* fMaph = blockA.getOutput();
    _EDoutput* edges = blockB.getOutput();
    bool* flags = blockC.getOutput();
    
    blockA.enqueue(src, cvstream);
    HANDLE_ERROR(cudaStreamSynchronize(custream));
    blockB.execute(fMaph);
    TDEF(time);
    blockC.enqueue(*edges, cvstream);
    HANDLE_ERROR(cudaStreamSynchronize(custream));
    TEND(time);
    TPRINTMS(time, "linear time(ms): ");

    cv::Mat outMap = cv::Mat::zeros(src.rows, src.cols, CV_8UC3);
    for(int i = 0; i < (edges->edge_offset_len - 1); i++)
    {
        int old_idx = -1;
        for(int j = edges->edge_offset[i]; j < edges->edge_offset[i+1]; j++)
        {
            if(flags[j])
            {
                std::cout << "idx: " << j << std::endl;
                if(old_idx >= 0)
                {
                    cv::line(outMap, edges->edge_set[old_idx], edges->edge_set[j], cv::Scalar(255, 255, 255), 1, 4);
                    std::cout << "old_idx: " << old_idx << std::endl;
                }
                old_idx = j;
                
            }
        }
    }
    cv::Mat eMap = cv::Mat(src.rows, src.cols, CV_8UC1, edges->eMap);
    cv::imshow("edge", eMap * 255);
    cv::imshow("outMap", outMap);
    cv::waitKey();
    return 0;
}