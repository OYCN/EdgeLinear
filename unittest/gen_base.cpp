#include "../common/common.h"
#include "../pipeline/BlockGetFlag.h"
#include "../pipeline/BlockConnect.h"

#include <fstream>

int main()
{
    cv::Mat src = cv::Mat::zeros(320, 320, CV_8UC3);
    int r = src.cols > src.rows ? src.rows / 3 : src.cols / 3;
    cv::circle(src, {src.cols/2, src.rows/2}, r, {255, 255, 255}, 1);
    cv::imshow("src.jpg", src);
    auto blockA = BlockGetFlag(src.rows, src.cols, 6, 2, 5, 1, 0);
    auto blockB = BlockConnect(src.rows, src.cols);

    cv::cuda::Stream cvstream;
    cudaStream_t custream = cv::cuda::StreamAccessor::getStream(cvstream);

    uchar* fMaph = blockA.getOutput();
    _EDoutput* edges = blockB.getOutput();
    
    blockA.enqueue(src, cvstream);
    HANDLE_ERROR(cudaStreamSynchronize(custream));
    blockB.execute(fMaph);

    size_t one_edge_len = edges->edge_offset[1] - edges->edge_offset[0];
    std::cout << "base len is " << one_edge_len << std::endl;

    std::ofstream fout;
    fout.open("base.dat", std::ios::binary);
    fout.write(reinterpret_cast<char*>(edges->edge_set), one_edge_len * sizeof(POINT));
    fout.close();

    return 0;
}