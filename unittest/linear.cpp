#include "../common/common.h"
#include "../pipeline/BlockGetFlag.h"
#include "../pipeline/BlockConnect.h"
#include "../pipeline/BlockLinear.h"

bool* linear(_EDoutput input, bool* flags_h, float th);

int main(int argc, char* argv[])
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

    size_t number_seg = 1;
    if(argc == 2) 
    {
        number_seg = std::atoi(argv[1]);
    }

    std::vector<POINT> one_edge;
    std::vector<int> offset;
    offset.push_back(0);
    size_t one_edge_len = edges->edge_offset[1] - edges->edge_offset[0];
    std::cout << "edge len per seg " << one_edge_len << std::endl;
    one_edge.resize(one_edge_len * number_seg);
    for(int i = 0; i < number_seg; i++)
    {
        memcpy(one_edge.data() + i * one_edge_len, edges->edge_set, one_edge_len * sizeof(POINT));
        offset.push_back(offset[i] + one_edge_len);
    }

    auto blockC = BlockLinear(one_edge_len * number_seg, 1, 5, false);
    bool* flags_d = blockC.getOutput();

    _EDoutput fakeEdge = {
        .edge_set = one_edge.data(),
        .edge_offset = offset.data(),
        .edge_offset_len = number_seg + 1
    };

    bool* flags_h;
    HANDLE_ERROR(cudaMallocHost(&flags_h, sizeof(bool)*(fakeEdge.edge_offset)[(fakeEdge.edge_offset_len)-1]));

    TDEF(all);
    TDEF(compute);
    TDEF(mem);
    TSTART(all);
    TSTART(compute);
    blockC.enqueue(fakeEdge, cvstream);
    HANDLE_ERROR(cudaStreamSynchronize(custream));
    TEND(compute);
    TSTART(mem);
    HANDLE_ERROR(cudaMemcpyAsync(flags_h, flags_d, sizeof(bool)*(fakeEdge.edge_offset)[(fakeEdge.edge_offset_len)-1], cudaMemcpyDeviceToHost, custream));
    HANDLE_ERROR(cudaStreamSynchronize(custream));
    TEND(mem);
    TEND(all);
    TPRINTUS(all, "linear all(ms): ");
    TPRINTUS(compute, "linear compute(ms): ");
    TPRINTUS(mem, "linear mem(ms): ");

    TDEF(cpu);
    TSTART(cpu);
    linear(fakeEdge, flags_h, 5);
    TEND(cpu);
    TPRINTUS(cpu, "linear cpu(ms): ");

    cv::Mat outMap = cv::Mat::zeros(src.rows, src.cols, CV_8UC3);
    // std::cout << fakeEdge.edge_offset_len <<std::endl;
    for(int i = 0; i < (fakeEdge.edge_offset_len - 1); i++)
    {
        int old_idx = -1;
        for(int j = fakeEdge.edge_offset[i]; j < fakeEdge.edge_offset[i+1]; j++)
        {
            if(flags_h[j])
            {   
                if(old_idx >= 0)
                {
                    cv::line(outMap, fakeEdge.edge_set[old_idx], fakeEdge.edge_set[j], cv::Scalar(255, 255, 255), 1, 4);
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

bool* linear(_EDoutput input, bool* flags_h, float th)
{
    float now_len = 0;
    float now_dis = 0;
    // A 为上一直线化的点，或起始点
    // B 为当前遍历的点
    // T 为上一个点
    POINT A, B, T;

    for(int i = 0; i < (input.edge_offset_len - 1); i++)
    {
		now_len = 0;
        A = input.edge_set[input.edge_offset[i]];
        // std::cout << "new line:" << std::endl;
        // std::cout << input.edge_offset[i] << ":(" << A.x << "," << A.y << ")" <<std::endl;
        // 起始点置位
        flags_h[input.edge_offset[i]] = true;
        for(int j = (input.edge_offset[i] + 1); j < input.edge_offset[i + 1]; j++)
        {
            flags_h[j] = false;
            B = input.edge_set[j];
            T = input.edge_set[j-1];
            float dx = T.x - B.x;
            float dy = T.y - B.y;
            now_len += sqrt(dx * dx + dy * dy);
            dx = A.x - B.x;
            dy = A.y - B.y;
            now_dis = sqrt(dx * dx + dy * dy);
            // 若本次超过阈值，上次的为最佳点
            if(fabs(now_len - now_dis) > th)
            {
                flags_h[j - 1] = true;
                // std::cout << j - 1 << ":(" << T.x << "," << T.y << ")" <<std::endl;
                // 上次点为起始点
                A = T;
                now_len = 0;
                // 需要重新计算本点
                j--;
            }
            // else
            // {
            //     flags_h[j - 1] = false;
            // }
        }
        // 结束点为最佳点
        flags_h[input.edge_offset[i + 1] - 1] = true;
        // std::cout << input.edge_offset[i + 1] - 1 << ":(" << B.x << "," << B.y << ")" <<std::endl;
	}
    return flags_h;
}
