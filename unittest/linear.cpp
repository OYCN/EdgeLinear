#include <fstream>
#include <random>
#include <cmath>
#include <string>
#include "../common/common.h"
#include "../pipeline/BlockLinear.h"

bool* linear(_EDoutput input, bool* flags_h, float th);

int main(int argc, char* argv[])
{
    std::ifstream fin;
    fin.open("base.dat", std::ios::binary);
    fin.seekg(0, fin.end);
    size_t size = fin.tellg();
    fin.seekg(0, fin.beg);
    std::vector<POINT> base;
    base.resize(size / sizeof(POINT));
    fin.read(reinterpret_cast<char*>(base.data()), size);
    std::cout << "base line is " << base.size() << std::endl;

    _EDoutput fakeEdge;

    int mod;
    // bin 边缘数 长度基本值 均值 方差
    if(argc != 4)
    {
        std::cout << argv[0] << " 边缘数 均值 方差" << std::endl;
        exit(0);
    }
    size_t a = std::atoi(argv[1]);
    size_t b = std::atoi(argv[2]);
    size_t c = std::atoi(argv[3]);

    std::default_random_engine engine; //引擎
    std::normal_distribution<double> norm(b, c); //均值, 方差

    HANDLE_ERROR(cudaMallocHost(&fakeEdge.edge_offset, (a + 1) * sizeof(int)));
    
    fakeEdge.edge_offset_len = a + 1;
    int* point_num = fakeEdge.edge_offset;
    std::vector<POINT> segs_v;
    *point_num = 0;
    point_num++;
    int max = 0;
    int min = a+1;
    for(int i = 0; i < a; i++)
    {
        unsigned len = std::lround(norm(engine)); //取整-最近的整数
        if(len < 2) len = 2;
        if(max < len) max = len;
        if(min > len) min = len;
        *point_num = *(point_num - 1) + len;
        point_num++;
        for(size_t _ = 0, idx = 0; _ < len; _++, idx++)
        {
            if(idx == base.size()) idx = 0;
            segs_v.push_back(base[idx]);
        }
    }

    std::cout << "max: " << max << ", min: " << min << std::endl;

    HANDLE_ERROR(cudaMallocHost(&fakeEdge.edge_set, segs_v.size() * sizeof(POINT)));
    for(int i = 0; i < segs_v.size(); i++)
    {
        fakeEdge.edge_set[i] = segs_v[i];
    }

    std::vector<POINT>().swap(segs_v);


    auto blockC = BlockLinear(fakeEdge.edge_offset[(fakeEdge.edge_offset_len)-1], 1, 5, false);
    cv::cuda::Stream cvstream;
    cudaStream_t custream = cv::cuda::StreamAccessor::getStream(cvstream);
    bool* flags_d = blockC.getOutput();

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

    // std::cout << fakeEdge.edge_offset_len <<std::endl;
    for(int i = 0; i < (fakeEdge.edge_offset_len - 1); i++)
    {
        cv::Mat outMap = cv::Mat::zeros(320, 320, CV_8UC3);
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
        // cv::imshow(std::string("outMap") + std::to_string(i), outMap);
        cv::imshow(std::string("outMap"), outMap);
        // if(cv::waitKey(1) == ' ') break;
    }
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
