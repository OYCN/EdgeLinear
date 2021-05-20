#include "../pipeline/BlockGetFlag.h"
#include "../pipeline/BlockConnect.h"
#include <numeric>
#include <algorithm>
#include <fstream>

int main(int argc, char* argv[])
{

    cv::Mat img;
    cv::VideoCapture cap;
    
	if(argc == 1)
		cap.open("/home/opluss/Documents/EdgeLinear/img/dataset.mp4");
	else
		cap.open(argv[1]);

    int rows = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    int cols = cap.get(CV_CAP_PROP_FRAME_WIDTH);

    auto runner1 = BlockGetFlag(rows, cols, 6, 2, 5, 1, 0);
    auto runner2 = BlockConnect(rows, cols);
    cv::cuda::Stream cvstream;
    cudaStream_t custream = cv::cuda::StreamAccessor::getStream(cvstream);
    uchar* res1 = runner1.getOutput();
    _EDoutput* res2 = runner2.getOutput();

    std::vector<double> means;
    std::vector<double> stdevs;
    std::vector<int> lens;
    while(cap.read(img))
    {

        runner1.enqueue(img, cvstream);
        HANDLE_ERROR(cudaStreamSynchronize(custream));
        runner2.execute(res1);

        std::vector<int> edges_len;
        for(int j = 0; j < (res2->edge_offset_len - 1); j++)
        {
            edges_len.push_back(res2->edge_offset[j + 1] - res2->edge_offset[j]);
        }
        int sum = std::accumulate(edges_len.begin(), edges_len.end(), 0);
        double mean =  sum / edges_len.size();
        double accum  = 0.0;
        std::for_each (edges_len.begin(), edges_len.end(), [&](const double d) {
            accum  += (d-mean)*(d-mean);
        });
        double stdev = sqrt(accum/(edges_len.size()-1)); //方差

        means.push_back(mean);
        stdevs.push_back(stdev);
        lens.push_back(res2->edge_offset_len - 1);
    }

    std::ofstream fout;
    fout.open("edges_info.csv");
    for(int j = 0; j < means.size(); j++)
    {
        std::string str1 = std::to_string(lens[j]);
        fout.write(str1.c_str(), str1.size());
        fout.write(",", 1);
        std::string str2 = std::to_string(means[j]);
        fout.write(str2.c_str(), str2.size());
        fout.write(",", 1);
        std::string str3 = std::to_string(stdevs[j]);
        fout.write(str3.c_str(), str3.size());
        fout.write("\n", 1);
    }
    fout.close();

    return 0;
}