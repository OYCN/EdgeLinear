#include "BlockLinear.h"

__global__ void kernel(POINT *edge_set_d, int *edge_offset_d, int edge_offset_len, bool *flags_d, float epsilon);

void BlockLinear::init()
{
    HANDLE_ERROR(cudaMalloc(&edge_set_d, sizeof(POINT)*rows*cols));
	HANDLE_ERROR(cudaMalloc(&edge_offset_d, sizeof(int)*(rows*cols+1)));
	HANDLE_ERROR(cudaMalloc(&flags_d, sizeof(bool)*rows*cols));
    if(returnH)
        HANDLE_ERROR(cudaMallocHost(&flags_h, sizeof(bool)*rows*cols));
}

void BlockLinear::deinit()
{
    HANDLE_ERROR(cudaFree(edge_set_d));
	HANDLE_ERROR(cudaFree(edge_offset_d));
	HANDLE_ERROR(cudaFree(flags_d));
    if(returnH)
        HANDLE_ERROR(cudaFreeHost(flags_h));
}

void BlockLinear::enqueue(_EDoutput fMaph, cv::cuda::Stream& cvstream)
{
    const dim3 dimBlock(32,1);
    const dim3 dimGrid((cols*rows+31) / 32, 1);

    cudaStream_t custream = cv::cuda::StreamAccessor::getStream(cvstream);

    HANDLE_ERROR(cudaMemcpyAsync(edge_set_d, fMaph.edge_set, sizeof(POINT)*(fMaph.edge_offset)[(fMaph.edge_offset_len)-1], cudaMemcpyHostToDevice, custream));
	HANDLE_ERROR(cudaMemcpyAsync(edge_offset_d, fMaph.edge_offset, sizeof(int)*(fMaph.edge_offset_len), cudaMemcpyHostToDevice, custream));
	HANDLE_ERROR(cudaMemsetAsync(flags_d, false, sizeof(bool)*rows*cols, custream));

    kernel<<<dimGrid, dimBlock, 0, custream>>>(edge_set_d, edge_offset_d, fMaph.edge_offset_len, flags_d, th);
	// HANDLE_ERROR(cudaDeviceSynchronize());
    if(returnH)
	    HANDLE_ERROR(cudaMemcpyAsync(flags_h, flags_d, sizeof(bool)*(fMaph.edge_offset)[(fMaph.edge_offset_len)-1], cudaMemcpyDeviceToHost, custream));


}

__global__ void kernel(POINT *edge_set_d, int *edge_offset_d, int edge_offset_len, bool *flags_d, float epsilon)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index>=(edge_offset_len-1)) return;

    float now_len = 0;
    float now_dis = 0;
    // A 为上一直线化的点，或起始点
    // B 为当前遍历的点
    // T 为上一个点
    POINT A, B, T;

    A = edge_set_d[edge_offset_d[index]];
    // 起始点置位
    flags_d[edge_offset_d[index]] = true;
    for(int j = (edge_offset_d[index] + 1); j < edge_offset_d[index + 1]; j++)
    {
        B = edge_set_d[j];
        T = edge_set_d[j-1];
        float dx = T.x - B.x;
        float dy = T.y - B.y;
        now_len += sqrt(dx * dx + dy * dy);
        dx = A.x - B.x;
        dy = A.y - B.y;
        now_dis = sqrt(dx * dx + dy * dy);
        // 若本次超过阈值，上次的为最佳点
        if(fabs(now_len - now_dis) > epsilon)
        {
            flags_d[j - 1] = true;
            // std::cout << j - 1 << ":(" << T.x << "," << T.y << ")" <<std::endl;
            // 上次点为起始点
            A = T;
            now_len = 0;
            // 需要重新计算本点
            j--;
        }
    }
    // 结束点为最佳点
    flags_d[edge_offset_d[index + 1] - 1] = true;
}
