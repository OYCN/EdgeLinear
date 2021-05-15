#include "LinearSum.h"

__global__ void kernel(POINT *edge_set_d, int *edge_offset_d, int edge_offset_len, bool *flags_d, float epsilon);

LinearSum::LinearSum(int _rows, int _cols, float _th)
    :rows(_rows), cols(_cols), th(_th)
{
    HANDLE_ERROR(cudaMalloc(&edge_set_d, sizeof(POINT)*rows*cols));
	HANDLE_ERROR(cudaMalloc(&edge_offset_d, sizeof(int)*(rows*cols+1)));
	HANDLE_ERROR(cudaMalloc(&flags_d, sizeof(bool)*rows*cols));
    HANDLE_ERROR(cudaMallocHost(&flags_h, sizeof(bool)*rows*cols));
    // flags_h = new bool[rows*cols];
}

LinearSum::~LinearSum()
{
    cudaFree(edge_set_d);
	cudaFree(edge_offset_d);
	cudaFree(flags_d);
    HANDLE_ERROR(cudaFreeHost(flags_h));
	// delete[] flags_h;
}

void LinearSum::initLoop(_EDoutput input)
{
    HANDLE_ERROR(cudaMemcpy(edge_set_d, input.edge_set, sizeof(POINT)*(input.edge_offset)[(input.edge_offset_len)-1], cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(edge_offset_d, input.edge_offset, sizeof(int)*(input.edge_offset_len), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemset(flags_d, false, sizeof(bool)*rows*cols));
}

bool* LinearSum::run(_EDoutput input)
{
    const dim3 dimBlock(32,1);
    const dim3 dimGrid(cols*rows / 32, 1);

    initLoop(input);

    kernel<<<dimGrid,dimBlock>>>(edge_set_d, edge_offset_d, input.edge_offset_len, flags_d, th);
	// HANDLE_ERROR(cudaDeviceSynchronize());
	HANDLE_ERROR(cudaMemcpy(flags_h, flags_d, sizeof(bool)*rows*cols, cudaMemcpyDeviceToHost));

    return flags_h;
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
