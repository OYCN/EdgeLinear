#include "LinearDis.h"

__global__ void kernel(POINT *edge_set_d, int *edge_offset_d, int edge_offset_len, bool *flags_d, float epsilon);

LinearDis::LinearDis(int _rows, int _cols, float _th)
    :rows(_rows), cols(_cols), th(_th)
{
    HANDLE_ERROR(cudaMalloc(&edge_set_d, sizeof(POINT)*rows*cols));
	HANDLE_ERROR(cudaMalloc(&edge_offset_d, sizeof(int)*(rows*cols+1)));
	HANDLE_ERROR(cudaMalloc(&flags_d, sizeof(bool)*rows*cols));
    flags_h = new bool[rows*cols];
}

LinearDis::~LinearDis()
{
    cudaFree(edge_set_d);
	cudaFree(edge_offset_d);
	cudaFree(flags_d);
    delete[] flags_h;
}

void LinearDis::initLoop(_EDoutput input)
{
    HANDLE_ERROR(cudaMemcpy(edge_set_d, input.edge_set, sizeof(POINT)*(input.edge_offset)[(input.edge_offset_len)-1], cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(edge_offset_d, input.edge_offset, sizeof(int)*(input.edge_offset_len), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemset(flags_d, false, sizeof(bool)*rows*cols));
}

bool* LinearDis::run(_EDoutput input)
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

    float max_dis = 0;
    float now_dis = 0;
    // A 为上一直线化的点，或起始点
    // B 为当前遍历的点
    // M 临时变量点
    POINT A, B, M;

    A = edge_set_d[edge_offset_d[index]];
    // 起始点置位
    flags_d[edge_offset_d[index]] = true;
    int start_idx = edge_offset_d[index] + 1;
    for(int j = (edge_offset_d[index] + 1); j < edge_offset_d[index + 1]; j++)
    {
        max_dis = 0;
        B = edge_set_d[j];
        float da = B.y - A.y;
        float db = A.x - B.x;
        float dc = B.x * A.y - A.x * B.y;
        float normal = sqrt(da * da + db * db);
        for(int idx = start_idx; idx < j; idx++)
        {
            M = edge_set_d[idx];
            now_dis = fabs((da * M.x + db * M.y + dc) / normal);
            if(now_dis > max_dis)
            {
                max_dis = now_dis;
            }
        }
        // 若本次超过阈值，上次的为最佳点
        if(max_dis > epsilon)
        {
            flags_d[j - 1] = true;
            // 上次点为起始点
            A = edge_set_d[j - 1];
            start_idx = j;
        }
    }
    // 结束点为最佳点
    flags_d[edge_offset_d[index + 1] - 1] = true;
}
