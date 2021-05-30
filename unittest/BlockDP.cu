#include "BlockDP.h"

#define sharedMemPerBlock 49152

__global__ void kernelDP(POINT *edge_set_d, int *edge_offset_d, int edge_offset_len, POINT *stack_d, bool *flags_d, float epsilon);

void BlockDP::init()
{
    HANDLE_ERROR(cudaMalloc(&edge_set_d, sizeof(POINT)*rows*cols));
	HANDLE_ERROR(cudaMalloc(&edge_offset_d, sizeof(int)*(rows*cols+1)));
	HANDLE_ERROR(cudaMalloc(&flags_d, sizeof(bool)*rows*cols));
    HANDLE_ERROR(cudaMalloc(&stack_d, sizeof(POINT)*rows*cols));
    if(returnH)
        HANDLE_ERROR(cudaMallocHost(&flags_h, sizeof(bool)*rows*cols));
}

void BlockDP::deinit()
{
    HANDLE_ERROR(cudaFree(edge_set_d));
	HANDLE_ERROR(cudaFree(edge_offset_d));
	HANDLE_ERROR(cudaFree(flags_d));
    HANDLE_ERROR(cudaFree(stack_d));
    if(returnH)
        HANDLE_ERROR(cudaFreeHost(flags_h));
}

void BlockDP::enqueue(_EDoutput fMaph, cv::cuda::Stream& cvstream)
{
    const dim3 dimBlock(32,1);
    const dim3 dimGrid((cols*rows+31) / 32, 1);

    cudaStream_t custream = cv::cuda::StreamAccessor::getStream(cvstream);

    HANDLE_ERROR(cudaMemcpyAsync(edge_set_d, fMaph.edge_set, sizeof(POINT)*(fMaph.edge_offset)[(fMaph.edge_offset_len)-1], cudaMemcpyHostToDevice, custream));
	HANDLE_ERROR(cudaMemcpyAsync(edge_offset_d, fMaph.edge_offset, sizeof(int)*(fMaph.edge_offset_len), cudaMemcpyHostToDevice, custream));
	HANDLE_ERROR(cudaMemsetAsync(flags_d, false, sizeof(bool)*rows*cols, custream));

    kernelDP<<<dimGrid, dimBlock, 0, custream>>>(edge_set_d, edge_offset_d, fMaph.edge_offset_len, stack_d, flags_d, th);
	// HANDLE_ERROR(cudaDeviceSynchronize());
    if(returnH)
	    HANDLE_ERROR(cudaMemcpyAsync(flags_h, flags_d, sizeof(bool)*(fMaph.edge_offset)[(fMaph.edge_offset_len)-1], cudaMemcpyDeviceToHost, custream));


}

__global__ void kernelDP(POINT *edge_set_d, int *edge_offset_d, int edge_offset_len, POINT *stack_d, bool *flags_d, float epsilon)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index>=(edge_offset_len-1)) return;
	float dmax = 0;
	float d;
	float da, db, dc, norm;
	unsigned int C = 0;
	POINT wp;
	int start = edge_offset_d[index];
	int end = edge_offset_d[index+1];
	bool *flags = flags_d + start;
	POINT *edge = edge_set_d + start;
	__shared__ POINT stack_s[sharedMemPerBlock / sizeof(POINT)];
	POINT* stack_s_start = stack_s + sharedMemPerBlock / sizeof(POINT) / blockDim.x * threadIdx.x;
	POINT* stack_s_end = stack_s + sharedMemPerBlock / sizeof(POINT) / blockDim.x * (threadIdx.x + 1);
	POINT *stack_base = stack_d + start;
	POINT *stack_top = stack_s_start;

	(*stack_top).x = 0;
	(*stack_top).y = end - start - 1;
	if(stack_top == (stack_s_end - 1)) stack_top = stack_base;
	else stack_top++;
	while (stack_top != stack_s_start)
	{
		if(stack_top == stack_base) stack_top = stack_s_end - 1;
		else stack_top--;
		wp = *(stack_top);
		dmax = 0;
		da = edge[wp.y].y - edge[wp.x].y;
		db = edge[wp.x].x - edge[wp.y].x;
		dc = edge[wp.y].x * edge[wp.x].y - edge[wp.x].x * edge[wp.y].y;
		norm = sqrt(da * da + db * db);
		for (unsigned int i = wp.x; i < wp.y; i++)
		{
			d = fabs((da * edge[i].x + db * edge[i].y + dc) / norm);
			if (d > dmax)
			{
				C = i;
				dmax = d;
			}
		}
		if (dmax >= epsilon)
		{
			(*stack_top).x = wp.x;
			(*stack_top).y = C;
			if(stack_top == (stack_s_end - 1)) stack_top = stack_base;
			else stack_top++;
			(*stack_top).x = C;
			(*stack_top).y = wp.y;
			if(stack_top == (stack_s_end - 1)) stack_top = stack_base;
			else stack_top++;
		}
		else
		{
			flags[wp.x] = true;
			flags[wp.y] = true;
		}
	}
}
