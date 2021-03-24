#include "DouglasPeucker.h"

#define sharedMemPerBlock 49152

__global__ void kernelDP(POINT *edge_set_d, int *edge_offset_d, int edge_offset_len, POINT *stack_d, bool *flags_d, float epsilon);

DouglasPeucker::DouglasPeucker(int _rows, int _cols, float _th)
    :rows(_rows), cols(_cols), th(_th)
{
    HANDLE_ERROR(cudaMalloc(&edge_set_d, sizeof(POINT)*rows*cols));
	HANDLE_ERROR(cudaMalloc(&edge_offset_d, sizeof(int)*(rows*cols+1)));
	HANDLE_ERROR(cudaMalloc(&flags_d, sizeof(bool)*rows*cols));
	HANDLE_ERROR(cudaMalloc(&stack_d, sizeof(POINT)*rows*cols));
    flags_h = new bool[rows*cols];
}

DouglasPeucker::~DouglasPeucker()
{
    cudaFree(edge_set_d);
	cudaFree(edge_offset_d);
	cudaFree(flags_d);
	cudaFree(stack_d);
	delete[] flags_h;
}

bool* DouglasPeucker::run(_EDoutput input)
{
    const dim3 dimBlock_DP(16,1);
    const dim3 dimGrid_DP(cols*rows / 16, 1);

    HANDLE_ERROR(cudaMemcpy(edge_set_d, input.edge_set, sizeof(POINT)*(input.edge_offset)[(input.edge_offset_len)-1], cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(edge_offset_d, input.edge_offset, sizeof(int)*(input.edge_offset_len), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemset(flags_d, false, sizeof(bool)*rows*cols));

    kernelDP<<<dimGrid_DP,dimBlock_DP>>>(edge_set_d, edge_offset_d, input.edge_offset_len, stack_d, flags_d, 5);
	// HANDLE_ERROR(cudaDeviceSynchronize());
	HANDLE_ERROR(cudaMemcpy(flags_h, flags_d, sizeof(bool)*rows*cols, cudaMemcpyDeviceToHost));

    return flags_h;
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
