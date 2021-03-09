#include "EDProcess_par.h"
#include "Timer.h"

/*
*Summary: 类所需内存的申请等初始化操作，一个实例仅需一次
*Parameters: 无
*Return: 无
*/
void Main::_InitPD()
{
	HANDLE_ERROR(cudaMalloc(&edge_set_d, sizeof(POINT)*rows*cols));
	HANDLE_ERROR(cudaMalloc(&edge_offset_d, sizeof(int)*(rows*cols+1)));
	HANDLE_ERROR(cudaMalloc(&flags_d, sizeof(bool)*rows*cols));
	HANDLE_ERROR(cudaMalloc(&stack_d, sizeof(POINT)*rows*cols));
	flags_h = new bool[rows*cols];
	// dimBlock_DP = dim3(1,1);
	// 仅支持一维
	dimBlock_DP = dim3(16,1);
	// dimBlock_DP = dim3(32,1);
	// dimGrid_DP = dim3(cols, rows);
	// dimGrid_DP = dim3(cols*rows / 32, 1);
	dimGrid_DP = dim3(cols*rows / 16, 1);
}

/*
*Summary: DP类所需内存的free，一个实例仅需一次
*Parameters: 无
*Return: 无
*/
void Main::_FreePD()
{
	cudaFree(edge_set_d);
	cudaFree(edge_offset_d);
	cudaFree(flags_d);
	cudaFree(stack_d);
	delete[] flags_h;
}

/*
*Summary: per deal
*Parameters: 无
*Return: 无
*/
void Main::PerProcDP()
{
	HANDLE_ERROR(cudaMemcpy(edge_set_d, edge_set, sizeof(POINT)*edge_offset[edge_offset_len-1], cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(edge_offset_d, edge_offset, sizeof(int)*edge_offset_len, cudaMemcpyHostToDevice));
	// HANDLE_ERROR(cudaMemset(flags_d, 0, sizeof(bool)*edge_offset[edge_offset_len-1]));
	HANDLE_ERROR(cudaMemset(flags_d, false, sizeof(bool)*rows*cols));
}

void testDP(int index, POINT *edge_set_d, int *edge_offset_d, int edge_offset_len, POINT *stack_d, bool *flags_d, float epsilon);
/*
*Summary: DP
*Parameters: 无
*Return: 无
*/
// int Main::runDP(VECTOR_H<VECTOR_H<POINT>> &line_all_gpu)
void Main::runDP(bool *&flag_in)
{
	PerProcDP();
	kernelDP<<<dimGrid_DP,dimBlock_DP>>>(edge_set_d, edge_offset_d, edge_offset_len, stack_d, flags_d, 5);
	HANDLE_ERROR(cudaDeviceSynchronize());
	HANDLE_ERROR(cudaMemcpy(flags_h, flags_d, sizeof(bool)*rows*cols, cudaMemcpyDeviceToHost));

	// POINT stack[rows*cols];
	// for(int i = 0; i < cols*rows; i++) {
	// 	testDP(i, edge_set, edge_offset, edge_offset_len, stack, flags_h, 5);
	// }

	flag_in = flags_h;
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

void testDP(int index, POINT *edge_set_d, int *edge_offset_d, int edge_offset_len, POINT *stack_d, bool *flags_d, float epsilon)
{
	if(index>=edge_offset_len) return;
	float dmax = 0;
	float d;
	float da, db, dc, norm;
	unsigned int C = 0;
	POINT wp;
	int start = edge_offset_d[index];
	int end = edge_offset_d[index+1];
	bool *flags = flags_d + start;
	POINT *edge = edge_set_d + start;
	POINT stack_s[sharedMemPerBlock / sizeof(POINT)];
	POINT* stack_s_start = stack_s + sharedMemPerBlock / sizeof(POINT) / 16 * (index % 16);
	POINT* stack_s_end = stack_s + sharedMemPerBlock / sizeof(POINT) / 16 * (index % 16 + 1);
	POINT *stack_base = stack_d + start;
	POINT *stack_top = stack_s_start;
	std::cout << "start_s\t" << stack_s_start << std::endl;
	std::cout << "end_s\t" << stack_s_end << std::endl;
	std::cout << "start_m\t" << stack_base << std::endl;
	std::cout << "push\t" << stack_top << std::endl;
	(*stack_top).x = 0;
	(*stack_top).y = end - start - 1;
	if(stack_top == (stack_s_end - 1)) stack_top = stack_base;
	else stack_top++;
	std::cout << "then\t" << stack_top << std::endl;
	while (stack_top != stack_s_start)
	{
		std::cout << "pop\t" << stack_top << std::endl;
		if(stack_top == stack_base) stack_top = stack_s_end - 1;
		else stack_top--;
		std::cout << "then\t" << stack_top << std::endl;
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
			std::cout << "push1\t" << stack_top << std::endl;
			(*stack_top).x = wp.x;
			(*stack_top).y = C;
			if(stack_top == (stack_s_end - 1)) stack_top = stack_base;
			else stack_top++;
			std::cout << "then\t" << stack_top << std::endl;
			std::cout << "push2\t" << stack_top << std::endl;
			(*stack_top).x = C;
			(*stack_top).y = wp.y;
			if(stack_top == (stack_s_end - 1)) stack_top = stack_base;
			else stack_top++;
			std::cout << "then\t" << stack_top << std::endl;
		}
		else
		{
			flags[wp.x] = true;
			flags[wp.y] = true;
		}
	}
}

