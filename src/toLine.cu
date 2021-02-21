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
	POINT *stack_base = stack_d + start;
	POINT *stack_top = stack_base;
	bool *flags = flags_d + start;
	POINT *edge = edge_set_d + start;

	(*stack_top).x = 0;
	(*stack_top).y = end - start - 1;
	stack_top++;
	while (stack_top != stack_base)
	{
		stack_top--;
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
			stack_top++;
			(*stack_top).x = C;
			(*stack_top).y = wp.y;
			stack_top++;
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
	if(index>=(edge_offset_len-1)) return;
	std::cout << "index=" << index << ", of_len=" << edge_offset_len << ", point_num = " << edge_offset_d[index+1]-edge_offset_d[index] << std::endl;
	float dmax = 0;
	float d;
	float da, db, dc, norm;
	unsigned int C = 0;
	POINT wp;
	int start = edge_offset_d[index];
	int end = edge_offset_d[index+1];
	POINT *stack_base = stack_d + start;
	POINT *stack_top = stack_base;
	bool *flags = flags_d + start;
	POINT *edge = edge_set_d + start;

	(*stack_top).x = 0;
	(*stack_top).y = end - start - 1;
	stack_top++;
	std::cout << "[GPU]stack push AB(" << 0 << ", " << end - start - 1 << ") TOP@" << (long long) stack_top << "/" << (long long)stack_base << std::endl;
	while (stack_top != stack_base)
	{
		stack_top--;
		wp = *(stack_top);
		std::cout << "[GPU]stack pop (" << wp.x << ", " << wp.y << ") TOP@" << (long long) stack_top << "/" << (long long)stack_base << std::endl;
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
			stack_top++;
			std::cout << "[GPU]stack push AC(" << wp.x << ", " << C << ") TOP@" << (long long) stack_top << "/" << (long long)stack_base << std::endl;
			(*stack_top).x = C;
			(*stack_top).y = wp.y;
			stack_top++;
			std::cout << "[GPU]stack push CB(" << C << ", " << wp.y << ") TOP@" << (long long) stack_top << "/" << (long long)stack_base << std::endl;
		}
		else
		{
			flags[wp.x] = true;
			flags[wp.y] = true;
		}
	}
}

