#include "DouglasPeucker.h"

#define sharedMemPerBlock 49152

void testDP(int index, POINT *edge_set_d, int *edge_offset_d, int edge_offset_len, POINT *stack_d, bool *flags_d, float epsilon);

DouglasPeucker::DouglasPeucker(int _rows, int _cols, float _th)
    :rows(_rows), cols(_cols), th(_th)
{
    flagh = new bool[rows*cols];
    stack_h = new POINT[rows*cols];
}

DouglasPeucker::~DouglasPeucker()
{
	delete[] flagh;
    delete[] stack_h;
}

bool* DouglasPeucker::run(EDoutput input)
{
    const dim3 dimBlock_DP(16,1);
    const dim3 dimGrid_DP(cols*rows / 16, 1);

    memset(flagh, false, sizeof(bool)*rows*cols);

	for(int i = 0; i < edge_offset_len; i++) {
		testDP(i, edge_set, edge_offset, edge_offset_len, stack_h, flags_h, 5);
	}

    return flagh;
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