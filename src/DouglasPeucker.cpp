#include "DouglasPeucker.h"

#define sharedMemPerBlock 49152

void testDP(int index, POINT *edge_set_d, int *edge_offset_d, int edge_offset_len, POINT *stack_d, bool *flags_d, float epsilon);

DouglasPeucker::DouglasPeucker(int _rows, int _cols, float _th)
    :rows(_rows), cols(_cols), th(_th)
{
    flags_h = new bool[rows*cols];
    stack_h = new POINT[rows*cols];
}

DouglasPeucker::~DouglasPeucker()
{
	delete[] flags_h;
    delete[] stack_h;
}

void DouglasPeucker::initLoop(_EDoutput input)
{
	memset(flags_h, false, sizeof(bool)*rows*cols);
}

bool* DouglasPeucker::run(_EDoutput input)
{
    const dim3 dimBlock_DP(16,1);
    const dim3 dimGrid_DP(cols*rows / 16, 1);

    initLoop(input);

	for(int i = 0; i < (input.edge_offset_len - 1); i++) {
		testDP(i, input.edge_set, input.edge_offset, input.edge_offset_len, stack_h, flags_h, th);
	}

    return flags_h;
}

void testDP(int index, POINT *edge_set_d, int *edge_offset_d, int edge_offset_len, POINT *stack_d, bool *flags_d, float epsilon)
{
	float dmax = 0;
	float d;
	float da, db, dc, norm;
	unsigned int C = 0;
	POINT wp;
	int start = edge_offset_d[index];
	int end = edge_offset_d[index+1];
	bool *flags = flags_d + start;
	POINT *edge = edge_set_d + start;
	POINT *stack_top = stack_d;
	(*stack_top).x = 0;
	(*stack_top).y = end - start - 1;
	stack_top++;
	while (stack_top != stack_d)
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