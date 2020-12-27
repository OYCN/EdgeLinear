#include "EDProcess_par.h"

/*
*Summary: 类所需内存的申请等初始化操作，一个实例仅需一次
*Parameters: 无
*Return: 无
*/
void Main::_InitPD()
{

}

/*
*Summary: 类所需内存的申请等初始化操作，一个实例仅需一次
*Parameters: 无
*Return: 无
*/
void Main::_FreePD()
{

}

#ifdef USE_CPUDP
void DouglasPeucker(const VECTOR_H<POINT> &edge, VECTOR_H<POINT> &line, float epsilon)
{
	
	float dmax = 0;
	float d;
	float da, db, dc, norm;
	unsigned int C = 0;
	POINT wp;
	VECTOR_H<POINT> stack;
	int edge_len = edge.size();
	bool *flags = new bool[edge_len];

	memset(flags, 0, sizeof(bool)*edge_len);

	stack.push_back(POINT( 0,edge_len - 1 ));
	while (!stack.empty())
	{
		wp = stack[stack.size() - 1];
		stack.pop_back();
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
		// cout << wp.x << " to " << wp.y << " da= " << da << " db= " << db << " dc= " << dc << " norm= " << norm << " maxdis: " << dmax << endl;
		if (dmax >= epsilon)
		{
			stack.push_back(POINT( wp.x, C ));
			stack.push_back(POINT( C, wp.y ));
		}
		else
		{
			flags[wp.x] = true;
			flags[wp.y] = true;
		}
	}
	
	for (unsigned int i = 0; i < edge_len; i++)
	{
		if (flags[i])
		{
			line.push_back(edge[i]);
		}
	}

    delete[] flags;
}
#endif

#ifdef USE_GPUDP
__global__ void DouglasPeucker(POINT **edge_seg_d, int *edge_offset_d, int edge_seg_len, POINT *stack, bool *flags_d, float epsilon)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index>=edge_seg_len) return;
	float dmax = 0;
	float d;
	float da, db, dc, norm;
	unsigned int C = 0;
	POINT wp;
	int &offset = edge_offset_d[index];
	int edge_len = edge_offset_d[index+1] - edge_offset_d[index];
	int stack_top = offset;
	bool *flags = flags_d + offset;
	POINT * &edge = edge_seg_d[index];

	// memset(flags, 0, sizeof(bool)*edge_len);

	// stack[stack_top-1] = { 0,edge_len - 1 };
	stack[stack_top].x = 0;
	stack[stack_top].y = edge_len - 1;
	stack_top++;
	while (stack_top != offset)
	{
		wp = stack[stack_top-1];
		stack_top--;
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
		// cout << wp.x << " to " << wp.y << " da= " << da << " db= " << db << " dc= " << dc << " norm= " << norm << " maxdis: " << dmax << endl;
		if (dmax >= epsilon)
		{
			// stack[stack_top-1] = { wp.x, C };
			stack[stack_top].x = wp.x;
			stack[stack_top].y = C;
			stack_top++;
			// stack[stack_top-1] = { C, wp.y };
			stack[stack_top].x = C;
			stack[stack_top].y = wp.y;
			stack_top++;
		}
		else
		{
			flags[wp.x] = true;
			flags[wp.y] = true;
		}
	}
}
#endif