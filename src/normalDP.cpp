#include "EDProcess_par.h"
#include <omp.h>

void cpuDP(VECTOR_H<VECTOR_H<POINT>> &edge_seg_vec, VECTOR_H<VECTOR_H<POINT>> &line_all_cpu)
{
	for(VECTOR_H<VECTOR_H<POINT>>::const_iterator e=edge_seg_vec.begin(); e != edge_seg_vec.end(); e++)
	{
		VECTOR_H<POINT> line;
		// cv::approxPolyDP(*e, line, 5, false);
		// mygpu::approxPolyDP(*e, line, 5, false);
		DouglasPeucker(*e, line, 5);
        line_all_cpu.push_back(line);
	}
}

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