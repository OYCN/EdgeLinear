#include "LinearDis.h"

LinearDis::LinearDis(int _rows, int _cols, float _th)
    :rows(_rows), cols(_cols), th(_th)
{
    flags_h = new bool[rows*cols];
}

LinearDis::~LinearDis()
{
    delete[] flags_h;
}

void LinearDis::initLoop()
{
    memset(flags_h, false, sizeof(bool)*rows*cols);
}

bool* LinearDis::run(_EDoutput input)
{
    float max_dis = 0;
    float now_dis = 0;
    // A 为上一直线化的点，或起始点
    // B 为当前遍历的点
    // M 目前最远的点
    POINT A, B, M;

    initLoop();

    for(int i = 0; i < (input.edge_offset_len - 1); i++)
    {
		max_dis = 0;
        A = input.edge_set[input.edge_offset[i]];
        // 起始点置位
        flags_h[input.edge_offset[i]] = true;
        for(int j = (input.edge_offset[i] + 1); j < input.edge_offset[i + 1]; j++)
        {
            B = input.edge_set[j];
            float da = B.y - A.y;
            float db = A.x - B.x;
            float dc = B.x * A.y - A.x * B.y;
            now_dis = fabs((da * M.x + db * M.y + dc) / sqrt(da * da + db * db));
            if(now_dis > max_dis)
            {
                max_dis = now_dis;
                M = B;
            }
            // 若本次超过阈值，上次的为最佳点
            if(max_dis > th)
            {
                flags_h[j - 1] = true;
                // 上次点为起始点
                A = input.edge_set[j - 1];
                max_dis = 0;
            }
        }
        // 结束点为最佳点
        flags_h[input.edge_offset[i + 1] - 1] = true;
	}
    return flags_h;
}
