#include "LinearSum.h"

LinearSum::LinearSum(int _rows, int _cols, float _th)
    :rows(_rows), cols(_cols), th(_th)
{
    flags_h = new bool[rows*cols];
}

LinearSum::~LinearSum()
{
    delete[] flags_h;
}

void LinearSum::initLoop(_EDoutput input)
{
    memset(flags_h, false, sizeof(bool)*rows*cols);
}

bool* LinearSum::run(_EDoutput input)
{
    float now_len = 0;
    float now_dis = 0;
    // A 为上一直线化的点，或起始点
    // B 为当前遍历的点
    // T 为上一个点
    POINT A, B, T;

    initLoop(input);

    for(int i = 0; i < (input.edge_offset_len - 1); i++)
    {
		now_len = 0;
        A = input.edge_set[input.edge_offset[i]];
        // std::cout << "new line:" << std::endl;
        // std::cout << input.edge_offset[i] << ":(" << A.x << "," << A.y << ")" <<std::endl;
        // 起始点置位
        flags_h[input.edge_offset[i]] = true;
        for(int j = (input.edge_offset[i] + 1); j < input.edge_offset[i + 1]; j++)
        {
            // flags_h[j - 1] = true;
            B = input.edge_set[j];
            T = input.edge_set[j-1];
            float dx = T.x - B.x;
            float dy = T.y - B.y;
            now_len += sqrt(dx * dx + dy * dy);
            dx = A.x - B.x;
            dy = A.y - B.y;
            now_dis = sqrt(dx * dx + dy * dy);
            // 若本次超过阈值，上次的为最佳点
            if(fabs(now_len - now_dis) > th)
            {
                flags_h[j - 1] = true;
                // std::cout << j - 1 << ":(" << T.x << "," << T.y << ")" <<std::endl;
                // 上次点为起始点
                A = T;
                now_len = 0;
                // 需要重新计算本点
                j--;
            }
            // else
            // {
            //     flags_h[j - 1] = false;
            // }
        }
        // 结束点为最佳点
        flags_h[input.edge_offset[i + 1] - 1] = true;
        // std::cout << input.edge_offset[i + 1] - 1 << ":(" << B.x << "," << B.y << ")" <<std::endl;
	}
    return flags_h;
}
