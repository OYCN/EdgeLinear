#ifndef _INC_MYPOINTSTRUCT_H
#define _INC_MYPOINTSTRUCT_H

#include "common.h"

namespace mygpu
{
	struct Point
	{
		int x;
		int y;
		__host__ __device__
		Point()
		{
			x = 0;
			y = 0;
		}
		__host__ __device__
		Point(int ix, int iy)
		{
			x = ix;
			y = iy;
		}
		__host__ __device__
		void operator = (const Point &obj)
		{
			x = obj.x;
			y = obj.y;
		}
		__host__ __device__
		bool operator != (const Point &obj)
		{
			if(obj.x==x && obj.y==y) return false;
			else return true; 
		}
		__host__ __device__
		operator cv::Point()
		{
			cv::Point pt(x, y);
			return pt;
		}
		__host__ __device__
		operator cv::Point() const
		{
			cv::Point pt(x, y);
			return pt;
		}
	};
}

struct _EDoutput
{
	POINT *edge_set;	// 全部边缘片段点集合
	int *edge_offset;	// 每个边缘片段的偏移，最后为边缘片段点集总长
	int edge_offset_len;	// 偏移数组长度
	uchar* eMap;
};


#endif // _INC_MYPOINTSTRUCT_H