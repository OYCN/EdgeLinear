#include "EDProcess_par.h"
#include <stack>

#define CUR_LENGTH 9
#define THR 0.1

float cur(cv::Point edge[CUR_LENGTH])
{
	double xm = 0, ym = 0, s1 = 0, s2 = 0, s3 = 0;//xm—宽度平均 ym—高度平均 s1—宽度方差 s2—高度方差 s3—协方差
	double e2=0;//特征值

	//算局部线条方向(与水平线夹角)-----------------------------------（还需进一步优化：前4点后4点的边缘没有处理）
	for (size_t A = 0; A < CUR_LENGTH; A++)
	{
		xm = (xm + edge[A].x);
		ym = (ym + edge[A].y);
	}
    xm = xm / CUR_LENGTH;//求横坐标均值
    ym = ym / CUR_LENGTH;//求纵坐标均值

	for (size_t A = 0; A < CUR_LENGTH; A++)
	{
		s1 = s1 + (edge[A].x - xm)*(edge[A].x - xm);
		s2 = s2 + (edge[A].y - ym)*(edge[A].y - ym);
		s3 = s3 + (edge[A].x - xm)*(edge[A].y - ym);
	}
    s1 = s1 / CUR_LENGTH;//宽度方差
    s2 = s2 / CUR_LENGTH;//高度方差
    s3 = s3 / CUR_LENGTH;//协方差
    
    e2 = (s1 + s2 - sqrt((s1 - s2)*(s1 - s2) + 4 * s3*s3)) / 2;
    return e2;
}

void getLine(const VECTOR_H<cv::Point> &edge, VECTOR_H<VECTOR_H<cv::Point>> &line)
{
    VECTOR_H<cv::Point> oneline(2);
    cv::Point dd[CUR_LENGTH];

    oneline[0] = edge[0];
    oneline[1].x = -1;

    if (edge.size() >= 9)
    {
        for (size_t A = 4; A < edge.size() - 4; A++)
        {
            for(int di=(-CUR_LENGTH/2); di<(CUR_LENGTH-CUR_LENGTH/2); di++)
            {
                dd[di+CUR_LENGTH/2] = edge[A-di];
            }
            if (cur(dd) < THR)//最小特征值小于0.25阈值的为直线
            {
                oneline[1] = dd[CUR_LENGTH-1];
            }
            else
            {
                if (oneline[1].x != -1)
                {
                    line.push_back(oneline);
                }
                oneline[0] = dd[CUR_LENGTH-1];
                oneline[1].x = -1;
            }
        }
        oneline[1] = edge[edge.size() - 1];
    }
    if (edge.size() < 9 && edge.size() > 4)
    {
        oneline[1] = edge[edge.size() - 1];;
    }
    if (oneline[1].x != -1)
    {
        line.push_back(oneline);
    }
}

VECTOR_H<VECTOR_H<cv::Point>> orgline(const VECTOR_H<VECTOR_H<cv::Point>> &edge_seg)
{
	cv::Point dd[9];
	//存断开的边缘
	VECTOR_H<VECTOR_H<cv::Point>> edge_seg2;
	VECTOR_H<cv::Point> edge_s2;
	int CC = 75;
	double ang = 0;
	double k = 0;//两个特征值
	int  ii = 0, jj = 0;
	for (int B = 0; B < edge_seg.size(); B++)
	{
		if (edge_seg[B].size() < 9 && edge_seg[B].size() > 4)
		{
			edge_seg2.push_back(edge_seg[B]);
		}
		if (edge_seg[B].size() >= 9)
		{
			edge_s2.push_back(edge_seg[B][0]);
			edge_s2.push_back(edge_seg[B][1]);
			edge_s2.push_back(edge_seg[B][2]);
			edge_s2.push_back(edge_seg[B][3]);
			for (size_t A = 4; A < edge_seg[B].size() - 4; A++)
			{
				dd[0] = edge_seg[B][A - 4];
				dd[1] = edge_seg[B][A - 3];
				dd[2] = edge_seg[B][A - 2];
				dd[3] = edge_seg[B][A - 1];
				dd[4] = edge_seg[B][A];
				dd[5] = edge_seg[B][A + 1];
				dd[6] = edge_seg[B][A + 2];
				dd[7] = edge_seg[B][A + 3];
				dd[8] = edge_seg[B][A + 4];

				if (cur(dd) < THR)//最小特征值小于0.25阈值的为直线
				{
					edge_s2.push_back(dd[4]);

					//ang = tr1.ang;
				}
				else
				{
					//if (edge_s2.size() != 0)
					if (edge_s2.size() > 4)
					{
						edge_seg2.push_back(edge_s2);
						VECTOR_H<cv::Point>().swap(edge_s2);
					}
					VECTOR_H<cv::Point>().swap(edge_s2);

				}
			}
			edge_s2.push_back(edge_seg[B][edge_seg[B].size() - 4]);
			edge_s2.push_back(edge_seg[B][edge_seg[B].size() - 3]);
			edge_s2.push_back(edge_seg[B][edge_seg[B].size() - 2]);
			edge_s2.push_back(edge_seg[B][edge_seg[B].size() - 1]);
		}
		//if (edge_s2.size() != 0)
		if (edge_s2.size() > 4)
		{
			edge_seg2.push_back(edge_s2);
			VECTOR_H<cv::Point>().swap(edge_s2);
		}
		VECTOR_H<cv::Point>().swap(edge_s2);
		//coding_edge.at<uchar>(edge_seg[B]) = ang + CC;
		//printf("x=%d,y=%d,k=%.9lf\n", edge_seg[B][A].x, edge_seg[B][A].y,k);

	}
	return edge_seg2;
}

static int approxPolyDP_( const cv::Point* src_contour, int count0, cv::Point* dst_contour,
              bool is_closed0, double eps, cv::AutoBuffer<cv::Range>& _stack )
{
    #define PUSH_SLICE(slice) \
        if( top >= stacksz ) \
        { \
            _stack.resize(stacksz*3/2); \
            stack = _stack.data(); \
            stacksz = _stack.size(); \
        } \
        stack[top++] = slice

    #define READ_PT(pt, pos) \
        pt = src_contour[pos]; \
        if( ++pos >= count ) pos = 0

    #define READ_DST_PT(pt, pos) \
        pt = dst_contour[pos]; \
        if( ++pos >= count ) pos = 0

    #define WRITE_PT(pt) \
        dst_contour[new_count++] = pt

    typedef cv::Point PT;
    int             init_iters = 3;
    cv::Range           slice(0, 0), right_slice(0, 0);
    PT              start_pt((int)-1000000, (int)-1000000), end_pt(0, 0), pt(0,0);
    int             A = 0, B, pos = 0, wpos, count = count0, new_count=0;
    int             is_closed = is_closed0;
    bool            le_eps = false;
    size_t top = 0, stacksz = _stack.size();
    cv::Range*          stack = _stack.data();

    if( count == 0  )
        return 0;

    eps *= eps;

    if( !is_closed )
    {
        right_slice.start = count;
        end_pt = src_contour[0];
        start_pt = src_contour[count-1];

        if( start_pt.x != end_pt.x || start_pt.y != end_pt.y )
        {
            slice.start = 0;
            slice.end = count - 1;
            PUSH_SLICE(slice);
        }
        else
        {
            is_closed = 1;
            init_iters = 1;
        }
    }

    if( is_closed )
    {
        // 1. Find approximately two farthest points of the contour
        right_slice.start = 0;

        for( A = 0; A < init_iters; A++ )
        {
            double dist, max_dist = 0;
            pos = (pos + right_slice.start) % count;
            READ_PT(start_pt, pos);

            for( B = 1; B < count; B++ )
            {
                double dx, dy;

                READ_PT(pt, pos);
                dx = pt.x - start_pt.x;
                dy = pt.y - start_pt.y;

                dist = dx * dx + dy * dy;

                if( dist > max_dist )
                {
                    max_dist = dist;
                    right_slice.start = B;
                }
            }

            le_eps = max_dist <= eps;
        }

        // 2. initialize the stack
        if( !le_eps )
        {
            right_slice.end = slice.start = pos % count;
            slice.end = right_slice.start = (right_slice.start + slice.start) % count;

            PUSH_SLICE(right_slice);
            PUSH_SLICE(slice);
        }
        else
            WRITE_PT(start_pt);
    }

    // 3. run recursive process
    while( top > 0 )
    {
        slice = stack[--top];
        end_pt = src_contour[slice.end];
        pos = slice.start;
        READ_PT(start_pt, pos);

        if( pos != slice.end )
        {
            double dx, dy, dist, max_dist = 0;

            dx = end_pt.x - start_pt.x;
            dy = end_pt.y - start_pt.y;

            assert( dx != 0 || dy != 0 );

            while( pos != slice.end )
            {
                READ_PT(pt, pos);
                dist = fabs((pt.y - start_pt.y) * dx - (pt.x - start_pt.x) * dy);

                if( dist > max_dist )
                {
                    max_dist = dist;
                    right_slice.start = (pos+count-1)%count;
                }
            }

            le_eps = max_dist * max_dist <= eps * (dx * dx + dy * dy);
        }
        else
        {
            le_eps = true;
            // read starting point
            start_pt = src_contour[slice.start];
        }

        if( le_eps )
        {
            WRITE_PT(start_pt);
        }
        else
        {
            right_slice.end = slice.end;
            slice.end = right_slice.start;
            PUSH_SLICE(right_slice);
            PUSH_SLICE(slice);
        }
    }

    if( !is_closed )
        WRITE_PT( src_contour[count-1] );

    // last stage: do final clean-up of the approximated contour -
    // remove extra points on the [almost] straight lines.
    is_closed = is_closed0;
    count = new_count;
    pos = is_closed ? count - 1 : 0;
    READ_DST_PT(start_pt, pos);
    wpos = pos;
    READ_DST_PT(pt, pos);

    for( A = !is_closed; A < count - !is_closed && new_count > 2; A++ )
    {
        double dx, dy, dist, successive_inner_product;
        READ_DST_PT( end_pt, pos );

        dx = end_pt.x - start_pt.x;
        dy = end_pt.y - start_pt.y;
        dist = fabs((pt.x - start_pt.x)*dy - (pt.y - start_pt.y)*dx);
        successive_inner_product = (pt.x - start_pt.x) * (end_pt.x - pt.x) +
        (pt.y - start_pt.y) * (end_pt.y - pt.y);

        if( dist * dist <= 0.5*eps*(dx*dx + dy*dy) && dx != 0 && dy != 0 &&
           successive_inner_product >= 0 )
        {
            new_count--;
            dst_contour[wpos] = start_pt = end_pt;
            if(++wpos >= count) wpos = 0;
            READ_DST_PT(pt, pos);
            A++;
            continue;
        }
        dst_contour[wpos] = start_pt = pt;
        if(++wpos >= count) wpos = 0;
        pt = end_pt;
    }

    if( !is_closed )
        dst_contour[wpos] = pt;

    return new_count;
}

void mygpu::approxPolyDP( cv::InputArray _curve, cv::OutputArray _approxCurve,
                      double epsilon, bool closed )
{

    //Prevent unreasonable error values (Douglas-Peucker algorithm)
    //from being used.
    if (epsilon < 0.0 || !(epsilon < 1e30))
    {
        CV_Error(CV_StsOutOfRange, "Epsilon not valid.");
    }

    cv::Mat curve = _curve.getMat();
    int npoints = curve.checkVector(2), depth = curve.depth();
    CV_Assert( npoints >= 0 && (depth == CV_32S || depth == CV_32F));

    if( npoints == 0 )
    {
        _approxCurve.release();
        return;
    }

    cv::AutoBuffer<cv::Point> _buf(npoints);
    cv::AutoBuffer<cv::Range> _stack(npoints);
    cv::Point* buf = _buf.data();
    int nout = 0;

    nout = approxPolyDP_(curve.ptr<cv::Point>(), npoints, buf, closed, epsilon, _stack);

    cv::Mat(nout, 1, CV_MAKETYPE(depth, 2), buf).copyTo(_approxCurve);
}

void DouglasPeucker(const VECTOR_H<cv::Point> &edge, VECTOR_H<cv::Point> &line, float epsilon)
{
	struct warp
	{
		unsigned int s;
		unsigned int e;
	} wp;
	float dmax = 0;
	float d;
	float da, db, dc, norm;
	unsigned int C = 0;
	VECTOR_H<warp> stack;
	bool *flags = new bool[edge.size()];

	memset(flags, 0, sizeof(bool)*edge.size());

	stack.push_back({ 0,edge.size() - 1 });
	while (!stack.empty())
	{
		wp = stack[stack.size() - 1];
		stack.pop_back();
		dmax = 0;
		da = edge[wp.e].y - edge[wp.s].y;
		db = edge[wp.s].x - edge[wp.e].x;
		dc = edge[wp.e].x * edge[wp.s].y - edge[wp.s].x * edge[wp.e].y;
		norm = sqrt(da * da + db * db);
		for (unsigned int i = wp.s; i < wp.e; i++)
		{
			d = fabs((da * edge[i].x + db * edge[i].y + dc) / norm);
			if (d > dmax)
			{
				C = i;
				dmax = d;
			}
		}
		// cout << wp.s << " to " << wp.e << " da= " << da << " db= " << db << " dc= " << dc << " norm= " << norm << " maxdis: " << dmax << endl;
		if (dmax >= epsilon)
		{
			stack.push_back({ wp.s, C });
			stack.push_back({ C, wp.e });
		}
		else
		{
			flags[wp.s] = true;
			flags[wp.e] = true;
		}
	}
	
	for (unsigned int i = 0; i < edge.size(); i++)
	{
		if (flags[i])
		{
			line.push_back(edge[i]);
		}
	}

    delete[] flags;
}
