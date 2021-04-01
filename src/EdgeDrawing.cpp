#include "EdgeDrawing.h"
// #include "Timer.h"

#define IDX(x, y) [(x) + (y)*cols]

void kernelC(uchar *src, uchar * gMap, uchar *fMap, int cols, int rows, int ANCHOR_TH, int K);

EdgeDrawing::EdgeDrawing(int _rows, int _cols, float _th, int _k)
    :rows(_rows), cols(_cols), th(_th), k(_k)
{
	gMaph = new uchar[rows*cols];
	fMaph = new uchar[rows*cols];
	eMaph = new uchar[rows*cols];
    eMaph_bk = new uchar[rows*cols];
	EDoutput.edge_set = new POINT[rows*cols];
	EDoutput.edge_offset = new int[rows*cols+1];
	edge_smart = new POINT[rows*cols];
    EDoutput.eMap = eMaph;
    for(int i = 0; i < rows; i++)
    for(int j = 0; j < cols; j++)
    {
        if(i == 0 || i == (rows - 1) || j == 0 || j == (cols - 1)) eMaph_bk[j + (i*cols)] = 1;
        else eMaph_bk[j + (i*cols)] = 0;
    }
}

EdgeDrawing::~EdgeDrawing()
{
	delete[] gMaph;
	delete[] fMaph;
    delete[] eMaph;
    delete[] eMaph_bk;
	delete[] EDoutput.edge_set;
	delete[] EDoutput.edge_offset;
	delete[] edge_smart;
}

void EdgeDrawing::initLoop()
{
    // memset(eMaph, 0, rows*cols*sizeof(uchar));
    memcpy(eMaph, eMaph_bk, rows*cols*sizeof(uchar));
}

_EDoutput* EdgeDrawing::run(cv::Mat& _src)
{
    TDEF(part)
    TDEF(init)
    TDEF(com)
    TDEF(cpu)

    TSTART(part)
    TSTART(init)
    initLoop();
    
    cv::cvtColor(_src, srch, CV_RGB2GRAY);
	cv::GaussianBlur(srch, srch, cv::Size(5, 5), 1, 0);
    TEND(init)
    TSTART(com)
    kernelC(srch.data, gMaph, fMaph, cols, rows, th, k);
    TEND(com)
    TEND(part)
    TSTART(cpu)
    smartConnecting();
    TEND(cpu)

    TPRINTMS(part, "part:")
    TPRINTMS(init, "\tinit:")
    TPRINTMS(com, "\tcom:")
    TPRINTMS(cpu, "cpu:")

	return &EDoutput;
}

void kernelC(uchar *src, uchar * gMap, uchar *fMap, int cols, int rows, int ANCHOR_TH, int K)
{
    int dx = 0;
    int dy = 0;
    float val = 0;

    // 求梯度
    for(int y = 1; y < (rows - 1); y++)
    for(int x = 1; x < (cols - 1); x++)
    {
        dx = src IDX(x+1,y-1)
            + 2 * src IDX(x+1,y)
            + src IDX(x+1,y+1)
            - src IDX(x-1,y-1)
            - 2 * src IDX(x-1,y)
            - src IDX(x-1,y+1);
        dx = abs(dx);

        dy = src IDX(x-1,y-1)
            + 2 * src IDX(x,y-1)
            + src IDX(x+1,y-1)
            - src IDX(x-1,y+1)
            - 2 * src IDX(x,y+1)
            - src IDX(x+1,y+1);
        dy = abs(dy);

        val = 0.5f*dx + 0.5f*dy;
        if (val > 255) val = 255.0f;
        gMap IDX(x, y) = (uchar)val;

		if(dx > dy)
		{
			fMap IDX(x, y) = 0x80;
		}
		else
		{
			fMap IDX(x, y) = 0;
		}
    }

    // 求flag
    for(int y = 1; y < (rows - 1); y++)
    for(int x = 1; x < (cols - 1); x++)
    {
        if(fMap IDX(x, y) != 0)
        {
            if((gMap IDX(x, y) - gMap IDX(x-1, y)) >= ANCHOR_TH
                && (gMap IDX(x, y) - gMap IDX(x+1, y)) >= ANCHOR_TH
                && ((x-1) % K) == 0 && ((y-1) % K) == 0)
            {
                fMap IDX(x, y) |= 0x40;
            }
        }
        else
        {
            if((gMap IDX(x, y) - gMap IDX(x, y-1)) >= ANCHOR_TH
                && (gMap IDX(x, y) - gMap IDX(x, y+1)) >= ANCHOR_TH
                && ((x-1) % K) == 0 && ((y-1) % K) == 0)
            {
                fMap IDX(x, y) |= 0x40;
            }
        }
    }

}

#define DU 0x08
#define DD 0x04
#define DL 0x02
#define DR 0x01
#define DLU 0x0A
#define DLD 0x06
#define DRU 0x09
#define DRD 0x05

#define SDIR_L(x) (x)|=DL
#define SDIR_R(x) (x)|=DR
#define SDIR_U(x) (x)|=DU
#define SDIR_D(x) (x)|=DD
#define RDIR(x) (x)=0
#define CDIR_L(x) ((x)&DL)
#define CDIR_R(x) ((x)&DR)
#define CDIR_U(x) ((x)&DU)
#define CDIR_D(x) ((x)&DD)

#define IDX(x, y) [(y) + ((x)*cols)]

// connecting 
void EdgeDrawing::smartConnecting()
{
	int &h = rows, &w = cols;
	int edge_smart_idx;
	uchar mydir = 0;
	EDoutput.edge_offset_len = 1;
	EDoutput.edge_offset[0] = 0;

	for (int j = 1; j < h - 1; j++)
	{
		for (int i = 1; i < w - 1; i++)
		{
			// 从每个未检测的 anchor point 出发
			if (!((fMaph IDX(j,i)>>6)&0x01) || eMaph IDX(j,i))
			{
				continue;
			}
			EDoutput.edge_offset[EDoutput.edge_offset_len] = EDoutput.edge_offset[EDoutput.edge_offset_len-1];
			//if(debug==0) {printf("a:%d e:%d\n", aMap IDX(0,0), eMap.data IDX(0,0));debug++;}
			
			edge_smart_idx = 0;
			edge_smart[edge_smart_idx] = POINT(i, j);
			edge_smart_idx++;

			// 判断走向  horizonal - left   vertical - up
			mydir = ((fMaph IDX(j,i)>>7)&0x01) ? DU : DL;
			goMove(i, j, mydir, edge_smart, edge_smart_idx);

			if (edge_smart_idx!=0)
			{
				for(int i=0; i<edge_smart_idx; i++)
				{
					EDoutput.edge_set[EDoutput.edge_offset[EDoutput.edge_offset_len-1]+i] = edge_smart[edge_smart_idx-i-1];
					EDoutput.edge_offset[EDoutput.edge_offset_len]++;
				}
			}

			// 判断走向  horizonal - right  vertical - down
			mydir = ((fMaph IDX(j,i)>>7)&0x01) ? DD : DR;
			goMove(i, j, mydir, EDoutput.edge_set, EDoutput.edge_offset[EDoutput.edge_offset_len]);
			EDoutput.edge_offset_len++;
		}
	}
	
}

void EdgeDrawing::goMove(int x, int y, uchar mydir, POINT *edge_s, int &idx)
{
	int h = rows, w = cols, s = cols;
	eMaph[y*s + x] = 0; // for the second scan

	while (x>0 && x <w - 1 && y>0 && y<h - 1 && !eMaph[y*s + x] && gMaph[y*s + x])
	{
		if((!((fMaph[y*s + x]>>7)&0x01) && (mydir==DU || mydir==DD))
			|| (((fMaph[y*s + x]>>7)&0x01) && (mydir==DL || mydir==DR)))
		{
			break;
		}

		eMaph[y*s + x] = 1;

		if (!((fMaph[y*s + x]>>7)&0x01))      // 水平
		{
			if (CDIR_L(mydir))
			{
				//####################################################
				uchar left = gMaph[y*s + x - 1];
				uchar left_up = gMaph[(y - 1)*s + (x - 1)];
				uchar left_down = gMaph[(y + 1)*s + (x - 1)];
				uchar eLeftCounter = eMaph[y*s + x - 1] + eMaph[(y - 1)*s + (x - 1)] + eMaph[(y + 1)*s + (x - 1)];

				// 处理不是单像素链点 
				if (eLeftCounter>=2)
				{
					eMaph[y*s + x] = 0;
					break;
				}
				else if (eLeftCounter==1)
				{
					edge_s[idx] = POINT(x, y);
					idx++;
					break;
				}
				else
				{
					if (left_up > left && left_up > left_down)
					{
						x = x - 1;
						y = y - 1;
						mydir = DLU;
					}
					else if (left_down > left && left_down > left_up)
					{
						x = x - 1;
						y = y + 1;
						mydir = DLD;
					}
					else
					{
						x = x - 1;
						mydir = DL;
					}
					edge_s[idx] = POINT(x, y);
					idx++;
				}
			}
			else if (CDIR_R(mydir))
			{
				uchar right = gMaph[y*s + x + 1];
				uchar right_up = gMaph[(y - 1)*s + (x + 1)];
				uchar right_down = gMaph[(y + 1)*s + (x + 1)];
				uchar eRightCounter = eMaph[y*s + x + 1] + eMaph[(y - 1)*s + (x + 1)] + eMaph[(y + 1)*s + (x + 1)];

				if (eRightCounter>=2)
				{
					eMaph[y*s + x] = 0;
					break;
				}
				else if (eRightCounter==1)
				{
					edge_s[idx] = POINT(x, y);
					idx++;
					break;
				}
				else
				{
					if (right_up > right && right_up > right_down)
					{
						x = x + 1;
						y = y - 1;
						mydir = DRU;
					}
					else if (right_down > right && right_down > right_up)
					{
						x = x + 1;
						y = y + 1;
						mydir = DRD;
					}
					else
					{
						x = x + 1;
						mydir = DR;
					}
					edge_s[idx] = POINT(x, y);
					idx++;
				}
			}
			else 
			{
				ERROR("cur_dir not in list");
			}
		}
		else                        //  vertical scanning
		{
			if (CDIR_U(mydir))
			{
				uchar up = gMaph[(y - 1)*s + x];
				uchar up_left = gMaph[(y - 1)*s + (x - 1)];
				uchar up_right = gMaph[(y - 1)*s + (x + 1)];
				uchar eUpCounter = eMaph[(y - 1)*s + x] + eMaph[(y - 1)*s + (x - 1)] + eMaph[(y - 1)*s + (x + 1)];

				if (eUpCounter>=2)
				{
					eMaph[y*s + x] = 0;
					break;
				}
				else if (eUpCounter==1)
				{
					edge_s[idx] = POINT(x, y);
					idx++;
					break;
				}
				else
				{
					if (up_left > up && up_left > up_right)
					{
						x = x - 1;
						y = y - 1;
						mydir = DLU;
					}
					else if (up_right > up && up_right > up_left)
					{
						x = x + 1;
						y = y - 1;
						mydir = DRU;
					}
					else
					{
						y = y - 1;
						mydir = DU;
					}
					edge_s[idx] = POINT(x, y);
					idx++;
				}
			}
			else if (CDIR_D(mydir))
			{
				uchar down = gMaph[(y + 1)*s + x];
				uchar down_left = gMaph[(y + 1)*s + (x - 1)];
				uchar down_right = gMaph[(y + 1)*s + (x + 1)];
				uchar eDownCounter = eMaph[(y + 1)*s + x] + eMaph[(y + 1)*s + (x - 1)] + eMaph[(y + 1)*s + (x + 1)];

				if (eDownCounter>=2)
				{
					eMaph[y*s + x] = 0;
					break;
				}
				else if (eDownCounter==1)
				{
					edge_s[idx] = POINT(x, y);
					idx++;
					break;
				}
				else
				{
					if (down_left > down && down_left > down_right)
					{
						x = x - 1;
						y = y + 1;
						mydir = DLD;
					}
					else if (down_right > down && down_right > down_left)
					{
						x = x + 1;
						y = y + 1;
						mydir = DRD;
					}
					else
					{
						y = y + 1;
						mydir = DD;
					}
					edge_s[idx] = POINT(x, y);
					idx++;
				}
			}
			else 
			{
				ERROR("cur_dir not in list");
			}
		}
	}
}
