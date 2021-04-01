#include "EdgeDrawing.h"


// 方案1 用于GPU计算结果
// 	0	0	0	0	0	0	0	0
// 					----- A方向（左/上）上/左、下/右
// 							----- B方向（右/下）上/左、下/右

// 方案2 用于dir
// 	0	0	0	0	0	0	0	0
// 					上	下	左	右

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
