#include "BlockConnect.h"

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

#define BIT(x, i) ((x>>i)&0x01)

#define IDX(x, y) [(y) + ((x)*cols)]

void BlockConnect::init()
{
    HANDLE_ERROR(cudaMallocHost(&edges.eMap, rows*cols*sizeof(uchar)));
	HANDLE_ERROR(cudaMallocHost(&eMaph_bk, rows*cols*sizeof(uchar)));
    HANDLE_ERROR(cudaMallocHost(&edges.edge_set, rows*cols*sizeof(POINT)));
	HANDLE_ERROR(cudaMallocHost(&edges.edge_offset, rows*cols*sizeof(int)));
	HANDLE_ERROR(cudaMallocHost(&edge_smart, rows*cols*sizeof(POINT)));
    for(int i = 0; i < rows; i++)
    for(int j = 0; j < cols; j++)
    {
        if(i == 0 || i == (rows - 1) || j == 0 || j == (cols - 1)) eMaph_bk[j + (i*cols)] = 1;
        else eMaph_bk[j + (i*cols)] = 0;
    }
}

void BlockConnect::deinit()
{
	HANDLE_ERROR(cudaFreeHost(edges.eMap));
	HANDLE_ERROR(cudaFreeHost(eMaph_bk));
	HANDLE_ERROR(cudaFreeHost(edges.edge_set));
	HANDLE_ERROR(cudaFreeHost(edges.edge_offset));
	HANDLE_ERROR(cudaFreeHost(edge_smart));
}

void BlockConnect::execute(uchar* fMaph)
{
    memcpy(edges.eMap, eMaph_bk, rows*cols*sizeof(uchar));
    smartConnecting(rows, cols, fMaph, edges.eMap, edge_smart, edges.edge_set, edges.edge_offset, edges.edge_offset_len);
}

// connecting 
void smartConnecting(int rows, int cols, uchar* fMaph, uchar* eMaph, POINT* edge_smart, POINT* edge_set, int* edge_offset, int& edge_offset_len)
{
	int edge_smart_idx;
	uchar mydir = 0;
	edge_offset_len = 1;
	edge_offset[0] = 0;

	for (int j = 1; j < rows - 1; j++)
	{
		for (int i = 1; i < cols - 1; i++)
		{
			// 从每个未检测的 anchor point 出发
			if (!((fMaph IDX(j,i)>>6)&0x01) || eMaph IDX(j,i))
			{
				continue;
			}
			edge_offset[edge_offset_len] = edge_offset[edge_offset_len-1];
			//if(debug==0) {printf("a:%d e:%d\n", aMap IDX(0,0), eMap.data IDX(0,0));debug++;}
			
			edge_smart_idx = 0;
			edge_smart[edge_smart_idx] = POINT(i, j);
			edge_smart_idx++;

			// 判断走向  horizonal - left   vertical - up
			mydir = ((fMaph IDX(j,i)>>7)&0x01) ? DU : DL;
			goMove(i, j, mydir, edge_smart, edge_smart_idx, eMaph, fMaph, rows, cols);

			if (edge_smart_idx!=0)
			{
				for(int i=0; i<edge_smart_idx; i++)
				{
					edge_set[edge_offset[edge_offset_len-1]+i] = edge_smart[edge_smart_idx-i-1];
					edge_offset[edge_offset_len]++;
				}
			}

			// 判断走向  horizonal - right  vertical - down
			mydir = ((fMaph IDX(j,i)>>7)&0x01) ? DD : DR;
			goMove(i, j, mydir, edge_set, edge_offset[edge_offset_len], eMaph, fMaph, rows, cols);
			edge_offset_len++;
		}
	}
	
}

void goMove(int x, int y, uchar mydir, POINT *edge_s, int &idx, uchar* eMaph, uchar* fMaph, int rows, int cols)
{
	eMaph[y*cols + x] = 0; // for the second scan

	while (x>0 && x <cols - 1 && y>0 && y<rows - 1 && !eMaph[y*cols + x])
	{
		uchar this_fmap = fMaph[y*cols + x];
		uchar this_dir = BIT(this_fmap, 7);
		if((!(this_dir) && (mydir==DU || mydir==DD))
			|| ((this_dir) && (mydir==DL || mydir==DR)))
		{
			break;
		}

		eMaph[y*cols + x] = 1;

		if (!(this_dir))      // 水平
		{
			if (CDIR_L(mydir))
			{
				uchar eLeftCounter = eMaph[y*cols + x - 1] + eMaph[(y - 1)*cols + (x - 1)] + eMaph[(y + 1)*cols + (x - 1)];

				// 处理不是单像素链点 
				if (eLeftCounter>=2)
				{
					eMaph[y*cols + x] = 0;
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
					x = x - 1;
					if (BIT(this_fmap, 5))
					{
						y = y - 1;
						mydir = DLU;
					}
					else if (BIT(this_fmap, 3))
					{
						y = y + 1;
						mydir = DLD;
					}
					else
					{
						mydir = DL;
					}
					edge_s[idx] = POINT(x, y);
					idx++;
				}
			}
			else if (CDIR_R(mydir))
			{
				uchar eRightCounter = eMaph[y*cols + x + 1] + eMaph[(y - 1)*cols + (x + 1)] + eMaph[(y + 1)*cols + (x + 1)];

				if (eRightCounter>=2)
				{
					eMaph[y*cols + x] = 0;
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
					x = x + 1;
					if (BIT(this_fmap, 2))
					{
						y = y - 1;
						mydir = DRU;
					}
					else if (BIT(this_fmap, 0))
					{
						y = y + 1;
						mydir = DRD;
					}
					else
					{
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
				uchar eUpCounter = eMaph[(y - 1)*cols + x] + eMaph[(y - 1)*cols + (x - 1)] + eMaph[(y - 1)*cols + (x + 1)];

				if (eUpCounter>=2)
				{
					eMaph[y*cols + x] = 0;
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
					y = y - 1;
					if (BIT(this_fmap, 5))
					{
						x = x - 1;
						mydir = DLU;
					}
					else if (BIT(this_fmap, 3))
					{
						x = x + 1;
						mydir = DRU;
					}
					else
					{
						mydir = DU;
					}
					edge_s[idx] = POINT(x, y);
					idx++;
				}
			}
			else if (CDIR_D(mydir))
			{
				uchar eDownCounter = eMaph[(y + 1)*cols + x] + eMaph[(y + 1)*cols + (x - 1)] + eMaph[(y + 1)*cols + (x + 1)];

				if (eDownCounter>=2)
				{
					eMaph[y*cols + x] = 0;
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
					y = y + 1;
					if (BIT(this_fmap, 2))
					{
						x = x - 1;
						mydir = DLD;
					}
					else if (BIT(this_fmap, 0))
					{
						x = x + 1;
						mydir = DRD;
					}
					else
					{
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