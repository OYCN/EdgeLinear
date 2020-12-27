#include "EDProcess_par.h"


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

static void goMove(int x, int y, uchar *gMap, cv::Mat eMap, uchar *fMap,int rows, int cols, uchar mydir, VECTOR_H<POINT>& edge_s);

// connecting 
cv::Mat smartConnecting(uchar *gMap, uchar *fMap, int rows, int cols, VECTOR_H<VECTOR_H<POINT>>& edge_s)
{
	cv::Mat eMap = cv::Mat::zeros(rows, cols, CV_8UC1);
	int h = rows, w = cols;
	VECTOR_H<POINT>edges;
	uchar mydir = 0;

	for (int j = 1; j < h - 1; j++)
	{
		for (int i = 1; i < w - 1; i++)
		{
			// 从每个未检测的 anchor point 出发
			if (!((fMap IDX(j,i)>>6)&0x01) || eMap.data IDX(j,i))
			{
				continue;
			}
			//if(debug==0) {printf("a:%d e:%d\n", aMap IDX(0,0), eMap.data IDX(0,0));debug++;}
			
			VECTOR_H<POINT>().swap(edges);

			edges.push_back(POINT(i, j)); // start point

			// 判断走向  horizonal - left   vertical - up
			mydir = ((fMap IDX(j,i)>>7)&0x01) ? DU : DL;
			goMove(i, j, gMap, eMap, fMap, rows, cols, mydir, edges);

			if (!edges.empty()) std::reverse(edges.begin(), edges.end());

			// 判断走向  horizonal - right  vertical - down
			mydir = ((fMap IDX(j,i)>>7)&0x01) ? DD : DR;
			goMove(i, j, gMap, eMap, fMap, rows, cols, mydir, edges);

			// 添加函数： 

			edge_s.push_back(edges);
		}
	}
	return eMap;
}

// not changed: x,y,gMap,dMap
// changed and read: eMap
// in/out: 

static void goMove(int x, int y, uchar *gMap, cv::Mat eMap, uchar *fMap, int rows, int cols, uchar mydir, VECTOR_H<POINT>& edge_s)
{
	int h = rows, w = cols, s = cols;
	eMap.data[y*s + x] = 0; // for the second scan

	while (x>0 && x <w - 1 && y>0 && y<h - 1 && !eMap.data[y*s + x] && gMap[y*s + x])
	{
		if((!((fMap[y*s + x]>>7)&0x01) && (mydir==DU || mydir==DD))
			|| (((fMap[y*s + x]>>7)&0x01) && (mydir==DL || mydir==DR)))
		{
			break;
		}

		eMap.data[y*s + x] = 1;

		if (!((fMap[y*s + x]>>7)&0x01))      // horizonal scanning
		{
			if (CDIR_L(mydir))
			{
				//####################################################
				uchar left = gMap[y*s + x - 1];
				uchar left_up = gMap[(y - 1)*s + (x - 1)];
				uchar left_down = gMap[(y + 1)*s + (x - 1)];
				uchar eLeftCounter = eMap.data[y*s + x - 1] + eMap.data[(y - 1)*s + (x - 1)] + eMap.data[(y + 1)*s + (x - 1)];

				// 处理不是单像素链点 
				if (eLeftCounter>=2)
				{
					eMap.data[y*s + x] = 0;
					break;
				}
				else if (eLeftCounter==1)
				{
					edge_s.push_back(POINT(x, y));
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
					edge_s.push_back(POINT(x, y));
				}
			}
			else if (CDIR_R(mydir))
			{
				uchar right = gMap[y*s + x + 1];
				uchar right_up = gMap[(y - 1)*s + (x + 1)];
				uchar right_down = gMap[(y + 1)*s + (x + 1)];
				uchar eRightCounter = eMap.data[y*s + x + 1] + eMap.data[(y - 1)*s + (x + 1)] + eMap.data[(y + 1)*s + (x + 1)];

				if (eRightCounter>=2)
				{
					eMap.data[y*s + x] = 0;
					break;
				}
				else if (eRightCounter==1)
				{
					edge_s.push_back(POINT(x, y));
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
					edge_s.push_back(POINT(x, y));
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
				uchar up = gMap[(y - 1)*s + x];
				uchar up_left = gMap[(y - 1)*s + (x - 1)];
				uchar up_right = gMap[(y - 1)*s + (x + 1)];
				uchar eUpCounter = eMap.data[(y - 1)*s + x] + eMap.data[(y - 1)*s + (x - 1)] + eMap.data[(y - 1)*s + (x + 1)];

				if (eUpCounter>=2)
				{
					eMap.data[y*s + x] = 0;
					break;
				}
				else if (eUpCounter==1)
				{
					edge_s.push_back(POINT(x, y));
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
					edge_s.push_back(POINT(x, y));
				}
			}
			else if (CDIR_D(mydir))
			{
				uchar down = gMap[(y + 1)*s + x];
				uchar down_left = gMap[(y + 1)*s + (x - 1)];
				uchar down_right = gMap[(y + 1)*s + (x + 1)];
				uchar eDownCounter = eMap.data[(y + 1)*s + x] + eMap.data[(y + 1)*s + (x - 1)] + eMap.data[(y + 1)*s + (x + 1)];

				if (eDownCounter>=2)
				{
					eMap.data[y*s + x] = 0;
					break;
				}
				else if (eDownCounter==1)
				{
					edge_s.push_back(POINT(x, y));
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
					edge_s.push_back(POINT(x, y));
				}
			}
			else 
			{
				ERROR("cur_dir not in list");
			}
		}
	}
}
