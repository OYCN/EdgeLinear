#include "EdgeDrawing.h"

#define LIDX(x, y) [(x) + (y)*lcols]
#define GIDX(x, y) [(x) + (y)*gcols]

__global__ void kernelC(uchar *blur, uchar * gMap, uchar *fMap, int cols, int rows, int ANCHOR_TH, int K);

EdgeDrawing::EdgeDrawing(int _rows, int _cols, float _th, int _k, int _GFSize, int _GFs1, int _GFs2)
    :rows(_rows), cols(_cols), th(_th), k(_k), GFSize(_GFSize), GFs1(_GFs1), GFs2(_GFs2)
{
    HANDLE_ERROR(cudaSetDevice(0));
    HANDLE_ERROR(cudaFree(0));
	HANDLE_ERROR(cudaMalloc(&gMapd, sizeof(uchar)*rows*cols));
	HANDLE_ERROR(cudaMalloc(&blurd, sizeof(uchar)*rows*cols));
	HANDLE_ERROR(cudaMalloc(&fMapd, sizeof(uchar)*rows*cols));
	HANDLE_ERROR(cudaMemset(gMapd, 0, sizeof(uchar)*rows*cols));
	HANDLE_ERROR(cudaMemset(fMapd, 0, sizeof(uchar)*rows*cols));

	#ifdef USE_OPENCV_GPU
	HANDLE_ERROR(cudaMalloc(&srcd, sizeof(uchar)*rows*cols*3));
	HANDLE_ERROR(cudaMalloc(&grayd, sizeof(uchar)*rows*cols));
	gmat_src = new cv::cuda::GpuMat(rows, cols, CV_8UC3, srcd);
	gmat_gray = new cv::cuda::GpuMat(rows, cols, CV_8UC1, grayd);
	gmat_blur = new cv::cuda::GpuMat(rows, cols, CV_8UC1, blurd);
	gauss = cv::cuda::createGaussianFilter(CV_8U, CV_8U, cv::Size(GFSize, GFSize), GFs1, GFs2);
	// 第一次貌似很慢
	cv::cuda::cvtColor(*gmat_src, *gmat_gray, CV_RGB2GRAY);
	gauss->apply(*gmat_gray, *gmat_blur);
	#endif

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
    cudaFree(gMapd);
	cudaFree(blurd);
	cudaFree(fMapd);
	#ifdef USE_OPENCV_GPU
	cudaFree(grayd);
	cudaFree(srcd);
	#endif
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
	// GPU Block 划分
    const dim3 dimBlock(32,32);;
    // GPU Grid 划分
    const dim3 dimGrid((cols+27)/28, (rows+27)/28);

	#ifdef USE_OPENCV_GPU
	gmat_src->upload(_src);
	cv::cuda::cvtColor(*gmat_src, *gmat_gray, CV_RGB2GRAY);
	gauss->apply(*gmat_gray, *gmat_blur);
	#else
	cv::cvtColor(_src, srch, CV_RGB2GRAY);
	cv::GaussianBlur(srch, srch, cv::Size(5, 5), 1, 0);
	HANDLE_ERROR(cudaMemcpy(blurd, srch.data, sizeof(uchar)*rows*cols, cudaMemcpyHostToDevice));
	#endif
	kernelC<<< dimGrid, dimBlock >>>(blurd, gMapd, fMapd, cols, rows, th, k);
    // HANDLE_ERROR(cudaDeviceSynchronize());
    initLoop();
    // HANDLE_ERROR(cudaMemcpy(gMaph, gMapd, sizeof(uchar)*rows*cols, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(fMaph, fMapd, sizeof(uchar)*rows*cols, cudaMemcpyDeviceToHost));
    smartConnecting();

	return &EDoutput;
}

__global__ void kernelC(uchar *blur, uchar * gMap, uchar *fMap, int gcols, int grows, int ANCHOR_TH, int K)
{
    const int &lx = threadIdx.x;
    const int &ly = threadIdx.y;
    const int &lcols = blockDim.x;
    const int &lrows = blockDim.y;
    int gx = blockIdx.x*(lcols - 4) + threadIdx.x;
    int gy = blockIdx.y*(lrows - 4) + threadIdx.y;
    int dx = 0;
    int dy = 0;
    float val = 0;
    uchar dir = 0;
	uchar flag1 = 0;
	uchar flag2 = 0;
	int &com = dx;
	uchar center = 0;
    uchar fmap = 0;
    uchar &a = flag1;
    uchar &b = flag2;
    uchar &c = center;
    // uchar r;
    __shared__ volatile uchar sblur[32*32];
    __shared__ volatile uchar sgMap[32*32];
    // 以上 4.362 ms
    // 数据写入共享内存 7.5 ms
    if(gx<gcols && gy<grows)
        sblur LIDX(lx, ly) = blur GIDX(gx, gy);
    __syncthreads();
    // 以上 11.767 ms
    // 梯度计算 17.6 ms
	if(lx!=0 && ly!=0 && lx<(lcols-1) && ly<(lrows-1) && gx<(gcols-1) && gy<(grows-1))
    {
        dx = sblur LIDX(lx+1,ly-1);
        dx += 2 * sblur LIDX(lx+1,ly);
        dx += sblur LIDX(lx+1,ly+1);
        dx -= sblur LIDX(lx-1,ly-1);
        dx -= 2 * sblur LIDX(lx-1,ly);
        dx -= sblur LIDX(lx-1,ly+1);
        dx = abs(dx);

        dy = sblur LIDX(lx-1,ly-1);
        dy += 2 * sblur LIDX(lx,ly-1);
        dy += sblur LIDX(lx+1,ly-1);
        dy -= sblur LIDX(lx-1,ly+1);
        dy -= 2 * sblur LIDX(lx,ly+1);
        dy -= sblur LIDX(lx+1,ly+1);
        dy = abs(dy);

        val = 0.5f*dx + 0.5f*dy;
        if (val > 255) val = 255.0f;

        // 1 -- vertical   0 -- horizonal
        dir = dx > dy;
        fmap |= (dir<<7)&0x80;

        center = (uchar)(val);
        sgMap LIDX(lx,ly) = center;
        gMap GIDX(gx,gy) = center;
        //debug
        // if(gx==732 && gy==1445)
        // {
        //     printf("%d\n", center);
        // }
    }
	__syncthreads();
    // 以上 29.3 ms
    // 锚点提取 21.341 ms
	if((lx>1 || gx==1) && (ly>1 || gy==1) && (lx<(lcols-2) || gx==(gcols-2)) && (ly<(lrows-2) || gy==(grows-2)) && gx<(gcols-1) && gy<(grows-1))
	{
		// h
		flag1 = !dir;
		com = center;
		com -= sgMap LIDX(lx, ly-1);
		flag1 &= com>=ANCHOR_TH;
		com = center;
		com -= sgMap LIDX(lx, ly+1);
		flag1 &= com>=ANCHOR_TH;
		// v
		flag2 = dir;
		com = center;
		com -= sgMap LIDX(lx-1,ly);
		flag2 &= com >= ANCHOR_TH;
		com = center;
		com -= sgMap LIDX(lx+1,ly);
		flag2 &= com >= ANCHOR_TH;
        fmap |= (((flag1 | flag2) && ((gx-1)%K)==0 && ((gy-1)%K)==0)<<6)&0x40;

        // fmap
        // 	0	0	0	0	0	0	0	0
        // 	|   |   左上 左 左下 右上 右 右下
        //  -dir|   上左 上 上右 下左 下 下右
        //      -keypoint

        // dir : 1 -- 垂直   0 -- 水平 
        // keypoint: 锚点
        // 水平:
        //      A方向 - 左
        //      B方向 - 右
        // 垂直:
        //      A方向 - 上
        //      B方向 - 下

        // a    b  c
        // 上左 上 上右
        // 左上 左 左下
        a = sgMap LIDX(lx - 1, ly - 1);
        if(dir) // 垂直
        {   
            b = sgMap LIDX(lx, ly - 1);
            c = sgMap LIDX(lx + 1, ly - 1);
        }
        else
        {
            b = sgMap LIDX(lx - 1, ly);
            c = sgMap LIDX(lx - 1, ly + 1);
        }
        
        fmap |= (a>b && a>c) << 5;
        fmap |= (b>a && b>c) << 4;
        fmap |= (c>a && c>b) << 3;

        // a    b  c
        // 下左 下 下右
        // 右上 右 右下
        if(dir) // 垂直
        {   
            a = sgMap LIDX(lx - 1, ly + 1);
            b = sgMap LIDX(lx, ly + 1);
        }
        else
        {
            a = sgMap LIDX(lx + 1, ly - 1);
            b = sgMap LIDX(lx + 1, ly);
        }
        c = sgMap LIDX(lx + 1, ly + 1);

        fmap |= (a>b && a>c) << 2;
        fmap |= (b>a && b>c) << 1;
        fmap |= (c>a && c>b) << 0;

        // // debug
        // if(gx==114 && gy==10)
        // {
        //     printf("%d\n", a);
        //     printf("%d\n", b);
        //     printf("%d\n", c);
        // }

        fMap GIDX(gx,gy) = fmap;
	}
    // 以上 50.641 ms

}

// dir
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

#define BIT(x, i) ((x>>i)&0x01)

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

	while (x>0 && x <w - 1 && y>0 && y<h - 1 && !eMaph[y*s + x])
	{
		uchar this_fmap = fMaph[y*s + x];
		uchar this_dir = BIT(this_fmap, 7);
		if((!(this_dir) && (mydir==DU || mydir==DD))
			|| ((this_dir) && (mydir==DL || mydir==DR)))
		{
			break;
		}

		eMaph[y*s + x] = 1;

		if (!(this_dir))      // 水平
		{
			if (CDIR_L(mydir))
			{
				//####################################################
				// uchar left = gMaph[y*s + x - 1];
				// uchar left_up = gMaph[(y - 1)*s + (x - 1)];
				// uchar left_down = gMaph[(y + 1)*s + (x - 1)];
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
					x = x - 1;
					// if (left_up > left && left_up > left_down)
					if (BIT(this_fmap, 5))
					{
						// if(!BIT(this_fmap, 5))
						// {
						// 	std::cout << "lu" << std::endl;
						// 	exit(0);
						// }
						y = y - 1;
						mydir = DLU;
					}
					// else if (left_down > left && left_down > left_up)
					else if (BIT(this_fmap, 3))
					{
						// if(!BIT(this_fmap, 3))
						// {
						// 	std::cout << "ld" << std::endl;
						// 	exit(0);
						// }
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
				// uchar right = gMaph[y*s + x + 1];
				// uchar right_up = gMaph[(y - 1)*s + (x + 1)];
				// uchar right_down = gMaph[(y + 1)*s + (x + 1)];
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
					x = x + 1;
					// if (right_up > right && right_up > right_down)
					if (BIT(this_fmap, 2))
					{
						y = y - 1;
						mydir = DRU;
					}
					// else if (right_down > right && right_down > right_up)
					else if (BIT(this_fmap, 0))
					{
						// if(!BIT(this_fmap, 0))
						// {
						// 	std::cout << "rd" << std::endl;
						// 	exit(0);
						// }
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
				// uchar up = gMaph[(y - 1)*s + x];
				// uchar up_left = gMaph[(y - 1)*s + (x - 1)];
				// uchar up_right = gMaph[(y - 1)*s + (x + 1)];
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
					y = y - 1;
					// if (up_left > up && up_left > up_right)
					if (BIT(this_fmap, 5))
					{
						// if(!BIT(this_fmap, 5))
						// {
						// 	std::cout << "ul" << std::endl;
						// 	exit(0);
						// }
						x = x - 1;
						mydir = DLU;
					}
					// else if (up_right > up && up_right > up_left)
					else if (BIT(this_fmap, 3))
					{
						// if(!BIT(this_fmap, 3))
						// {
						// 	std::cout << "ur" << std::endl;
						// 	exit(0);
						// }
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
				// uchar down = gMaph[(y + 1)*s + x];
				// uchar down_left = gMaph[(y + 1)*s + (x - 1)];
				// uchar down_right = gMaph[(y + 1)*s + (x + 1)];
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
					y = y + 1;
					// if (down_left > down && down_left > down_right)
					if (BIT(this_fmap, 2))
					{
						// if(!BIT(this_fmap, 2))
						// {
						// 	std::cout << "dl" << std::endl;
						// 	exit(0);
						// }
						x = x - 1;
						mydir = DLD;
					}
					// else if (down_right > down && down_right > down_left)
					else if (BIT(this_fmap, 0))
					{
						// if(!BIT(this_fmap, 0))
						// {
						// 	std::cout << "dr" << std::endl;
						// 	exit(0);
						// }
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
