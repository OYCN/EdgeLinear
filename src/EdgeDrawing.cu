#include "EdgeDrawing.h"
// #include "Timer.h"

#define LIDX(x, y) [(x) + (y)*lcols]
#define GIDX(x, y) [(x) + (y)*gcols]

__global__ void kernelC(uchar *blur, uchar * gMap, uchar *fMap, int cols, int rows, int ANCHOR_TH, int K);

EdgeDrawing::EdgeDrawing(int _rows, int _cols, float _th, int _k)
    :rows(_rows), cols(_cols), th(_th), k(_k)
{
    HANDLE_ERROR(cudaFree(0));
	HANDLE_ERROR(cudaMalloc(&gMapd, sizeof(uchar)*rows*cols));
	HANDLE_ERROR(cudaMalloc(&srcd, sizeof(uchar)*rows*cols));
	HANDLE_ERROR(cudaMalloc(&fMapd, sizeof(uchar)*rows*cols));
	HANDLE_ERROR(cudaMemset(gMapd, 0, sizeof(uchar)*rows*cols));
	HANDLE_ERROR(cudaMemset(fMapd, 0, sizeof(uchar)*rows*cols));
	gMaph = new uchar[rows*cols];
	fMaph = new uchar[rows*cols];
	eMaph = new uchar[rows*cols];
	EDoutput.edge_set = new POINT[rows*cols];
	EDoutput.edge_offset = new int[rows*cols+1];
	edge_smart = new POINT[rows*cols];
    EDoutput.eMap = eMaph;
}

EdgeDrawing::~EdgeDrawing()
{
    cudaFree(gMapd);
	cudaFree(srcd);
	cudaFree(fMapd);
	delete[] gMaph;
	delete[] fMaph;
    delete[] eMaph;
	delete[] EDoutput.edge_set;
	delete[] EDoutput.edge_offset;
	delete[] edge_smart;
}

void EdgeDrawing::initLoop()
{
    memset(eMaph, 0, rows*cols*sizeof(uchar));
}

_EDoutput* EdgeDrawing::run(cv::Mat& _src)
{
    // TDEF(gpu)
    // TDEF(init)
    // TDEF(h2d)
    // TDEF(kernel)
    // TDEF(d2h)
    // TDEF(cpu)
    // TSTART(gpu)
    // TSTART(init)
	// GPU Block 划分
    const dim3 dimBlock(32,32);;
    // GPU Grid 划分
    const dim3 dimGrid((cols+27)/28, (rows+27)/28);

    initLoop();
    
    cv::cvtColor(_src, srch, CV_RGB2GRAY);
	cv::GaussianBlur(srch, srch, cv::Size(5, 5), 1, 0);
    // TEND(init)
    // TSTART(h2d)
	HANDLE_ERROR(cudaMemcpy(srcd, srch.data, sizeof(uchar)*rows*cols, cudaMemcpyHostToDevice));
    // TEND(h2d)
    // TSTART(kernel)
    kernelC<<< dimGrid, dimBlock >>>(srcd, gMapd, fMapd, cols, rows, th, k);
    // HANDLE_ERROR(cudaDeviceSynchronize());
    // TEND(kernel)
    // TSTART(d2h)
    HANDLE_ERROR(cudaMemcpy(gMaph, gMapd, sizeof(uchar)*rows*cols, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(fMaph, fMapd, sizeof(uchar)*rows*cols, cudaMemcpyDeviceToHost));
    // TEND(d2h)
    // cv::Mat fMap(rows ,cols, CV_8UC1, (unsigned char*)(fMaph));
	// cv::imshow("fMap", fMap);
    // TEND(gpu)
    // TSTART(cpu)
    smartConnecting();
    // TEND(cpu)
    // TPRINTMS(gpu, "gpu:")
    // TPRINTMS(init, "\tinit:")
    // TPRINTMS(h2d, "\th2d:")
    // TPRINTMS(kernel, "\tkernel:")
    // TPRINTMS(d2h, "\td2h:")
    // TPRINTMS(cpu, "cpu:")

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
	int com = 0;
	uchar center = 0;
    uchar fmap = 0;
    // uchar &a = flag1;
    // uchar &b = flag2;
    // uchar &c = center;
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

        center = (uchar)(val);
        sgMap LIDX(lx,ly) = center;
        gMap GIDX(gx,gy) = center;

        // 1 -- vertical   0 -- horizonal
        dir = dx > dy;
        fmap |= (dir<<7)&0x80;
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

        // 方向分析
        // 方案1 用于GPU计算结果
        // 	0	0	0	0	0	0	0	0
        // 					----- A方向（左/上）上/左、下/右
        // 							----- B方向（右/下）上/左、下/右

        // dir : 1 -- vertical   0 -- horizonal

        //      第一轮       第二轮
        //   h-0    v-1    h-0   v-1
        // a 左上 / 上左 | 右上 / 下左
        // b 左   / 上   | 右   / 下
        // c 左下 / 上右 | 右下 / 下右
        // a = sgMap LIDX(lx-1, ly-1);
        // b = sgMap LIDX(lx-1, ly) * !dir;
        // b += sgMap LIDX(lx, ly-1) * dir;
        // c = sgMap LIDX(lx-1, ly+1) * !dir;
        // c += sgMap LIDX(lx+1, ly-1) * dir;

        // fmap |= (a>b && a>c)

        fMap GIDX(gx,gy) = fmap;
	}
    // 以上 50.641 ms

}