#include "BlockGetFlag.h"

void BlockGetFlag::init()
{
    HANDLE_ERROR(cudaMalloc(&srcd, sizeof(uchar)*rows*cols*3));
    gmat_src = new cv::cuda::GpuMat(rows, cols, CV_8UC3, srcd);
    HANDLE_ERROR(cudaMalloc(&grayd, sizeof(uchar)*rows*cols));
    gmat_gray = new cv::cuda::GpuMat(rows, cols, CV_8UC1, grayd);
    HANDLE_ERROR(cudaMalloc(&blurd, sizeof(uchar)*rows*cols));
	gmat_blur = new cv::cuda::GpuMat(rows, cols, CV_8UC1, blurd);

    HANDLE_ERROR(cudaMalloc(&fMapd, sizeof(uchar)*rows*cols));

    HANDLE_ERROR(cudaMallocHost(&fMaph, rows*cols*sizeof(uchar)));

    gauss = cv::cuda::createGaussianFilter(CV_8U, CV_8U, cv::Size(GFSize, GFSize), GFs1, GFs2);
    cv::cuda::cvtColor(*gmat_src, *gmat_gray, CV_RGB2GRAY);
	gauss->apply(*gmat_gray, *gmat_blur);
    
}

void BlockGetFlag::deinit()
{
    HANDLE_ERROR(cudaFree(srcd));
    HANDLE_ERROR(cudaFree(grayd));
    HANDLE_ERROR(cudaFree(blurd));
    HANDLE_ERROR(cudaFree(fMapd));
    HANDLE_ERROR(cudaFreeHost(fMaph));
}

void BlockGetFlag::enqueue(cv::Mat& sMaph, cv::cuda::Stream& cvstream)
{
    // GPU Block 划分
    const dim3 dimBlock(32,32);;
    // GPU Grid 划分
    const dim3 dimGrid((cols+27)/28, (rows+27)/28);

    cudaStream_t custream = cv::cuda::StreamAccessor::getStream(cvstream);
    HANDLE_ERROR(cudaMemcpyAsync(srcd, sMaph.data, sizeof(uchar)*rows*cols*3, cudaMemcpyHostToDevice, custream));
	cv::cuda::cvtColor(*gmat_src, *gmat_gray, CV_RGB2GRAY, 0, cvstream);
	gauss->apply(*gmat_gray, *gmat_blur, cvstream);
    kernelC<<< dimGrid, dimBlock, 0,custream >>>(blurd, fMapd, cols, rows, th, k);
    HANDLE_ERROR(cudaMemcpyAsync(fMaph, fMapd, sizeof(uchar)*rows*cols, cudaMemcpyDeviceToHost, custream));
}

__global__ void kernelC(uchar *blur, uchar *fMap, int gcols, int grows, int ANCHOR_TH, int K)
{
    #define LIDX(x, y) [(x) + (y)*lcols]
    #define GIDX(x, y) [(x) + (y)*gcols]

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
        // gMap GIDX(gx,gy) = center;
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
