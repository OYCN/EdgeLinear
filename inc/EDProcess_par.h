#ifndef _EDPROCESS_H_
#define _EDPROCESS_H_

// 包含文件
#include <opencv2/opencv.hpp>
// #include <opencv2/core/cuda.hpp>
#include <iostream>
#include <vector>
#include "cuda.h"
#include "cuda_runtime.h"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"

// 编译配置
#define DEFIMG "./img/1.png"	// 默认图片
#define USE_CHECK	// 运行GPUDP与CPUDP并进行结果比较
// #define USE_CPUDP	// 使用CPU-DP多边形化算法
#define USE_GPUDP	// 使用GPU-DP多边形化算法
// #define USE_UNIMEM	// 使用统一内存寻址
#define SHOW_IMG	// 是否显示图片
// #define TIM_PROC	// 是否显示ED proc的时间
#define TIM_GPUDP	// 是否显示GPU DP的时间

#ifdef USE_CHECK
#define USE_CPUDP
#define USE_GPUDP
#endif

// 类型定义
#ifndef USE_CHECK
#ifdef USE_CPUDP
#define VECTOR_H std::vector
#define POINT cv::Point
#endif
#endif

#ifdef USE_GPUDP
#define VECTOR_H thrust::host_vector
#define VECTOR_D thrust::device_vector
#define POINT mygpu::Point
#endif 

#define ERROR(x) {printf( "%s in %s at line %d\n", (x), __FILE__, __LINE__ );exit( EXIT_FAILURE );}
typedef unsigned char uchar;
static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

// 类定义
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
		// __host__ __device__
		// void operator = (std::initializer_list <int> &il)
		// {
		// 	const int *v = il.begin();
		// 	x = *v;
		// 	y = *(v+1);
		// }
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

class ED
{
	public:
		~ED();
		ED();
		cv::Mat Process(cv::Mat& blurImg, VECTOR_H<VECTOR_H<POINT>>& edge_s, const int anchor_th = 6, const int k = 2);
	private:
		void initinal(cv::Mat& blurImg, const int anchor_th, const int k);
		#ifndef USE_UNIMEM
		uchar *gMapd, *blurd, *fMapd;
		uchar *gMaph, *fMaph;
		#endif
		#ifdef USE_UNIMEM
		uchar *gMap, *fMap, *blurd;
		#endif
		int cols, rows;
		int K, ANCHOR_TH;
};

// 全局变量
// #ifdef USE_GPUDP
// cudaStream_t streams[STREAM_LEN];
// char *stream_dflag[STREAM_LEN];
// char *stream_hflag[STREAM_LEN];
// int stream_len[STREAM_LEN];
// #endif

// 函数定义
cv::Mat smartConnecting(uchar *gMap, uchar *fMap, int rows, int cols, VECTOR_H<VECTOR_H<POINT>>& edge_s);
// __global__ void kernelA(uchar *blur, uchar* gMap, uchar* dMap, int cols, int rows);
// __global__ void kernelB(uchar * gMap, uchar * dMap, uchar *aMap, int cols, int rows, int ANCHOR_TH, int K);
__global__ void kernelC(uchar *blur, uchar * gMap, uchar *fMap, int cols, int rows, int ANCHOR_TH, int K);
#ifdef USE_CPUDP
void DouglasPeucker(const VECTOR_H<POINT> &edge, VECTOR_H<POINT> &line, float epsilon);
#endif
#ifdef USE_GPUDP
__global__ void DouglasPeucker(POINT **edge_seg_d, int *edge_offset_d, int edge_seg_len, POINT *stack, bool *flags_d, float epsilon);
#endif

#endif