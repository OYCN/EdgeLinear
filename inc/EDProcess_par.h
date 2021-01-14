#ifndef _EDPROCESS_H_
#define _EDPROCESS_H_

// 包含文件
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <iostream>
#include <vector>
#include "cuda.h"
#include "cuda_runtime.h"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"

// 编译配置
#define DEFIMG "./img/7.jpg"	// 默认图片
#define USE_CHECK	// 运行GPUDP与CPUDP并进行结果比较
#define SHOW_IMG	// 是否显示图片
#define TIM_GPUDP	// 是否显示GPU DP的时间
// #define DEBUG
// #define USE_OPENCV_GPU

#ifdef USE_CHECK
#define USE_CPUDP
#define USE_GPUDP
#endif

// 类型定义
#define VECTOR_H thrust::host_vector
#define VECTOR_D thrust::device_vector
#define POINT mygpu::Point

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

class Main
{
	public:
		
		Main(int _rows=0, int _cols=0, int _anchor_th=6, int _k=2);
		~Main();
		cv::Mat Process(cv::Mat& src, POINT *&edge_seg, int *&edge_seg_offset, int &edge_seg_len);
		int runDP(VECTOR_H<VECTOR_H<POINT>> &line_all_gpu);
	private:
	// ========== ED ==========
		uchar *gMapd, *fMapd, *blurd;
		uchar *gMaph, *fMaph;
		cv::Mat eMaph;
		#ifndef USE_OPENCV_GPU
		cv::Mat grayImg;
		cv::Mat blurImg;
		#endif
		#ifdef USE_OPENCV_GPU
		cv::cuda::GpuMat src_d;
		cv::cuda::GpuMat grayImg;
		cv::cuda::GpuMat blurImg;
		#endif
		int cols, rows;
		int k, anchor_th;

		dim3 dimBlock;
		dim3 dimGrid;
		dim3 dimGridOld;

		POINT *edge_smart;	// 用于储存smart函数中临时数据
		int edge_smart_idx;
	// ======== ED&PD =========	// 初始化于 ED
		POINT *edge_set;	// 全部边缘点集合
		int *edge_offset;	// 每个边缘的偏移
		int edge_offset_len;	// 偏移数组长度
	// ========== PD ==========
		POINT *edge_set_d;
		int *edge_offset_d;
		bool *flags_h;
		bool *flags_d;
		POINT *stack_d;

		void _InitED();
		void _FreeED();
		void _InitPD();
		void _FreePD();
		void PerProcED(cv::Mat &src);
		void PerProcDP();
		void goMove(int x, int y, uchar mydir, POINT *edge_s, int &idx);
		cv::Mat smartConnecting();
		
};

// 函数定义
__global__ void kernelC(uchar *blur, uchar * gMap, uchar *fMap, int cols, int rows, int ANCHOR_TH, int K);
__global__ void kernelDP(POINT *edge_set_d, int *edge_offset_d, int edge_offset_len, POINT *stack_d, bool *flags_d, float epsilon);
void DouglasPeucker(const VECTOR_H<POINT> &edge, VECTOR_H<POINT> &line, float epsilon);

#endif