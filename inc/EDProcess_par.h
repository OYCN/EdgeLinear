#ifndef _EDPROCESS_H_
#define _EDPROCESS_H_

// 包含文件
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "cuda.h"
#include "cuda_runtime.h"

// 编译配置
#define DEFIMG "./img/2.jpg"	// 默认图片
#define USE_DP	// 使用DP多边形化算法
// #define USE_CUR	// 使用协方差矩阵平滑算法
// #define USE_UNIMEM	// 使用统一内存寻址
// #define JUST_ED	//仅测试ED程序
#define SHOW_IMG	// 是否显示图片
// #define TIM_PROC	// 是否显示ED proc的时间

// 配置初始化
#ifndef USE_DP
#ifndef USE_CUR
#define JUST_ED
#endif
#endif

// 类型定义
#define VECTOR_H std::vector
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
class ED
{
public:
	~ED();
	ED();
	cv::Mat Process(cv::Mat& blurImg, VECTOR_H<VECTOR_H<cv::Point>>& edge_s, const int anchor_th = 6, const int k = 2);

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

// 函数定义
cv::Mat smartConnecting(uchar *gMap, uchar *fMap, int rows, int cols, VECTOR_H<VECTOR_H<cv::Point>>& edge_s);
// __global__ void kernelA(uchar *blur, uchar* gMap, uchar* dMap, int cols, int rows);
// __global__ void kernelB(uchar * gMap, uchar * dMap, uchar *aMap, int cols, int rows, int ANCHOR_TH, int K);
__global__ void kernelC(uchar *blur, uchar * gMap, uchar *fMap, int cols, int rows, int ANCHOR_TH, int K);
float cur(cv::Point edge[]);
void getLine(const VECTOR_H<cv::Point> &edge, VECTOR_H<VECTOR_H<cv::Point>> &line);
VECTOR_H<VECTOR_H<cv::Point>> orgline(const VECTOR_H<VECTOR_H<cv::Point>> &edge_seg);
void DouglasPeucker(const VECTOR_H<cv::Point> &edge, VECTOR_H<cv::Point> &line, float epsilon);
namespace mygpu
{
	void approxPolyDP( cv::InputArray _curve, cv::OutputArray _approxCurve, double epsilon, bool closed );
} // namespace mygpu


#endif