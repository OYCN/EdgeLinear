#ifndef _INC_MAINCLASS_H
#define _INC_MAINCLASS_H

#include <common.h>

class MainClass
{
	public:
		
		MainClass(int _rows=0, int _cols=0, int _anchor_th=6, int _k=2);
		~MainClass();
        // 设置ED环节阈值
		void setTH(int value);
		int getTH();
        // 边缘提取
		cv::Mat getEdges(cv::Mat& src, POINT *&edge_seg, int *&edge_seg_offset, int &edge_seg_len);
		// 多边形化
		void toLines(bool *&flag_in);
	private:
	// ========== ED ==========
        // 梯度图、flag图、预处理图
		uchar *gMapd, *fMapd, *blurd;
		uchar *gMaph, *fMaph;
        // 边缘图
		cv::Mat eMaph;
        // 灰度图
		cv::Mat grayImg;
        //  模糊图
		cv::Mat blurImg;
        // 列、行
		int cols, rows;
        // 稀疏度、阈值
		int k, anchor_th;
        // GPU划分
		dim3 dimBlock_ED;
		dim3 dimGrid_ED;
		dim3 dimBlock_DP;
		dim3 dimGrid_DP;
        // 用于储存锚点连接函数中临时数据
		POINT *edge_smart;
		int edge_smart_idx;
	// ======== ED输出 =========
		POINT *edge_set;	// 全部边缘点集合
		int *edge_offset;	// 每个边缘的偏移
		int edge_offset_len;	// 偏移数组长度
	// ========== 多边形化 ==========
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

#endif // _INC_MAINCLASS_H