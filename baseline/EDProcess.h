#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

#define Y_DELETE 1
#define N_DELETE 2

//#define OP_SOBEL
//#define MY_PREWITT
#define MY_SOBEL

enum CURR_DIR
{
	LEFT,  LEFT_UP,   LEFT_DOWN,
	RIGHT, RIGHT_UP,  RIGHT_DOWN,
	UP,    UP_LEFT,   UP_RIGHT,
	DOWN,  DOWN_LEFT, DOWN_RIGHT
};

void getEdgeInfor(cv::Mat blurImg, cv::Mat gMap, cv::Mat dMap);         
cv::Mat getAnchorPoint(cv::Mat gMap, cv::Mat dMap, int K, int ANCHOR_TH);
cv::Mat smartConnecting(cv::Mat gMap, cv::Mat dMap, cv::Mat aMap, std::vector<std::vector<cv::Point>>& edge_s);

class ED
{
public:
	~ED();
	cv::Mat Process(cv::Mat& blurImg, std::vector<std::vector<cv::Point>>& edge_s);

private:
	void initinal(cv::Mat& blurImg, const int k = 2, const int anchor_th = 6);
	cv::Mat aMap, gMap, dMap;
	int K, ANCHOR_TH;
};


