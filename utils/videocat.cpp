#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    if(argc != 5) {
        cout << argv[0] << " <input_dir1> <input_dir2> <output_video_name> <fps>" << endl;
		return 0;
    }

	cv::VideoCapture capture1;
    capture1.open(argv[1]);
    if(!capture1.isOpened())
	{
		printf("[%s][%d]could not load video data...\n",__FUNCTION__,__LINE__);
		return -1;
	}
    int height1 = capture1.get(CV_CAP_PROP_FRAME_HEIGHT);
    int width1 = capture1.get(CV_CAP_PROP_FRAME_WIDTH);

	cv::VideoCapture capture2;
    capture2.open(argv[2]);
    if(!capture2.isOpened())
	{
		printf("[%s][%d]could not load video data...\n",__FUNCTION__,__LINE__);
		return -1;
	}
    int height2 = capture2.get(CV_CAP_PROP_FRAME_HEIGHT);
    int width2 = capture2.get(CV_CAP_PROP_FRAME_WIDTH);

	int fps = atoi(argv[4]);

	if(width1 != width2) return -1;
	if(height1 != height2) return -1;

	VideoWriter video(argv[3], VideoWriter::fourcc('M','J','P','G'), fps, Size(width1, height1));

	Mat image;
	size_t idx = 0;
	while (capture1.read(image))
	{
		video << image;
		cout << "dealing " << idx++ << " at " << argv[1] << std::endl;
	}
	while (capture2.read(image))
	{
		video << image;
		cout << "dealing " << idx++ << " at " << argv[2] << std::endl;
	}
	
	cout << "\ngen: " << argv[3] << endl;
}