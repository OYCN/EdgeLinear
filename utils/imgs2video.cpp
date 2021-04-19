#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    if(argc != 6) {
        cout << argv[0] << " <input_dir> <output_video_name> <width> <height> <fps>" << endl;
		return 0;
    }
	int width = atoi(argv[3]);
	int height = atoi(argv[4]);
	int fps = atoi(argv[5]);

	VideoWriter video(argv[2], VideoWriter::fourcc('M','J','P','G'), fps, Size(width, height));

	String img_path = argv[1];
	vector<String> img;

	glob(img_path, img, false);

	size_t count = img.size();
	for (size_t i = 0; i < count; i++)
	{
        // cout << "\rdealing " << img[i] << "       ";
		Mat image = imread(img[i]);
		if (!image.empty())
		{
			// resize(image, image, Size(width, height));
			video << image;
			cout << "\rdealing " << img[i] << "       ";
		}
	}
	cout << "\ngen: " << argv[2] << endl;
}