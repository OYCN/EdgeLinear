#include "EDProcess.h"


ED::~ED()
{
	aMap.release();
	gMap.release();
	dMap.release(); 
}

void ED::initinal(cv::Mat& blurImg, const int k , const int anchor_th)
{
	K = k;
	ANCHOR_TH = anchor_th;
	aMap = cv::Mat::zeros(blurImg.size(), blurImg.type());
	gMap = cv::Mat::zeros(blurImg.size(), blurImg.type());
	dMap = cv::Mat::zeros(blurImg.size(), blurImg.type());
}

cv::Mat ED::Process(cv::Mat& blurImg, std::vector<std::vector<cv::Point>>& edge_seg)
{
	initinal(blurImg);
	getEdgeInfor(blurImg, gMap, dMap);         
	aMap = getAnchorPoint(gMap, dMap,  K, ANCHOR_TH);
	cv::Mat eMap = smartConnecting(gMap, dMap, aMap, edge_seg);
	return eMap;
}

static int goLeft(int& x, int& y, cv::Mat eMap, cv::Mat gMap, cv::Mat dMap, CURR_DIR& cur_dir);
static int goRight(int& x, int& y, cv::Mat eMap, cv::Mat gMap, cv::Mat dMap, CURR_DIR& cur_dir);
static int goUp(int& x, int& y, cv::Mat eMap, cv::Mat gMap, cv::Mat dMap, CURR_DIR& cur_dir);
static int goDown(int& x, int& y, cv::Mat eMap, cv::Mat gMap,cv::Mat dMap,  CURR_DIR& cur_dir);
static void goMove(int x, int y, cv::Mat gMap, cv::Mat eMap, cv::Mat dMap, CURR_DIR cur_dir, std::vector<cv::Point>& edge_s);

// get gradient map, direction map
void getEdgeInfor(cv::Mat blurImg, cv::Mat gMap, cv::Mat dMap)
{
#if 0
	int w = blurImg.cols, h = blurImg.rows, s = blurImg.step;
	for (int j = 1; j < h-1; ++j)
	{
		uchar* blur_str = blurImg.ptr<uchar>(j);
		uchar* grad_str = gMap.ptr<uchar>(j);
		uchar* dMap_str = dMap.ptr<uchar>(j);
		for (int i = 1; i< w-1; ++i)
		{	
			uchar dx = abs((blur_str[i+1] - blur_str[i]) + (blur_str[i+s+1] - blur_str[i+s]));
			uchar dy = abs((blur_str[i+s] - blur_str[i]) + (blur_str[i+s+1] - blur_str[i+1]));

			grad_str[i] = dx + dy;

			// 255 -- vertical   0 -- horizonal
			if(dx > dy) dMap_str[i] = 255;
			else  dMap_str[i] = 0; 
		}
	}
#endif

#ifdef OP_SOBEL  // opencv Sobel
    cv::Mat grad_x(blurImg.size(), blurImg.type());
	cv::Mat grad_y(blurImg.size(), blurImg.type());

	cv::Sobel(blurImg, grad_x, CV_32F, 1, 0); 
	convertScaleAbs(grad_x, grad_x);  
	cv::Sobel( blurImg, grad_y, CV_32F, 0, 1);     
	convertScaleAbs(grad_y, grad_y);  
	addWeighted( grad_x, 0.5, grad_y, 0.5, 0, gMap); 
	// 255 -- vertical   0 -- horizonal
	for(int j = 0; j < blurImg.rows; j++){
		uchar* dMap_str = dMap.ptr<uchar>(j);
		uchar* gg_x = grad_x.ptr<uchar>(j);
		uchar* gg_y = grad_y.ptr<uchar>(j);
		for(int i = 0; i < blurImg.cols; i++){
			if(gg_x[i] > gg_y[i]) dMap_str[i] = 255;
			else  dMap_str[i] = 0; 
		}
	}
	grad_x.release();
	grad_y.release();
#endif
#ifdef MY_SOBEL // my Sobel
	for(int j = 1; j < blurImg.rows-1; j++)
	{
		uchar* blur_str = (uchar*)(blurImg.data + j * blurImg.step);
		uchar* gMap_str = (uchar*)(gMap.data + j * blurImg.step);
		uchar* dMap_str = dMap.ptr<uchar>(j);
		for(int i = 1; i < blurImg.cols-1; i++)
		{
			int dx = abs(blur_str[i-blurImg.step+1] + 2*blur_str[i+1] + blur_str[i+blurImg.step+1] - 
				        (blur_str[i-blurImg.step-1] + 2*blur_str[i-1] + blur_str[i+blurImg.step-1]));
		    int dy = abs(blur_str[i-blurImg.step-1] + 2*blur_str[i-blurImg.step] + blur_str[i-blurImg.step+1] -
				        (blur_str[i+blurImg.step-1] + 2*blur_str[i+blurImg.step] + blur_str[i+blurImg.step+1]));
	
			float val = 0.5f*dx + 0.5f*dy;
			if(val > 255) val = 255.0f;
			//else if(val < 36) val = 0.0f;

			gMap_str[i] = (int)(val);

			// 255 -- vertical   0 -- horizonal
			if(dx > dy) dMap_str[i] = 255;
			else dMap_str[i] = 0; 
		}
	}
	//cv::imshow("gMap", gMap);
	//cv::Mat bw;
	//cv::threshold(gMap, bw, 20, 255, cv::THRESH_BINARY); 
	//cv::imshow("bw", bw);
	//cv::imwrite("2.jpg", bw);
	//cv::waitKey(0);
#endif
#ifdef MY_PREWITT  // my Prewitt
	for(int j = 1; j < blurImg.rows-1; j++)
	{
		uchar* b_str = (uchar*)(blurImg.data + j * blurImg.step);
		uchar* gMap_str = (uchar*)(gMap.data + j * blurImg.step);
		uchar* dMap_str = dMap.ptr<uchar>(j);
		for(int i = 1; i < blurImg.cols-1; i++)
		{
			int dx = abs(b_str[i+1+blurImg.step] + UNIT_RESULTb_str[i+1] + b_str[i+1-blurImg.step] -
				     (b_str[i-1+blurImg.step] + b_str[i-1] + b_str[i-1-blurImg.step]));
			int dy = abs(b_str[i-1-blurImg.step] + b_str[i-blurImg.step] + b_str[i+1-blurImg.step] -
				     (b_str[i-1+blurImg.step] + b_str[i+blurImg.step] + b_str[i+1+blurImg.step]));

			float val = 0.7f*dx + 0.7f*dy;
			if(val > 255) val = 255.0f;

			gMap_str[i] = (int)val;

			// 255 -- vertical   0 -- horizonal
			if(dx > dy) dMap_str[i] = 255;
			else dMap_str[i] = 0; 
		}
	}
	//cv::imshow("gMap", gMap);
#endif
}

// get anchor points
cv::Mat getAnchorPoint(cv::Mat gMap, cv::Mat dMap, int K, int ANCHOR_TH)
{
	// anchor points map
	cv::Mat aMap = cv::Mat::zeros(gMap.size(), gMap.type()); 
	int h = gMap.rows, w = gMap.cols, s = gMap.step;
	///------------------------------------------------------
	cv::Mat am = cv::Mat::zeros(aMap.size(), CV_8UC3);
	am = ~am;

	for(int j = 1;j < h-1; j+=K)
	{
		uchar* gMap_str = gMap.ptr<uchar>(j);
		uchar* dMap_str = dMap.ptr<uchar>(j);
		uchar* aMap_str = aMap.ptr<uchar>(j);
		for(int i= 1; i < w-1; i+=K)
		{
			if(!dMap_str[i])  //  horizonal
			{
				if((gMap_str[i] - gMap_str[i-s] >= ANCHOR_TH) && 
				  (gMap_str[i] - gMap_str[i+s] >= ANCHOR_TH))
				{
					aMap_str[i] = 255;
					//cv::circle(am, cv::Point(i, j), 0, cv::Scalar(0, 0, 255), 2, 8, 0);
				}
			}
			else              //   vertical
			{
				if((gMap_str[i] - gMap_str[i-1] >= ANCHOR_TH) &&
				  (gMap_str[i] - gMap_str[i+1] >= ANCHOR_TH))
				{
					aMap_str[i] = 255;
					//cv::circle(am, cv::Point(i, j), 0, cv::Scalar(0, 0, 255), 2, 8, 0);
				}
			}
		}
	}


	//imshow("anchor map", am);
	//cv::imwrite("3.jpg", am);
	return aMap;
}
// connecting 
cv::Mat smartConnecting(cv::Mat gMap, cv::Mat dMap, cv::Mat aMap, std::vector<std::vector<cv::Point>>& edge_s)
{
	cv::Mat eMap = cv::Mat::zeros(gMap.size(), gMap.type());

	#ifdef UNIT_RESULT
	// unit result 
	for(int i = 0; i < gMap.rows; i++)
	for(int j = 0; j < gMap.cols; j++)
	{
		if(i == 0 || j == 0 || i == (gMap.rows - 1) || j == (gMap.cols - 1))
		{
			eMap.data[i * gMap.cols + j] = 255;
		}
	}
	#endif // UNIT_RESULT

	int h = gMap.rows, w = gMap.cols;
	std::vector<cv::Point>edges;

	for(int j = 1;j < h-1; j++)
	{
		uchar* gMap_str = gMap.ptr<uchar>(j);
		uchar* dMap_str = dMap.ptr<uchar>(j);
		uchar* eMap_str = eMap.ptr<uchar>(j);
		uchar* aMap_str = aMap.ptr<uchar>(j);
		for(int i= 1; i < w-1; i++)
		{
			// ��ÿ��δ���� anchor point ����
			if(!aMap_str[i] || eMap_str[i])
				continue;

			std::vector<cv::Point>().swap(edges);

			CURR_DIR current_dir;

			edges.push_back(cv::Point(i, j)); // start point

			// �ж�����  horizonal - left   vertical - up
			current_dir = dMap_str[i] ? UP : LEFT;  
			goMove(i, j, gMap, eMap, dMap, current_dir, edges);

			if(!edges.empty()) reverse(edges.begin(), edges.end());

			// �ж�����  horizonal - right  vertical - down
			current_dir = dMap_str[i] ? DOWN : RIGHT;
			goMove(i, j, gMap, eMap, dMap, current_dir, edges);

			edge_s.push_back(edges);
 		}
	}
	return eMap;
}

static void goMove(int x, int y, cv::Mat gMap, cv::Mat eMap, cv::Mat dMap, CURR_DIR cur_dir, std::vector<cv::Point>& edge_s)
{
	int flag = 0;
	int h = gMap.rows, w = gMap.cols, s = gMap.step;
	#ifdef UNIT_RESULT
	if(!(y == 0 || x == 0 || y == (gMap.rows - 1) || x == (gMap.cols - 1)))
	#endif // UNIT_RESULT
	eMap.data[y*s+x] = 0; // for the second scan

	while(x>0 && x <w-1 && y>0 && y<h-1 && !eMap.data[y*s+x] && gMap.data[y*s+x])
	{
		if(flag == Y_DELETE){
			break;
		}
		if(!dMap.data[y*s+x])      // horizonal scanning
		{
			switch(cur_dir)
			{
			case LEFT: case LEFT_UP: case LEFT_DOWN: case UP_LEFT: case DOWN_LEFT:
				flag = goLeft(x, y, eMap, gMap, dMap, cur_dir);
				if(flag == Y_DELETE) {eMap.data[y*s+x] = 0;}
				else
					edge_s.push_back(cv::Point(x,y));
				break;
			case RIGHT: case RIGHT_UP: case RIGHT_DOWN: case UP_RIGHT: case DOWN_RIGHT:
				flag = goRight(x, y, eMap, gMap, dMap, cur_dir);
				if(flag == Y_DELETE) {eMap.data[y*s+x] = 0;}
				else
					edge_s.push_back(cv::Point(x,y));
				break;
			default:
				{
					x = -1;
					y = -1;
				}
				break;
			}
		}
		else                        //  vertical scanning
		{
			switch(cur_dir)
			{
			case UP: case UP_LEFT: case UP_RIGHT: case LEFT_UP: case RIGHT_UP:
				flag = goUp(x, y, eMap, gMap, dMap, cur_dir);
				if(flag == Y_DELETE) {
					eMap.data[y*s+x] = 0;
				} 
				else
					edge_s.push_back(cv::Point(x,y));
				break;
			case DOWN: case DOWN_LEFT: case DOWN_RIGHT: case LEFT_DOWN: case RIGHT_DOWN:
				flag = goDown(x, y, eMap, gMap, dMap, cur_dir);
				if(flag == Y_DELETE) {
					eMap.data[y*s+x] = 0;
				}
				else 
					edge_s.push_back(cv::Point(x,y));
				break;
			default:
				{
					x = -1;
					y = -1;
				}
				break;
			}
		}
	}
}
static int goLeft(int& x, int& y, cv::Mat eMap, cv::Mat gMap, cv::Mat dMap, CURR_DIR& cur_dir)
{
	int h = gMap.rows, w = gMap.cols, s = gMap.step;
	eMap.data[y*s+x] = 255;
	//eMap.data[y*s+x] = gMap.data[y*s+x];

	uchar left = gMap.data[y*s+x-1];
	uchar left_up = gMap.data[(y-1)*s+(x-1)];
	uchar left_down = gMap.data[(y+1)*s+(x-1)];

	uchar eLeft = eMap.data[y*s+x-1];
	uchar eLeft_up = eMap.data[(y-1)*s+(x-1)];
	uchar eLeft_down = eMap.data[(y+1)*s+(x-1)];
	
	// �������ǵ��������� 
	if(eLeft && eLeft_up){
	    return Y_DELETE;
	}
	if(eLeft && eLeft_down){
		return Y_DELETE;
	}
	if(eLeft_up && eLeft_down){
		return Y_DELETE;
	}
	if(eLeft || eLeft_up || eLeft_down)
		return N_DELETE;

	if(left_up > left && left_up > left_down)
	{
		x = x - 1;
		y = y - 1;
		cur_dir = LEFT_UP;
	}
	else if(left_down > left && left_down > left_up)
	{
		x = x - 1;
		y = y + 1;
		cur_dir = LEFT_DOWN;
	}
	else
	{
		x = x - 1;
		cur_dir = LEFT;
	}
	return N_DELETE;
}
static int goRight(int& x, int& y, cv::Mat eMap, cv::Mat gMap, cv::Mat dMap, CURR_DIR& cur_dir)
{
	int h = gMap.rows, w = gMap.cols, s = gMap.step;
	eMap.data[y*s+x] = 255;
	//eMap.data[y*s+x] = gMap.data[y*s+x];

	uchar right = gMap.data[y*s+x+1];
	uchar right_up = gMap.data[(y-1)*s+(x+1)];
	uchar right_down = gMap.data[(y+1)*s+(x+1)];

	uchar eRight = eMap.data[y*s+x+1];
	uchar eRight_up = eMap.data[(y-1)*s+(x+1)];
	uchar eRight_down = eMap.data[(y+1)*s+(x+1)];

	if(eRight && eRight_up){
		return Y_DELETE;
	}
	if(eRight && eRight_down){
		return Y_DELETE;
	}
	if(eRight_up && eRight_down){
		return Y_DELETE;
	}
	if(eRight || eRight_up || eRight_down)
		return N_DELETE;

	if(right_up > right && right_up > right_down)
	{
		x = x + 1;
		y = y - 1;
		cur_dir = RIGHT_UP;
	}
	else if(right_down > right && right_down > right_up)
	{
		x = x + 1;
		y = y + 1;
		cur_dir = RIGHT_DOWN;
	}
	else 
	{
		x = x + 1;
		cur_dir = RIGHT;
	}
	return N_DELETE;
}
static int goUp(int& x, int& y, cv::Mat eMap, cv::Mat gMap, cv::Mat dMap, CURR_DIR& cur_dir)
{
	int h = gMap.rows, w = gMap.cols, s = gMap.step;
	eMap.data[y*s+x] = 255;
	//eMap.data[y*s+x] = gMap.data[y*s+x];
		
	uchar up = gMap.data[(y-1)*s+x];
	uchar up_left = gMap.data[(y-1)*s+(x-1)];
	uchar up_right = gMap.data[(y-1)*s+(x+1)];

	uchar eUp = eMap.data[(y-1)*s+x];
	uchar eUp_left = eMap.data[(y-1)*s+(x-1)];
	uchar eUp_right = eMap.data[(y-1)*s+(x+1)];

	if(eUp && eUp_left){
		return Y_DELETE;
	}
	if(eUp && eUp_right){
		return Y_DELETE;
	}
	if(eUp_left && eUp_right){
		return Y_DELETE;
	}
	if(eUp || eUp_left || eUp_right)
		return N_DELETE;

	if(up_left > up && up_left > up_right)
	{
		x = x - 1;
		y = y - 1;
		cur_dir = UP_LEFT;
	}
	else if(up_right > up && up_right > up_left)
	{
		x = x + 1;
		y = y - 1;
		cur_dir = UP_RIGHT;
	}
	else
	{
		y = y - 1;
		cur_dir = UP;
	}
	return N_DELETE;
}

static int goDown(int& x, int& y, cv::Mat eMap, cv::Mat gMap, cv::Mat dMap, CURR_DIR& cur_dir)
{
	int h = gMap.rows, w = gMap.cols, s = gMap.step;
	eMap.data[y*s+x] = 255;
	//eMap.data[y*s+x] = gMap.data[y*s+x];
		
	uchar down = gMap.data[(y+1)*s+x];
	uchar down_left = gMap.data[(y+1)*s+(x-1)];
	uchar down_right = gMap.data[(y+1)*s+(x+1)];

	uchar eDown = eMap.data[(y+1)*s+x];
	uchar eDown_left = eMap.data[(y+1)*s+(x-1)];
	uchar eDown_right = eMap.data[(y+1)*s+(x+1)];

	if(eDown && eDown_left){
		return Y_DELETE;
	}
	if(eDown && eDown_right){
		return Y_DELETE;
	}
	if(eDown_left && eDown_right){
		return Y_DELETE;
	}
	if(eDown || eDown_left || eDown_right)
		return N_DELETE;

	if(down_left > down && down_left > down_right)
	{
		x = x - 1;
		y = y + 1;
		cur_dir = DOWN_LEFT;
	}
	else if(down_right > down && down_right > down_left)
	{
		x = x + 1;
		y = y + 1;
		cur_dir = DOWN_RIGHT;
	}
	else
	{
		y = y + 1;
		cur_dir = DOWN;
	}
	return N_DELETE;
}