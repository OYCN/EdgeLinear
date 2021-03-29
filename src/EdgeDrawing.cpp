#include "EdgeDrawing.h"
// #include "Timer.h"

#define IDX(x, y) [(x) + (y)*cols]

void kernelC(uchar *src, uchar * gMap, uchar *fMap, int cols, int rows, int ANCHOR_TH, int K);

EdgeDrawing::EdgeDrawing(int _rows, int _cols, float _th, int _k)
    :rows(_rows), cols(_cols), th(_th), k(_k)
{
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
    // TDEF(part)
    // TDEF(init)
    // TDEF(com)
    // TDEF(cpu)

    // TSTART(part)
    // TSTART(init)
    initLoop();
    
    cv::cvtColor(_src, srch, CV_RGB2GRAY);
	cv::GaussianBlur(srch, srch, cv::Size(5, 5), 1, 0);
    // TEND(init)
    // TSTART(com)
    kernelC(srch.data, gMaph, fMaph, cols, rows, th, k);
    // TEND(com)
    // TEND(part)
	// cv::Mat fMap(rows ,cols, CV_8UC1, (unsigned char*)(fMaph));
	// cv::imshow("fMap", fMap);
    // TSTART(cpu)
    smartConnecting();
    // TEND(cpu)

    // TPRINTMS(part, "part:")
    // TPRINTMS(init, "\tinit:")
    // TPRINTMS(com, "\tcom:")
    // TPRINTMS(cpu, "cpu:")

	return &EDoutput;
}

void kernelC(uchar *src, uchar * gMap, uchar *fMap, int cols, int rows, int ANCHOR_TH, int K)
{
    int dx = 0;
    int dy = 0;
    float val = 0;

    // 求梯度
    for(int y = 1; y < (rows - 1); y++)
    for(int x = 1; x < (cols - 1); x++)
    {
        dx = src IDX(x+1,y-1)
            + 2 * src IDX(x+1,y)
            + src IDX(x+1,y+1)
            - src IDX(x-1,y-1)
            - 2 * src IDX(x-1,y)
            - src IDX(x-1,y+1);
        dx = abs(dx);

        dy = src IDX(x-1,y-1)
            + 2 * src IDX(x,y-1)
            + src IDX(x+1,y-1)
            - src IDX(x-1,y+1)
            - 2 * src IDX(x,y+1)
            - src IDX(x+1,y+1);
        dy = abs(dy);

        val = 0.5f*dx + 0.5f*dy;
        if (val > 255) val = 255.0f;
        gMap IDX(x, y) = (uchar)val;

		if(dx > dy)
		{
			fMap IDX(x, y) = 0x80;
		}
		else
		{
			fMap IDX(x, y) = 0;
		}
    }

    // 求flag
    for(int y = 1; y < (rows - 1); y++)
    for(int x = 1; x < (cols - 1); x++)
    {
        if(fMap IDX(x, y) != 0)
        {
            if((gMap IDX(x, y) - gMap IDX(x-1, y)) >= ANCHOR_TH
                && (gMap IDX(x, y) - gMap IDX(x+1, y)) >= ANCHOR_TH
                && ((x-1) % K) == 0 && ((y-1) % K) == 0)
            {
                fMap IDX(x, y) |= 0x40;
            }
        }
        else
        {
            if((gMap IDX(x, y) - gMap IDX(x, y-1)) >= ANCHOR_TH
                && (gMap IDX(x, y) - gMap IDX(x, y+1)) >= ANCHOR_TH
                && ((x-1) % K) == 0 && ((y-1) % K) == 0)
            {
                fMap IDX(x, y) |= 0x40;
            }
        }
    }

}