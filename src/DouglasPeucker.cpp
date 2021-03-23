#include "DouglasPeucker.h"

DouglasPeucker::DouglasPeucker(int _rows, int _cols, float _th)
    :rows(_rows), cols(_cols), th(_th)
{

}

DouglasPeucker::~DouglasPeucker()
{
    delete[] flagh;
}