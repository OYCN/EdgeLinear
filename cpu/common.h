#ifndef _INC_COMMON_H
#define _INC_COMMON_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include "Timer.h"
#include "cuda.h"
#include "cuda_runtime.h"

#define VECTOR_H std::vector
#define POINT mygpu::Point

#include "myStruct.h"

#define ERROR(x) {printf( "%s in %s at line %d\n", (x), __FILE__, __LINE__ );exit( EXIT_FAILURE );}
typedef unsigned char uchar;

#endif // _INC_COMMON_H