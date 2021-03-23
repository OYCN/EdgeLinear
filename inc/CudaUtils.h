#ifndef _INC_CUDAUTILS_H
#define _INC_CUDAUTILS_H

#include "cuda.h"
#include "cuda_runtime.h"
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

#endif // _INC_CUDAUTILS_H