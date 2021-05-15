project(EdgeLinear LANGUAGES CXX CUDA)

cmake_minimum_required(VERSION 3.5)

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.2)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
link_libraries(${OpenCV_LIBS})

include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

add_compile_definitions(USE_OPENCV_GPU)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

add_subdirectory(baseline)
add_subdirectory(utils)
add_subdirectory(cpu)
add_subdirectory(gpu)
add_subdirectory(pipeline)
add_subdirectory(unittest)