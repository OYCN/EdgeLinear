add_executable(
    UT-getFlag

    ../pipeline/BlockGetFlag.cu

    getFlag.cpp
)

target_compile_definitions(UT-getFlag PRIVATE LINUX TIMING)

add_executable(
    UT-ED

    ../pipeline/BlockGetFlag.cu
    ../pipeline/BlockConnect.cpp

    edtest.cpp
)

target_compile_definitions(UT-ED PRIVATE LINUX TIMING)

add_executable(
    UT-Linear

    ../pipeline/BlockLinear.cu

    BlockDP.cu

    linear.cpp
)

target_compile_definitions(UT-Linear PRIVATE LINUX TIMING)

add_executable(
    UT-GenBase

    ../pipeline/BlockGetFlag.cu
    ../pipeline/BlockConnect.cpp
    ../pipeline/BlockLinear.cu

    gen_base.cpp
)


add_executable(
    UT-random

    random.cpp
)

add_executable(
    UT-info

    ../pipeline/BlockGetFlag.cu
    ../pipeline/BlockConnect.cpp

    edges_info.cpp
)