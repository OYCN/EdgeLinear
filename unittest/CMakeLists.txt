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

    linear.cpp
)

target_compile_definitions(UT-Linear PRIVATE LINUX TIMING)
