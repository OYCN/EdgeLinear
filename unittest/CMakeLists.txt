add_executable(
    UT-getFlag

    BlockGetFlag.cu

    getFlag.cpp
)

target_compile_definitions(UT-getFlag PRIVATE LINUX TIMING)

add_executable(
    UT-getFlag-mod

    BlockGetFlag_mod.cu

    getFlag_mod.cpp
)

target_compile_definitions(UT-getFlag-mod PRIVATE LINUX TIMING)

add_executable(
    UT-ED

    BlockGetFlag.cu
    BlockConnect.cpp

    edtest.cpp
)

target_compile_definitions(UT-ED PRIVATE LINUX TIMING)

add_executable(
    UT-Linear

    BlockLinear.cu

    linear.cpp
)

target_compile_definitions(UT-Linear PRIVATE LINUX TIMING)
