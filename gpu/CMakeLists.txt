
set(COMMON
    Config.cpp
    EdgeDrawing.cu
    main.cpp
)

add_executable(
    EDmain-gpu

    ${COMMON}
)

add_executable(
    EDmain-gpu-cv

    ${COMMON}
)

target_compile_definitions(EDmain-gpu-cv PRIVATE USE_OPENCV_GPU)

add_executable(
    EDLSmain-gpu

    LinearSum.cu
    ${COMMON}
)

target_compile_definitions(EDLSmain-gpu PRIVATE _LS)

add_executable(
    EDDPmain-gpu

    DouglasPeucker.cu
    ${COMMON}
)

target_compile_definitions(EDDPmain-gpu PRIVATE _DP)