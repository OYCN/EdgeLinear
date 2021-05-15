
set(COMMON
    Config.cpp
    EdgeDrawing.cpp
    main.cpp
)

add_executable(
    EDmain-cpu

    ${COMMON}
)

add_executable(
    EDLSmain-cpu

    LinearSum.cpp
    ${COMMON}
)

target_compile_definitions(EDLSmain-cpu PRIVATE _LS)

add_executable(
    EDDPmain-cpu

    DouglasPeucker.cpp
    ${COMMON}
)

target_compile_definitions(EDDPmain-cpu PRIVATE _DP)