add_executable(
    Pipeline

    BlockGetFlag.cu
    BlockConnect.cpp
    BlockLinear.cu

    BlockPipeline.cpp
    BlockWarper.cpp

    main.cpp
)