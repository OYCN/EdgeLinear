INCDIR = pipeline
CXX = g++
CUDAXX = nvcc
CUDA = /usr/local/cuda/include
OPENCVENV = $(shell pkg-config opencv --cflags --libs)
DEF = -DLINUX -DTIMING -DUSE_OPENCV_GPU
SHELL=/bin/bash
INC = -I$(INCDIR) -I$(CUDA)
FLAG = -g
CUDA_FLAG = -Xptxas="-v"
# LIB = -L/usr/local/cuda/lib64/

.PHONY : all clean QueryDev

all: BlockGetFlag.o \
	 BlockConnect.o \
	 BlockLinear.o

BlockGetFlag.o: pipeline/BlockGetFlag.cu pipeline/BlockGetFlag.h
	@echo "compile tmp/BlockGetFlag.o"
	@$(CUDAXX) -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV) $(CUDA_FLAG)
	@rm $@

BlockLinear.o: pipeline/BlockLinear.cu pipeline/BlockLinear.h
	@echo "compile tmp/BlockLinear.o"
	@$(CUDAXX) -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV) $(CUDA_FLAG)
	@rm $@
