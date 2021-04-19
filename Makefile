INCDIR = inc
CXX = g++
CUDAXX = nvcc
CUDA = /usr/local/cuda/include
OPENCVENV = $(shell pkg-config opencv --cflags --libs)
INC_FILE = $(wildcard $(INCDIR)/*.h)
DEF = -DLINUX -DTIMING -DUSE_OPENCV_GPU
SHELL=/bin/bash
INC = -I$(INCDIR) -I$(CUDA)
FLAG = -g -O3
CUDA_FLAG = -Xptxas="-v"
# LIB = -L/usr/local/cuda/lib64/

.PHONY : all clean QueryDev

all: bin/EDmain-gpu\
	bin/EDmain-cpu\
	bin/EDDPmain-gpu\
	bin/EDDPmain-cpu\
	bin/EDLSmain-gpu\
	bin/EDLSmain-cpu\
	bin/EDbaseline\
	bin/EDDPbaseline\
	bin/EDLSbaseline\
	bin/imgs2video
	
clean:
	@echo "clean"
	@rm -f bin/ED*
	@rm -f tmp/*

# utils Part
tmp/Config.o: src/Config.cpp $(INC_FILE) 
	@if [ ! -e tmp ];then mkdir tmp; fi
	@echo "compile tmp/Config.o"
	@$(CXX) -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV)

# ED Part:

tmp/EdgeDrawing_cpu.o: src/EdgeDrawing.cpp $(INC_FILE) 
	@if [ ! -e tmp ];then mkdir tmp; fi
	@echo "compile tmp/EdgeDrawing_cpu.o"
	@$(CXX) -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV)

tmp/EdgeDrawing_gpu.o: src/EdgeDrawing.cu $(INC_FILE) 
	@if [ ! -e tmp ];then mkdir tmp; fi
	@echo "compile tmp/EdgeDrawing_gpu.o"
	@$(CUDAXX) -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV) $(CUDA_FLAG)

# DP Part:

tmp/DouglasPeucker_cpu.o: src/DouglasPeucker.cpp $(INC_FILE) 
	@if [ ! -e tmp ];then mkdir tmp; fi
	@echo "compile tmp/DouglasPeucker_cpu.o"
	@$(CXX) -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV)

tmp/DouglasPeucker_gpu.o: src/DouglasPeucker.cu $(INC_FILE) 
	@if [ ! -e tmp ];then mkdir tmp; fi
	@echo "compile tmp/DouglasPeucker_gpu.o"
	@$(CUDAXX) -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV) $(CUDA_FLAG)

# LS Part:

tmp/LinearSum_cpu.o: src/LinearSum.cpp $(INC_FILE) 
	@if [ ! -e tmp ];then mkdir tmp; fi
	@echo "compile tmp/LinearSum_cpu.o"
	@$(CXX) -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV)

tmp/LinearSum_gpu.o: src/LinearSum.cu $(INC_FILE) 
	@if [ ! -e tmp ];then mkdir tmp; fi
	@echo "compile tmp/LinearSum_gpu.o"
	@$(CUDAXX) -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV) $(CUDA_FLAG)

# LD Part:

tmp/LinearDis_cpu.o: src/LinearDis.cpp $(INC_FILE) 
	@if [ ! -e tmp ];then mkdir tmp; fi
	@echo "compile tmp/LinearDis_cpu.o"
	@$(CXX) -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV)

tmp/LinearDis_gpu.o: src/LinearDis.cu $(INC_FILE) 
	@if [ ! -e tmp ];then mkdir tmp; fi
	@echo "compile tmp/LinearDis_gpu.o"
	@$(CUDAXX) -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV) $(CUDA_FLAG)

# COMMON Part

tmp/EDmain.o: src/main.cpp $(INC_FILE) 
	@if [ ! -e tmp ];then mkdir tmp; fi
	@echo "compile tmp/EDmain.o"
	@$(CXX) -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV)

tmp/EDDPmain.o: src/main.cpp $(INC_FILE) 
	@if [ ! -e tmp ];then mkdir tmp; fi
	@echo "compile tmp/EDDPmain.o"
	@$(CXX) -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV) -D_DP

tmp/EDLSmain.o: src/main.cpp $(INC_FILE) 
	@if [ ! -e tmp ];then mkdir tmp; fi
	@echo "compile tmp/EDLSmain.o"
	@$(CXX) -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV) -D_LS

tmp/EDLDmain.o: src/main.cpp $(INC_FILE) 
	@if [ ! -e tmp ];then mkdir tmp; fi
	@echo "compile tmp/EDLDmain.o"
	@$(CXX) -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV) -D_LD

# ED bin Part

bin/EDmain-gpu: \
		tmp/Config.o\
		tmp/EdgeDrawing_gpu.o\
		tmp/EDmain.o
	@echo "Linking bin/EDmain-gpu"
	@$(CUDAXX) -o $@ $^ $(INC) $(DEF) $(FLAG) $(OPENCVENV)

bin/EDmain-cpu: \
		tmp/Config.o\
		tmp/EdgeDrawing_cpu.o\
		tmp/EDmain.o
	@echo "Linking bin/EDmain-gpuEDmain-cpu"
	@$(CUDAXX) -o $@ $^ $(INC) $(DEF) $(FLAG) $(OPENCVENV)

# ED & DP bin Part

bin/EDDPmain-gpu: \
		tmp/Config.o\
		tmp/EdgeDrawing_gpu.o\
		tmp/DouglasPeucker_gpu.o\
		tmp/EDDPmain.o
	@echo "Linking bin/EDDPmain-gpu"
	@$(CUDAXX) -o $@ $^ $(INC) $(DEF) $(FLAG) $(OPENCVENV)

bin/EDDPmain-cpu: \
		tmp/Config.o\
		tmp/EdgeDrawing_cpu.o\
		tmp/DouglasPeucker_cpu.o\
		tmp/EDDPmain.o
	@echo "Linking bin/EDDPmain-cpu"
	@$(CUDAXX) -o $@ $^ $(INC) $(DEF) $(FLAG) $(OPENCVENV)

# ED & LS bin Part

bin/EDLSmain-gpu: \
		tmp/Config.o\
		tmp/EdgeDrawing_gpu.o\
		tmp/LinearSum_gpu.o\
		tmp/EDLSmain.o
	@echo "Linking bin/EDLSmain-gpu"
	@$(CUDAXX) -o $@ $^ $(INC) $(DEF) $(FLAG) $(OPENCVENV)

bin/EDLSmain-cpu: \
		tmp/Config.o\
		tmp/EdgeDrawing_cpu.o\
		tmp/LinearSum_cpu.o\
		tmp/EDLSmain.o
	@echo "Linking bin/EDLSmain-cpu"
	@$(CUDAXX) -o $@ $^ $(INC) $(DEF) $(FLAG) $(OPENCVENV)

# ED & LD bin Part

bin/EDLDmain-gpu: \
		tmp/Config.o\
		tmp/EdgeDrawing_gpu.o\
		tmp/LinearDis_gpu.o\
		tmp/EDLDmain.o
	@echo "Linking bin/EDLDmain-gpu"
	@$(CUDAXX) -o $@ $^ $(INC) $(DEF) $(FLAG) $(OPENCVENV)

bin/EDLDmain-cpu: \
		tmp/Config.o\
		tmp/EdgeDrawing_cpu.o\
		tmp/LinearDis_cpu.o\
		tmp/EDLDmain.o
	@echo "Linking bin/EDLDmain-cpu"
	@$(CUDAXX) -o $@ $^ $(INC) $(DEF) $(FLAG) $(OPENCVENV)

bin/EDbaseline: \
		baseline/EDProcess.cpp\
		baseline/EDmain.cpp
	@echo "Linking bin/EDmain"
	@$(CXX) -o $@ $^ -Ibaseline $(FLAG) $(OPENCVENV)

bin/EDDPbaseline: \
		baseline/EDProcess.cpp\
		baseline/EDDPmain.cpp
	@echo "Linking bin/EDDPmain"
	@$(CXX) -o $@ $^ -Ibaseline $(FLAG) $(OPENCVENV)

bin/EDLSbaseline: \
		baseline/EDProcess.cpp\
		baseline/EDLSmain.cpp
	@echo "Linking bin/EDLSmain"
	@$(CXX) -o $@ $^ -Ibaseline $(FLAG) $(OPENCVENV)

QueryDev: src/CudaQueryDev.cpp
	@echo "C&L bin/QueryDev"
	@$(CUDAXX) -o bin/QueryDev src/CudaQueryDev.cpp $(INC) $(DEF) $(FLAG) $(OPENCVENV)
	@bin/QueryDev
	@rm -f bin/QueryDev

GaussianKernel: src/GaussianKernel.cpp
	@echo "C&L bin/GaussianKernel"
	@$(CXX) -o bin/GaussianKernel src/GaussianKernel.cpp
	@bin/GaussianKernel
	@rm -f bin/GaussianKernel

bin/imgs2video: utils/imgs2video.cpp
	@echo "C&L bin/imgs2video"
	@$(CXX) -o bin/imgs2video utils/imgs2video.cpp $(OPENCVENV)
