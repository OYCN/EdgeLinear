INCDIR = inc
CUDA = /usr/local/cuda-11.2/include
OPENCVENV = $(shell pkg-config opencv --cflags --libs)
INC_FILE = $(wildcard $(INCDIR)/*.h)
DEF = -DLINUX -DTIMING
SHELL=/bin/bash
INC = -I$(INCDIR) -I$(CUDA)
FLAG = -g -O3
# LIB = -L/usr/local/cuda-11.2/lib64/

.PHONY : all clean

all: bin/EDmain-gpu\
	bin/EDmain-cpu\
	bin/EDDPmain-gpu\
	bin/EDDPmain-cpu\
	bin/EDLSmain-cpu\
	bin/EDLDmain-cpu


# ED Part:

tmp/EdgeDrawing_cpu.o: src/EdgeDrawing.cpp $(INC_FILE)
	g++ -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV)

tmp/EdgeDrawing_gpu.o: src/EdgeDrawing.cu $(INC_FILE)
	nvcc -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV)

tmp/smartConnecting.o: src/smartConnecting.cpp $(INC_FILE)
	g++ -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV)

# DP Part:

tmp/DouglasPeucker_cpu.o: src/DouglasPeucker.cpp $(INC_FILE)
	g++ -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV)

tmp/DouglasPeucker_gpu.o: src/DouglasPeucker.cu $(INC_FILE)
	nvcc -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV)

# LS Part:

tmp/LinearSum_cpu.o: src/LinearSum.cpp $(INC_FILE)
	g++ -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV)

tmp/LinearSum_gpu.o: src/LinearSum.cu $(INC_FILE)
	nvcc -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV)

# LD Part:

tmp/LinearDis_cpu.o: src/LinearDis.cpp $(INC_FILE)
	g++ -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV)

tmp/LinearDis_gpu.o: src/LinearDis.cu $(INC_FILE)
	nvcc -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV)

# COMMON Part

tmp/EDmain.o: src/main.cpp $(INC_FILE)
	g++ -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV)

tmp/EDDPmain.o: src/main.cpp $(INC_FILE)
	g++ -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV) -D_DP

tmp/EDLSmain.o: src/main.cpp $(INC_FILE)
	g++ -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV) -D_LS

tmp/EDLDmain.o: src/main.cpp $(INC_FILE)
	g++ -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV) -D_LD

# ED bin Part

bin/EDmain-gpu: tmp/EdgeDrawing_gpu.o\
		 tmp/smartConnecting.o\
		 tmp/EDmain.o
	nvcc -o $@ $^ $(INC) $(DEF) $(FLAG) $(OPENCVENV)

bin/EDmain-cpu: tmp/EdgeDrawing_cpu.o\
		 tmp/smartConnecting.o\
		 tmp/EDmain.o
	nvcc -o $@ $^ $(INC) $(DEF) $(FLAG) $(OPENCVENV)

# ED & DP bin Part

bin/EDDPmain-gpu: tmp/EdgeDrawing_gpu.o\
		 tmp/smartConnecting.o\
		 tmp/DouglasPeucker_gpu.o\
		 tmp/EDDPmain.o
	nvcc -o $@ $^ $(INC) $(DEF) $(FLAG) $(OPENCVENV)

bin/EDDPmain-cpu: tmp/EdgeDrawing_cpu.o\
		 tmp/smartConnecting.o\
		 tmp/DouglasPeucker_cpu.o\
		 tmp/EDDPmain.o
	nvcc -o $@ $^ $(INC) $(DEF) $(FLAG) $(OPENCVENV)

# ED & LS bin Part

bin/EDLSmain-gpu: tmp/EdgeDrawing_gpu.o\
		 tmp/smartConnecting.o\
		 tmp/LinearSum_gpu.o\
		 tmp/EDLSmain.o
	nvcc -o $@ $^ $(INC) $(DEF) $(FLAG) $(OPENCVENV)

bin/EDLSmain-cpu: tmp/EdgeDrawing_cpu.o\
		 tmp/smartConnecting.o\
		 tmp/LinearSum_cpu.o\
		 tmp/EDLSmain.o
	nvcc -o $@ $^ $(INC) $(DEF) $(FLAG) $(OPENCVENV)

# ED & LD bin Part

bin/EDLDmain-gpu: tmp/EdgeDrawing_gpu.o\
		 tmp/smartConnecting.o\
		 tmp/LinearDis_gpu.o\
		 tmp/EDLDmain.o
	nvcc -o $@ $^ $(INC) $(DEF) $(FLAG) $(OPENCVENV)

bin/EDLDmain-cpu: tmp/EdgeDrawing_cpu.o\
		 tmp/smartConnecting.o\
		 tmp/LinearDis_cpu.o\
		 tmp/EDLDmain.o
	nvcc -o $@ $^ $(INC) $(DEF) $(FLAG) $(OPENCVENV)

# utils Part

clean:
	rm -f $(wildcard ./tmp/*.o) $(wildcard ./bin/*)

