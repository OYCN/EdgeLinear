INCDIR = inc
CUDA = /usr/local/cuda-11.2/include
OPENCVENV = $(shell pkg-config opencv --cflags --libs)
INC_FILE = $(wildcard $(INCDIR)/*.h)
DEF = -DLINUX -DTIMING
SHELL=/bin/bash
INC = -I$(INCDIR) -I$(CUDA)
# LIB = -L/usr/local/cuda-11.2/lib64/

.PHONY : all clean

all: bin/EDmain-gpu\
	bin/EDmain-cpu\
	bin/EDDPmain-gpu\
	bin/EDDPmain-cpu\
	bin/EDLSmain-gpu\
	bin/EDLSmain-cpu\
	bin/EDLDmain-gpu\
	bin/EDLDmain-cpu


# ED Part:

tmp/EdgeDrawing_cpu.o: src/EdgeDrawing.cpp $(INC_FILE)
	g++ -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV) -O3

tmp/EdgeDrawing_gpu.o: src/EdgeDrawing.cu $(INC_FILE)
	nvcc -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV) -O3

tmp/smartConnecting.o: src/smartConnecting.cpp $(INC_FILE)
	g++ -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV) -O3

tmp/EDmain.o: src/EDmain.cpp $(INC_FILE)
	g++ -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV) -O3

# DP Part:

tmp/DouglasPeucker_cpu.o: src/DouglasPeucker.cpp $(INC_FILE)
	g++ -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV) -O3

tmp/DouglasPeucker_gpu.o: src/DouglasPeucker.cu $(INC_FILE)
	nvcc -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV) -O3

tmp/EDDPmain.o: src/EDDPmain.cpp $(INC_FILE)
	g++ -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV) -O3

# LS Part:

tmp/LinearSum_cpu.o: src/LinearSum.cpp $(INC_FILE)
	g++ -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV) -O3

tmp/LinearSum_gpu.o: src/LinearSum.cu $(INC_FILE)
	nvcc -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV) -O3

tmp/EDLSmain.o: src/EDLSmain.cpp $(INC_FILE)
	g++ -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV) -O3

# LD Part:

tmp/LinearDis_cpu.o: src/LinearDis.cpp $(INC_FILE)
	g++ -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV) -O3

tmp/LinearDis_gpu.o: src/LinearDis.cu $(INC_FILE)
	nvcc -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV) -O3

tmp/EDLDmain.o: src/EDLDmain.cpp $(INC_FILE)
	g++ -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV) -O3

# ED bin Part

bin/EDmain-gpu: tmp/EdgeDrawing_gpu.o\
		 tmp/smartConnecting.o\
		 tmp/EDmain.o
	nvcc -o $@ $^ $(INC) $(DEF) $(FLAG) $(OPENCVENV) -O3 -g

bin/EDmain-cpu: tmp/EdgeDrawing_cpu.o\
		 tmp/smartConnecting.o\
		 tmp/EDmain.o
	g++ -o $@ $^ $(INC) $(DEF) $(FLAG) $(OPENCVENV) -O3 -g

# ED & DP bin Part

bin/EDDPmain-gpu: tmp/EdgeDrawing_gpu.o\
		 tmp/smartConnecting.o\
		 tmp/DouglasPeucker_gpu.o\
		 tmp/EDDPmain.o
	nvcc -o $@ $^ $(INC) $(DEF) $(FLAG) $(OPENCVENV) -O3 -g

bin/EDDPmain-cpu: tmp/EdgeDrawing_cpu.o\
		 tmp/smartConnecting.o\
		 tmp/DouglasPeucker_cpu.o\
		 tmp/EDDPmain.o
	g++ -o $@ $^ $(INC) $(DEF) $(FLAG) $(OPENCVENV) -O3 -g

# ED & LS bin Part

bin/EDLSmain-gpu: tmp/EdgeDrawing_gpu.o\
		 tmp/smartConnecting.o\
		 tmp/LinearSum_gpu.o\
		 tmp/EDLSmain.o
	nvcc -o $@ $^ $(INC) $(DEF) $(FLAG) $(OPENCVENV) -O3 -g

bin/EDLSmain-cpu: tmp/EdgeDrawing_cpu.o\
		 tmp/smartConnecting.o\
		 tmp/LinearSum_cpu.o\
		 tmp/EDLSmain.o
	g++ -o $@ $^ $(INC) $(DEF) $(FLAG) $(OPENCVENV) -O3 -g

# ED & LD bin Part

bin/EDLDmain-gpu: tmp/EdgeDrawing_gpu.o\
		 tmp/smartConnecting.o\
		 tmp/LinearDis_gpu.o\
		 tmp/EDLDmain.o
	nvcc -o $@ $^ $(INC) $(DEF) $(FLAG) $(OPENCVENV) -O3 -g

bin/EDLDmain-cpu: tmp/EdgeDrawing_cpu.o\
		 tmp/smartConnecting.o\
		 tmp/LinearDis_cpu.o\
		 tmp/EDLDmain.o
	g++ -o $@ $^ $(INC) $(DEF) $(FLAG) $(OPENCVENV) -O3 -g

# utils Part

clean:
	rm -f $(wildcard ./tmp/*.o) $(wildcard ./bin/*)

