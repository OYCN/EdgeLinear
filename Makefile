INCDIR = inc
CUDA = /usr/local/cuda-11.2/include
OPENCVENV = $(shell pkg-config opencv --cflags --libs)
INC_FILE = $(wildcard $(INCDIR)/*.h)
DEF = -DLINUX -DTIMING
SHELL=/bin/bash
INC = -I$(INCDIR) -I$(CUDA)
# LIB = -L/usr/local/cuda-11.2/lib64/

.PHONY : all clean

all: bin/EDmain

# ED Part:

tmp/EdgeDrawing_cpu.o: src/EdgeDrawing.cpp $(INC_FILE)
	g++ -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV) -O3

tmp/EdgeDrawing_gpu.o: src/EdgeDrawing.cu $(INC_FILE)
	nvcc -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV) -O3

tmp/smartConnecting.o: src/smartConnecting.cpp $(INC_FILE)
	g++ -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV) -O3

tmp/EDmain.o: src/EDmain.cpp $(INC_FILE)
	g++ -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV) -O3

tmp/DouglasPeucker_cpu.o: src/DouglasPeucker.cpp $(INC_FILE)
	g++ -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV) -O3

tmp/DouglasPeucker_gpu.o: src/DouglasPeucker.cu $(INC_FILE)
	nvcc -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV) -O3

tmp/EDDPmain.o: src/EDDPmain.cpp $(INC_FILE)
	g++ -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV) -O3

bin/EDmain-gpu: tmp/EdgeDrawing_gpu.o\
		 tmp/smartConnecting.o\
		 tmp/EDmain.o
	nvcc -o $@ $^ $(INC) $(DEF) $(FLAG) $(OPENCVENV) -O3 -g

bin/EDmain-cpu: tmp/EdgeDrawing_cpu.o\
		 tmp/smartConnecting.o\
		 tmp/EDmain.o
	g++ -o $@ $^ $(INC) $(DEF) $(FLAG) $(OPENCVENV) -O3 -g

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

clean:
	rm -f $(wildcard ./tmp/*.o) $(wildcard ./bin/*)
	
run: bin/run
	@echo -e "\033[33mRun:\033[0m"
	@bin/run
	@echo -e "\033[33mend\033[0m"
