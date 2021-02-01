INCDIR = inc
CUDA = /usr/local/cuda-11.2/include
OPENCVENV = $(shell pkg-config opencv --cflags --libs)
INC_FILE = $(wildcard $(INCDIR)/*.h)
DEF = -DLINUX -DTIMING
SHELL=/bin/bash
INC = -I$(INCDIR) -I$(CUDA)
# LIB = -L/usr/local/cuda-11.2/lib64/

.PHONY : all clean

all: bin/run run

tmp/EDProcess_base.o: src/EDProcess_base.cu
	nvcc -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV) -O3 -g

tmp/getAll.o: src/getAll.cu
	nvcc -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV) -O3 -g

tmp/main.o: src/main.cpp
	g++ -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV) -O3 -g

tmp/toLine.o: src/toLine.cu
	nvcc -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV) -O3 -g

tmp/smartConnecting.o: src/smartConnecting.cpp
	g++ -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV) -O3 -g

tmp/normalDP.o: src/normalDP.cpp
	g++ -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV) -O3 -g -fopenmp

bin/run: tmp/EDProcess_base.o\
		 tmp/getAll.o\
		 tmp/main.o\
		 tmp/toLine.o\
		 tmp/smartConnecting.o\
		 tmp/normalDP.o
	nvcc -o $@ $^ $(INC) $(DEF) $(FLAG) $(OPENCVENV) -O3 -g

clean:
	rm -f $(wildcard ./tmp/*.o) $(wildcard ./bin/*)
	
run: bin/run
	@echo -e "\033[33mRun:\033[0m"
	@bin/run
	@echo -e "\033[33mend\033[0m"
