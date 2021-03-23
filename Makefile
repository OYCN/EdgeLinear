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

tmp/EdgeDrawing.o: src/EdgeDrawing.cpp $(INC_FILE)
	nvcc -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV) -O3

tmp/smartConnecting.o: src/smartConnecting.cpp $(INC_FILE)
	nvcc -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV) -O3

tmp/EDkernel.o: src/EDkernel.cu $(INC_FILE)
	nvcc -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV) -O3

tmp/EDmain.o: src/EDmain.cpp $(INC_FILE)
	nvcc -c $< -o $@ $(INC) $(DEF) $(FLAG) $(OPENCVENV) -O3

bin/EDmain: tmp/EdgeDrawing.o\
		 tmp/smartConnecting.o\
		 tmp/EDkernel.o\
		 tmp/EDmain.o
	nvcc -o $@ $^ $(INC) $(DEF) $(FLAG) $(OPENCVENV) -O3 -g

clean:
	rm -f $(wildcard ./tmp/*.o) $(wildcard ./bin/*)
	
run: bin/run
	@echo -e "\033[33mRun:\033[0m"
	@bin/run
	@echo -e "\033[33mend\033[0m"
