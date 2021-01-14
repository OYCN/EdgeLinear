CXX = nvcc
TARGET = run
SRCDIR = src
INCDIR = inc
OBJDIR = tmp
BINDIR = bin
OPENCVENV = $(shell pkg-config opencv --cflags --libs)
SRC_C = $(wildcard $(SRCDIR)/*.cpp $(SRCDIR)/*.cu)
INC_FILE = $(wildcard $(INCDIR)/*.h)
DEF = -DLINUX -DTIMING
FLAG = 
SHELL=/bin/bash

.PHONY : all clean

all : $(BINDIR)/$(TARGET) $(TARGET)

$(BINDIR)/$(TARGET): $(SRC_C) $(INC_FILE)
	@$(CXX) -o $(BINDIR)/$(TARGET) $(SRC_C) -I$(INCDIR) $(DEF) $(FLAG) $(OPENCVENV) -O3 -g

clean:
	rm -f $(wildcard ./src/*.o) $(wildcard ./bin/*)
$(TARGET):$(BINDIR)/$(TARGET)
	@echo -e "\033[33mRun:\033[0m"
	@./$(BINDIR)/$(TARGET)
	@echo -e "\033[33mend\033[0m"
