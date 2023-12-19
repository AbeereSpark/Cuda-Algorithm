ifdef GMP_HOME
  INC := -I$(GMP_HOME)/include
  LIB := -L$(GMP_HOME)/lib
endif
ifndef GMP_HOME
  INC :=
  LIB :=
endif

CC = nvcc
CFLAGS = -lgmp -lgmpxx
TARGET = key
SOURCE = key.cu
KEY = 03dc2e14c81f317567ced3da0f436ead9a88c2d474fd24b264d00a68b10820279c
FILENAME = FOUND.txt

pick:
	@echo
	@echo Please run one of the following:
	@echo "   make kepler"
	@echo "   make maxwell"
	@echo "   make pascal"
	@echo "   make volta"
	@echo "   make turing"
	@echo "   make ampere"
	@echo "   make ada"
	@echo

clean:
	rm -f add

kepler:
	nvcc $(INC) $(LIB) -I../../include -arch=sm_35 key.cu -o key -lgmp -rdc=true

maxwell:
	nvcc $(INC) $(LIB) -I../../include -arch=sm_50 key.cu -o key -lgmp -rdc=true

pascal:
	nvcc $(INC) $(LIB) -I../../include -arch=sm_60 key.cu -o key -lgmp -rdc=true

volta: 
	nvcc $(INC) $(LIB) -I../../include -arch=sm_70 key.cu -o key -lgmp -rdc=true

turing: 
	nvcc $(INC) $(LIB) -I../../include -arch=sm_75 key.cu -o key -lgmp -rdc=true

ampere: 
	nvcc $(INC) $(LIB) -I../../include -arch=sm_80 key.cu -o key -lgmp -rdc=true

ada: 
	nvcc $(INC) $(LIB) -I../../include -arch=sm_89 key.cu -o key -lgmp -rdc=true

run_add: $(TARGET)
	ncu ./$(TARGET) $(KEY) $(value) A $(ITERATIONS) $(FILENAME)

run_sub: $(TARGET)
	ncu ./$(TARGET) $(KEY) $(value) S $(ITERATIONS) $(FILENAME)