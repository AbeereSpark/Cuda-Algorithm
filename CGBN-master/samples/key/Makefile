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
KEY = 02b968ca5555330e4fc778efb2af7d9de81c860002dab90d247b0329c3071cc218
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
	nvcc $(INC) $(LIB) -I../../include -arch=sm_35 key.cu -o key -lgmp

maxwell:
	nvcc $(INC) $(LIB) -I../../include -arch=sm_50 key.cu -o key -lgmp

pascal:
	nvcc $(INC) $(LIB) -I../../include -arch=sm_60 key.cu -o key -lgmp

volta: 
	nvcc $(INC) $(LIB) -I../../include -arch=sm_70 key.cu -o key -lgmp

turing: 
	nvcc $(INC) $(LIB) -I../../include -arch=sm_75 key.cu -o key -lgmp

ampere: 
	nvcc $(INC) $(LIB) -I../../include -arch=sm_80 key.cu -o key -lgmp

ada: 
	nvcc $(INC) $(LIB) -I../../include -arch=sm_89 key.cu -o key -lgmp -rdc=true

run_add: $(TARGET)
	ncu ./$(TARGET) $(KEY) $(value) A $(ITERATIONS) $(FILENAME)

run_sub: $(TARGET)
	ncu ./$(TARGET) $(KEY) $(value) S $(ITERATIONS) $(FILENAME)