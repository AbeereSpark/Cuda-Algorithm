CC = nvcc
CFLAGS = -lgmp -lgmpxx
TARGET = key
SOURCE = key.cu
KEY = 02b968ca5555330e4fc778efb2af7d9de81c860002dab90d247b0329c3071cc221
FILENAME = FOUND.txt

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CC) -o $(TARGET) $(SOURCE) $(CFLAGS)

run_add: $(TARGET)
	./$(TARGET) $(KEY) $(value) A $(ITERATIONS) $(FILENAME)

run_sub: $(TARGET)
	./$(TARGET) $(KEY) $(value) S $(ITERATIONS) $(FILENAME)

clean:
	rm -f $(TARGET)
