CC = nvcc
CFLAGS = -lgmp -lgmpxx
TARGET = key
SOURCE = key.cu
KEY = 023ae2be219c5b30277db2c1da0bd781474ba8fd9fe3079a3acbaca16775d63de9
FILENAME = FOUND.txt

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CC) -o $(TARGET) $(SOURCE) $(CFLAGS)

run_add: $(TARGET)
	./$(TARGET) $(KEY) A $(value) $(FILENAME)

run_sub: $(TARGET)
	./$(TARGET) $(KEY) S $(value) $(FILENAME)

clean:
	rm -f $(TARGET)
