CC = nvcc
CFLAGS = -lgmp -lgmpxx
TARGET = key
SOURCE = key.cu
MATCH_FILE = FOUND.txt

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CC) -o $(TARGET) $(SOURCE) $(CFLAGS)

run: $(TARGET)
	./$(TARGET) 023ae2be219c5b30277db2c1da0bd781474ba8fd9fe3079a3acbaca16775d63de9 1 A 6 $(MATCH_FILE)

clean:
	rm -f $(TARGET) $(MATCH_FILE)
