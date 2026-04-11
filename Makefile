NVCC = nvcc
TARGET = assignment
SRC = assignment.cu

# -O3: Aggressive host-side optimization
# -arch=sm_75: Optimized for Tesla T4 (AWS g4dn instances)
# -Xcompiler -Wall: Passes 'all warnings' flag to the host C++ compiler
CFLAGS = -O3 -arch=sm_75 -Xcompiler -Wall

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) $(SRC) -o $(TARGET)

# Cleans the binary and any generated CSV reports
clean:
	rm -f $(TARGET)
	rm -f cpu_n*.csv gpu_n*.csv

# Quick test run with a standard configuration
# Usage: make test
test: $(TARGET)
	./$(TARGET) 10000 256

# Quick test run for mini-batch mode
# Usage: make test-mini
test-mini: $(TARGET)
	./$(TARGET) 10000 256 --mini-batch

.PHONY: all clean test test-mini