#!/bin/bash

DATA_FILE="data_batch_1.bin"
TAR_FILE="cifar-10-binary.tar.gz"
URL="https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"

if [ ! -f "$DATA_FILE" ]; then
    echo "Dataset not found. Downloading CIFAR-10..."
    # wget $URL -O $TAR_FILE
    tar -xzf $TAR_FILE
    # Move the specific batch file to the current directory
    cp cifar-10-batches-bin/data_batch_1.bin .
    # Cleanup extra files
    # rm -rf cifar-10-batches-bin $TAR_FILE
    echo "Dataset ready."
fi

# Clean and Build
echo "Building project..."
make clean
make

if [ ! -f "./assignment" ]; then
    echo "Error: Compilation failed."
    exit 1
fi

echo -e "\nStarting Benchmarks on Tesla T4..."
echo "Format: ./assignment <total_threads> <block_size> [--mini-batch]"

# ============================================================
# PHASE 1: BLOCK SIZE EFFECT (CONSTANT WORKLOAD)
# Goal: isolate occupancy + scheduling effects
# ============================================================

echo -e "\n[PHASE 1] Block Size Scaling (N = 10000 fixed)"

./assignment 10000 32
./assignment 10000 64
./assignment 10000 128
./assignment 10000 256
./assignment 10000 512
./assignment 10000 1024


# ============================================================
# PHASE 2: THREAD / GRID SCALING (CONSTANT BLOCK SIZE)
# Goal: show GPU saturation & throughput scaling
# ============================================================

echo -e "\n[PHASE 2] Grid Scaling (Block Size = 256 fixed)"

./assignment 1000 256
./assignment 10000 256
./assignment 50000 256
./assignment 100000 256
./assignment 1000000 256


# ============================================================
# PHASE 3: STANDARD vs MINI-BATCH (CONSTANT PARAMS)
# Goal: isolate algorithmic difference only
# ============================================================

echo -e "\n[PHASE 3] Standard vs Mini-Batch (N = 10000, Block = 256)"

./assignment 10000 256
./assignment 10000 256 --mini-batch


# ============================================================
# PHASE 4: EXTREME OCCUPANCY TEST
# Goal: stress scheduler and memory system
# ============================================================

echo -e "\n[PHASE 4] Extreme Scaling Stress Test"

./assignment 1000000 128
./assignment 1000000 256
./assignment 1000000 512


echo -e "\n--- CSV Reports Generated ---"
ls -lh *.csv

echo -e "\nBenchmarks complete."