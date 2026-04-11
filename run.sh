#!/bin/bash

DATA_FILE="data_batch_1.bin"
TAR_FILE="cifar-10-binary.tar.gz"
URL="https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"

if [ ! -f "$DATA_FILE" ]; then
    echo "Dataset not found. Downloading CIFAR-10..."
    wget $URL -O $TAR_FILE
    tar -xzf $TAR_FILE
    # Move the specific batch file to the current directory
    cp cifar-10-batches-bin/data_batch_1.bin .
    # Cleanup extra files
    rm -rf cifar-10-batches-bin $TAR_FILE
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

# PHASE 1: Baseline Hardware Limits
# Tests threads-per-block limits and basic overhead.
echo -e "\n[TEST 1] Small Workload | Block Size 64"
./assignment 1024 64

echo -e "\n[TEST 2] Small Workload | Block Size 1024 (Max Threads/Block)"
./assignment 1024 1024

# PHASE 2: Throughput Scaling
# Demonstrates how the T4 scales as we saturate its 2,560 CUDA cores.
echo -e "\n[TEST 3] 10k Images | Block Size 256"
./assignment 10000 256

echo -e "\n[TEST 4] 100k Images | Block Size 256"
./assignment 100000 256

# PHASE 3: Algorithm Comparison
# Explicitly uses the new flag to compare performance.
echo -e "\n[TEST 5] 10k Images | STANDARD K-MEANS"
./assignment 10000 256

echo -e "\n[TEST 6] 10k Images | MINI-BATCH K-MEANS"
./assignment 10000 256 --mini-batch

# PHASE 4: Scheduler & Occupancy Stress
# Compares high occupancy vs high scheduler overhead.
echo -e "\n[TEST 7] 1M Images | Block Size 1024 (High Occupancy)"
./assignment 1000000 1024

echo -e "\n[TEST 8] 1M Images | Block Size 32 (High Scheduler Overhead)"
./assignment 1000000 32

echo -e "\n--- CSV Reports Generated ---"
ls -lh *.csv

echo -e "\nBenchmarks complete."