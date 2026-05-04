#!/bin/bash

# Configuration
FULL_DATASET="cifar10_full.bin"
TAR_FILE="cifar-10-binary.tar.gz"
URL="https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"

# 1. Prepare Data
if [ ! -f "$FULL_DATASET" ]; then
    echo "Dataset not found. Downloading and merging CIFAR-10..."
    
    # Download and extract
    wget -q $URL -O $TAR_FILE
    tar -xzf $TAR_FILE
    
    # Concatenate all 5 batches into one contiguous file
    # This allows you to handle N up to 50,000 safely
    cat cifar-10-batches-bin/data_batch_*.bin > $FULL_DATASET
    
    # Cleanup
    rm -rf cifar-10-batches-bin $TAR_FILE
    echo "Dataset ready: $FULL_DATASET (50,000 images)"
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


# # ============================================================
# # PHASE 3: STANDARD vs MINI-BATCH (CONSTANT PARAMS)
# # Goal: isolate algorithmic difference only
# # ============================================================

echo -e "\n[PHASE 3] Mini-Batch Scaling (Block Size = 256)"

./assignment 1000 256 --mini-batch
./assignment 10000 256 --mini-batch
./assignment 50000 256 --mini-batch
./assignment 100000 256 --mini-batch
./assignment 1000000 256 --mini-batch


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