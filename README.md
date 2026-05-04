# GPU-Accelerated K-Means Clustering for CIFAR-10
*Author: Molly Pate*  
*Course: EN.605.617*

---

## Project Overview

This project implements and benchmarks a GPU-accelerated version of the K-Means clustering algorithm using CUDA, targeting the CIFAR-10 dataset represented as high-dimensional feature vectors (3072 dimensions per image).

K-Means is computationally expensive due to repeated distance calculations between each data point and all cluster centroids, resulting in an O(N × K × DIM) complexity per iteration. This becomes particularly costly for image-scale datasets.

The primary goal of this project is to explore how GPU parallelism and memory hierarchy optimizations can significantly reduce runtime, and how approximate methods such as Mini-Batch K-Means trade convergence stability for performance.

Both CPU and GPU implementations are included to provide a direct performance comparison.

---

## Key Contributions

This implementation explores both **algorithmic and architectural optimizations**, including:

- Full-batch vs Mini-Batch K-Means
- CUDA kernel-level parallelization
- Memory hierarchy optimization (global => shared memory)
- Atomic-based parallel reductions
- Data tiling for high-dimensional centroid computation
- Streaming execution (where applicable in benchmarking framework)

---

## CUDA Architecture & Optimizations

### 1. Shared Memory Tiling (Centroid Blocking)
To reduce repeated global memory accesses, centroids are loaded in tiles (`TILE_K`) into **shared memory**. Each thread block processes a subset of centroids at a time, improving reuse and significantly reducing memory bandwidth pressure.

This is especially important given:
- High dimensionality (DIM = 3072)
- Frequent centroid access per distance computation

---

### 2. High-Dimensional Distance Optimization
Each thread computes the distance between one data point and all centroids. To handle the large feature space efficiently:

- Distance computation is fully parallelized across threads
- Strided loops distribute DIM computation
- Memory access patterns are structured for coalescing where possible

---

### 3. Atomic Aggregation for Cluster Updates
Cluster assignment updates require concurrent writes:

- `atomicAdd` is used for safe accumulation of:
  - cluster sums (`cluster_sums`)
  - cluster counts (`cluster_counts`)

While atomic operations introduce contention, they simplify correctness and maintain scalability across large datasets.

---

### 4. Mini-Batch K-Means (Algorithmic Optimization)
#### Why try Mini-Batching: 
**My Theory:** Mini-Batch K-Means is included in this project as a performance-oriented alternative to full-batch K-Means, designed to improve scalability on large, high-dimensional datasets like CIFAR-10. Instead of computing centroid updates using the entire dataset in each iteration, the algorithm processes a randomly sampled subset of data points, significantly reducing per-iteration computational cost from O(N × K × DIM) to O(batch_size × K × DIM). In this implementation, a random offset is used each iteration to select a contiguous batch, and centroid updates are performed using a learning-rate-based rule that incrementally adjusts centroids toward the batch mean. This stochastic update strategy reduces memory bandwidth pressure, lowers atomic contention, and improves overall GPU throughput. While this introduces noisier updates and potentially less stable convergence compared to full-batch K-Means, the tradeoff is acceptable for CIFAR-10 clustering, where approximate structure is sufficient for downstream analysis such as visualization or dimensionality reduction. 

In the mini-batch version:

- Only a random subset of the dataset is processed per iteration
- Centroids are updated using a learning rate:
  
  `c = c + alpha(batch_mean - c)`

This reduces computation from full dataset passes to partial updates, improving runtime at the cost of noisier convergence behavior.

---

### 5. Kernels & Parallel Execution 
The GPU pipeline separates responsibilities:

- `kmeans_minibatch_kernel`: assignment + accumulation
- `update_centroids_full`: centroid recomputation (standard GPU mode)
- `update_centroids_minibatch`: incremental centroid updates

This separation allows tuning between:
- deterministic convergence (full batch)
- stochastic convergence (mini-batch)

---

## Key Bottlenecks & Design Tradeoffs

### 1. High Dimensionality (DIM = 3072)
- Dominates runtime due to repeated floating-point operations
- Limits effectiveness of some warp-level optimizations
- Justifies use of tiling and shared memory

---

### 2. Global Memory Bandwidth
- Centroid reads are frequent and expensive
- Mitigated using shared memory staging (TILE_K strategy)

---

### 3. Atomic Contention
- Cluster updates create serialization under high load
- Tradeoff between correctness and performance

---

### 4. Synchronization Overhead
- `__syncthreads()` required between tile loads and computation phases
- Necessary for correctness but introduces latency

---

### 5. Constant vs Compile-Time Parameters
Key parameters are defined as constants:

- `K`, `DIM`, `TILE_K`, `ITER`

This enables:
- Compiler optimizations (loop unrolling, constant propagation)
- Static shared memory allocation
- Reduced runtime overhead from dynamic indexing

---

## How to Run
For the most basic setup, to run a series of tests, simply at the root of the directly run start a series of tests across several numbers of images, and implementation techniques. 
```bash
    ./run.sh
```
This will produce:
1. Timing comparisons
2. CSVs for the clusters, which will be post-processed for visual comparison (COMING SOON)

Can also run a single test via:
```bash
    make
    ./assignment <total_images> <block_size> [--mini-batch]
```
Where the total number of images and block sizes must be specified and the optional `--mini-batch` flag can be added to swap to this implementation 

## Expected Output
1. Performance Timing Metrics (Console)
The application outputs the results of the comparison between the single-threaded CPU baseline and the parallelized GPU implementation:

    - **CPU Execution Time:** Baseline performance for assignment and update phases.
    - **GPU Execution Time:** Total time including memory residency and kernel execution.
    - **Speedup Factor:** The calculated performance gain (ex: 8.55x).

These were manually exported into CSV format and graphed with `plot_results.py`
- results in phaseX_xxx.csv
- proof of runs in `proof_of_run` directory

2. Validation & Export
The program generates CSV reports (e.g., gpu_n1000000_b32_standard.csv) for every run. These files contain:

    - **ImageID:** Index of the CIFAR-10 image.
    - **ClusterID:** The final assigned cluster (0-9).

These were post-processed in python with `compare.py` simply for correctness, not as a main part of the analysis for this project.
