# GPU-Accelerated K-Means Clustering for CIFAR-10
* Author: Molly Pate
* EN.605.617

## Project Motivation
This project focuses on the design and optimization of a GPU-accelerated implementation of the K-Means clustering algorithm using CUDA. As an unsupervised learning algorithm, K-Means is traditionally computationally expensive due to repeated distance calculations in high-dimensional space.

The core motivation is to leverage the massively parallel architecture of the GPU to reduce processing latency. By evaluating the trade-offs between Standard K-Means (global convergence) and Mini-Batch K-Means (stochastic efficiency), this project explores the limits of scalability and hardware utilization.

## Architecture
This design leverages some of the concepts to help with large data bottlenecks, as described over several modules (Note: This largely builds off of the memory management module 5).
1. **Read-Only Cache:** Utilizes the `__restrict__` qualifier to bypass the 64KB hardware limits of constant memory, ensuring fast access to centroid data.
2. **Shared Memory Tiling:** Implements L1-speed staging for 3072-dimension vectors. Blocks load subsets of data into shared memory to facilitate reuse and reduce expensive global memory transactions.
3. **Unified Managed Memory:** Simplifies data coherence between the CPU and GPU for atomic updates and centroid recomputation.
4. **Asynchronous Streaming:** Employs multiple CUDA streams to overlap data transfers with kernel execution, transitioning the program from being communication-bound to compute-bound.

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

2. Validation & Export
The program generates CSV reports (e.g., gpu_n1000000_b32_standard.csv) for every run. These files contain:

    - **ImageID:** Index of the CIFAR-10 image.
    - **ClusterID:** The final assigned cluster (0-9).

These reports are designed to be imported into external tools for Principal Component Analysis (PCA) to visualize the high-dimensional clusters in a 2D space. (**COMING SOON**)