#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <float.h>
#include <string.h>

#define K 10
#define DIM 3072
#define ITER 20
#define TILE_SIZE 65536 // Max images per GPU upload (approx 800MB)

// ... [Keep CUDA_CHECK, warpReduceSum, and kernels from previous iteration] ...

// Helper: Accumulate partial sums on GPU
__global__ void kmeans_accumulate_kernel(
    const float* __restrict__ x,
    const int* __restrict__ labels,
    float* sum,
    int* cnt,
    int n) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int k = labels[i];
    atomicAdd(&cnt[k], 1);
    for (int d = 0; d < DIM; d++) {
        atomicAdd(&sum[k * DIM + d], x[i * DIM + d]);
    }
}

// ---------------- GPU DRIVER ----------------

float gpu_kmeans(const float* x, float* c, int n, int block_size, int mini_batch)
{
    float *d_x, *d_c, *d_sum;
    int *d_cnt, *d_labels;

    // Allocate Device Memory (Fixed Tile Size)
    int actual_tile = (n < TILE_SIZE) ? n : TILE_SIZE;
    CUDA_CHECK(cudaMalloc(&d_x, actual_tile * DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, K * DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum, K * DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cnt, K * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_labels, actual_tile * sizeof(int)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int it = 0; it < ITER; it++) {
        // Reset accumulators
        CUDA_CHECK(cudaMemset(d_sum, 0, K * DIM * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_cnt, 0, K * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_c, c, K * DIM * sizeof(float), cudaMemcpyHostToDevice));

        int total_processed = 0;
        int max_iters = mini_batch ? 1 : (n + actual_tile - 1) / actual_tile;

        for (int t = 0; t < max_iters; t++) {
            // Pick data: Random offset if mini-batch, sequential if full
            int offset = mini_batch ? (rand() % (n - actual_tile)) : (t * actual_tile);
            
            CUDA_CHECK(cudaMemcpy(d_x, x + (offset * DIM), actual_tile * DIM * sizeof(float), cudaMemcpyHostToDevice));

            // Assignment
            int blocks = (actual_tile + (block_size/32) - 1) / (block_size/32);
            kmeans_assignment_kernel<<<blocks, block_size>>>(d_x, d_c, d_labels, actual_tile);
            
            // Accumulation
            kmeans_accumulate_kernel<<< (actual_tile + 255)/256, 256 >>>(d_x, d_labels, d_sum, d_cnt, actual_tile);
        }

        // Update Centroids on Host
        float h_sum[K * DIM]; int h_cnt[K];
        CUDA_CHECK(cudaMemcpy(h_sum, d_sum, K * DIM * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_cnt, d_cnt, K * sizeof(int), cudaMemcpyDeviceToHost));

        for (int k = 0; k < K; k++) {
            if (h_cnt[k] > 0) {
                for (int d = 0; d < DIM; d++) c[k * DIM + d] = h_sum[k * DIM + d] / h_cnt[k];
            }
        }
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    
    // Cleanup
    cudaFree(d_x); cudaFree(d_c); cudaFree(d_sum); cudaFree(d_cnt); cudaFree(d_labels);
    return ms;
}

// ---------------- MAIN ----------------

int main(int argc, char** argv)
{
    int n = atoi(argv[1]);
    int block_size = atoi(argv[2]);
    int mini_batch = 0;

    for(int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--mini-batch") == 0) mini_batch = 1;
    }

    // ... [Allocation & Init] ...
    
    float gpu_ms = gpu_kmeans(x, c2, n, block_size, mini_batch);
    
    // ... [Print & Cleanup] ...
    return 0;
}