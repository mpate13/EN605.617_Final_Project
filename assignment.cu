#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <float.h>
#include <string.h>
#include <time.h>

#define K 10
#define DIM 3072
#define ITER 20
#define WARP_SIZE 32
#define TILE_SIZE 65536 // Max images per GPU upload

// ---------------- ERROR CHECK ----------------
#define CUDA_CHECK(x) \
    if ((x) != cudaSuccess) { \
        printf("CUDA ERROR: %s (%s:%d)\n", \
        cudaGetErrorString(x), __FILE__, __LINE__); \
        exit(1); \
    }

// ---------------- DEVICE FUNCTIONS ----------------
__device__ __forceinline__
float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// ---------------- KERNELS ----------------

__global__ void kmeans_assignment_kernel(
    const float* __restrict__ x,
    const float* __restrict__ c,
    int* labels,
    int n) 
{
    int warp_id = blockIdx.x * (blockDim.x / WARP_SIZE) + (threadIdx.x / WARP_SIZE);
    int lane = threadIdx.x % WARP_SIZE;
    if (warp_id >= n) return;

    const float* xi = x + warp_id * DIM;
    float bestDist = FLT_MAX;
    int bestk = 0;

    for (int k = 0; k < K; k++) {
        float partial = 0;
        const float* ck = c + k * DIM;
        for (int d = lane; d < DIM; d += WARP_SIZE) {
            float diff = xi[d] - ck[d];
            partial += diff * diff;
        }
        partial = warpReduceSum(partial);
        if (lane == 0) {
            if (partial < bestDist) {
                bestDist = partial;
                bestk = k;
            }
        }
    }
    if (lane == 0) labels[warp_id] = bestk;
}

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
        CUDA_CHECK(cudaMemset(d_sum, 0, K * DIM * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_cnt, 0, K * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_c, c, K * DIM * sizeof(float), cudaMemcpyHostToDevice));

        int max_iters = mini_batch ? 1 : (n + actual_tile - 1) / actual_tile;

        for (int t = 0; t < max_iters; t++) {
            int offset = mini_batch ? (rand() % (n - actual_tile)) : (t * actual_tile);
            
            CUDA_CHECK(cudaMemcpy(d_x, x + (offset * DIM), actual_tile * DIM * sizeof(float), cudaMemcpyHostToDevice));

            int blocks = (actual_tile + (block_size/WARP_SIZE) - 1) / (block_size/WARP_SIZE);
            kmeans_assignment_kernel<<<blocks, block_size>>>(d_x, d_c, d_labels, actual_tile);
            
            kmeans_accumulate_kernel<<< (actual_tile + 255)/256, 256 >>>(d_x, d_labels, d_sum, d_cnt, actual_tile);
        }

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
    
    cudaFree(d_x); cudaFree(d_c); cudaFree(d_sum); cudaFree(d_cnt); cudaFree(d_labels);
    return ms;
}

// ---------------- INIT & MAIN ----------------

void init(float* x, int n) {
    for (int i = 0; i < n * DIM; i++) x[i] = (float)rand() / RAND_MAX;
}

int main(int argc, char** argv) {
    if (argc < 3) { printf("Usage: %s <N> <block_size> [--mini-batch]\n", argv[0]); return 1; }

    int n = atoi(argv[1]);
    int block_size = atoi(argv[2]);
    int mini_batch = 0;
    for(int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--mini-batch") == 0) mini_batch = 1;
    }

    float* x = (float*)malloc(n * DIM * sizeof(float));
    float* c2 = (float*)malloc(K * DIM * sizeof(float));
    init(x, n);
    for (int i = 0; i < K * DIM; i++) c2[i] = x[i];

    float gpu_ms = gpu_kmeans(x, c2, n, block_size, mini_batch);
    
    printf("GPU: %.2f ms\n", gpu_ms);

    free(x); free(c2);
    return 0;
}