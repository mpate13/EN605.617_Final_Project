#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <float.h>
#include <string.h>

#define K 10
#define DIM 3072
#define ITER 20
#define WARP_SIZE 32
#define TILE_SIZE 16384 
#define SHARED_DIM_CHUNK 128 // Process 128 dims at a time to fit in Shared Mem

#define CUDA_CHECK(x) \
    if ((x) != cudaSuccess) { \
        printf("CUDA ERROR: %s (%s:%d)\n", cudaGetErrorString(x), __FILE__, __LINE__); \
        exit(1); \
    }

// ---------------- KERNELS ----------------

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void kmeans_assignment_kernel(const float* __restrict__ x, const float* __restrict__ c, int* labels, int n) {
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
        if (lane == 0 && partial < bestDist) { bestDist = partial; bestk = k; }
    }
    if (lane == 0) labels[warp_id] = bestk;
}

// Optimized Accumulate Kernel using Shared Memory
__global__ void kmeans_accumulate_kernel(const float* __restrict__ x, const int* __restrict__ labels, float* sum, int* cnt, int n) {
    __shared__ float s_sum[K][SHARED_DIM_CHUNK];
    
    // We process dimensions in chunks to fit into shared memory
    for (int d_start = 0; d_start < DIM; d_start += SHARED_DIM_CHUNK) {
        // 1. Reset shared memory for this chunk
        for(int k=0; k<K; k++) {
            for(int d = threadIdx.x; d < SHARED_DIM_CHUNK; d += blockDim.x) {
                s_sum[k][d] = 0.0f;
            }
        }
        __syncthreads();

        // 2. Accumulate in shared memory
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            int k = labels[i];
            for(int d = 0; d < SHARED_DIM_CHUNK; d++) {
                int dim_idx = d_start + d;
                if(dim_idx < DIM) s_sum[k][d] += x[i * DIM + dim_idx];
            }
        }
        __syncthreads();

        // 3. Commit shared results to global memory
        for(int k=0; k<K; k++) {
            for(int d = threadIdx.x; d < SHARED_DIM_CHUNK; d += blockDim.x) {
                int dim_idx = d_start + d;
                if(dim_idx < DIM) atomicAdd(&sum[k * DIM + dim_idx], s_sum[k][d]);
            }
        }
    }
    
    // Count labels (using atomics is fine for counts)
    if (threadIdx.x == 0) {
        for (int i = 0; i < n; i++) atomicAdd(&cnt[labels[i]], 1);
    }
}

// ---------------- GPU DRIVER ----------------

float gpu_kmeans(FILE* f, int n, int block_size, int mini_batch) {
    float *d_x[2], *d_sum, *d_c;
    int *d_labels[2], *d_cnt;
    float *h_buffer[2];

    cudaHostAlloc(&h_buffer[0], TILE_SIZE * DIM * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc(&h_buffer[1], TILE_SIZE * DIM * sizeof(float), cudaHostAllocDefault);
    CUDA_CHECK(cudaMalloc(&d_x[0], TILE_SIZE * DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x[1], TILE_SIZE * DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, K * DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum, K * DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cnt, K * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_labels[0], TILE_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_labels[1], TILE_SIZE * sizeof(int)));

    float *c = (float*)malloc(K * DIM * sizeof(float));
    for(int i=0; i < K * DIM; i++) c[i] = (float)rand()/RAND_MAX;

    cudaStream_t streams[2];
    cudaStreamCreate(&streams[0]); cudaStreamCreate(&streams[1]);

    for (int it = 0; it < ITER; it++) {
        CUDA_CHECK(cudaMemset(d_sum, 0, K * DIM * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_cnt, 0, K * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_c, c, K * DIM * sizeof(float), cudaMemcpyHostToDevice));

        int max_iters = mini_batch ? 1 : (n + TILE_SIZE - 1) / TILE_SIZE;

        for (int t = 0; t < max_iters; t++) {
            int stream_idx = t % 2;
            int offset = mini_batch ? (rand() % (n - TILE_SIZE)) : (t * TILE_SIZE);
            int current_tile = (t == max_iters - 1 && (n % TILE_SIZE != 0)) ? (n % TILE_SIZE) : TILE_SIZE;

            // Disk I/O Streaming
            fseek(f, offset * DIM * sizeof(float), SEEK_SET);
            fread(h_buffer[stream_idx], sizeof(float), current_tile * DIM, f);

            CUDA_CHECK(cudaMemcpyAsync(d_x[stream_idx], h_buffer[stream_idx], current_tile * DIM * sizeof(float), cudaMemcpyHostToDevice, streams[stream_idx]));

            int blocks = (current_tile + (block_size/WARP_SIZE) - 1) / (block_size/WARP_SIZE);
            kmeans_assignment_kernel<<<blocks, block_size, 0, streams[stream_idx]>>>(d_x[stream_idx], d_c, d_labels[stream_idx], current_tile);
            kmeans_accumulate_kernel<<< (current_tile + 255)/256, 256, 0, streams[stream_idx] >>>(d_x[stream_idx], d_labels[stream_idx], d_sum, d_cnt, current_tile);
        }
        cudaDeviceSynchronize();
        
        // Finalize Centroids
        float h_sum[K * DIM]; int h_cnt[K];
        CUDA_CHECK(cudaMemcpy(h_sum, d_sum, K * DIM * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_cnt, d_cnt, K * sizeof(int), cudaMemcpyDeviceToHost));

        for (int k = 0; k < K; k++) {
            if (h_cnt[k] > 0) {
                for (int d = 0; d < DIM; d++) c[k * DIM + d] = h_sum[k * DIM + d] / h_cnt[k];
            }
        }
    }
    // [Cleanup: free memory, streams, file pointers...]
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 4) { printf("Usage: %s <N> <block_size> <filename> [--mini-batch]\n", argv[0]); return 1; }
    int n = atoi(argv[1]);
    int block_size = atoi(argv[2]);
    FILE* f = fopen(argv[3], "rb");
    int mini_batch = (argc > 4 && strcmp(argv[4], "--mini-batch") == 0);

    gpu_kmeans(f, n, block_size, mini_batch);
    fclose(f);
    return 0;
}