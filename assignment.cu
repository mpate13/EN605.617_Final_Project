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
#define SHARED_DIM_CHUNK 128 

#define CUDA_CHECK(x) \
    if ((x) != cudaSuccess) { \
        printf("CUDA ERROR: %s (%s:%d)\n", cudaGetErrorString(x), __FILE__, __LINE__); \
        exit(1); \
    }

// Procedural generation to avoid OOM for large N
void generate_data_chunk(float* buffer, int n_elements) {
    for (int i = 0; i < n_elements * DIM; i++) {
        buffer[i] = (float)rand() / RAND_MAX;
    }
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

__global__ void kmeans_accumulate_kernel(const float* __restrict__ x, const int* __restrict__ labels, float* sum, int* cnt, int n) {
    __shared__ float s_sum[K][SHARED_DIM_CHUNK];
    
    // Tiled shared memory accumulation to reduce atomic contention
    for (int d_start = 0; d_start < DIM; d_start += SHARED_DIM_CHUNK) {
        for(int k=0; k<K; k++) {
            for(int d = threadIdx.x; d < SHARED_DIM_CHUNK; d += blockDim.x) s_sum[k][d] = 0.0f;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            int k = labels[i];
            for(int d = 0; d < SHARED_DIM_CHUNK; d++) {
                int dim_idx = d_start + d;
                if(dim_idx < DIM) s_sum[k][d] += x[i * DIM + dim_idx];
            }
        }
        __syncthreads();

        for(int k=0; k<K; k++) {
            for(int d = threadIdx.x; d < SHARED_DIM_CHUNK; d += blockDim.x) {
                int dim_idx = d_start + d;
                if(dim_idx < DIM) atomicAdd(&sum[k * DIM + dim_idx], s_sum[k][d]);
            }
        }
    }
    
    if (threadIdx.x == 0) {
        for (int i = 0; i < n; i++) atomicAdd(&cnt[labels[i]], 1);
    }
}

// ---------------- GPU DRIVER ----------------

float gpu_kmeans(int n, int block_size, int mini_batch) {
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

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int it = 0; it < ITER; it++) {
        CUDA_CHECK(cudaMemset(d_sum, 0, K * DIM * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_cnt, 0, K * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_c, c, K * DIM * sizeof(float), cudaMemcpyHostToDevice));

        int max_iters = mini_batch ? 1 : (n + TILE_SIZE - 1) / TILE_SIZE;

        for (int t = 0; t < max_iters; t++) {
            int stream_idx = t % 2;
            int current_tile = (t == max_iters - 1 && (n % TILE_SIZE != 0)) ? (n % TILE_SIZE) : TILE_SIZE;

            generate_data_chunk(h_buffer[stream_idx], current_tile);
            CUDA_CHECK(cudaMemcpyAsync(d_x[stream_idx], h_buffer[stream_idx], current_tile * DIM * sizeof(float), cudaMemcpyHostToDevice, streams[stream_idx]));

            int blocks = (current_tile + (block_size/WARP_SIZE) - 1) / (block_size/WARP_SIZE);
            kmeans_assignment_kernel<<<blocks, block_size, 0, streams[stream_idx]>>>(d_x[stream_idx], d_c, d_labels[stream_idx], current_tile);
            kmeans_accumulate_kernel<<< (current_tile + 255)/256, 256, 0, streams[stream_idx] >>>(d_x[stream_idx], d_labels[stream_idx], d_sum, d_cnt, current_tile);
        }
        cudaDeviceSynchronize();
        
        float h_sum[K * DIM]; int h_cnt[K];
        CUDA_CHECK(cudaMemcpy(h_sum, d_sum, K * DIM * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_cnt, d_cnt, K * sizeof(int), cudaMemcpyDeviceToHost));

        for (int k = 0; k < K; k++) {
            if (h_cnt[k] > 0) {
                for (int d = 0; d < DIM; d++) c[k * DIM + d] = h_sum[k * DIM + d] / h_cnt[k];
            }
        }
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms; cudaEventElapsedTime(&ms, start, stop);

    // Cleanup
    cudaFree(d_x[0]); cudaFree(d_x[1]); cudaFree(d_c); cudaFree(d_sum); cudaFree(d_cnt); cudaFree(d_labels[0]); cudaFree(d_labels[1]);
    cudaFreeHost(h_buffer[0]); cudaFreeHost(h_buffer[1]);
    free(c);
    return ms;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <N> <block_size> [--mini-batch]\n", argv[0]); 
        return 1; 
    }
    int n = atoi(argv[1]);
    int block_size = atoi(argv[2]);
    int mini_batch = (argc > 3 && strcmp(argv[3], "--mini-batch") == 0);

    float elapsed_ms = gpu_kmeans(n, block_size, mini_batch);
    printf("GPU Time: %.2f ms\n", elapsed_ms);
    
    return 0;
}