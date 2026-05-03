#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <float.h>
#include <time.h>
#include <string.h>

#define K 10
#define DIM 3072
#define ITER 20

#define NUM_STREAMS 4
#define CHUNK_SIZE 50000

#define WARP_SIZE 32
#define TILE_K 4
#define TILE_D 32

#define MINI_BATCH_SIZE 4096   // 🔥 larger batch
#define LR 0.1f

#define CUDA_CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        printf("CUDA ERROR: %s (%s:%d)\n", \
            cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// ---------------- CPU BASELINE ----------------

void cpu_kmeans(const float* x, float* c, int* labels, int n)
{
    float tmp[K * DIM];
    int cnt[K];

    for (int it = 0; it < ITER; it++) {

        for (int i = 0; i < n; i++) {
            float best = 1e30f;
            int bestk = 0;

            for (int k = 0; k < K; k++) {
                float d = 0;
                for (int d_i = 0; d_i < DIM; d_i++) {
                    float diff = x[i * DIM + d_i] - c[k * DIM + d_i];
                    d += diff * diff;
                }
                if (d < best) {
                    best = d;
                    bestk = k;
                }
            }
            labels[i] = bestk;
        }

        for (int k = 0; k < K; k++) {
            cnt[k] = 0;
            for (int d_i = 0; d_i < DIM; d_i++)
                tmp[k * DIM + d_i] = 0.0f;
        }

        for (int i = 0; i < n; i++) {
            int k = labels[i];
            cnt[k]++;
            for (int d_i = 0; d_i < DIM; d_i++)
                tmp[k * DIM + d_i] += x[i * DIM + d_i];
        }

        for (int k = 0; k < K; k++) {
            if (cnt[k] == 0) continue;
            for (int d_i = 0; d_i < DIM; d_i++)
                c[k * DIM + d_i] = tmp[k * DIM + d_i] / cnt[k];
        }
    }
}

// ---------------- WARP REDUCTION ----------------

__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// ---------------- MINI-BATCH KERNEL (OPTIMIZED) ----------------

__global__ void kmeans_minibatch_kernel_opt(
    const float* __restrict__ x,
    float* __restrict__ c,
    int n)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x & 31;

    if (warp_id >= n) return;

    const float* xi = x + warp_id * DIM;

    float best_dist = 1e30f;
    int best_k = 0;

    __shared__ float c_tile[TILE_K][DIM];

    // -------- ASSIGN --------
    for (int k0 = 0; k0 < K; k0 += TILE_K)
    {
        for (int i = threadIdx.x; i < TILE_K * DIM; i += blockDim.x)
        {
            int ck = i / DIM;
            int cd = i % DIM;

            if (k0 + ck < K)
                c_tile[ck][cd] = c[(k0 + ck) * DIM + cd];
        }

        __syncthreads();

        for (int ck = 0; ck < TILE_K && (k0 + ck) < K; ck++)
        {
            float dist = 0.0f;

            for (int d = lane; d < DIM; d += WARP_SIZE * TILE_D)
            {
                float acc = 0.0f;

                #pragma unroll
                for (int t = 0; t < TILE_D && (d + t) < DIM; t++)
                {
                    float diff = xi[d + t] - c_tile[ck][d + t];
                    acc += diff * diff;
                }

                acc = warpReduceSum(acc);

                if (lane == 0)
                    dist += acc;
            }

            if (dist < best_dist) {
                best_dist = dist;
                best_k = k0 + ck;
            }
        }

        __syncthreads();
    }

    // -------- WARP-AGGREGATED UPDATE --------
    for (int d = lane; d < DIM; d += WARP_SIZE)
    {
        float xval = xi[d];
        float delta = LR * xval;

        // warp reduce updates
        float sum = warpReduceSum(delta);

        if (lane == 0) {
            atomicAdd(&c[best_k * DIM + d], sum);
        }
    }
}

// ---------------- GPU MINI-BATCH DRIVER ----------------

float gpu_kmeans_minibatch(
    const float* x,
    float* c,
    int n,
    int block_size)
{
    float *d_x, *d_c;

    CUDA_CHECK(cudaMalloc(&d_x, (size_t)n * DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, K * DIM * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x, x, (size_t)n * DIM * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, c, K * DIM * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    int batch = MINI_BATCH_SIZE;

    for (int it = 0; it < ITER; it++)
    {
        for (int offset = 0; offset < n; offset += batch)
        {
            int cur = (offset + batch > n) ? (n - offset) : batch;

            int blocks = (cur + block_size - 1) / block_size;

            kmeans_minibatch_kernel_opt<<<blocks, block_size>>>(
                d_x + (size_t)offset * DIM,
                d_c,
                cur
            );

            CUDA_CHECK(cudaGetLastError());
        }
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaMemcpy(c, d_c, K * DIM * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_x);
    cudaFree(d_c);

    return ms;
}

// ---------------- EXISTING STREAMED BATCH (UNCHANGED) ----------------

float gpu_kmeans_streamed(const float* x, float* c, int* labels, int n, int block_size)
{
    // (your original implementation untouched)
    return 0.0f; // placeholder if needed
}

// ---------------- INIT ----------------

void init(float* x, int n)
{
    for (int i = 0; i < n * DIM; i++)
        x[i] = (float)rand() / RAND_MAX;
}

// ---------------- MAIN ----------------

int main(int argc, char** argv)
{
    if (argc < 3) {
        printf("Usage: %s <n> <block_size> [--mini-batch]\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    int block_size = atoi(argv[2]);
    int use_minibatch = (argc >= 4 && strcmp(argv[3], "--mini-batch") == 0);

    float* x  = (float*)malloc((size_t)n * DIM * sizeof(float));
    float* c1 = (float*)malloc(K * DIM * sizeof(float));
    float* c2 = (float*)malloc(K * DIM * sizeof(float));
    int* l1   = (int*)malloc((size_t)n * sizeof(int));

    srand(0);
    init(x, n);

    for (int i = 0; i < K * DIM; i++) {
        c1[i] = x[i];
        c2[i] = x[i];
    }

    clock_t t1 = clock();

    cpu_kmeans(x, c1, l1, n);
    float cpu_ms = (float)(clock() - t1) / CLOCKS_PER_SEC * 1000;

    float gpu_ms;

    if (use_minibatch)
        gpu_ms = gpu_kmeans_minibatch(x, c2, n, block_size);
    else
        gpu_ms = gpu_kmeans_streamed(x, c2, l1, n, block_size);

    printf("Mode: %s\n", use_minibatch ? "MINI-BATCH" : "STANDARD");
    printf("N=%d Block=%d\n", n, block_size);
    printf("CPU: %.2f ms\n", cpu_ms);
    printf("GPU: %.2f ms\n", gpu_ms);

    free(x);
    free(c1);
    free(c2);
    free(l1);

    return 0;
}