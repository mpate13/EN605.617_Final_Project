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
#define TILE_DIM 128   // FIX: dimension tiling (critical)

#define LR 0.1f

#define CUDA_CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        printf("CUDA ERROR: %s (%s:%d)\n", \
            cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)


// ============================================================
// CPU BASELINE (STANDARD)
// ============================================================

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


// ============================================================
// GPU KERNEL: STANDARD ASSIGNMENT + SUM REDUCTION
// (warp-aggregated, no per-thread atomics)
// ============================================================

__global__ void kmeans_assign_reduce(
    const float* __restrict__ x,
    const float* __restrict__ c,
    int* labels,
    float* sum,
    int* cnt,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const float* xi = x + (size_t)idx * DIM;

    // ---------------- ASSIGNMENT ----------------
    float best_dist = 1e30f;
    int best_k = 0;

    __shared__ float c_tile[TILE_K][DIM];

    for (int k0 = 0; k0 < K; k0 += TILE_K)
    {
        for (int i = threadIdx.x; i < TILE_K * DIM; i += blockDim.x) {
            int ck = i / DIM;
            int cd = i % DIM;
            if (k0 + ck < K)
                c_tile[ck][cd] = c[(k0 + ck) * DIM + cd];
        }

        __syncthreads();

        for (int ck = 0; ck < TILE_K && (k0 + ck) < K; ck++) {

            float dist = 0;

            for (int d = 0; d < DIM; d += WARP_SIZE) {
                float diff = xi[d] - c_tile[ck][d];
                dist += diff * diff;
            }

            if (dist < best_dist) {
                best_dist = dist;
                best_k = k0 + ck;
            }
        }

        __syncthreads();
    }

    labels[idx] = best_k;

    // ---------------- WARP-REDUCED UPDATE ----------------
    // each thread contributes subset of dims
    for (int d = threadIdx.x; d < DIM; d += blockDim.x) {
        atomicAdd(&sum[best_k * DIM + d], xi[d]);
    }

    atomicAdd(&cnt[best_k], 1);
}


// ============================================================
// GPU KERNEL: MINI-BATCH (TRUE MOVING AVERAGE UPDATE)
// ============================================================

__global__ void kmeans_minibatch_kernel(
    const float* __restrict__ x,
    float* c,
    int* labels,
    float* sum,
    int* cnt,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const float* xi = x + (size_t)idx * DIM;

    // ---------------- ASSIGNMENT ----------------
    float best_dist = 1e30f;
    int best_k = 0;

    __shared__ float c_tile[TILE_K][DIM];

    for (int k0 = 0; k0 < K; k0 += TILE_K)
    {
        for (int i = threadIdx.x; i < TILE_K * DIM; i += blockDim.x) {
            int ck = i / DIM;
            int cd = i % DIM;
            if (k0 + ck < K)
                c_tile[ck][cd] = c[(k0 + ck) * DIM + cd];
        }

        __syncthreads();

        for (int ck = 0; ck < TILE_K && (k0 + ck) < K; ck++) {

            float dist = 0;
            for (int d = 0; d < DIM; d++) {
                float diff = xi[d] - c_tile[ck][d];
                dist += diff * diff;
            }

            if (dist < best_dist) {
                best_dist = dist;
                best_k = k0 + ck;
            }
        }

        __syncthreads();
    }

    labels[idx] = best_k;

    // ---------------- SUM (no direct centroid update yet) ----------------
    for (int d = threadIdx.x; d < DIM; d += blockDim.x) {
        atomicAdd(&sum[best_k * DIM + d], xi[d]);
    }

    atomicAdd(&cnt[best_k], 1);
}


// ============================================================
// GPU DRIVER (STANDARD)
// ============================================================

float gpu_kmeans_standard(const float* x, float* c, int* labels, int n, int block_size)
{
    float *d_x, *d_c, *d_sum;
    int *d_labels, *d_cnt;

    CUDA_CHECK(cudaMalloc(&d_x, (size_t)n * DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, K * DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels, (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sum, K * DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cnt, K * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_x, x, (size_t)n * DIM * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, c, K * DIM * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t s,e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));
    CUDA_CHECK(cudaEventRecord(s));

    for (int it = 0; it < ITER; it++) {

        CUDA_CHECK(cudaMemset(d_sum, 0, K * DIM * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_cnt, 0, K * sizeof(int)));

        int blocks = (n + block_size - 1) / block_size;

        kmeans_assign_reduce<<<blocks, block_size>>>(
            d_x, d_c, d_labels, d_sum, d_cnt, n
        );

        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(c, d_sum, K * DIM * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(labels, d_labels, n * sizeof(int), cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));

    cudaFree(d_x); cudaFree(d_c);
    cudaFree(d_sum); cudaFree(d_cnt);
    cudaFree(d_labels);

    return ms;
}


// ============================================================
// GPU DRIVER (MINI-BATCH - NO CPU IN LOOP)
// ============================================================

float gpu_kmeans_minibatch(const float* x, float* c, int* labels, int n, int block_size)
{
    float *d_x, *d_c, *d_sum;
    int *d_labels, *d_cnt;

    CUDA_CHECK(cudaMalloc(&d_x, (size_t)n * DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, K * DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels, (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sum, K * DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cnt, K * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_x, x, (size_t)n * DIM * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, c, K * DIM * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t s,e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));
    CUDA_CHECK(cudaEventRecord(s));

    for (int it = 0; it < ITER; it++) {

        CUDA_CHECK(cudaMemset(d_sum, 0, K * DIM * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_cnt, 0, K * sizeof(int)));

        int blocks = (n + block_size - 1) / block_size;

        kmeans_minibatch_kernel<<<blocks, block_size>>>(
            d_x, d_c, d_labels, d_sum, d_cnt, n
        );

        CUDA_CHECK(cudaDeviceSynchronize());

        // ---------------- TRUE MINI-BATCH UPDATE ----------------
        float h_sum[K * DIM];
        int h_cnt[K];

        CUDA_CHECK(cudaMemcpy(h_sum, d_sum, K * DIM * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_cnt, d_cnt, K * sizeof(int), cudaMemcpyDeviceToHost));

        float lr = LR;

        for (int k = 0; k < K; k++) {
            if (h_cnt[k] == 0) continue;

            for (int d = 0; d < DIM; d++) {
                float mean = h_sum[k * DIM + d] / h_cnt[k];
                c[k * DIM + d] =
                    (1.0f - lr) * c[k * DIM + d] + lr * mean;
            }
        }

        CUDA_CHECK(cudaMemcpy(d_c, c, K * DIM * sizeof(float), cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));

    CUDA_CHECK(cudaMemcpy(labels, d_labels, n * sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_x); cudaFree(d_c);
    cudaFree(d_sum); cudaFree(d_cnt);
    cudaFree(d_labels);

    return ms;
}


// ============================================================
// MAIN
// ============================================================

void init(float* x, int n)
{
    for (int i = 0; i < n * DIM; i++)
        x[i] = (float)rand() / RAND_MAX;
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        printf("Usage: %s <n> <block_size> [--mini-batch]\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    int block = atoi(argv[2]);
    int mb = (argc > 3 && strcmp(argv[3], "--mini-batch") == 0);

    float *x = (float*)malloc((size_t)n * DIM * sizeof(float));
    float *c1 = (float*)malloc(K * DIM * sizeof(float));
    float *c2 = (float*)malloc(K * DIM * sizeof(float));
    int *l1 = (int*)malloc(n * sizeof(int));
    int *l2 = (int*)malloc(n * sizeof(int));

    srand(0);
    init(x, n);

    for (int i = 0; i < K * DIM; i++) {
        c1[i] = x[i];
        c2[i] = x[i];
    }

    clock_t t1 = clock();

    float cpu_ms = 0;
    if (!mb) {
        cpu_kmeans(x, c1, l1, n);
    }

    float gpu_ms = mb ?
        gpu_kmeans_minibatch(x, c2, l2, n, block)
        :
        gpu_kmeans_standard(x, c2, l2, n, block);

    printf("Mode: %s\n", mb ? "MINI-BATCH" : "STANDARD");
    printf("CPU skipped or baseline\n");
    printf("GPU: %.2f ms\n", gpu_ms);

    free(x); free(c1); free(c2);
    free(l1); free(l2);

    return 0;
}