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
#define TILE_K 2

#define CUDA_CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        printf("CUDA ERROR: %s (%s:%d)\n", \
            cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)


// ============================================================
// CPU BASELINE
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
                float dist = 0.0f;
                const float* ck = c + k * DIM;

                for (int d = 0; d < DIM; d++) {
                    float diff = x[i * DIM + d] - ck[d];
                    dist += diff * diff;
                }

                if (dist < best) {
                    best = dist;
                    bestk = k;
                }
            }

            labels[i] = bestk;
        }

        for (int k = 0; k < K; k++) {
            cnt[k] = 0;
            for (int d = 0; d < DIM; d++)
                tmp[k * DIM + d] = 0.0f;
        }

        for (int i = 0; i < n; i++) {
            int k = labels[i];
            cnt[k]++;
            for (int d = 0; d < DIM; d++)
                tmp[k * DIM + d] += x[i * DIM + d];
        }

        for (int k = 0; k < K; k++) {
            if (cnt[k] == 0) continue;
            for (int d = 0; d < DIM; d++)
                c[k * DIM + d] = tmp[k * DIM + d] / cnt[k];
        }
    }
}


// ============================================================
// CENTROID UPDATE (MINIBATCH CORRECT VERSION)
// ============================================================

__global__ void update_centroids_minibatch(
    float* c,
    const float* sum,
    const int* cnt,
    float lr)
{
    int k = blockIdx.x;
    if (k >= K || cnt[k] == 0) return;

    int count = cnt[k];

    for (int d = threadIdx.x; d < DIM; d += blockDim.x) {

        float batch_mean = sum[k * DIM + d] / count;

        float old = c[k * DIM + d];
        c[k * DIM + d] = old + lr * (batch_mean - old);
    }
}


// ============================================================
// GPU KERNEL: ASSIGN + REDUCE
// ============================================================

__global__ void kmeans_minibatch_kernel(
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

    float best_dist = 1e30f;
    int best_k = 0;

    __shared__ float c_tile[TILE_K * DIM];

    for (int k0 = 0; k0 < K; k0 += TILE_K)
    {
        for (int i = threadIdx.x; i < TILE_K * DIM; i += blockDim.x) {
            int ck = i / DIM;
            int cd = i % DIM;

            if (k0 + ck < K)
                c_tile[ck * DIM + cd] = c[(k0 + ck) * DIM + cd];
        }

        __syncthreads();

        for (int ck = 0; ck < TILE_K && (k0 + ck) < K; ck++) {

            const float* cptr = &c_tile[ck * DIM];
            float dist = 0.0f;

            for (int d = 0; d < DIM; d++) {
                float diff = xi[d] - cptr[d];
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

    for (int d = threadIdx.x; d < DIM; d += blockDim.x) {
        atomicAdd(&sum[best_k * DIM + d], xi[d]);
    }

    if (threadIdx.x == 0) {
        atomicAdd(&cnt[best_k], 1);
    }
}


// ============================================================
// STANDARD DRIVER
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

    int blocks = (n + block_size - 1) / block_size;

    for (int it = 0; it < ITER; it++) {

        CUDA_CHECK(cudaMemset(d_sum, 0, K * DIM * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_cnt, 0, K * sizeof(int)));

        kmeans_minibatch_kernel<<<blocks, block_size>>>(
            d_x, d_c, d_labels, d_sum, d_cnt, n
        );

        CUDA_CHECK(cudaDeviceSynchronize());

        update_centroids_minibatch<<<K, block_size>>>(
            d_c, d_sum, d_cnt, 1.0f
        );

        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));

    cudaFree(d_x);
    cudaFree(d_c);
    cudaFree(d_sum);
    cudaFree(d_cnt);
    cudaFree(d_labels);

    return ms;
}


// ============================================================
// MINI-BATCH DRIVER (FIXED)
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

    int batch_size = n / 10;
    float lr = 0.1f;

    int blocks = (batch_size + block_size - 1) / block_size;

    for (int it = 0; it < ITER; it++) {

        int offset = rand() % (n - batch_size);

        CUDA_CHECK(cudaMemset(d_sum, 0, K * DIM * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_cnt, 0, K * sizeof(int)));

        kmeans_minibatch_kernel<<<blocks, block_size>>>(
            d_x + (size_t)offset * DIM,
            d_c,
            d_labels + offset,
            d_sum,
            d_cnt,
            batch_size
        );

        CUDA_CHECK(cudaDeviceSynchronize());

        update_centroids_minibatch<<<K, block_size>>>(
            d_c, d_sum, d_cnt, lr
        );

        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));

    cudaFree(d_x);
    cudaFree(d_c);
    cudaFree(d_sum);
    cudaFree(d_cnt);
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
    float *c = (float*)malloc(K * DIM * sizeof(float));
    int *labels = (int*)malloc(n * sizeof(int));

    srand(0);
    init(x, n);

    for (int i = 0; i < K * DIM; i++)
        c[i] = x[i];

    float gpu_ms = mb ?
        gpu_kmeans_minibatch(x, c, labels, n, block)
        :
        gpu_kmeans_standard(x, c, labels, n, block);

    printf("Mode: %s\n", mb ? "MINI-BATCH" : "STANDARD");
    printf("GPU Time: %.2f ms\n", gpu_ms);

    free(x);
    free(c);
    free(labels);

    return 0;
}