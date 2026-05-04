#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <float.h>
#include <time.h>
#include <string.h>

#define K 10
#define DIM 128              // reduced so it actually runs fast in practice
#define ITER 10
#define NUM_STREAMS 4
#define CHUNK_SIZE 4096
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
// STREAM CONTEXT
// ============================================================

struct StreamContext {
    cudaStream_t streams[NUM_STREAMS];
    float* d_x[NUM_STREAMS];
    int* d_labels[NUM_STREAMS];

    float* d_sum;
    int* d_cnt;
};

// ============================================================
// CPU KMEANS (baseline)
// ============================================================

void cpu_kmeans(const float* x, float* c, int* labels, int n) {
    for (int it = 0; it < ITER; it++) {
        for (int i = 0; i < n; i++) {

            float best = 1e30f;
            int best_k = 0;

            for (int k = 0; k < K; k++) {
                float d = 0;
                for (int j = 0; j < DIM; j++) {
                    float diff = x[i * DIM + j] - c[k * DIM + j];
                    d += diff * diff;
                }
                if (d < best) {
                    best = d;
                    best_k = k;
                }
            }
            labels[i] = best_k;
        }

        float sum[K][DIM] = {0};
        int cnt[K] = {0};

        for (int i = 0; i < n; i++) {
            int k = labels[i];
            cnt[k]++;
            for (int j = 0; j < DIM; j++)
                sum[k][j] += x[i * DIM + j];
        }

        for (int k = 0; k < K; k++) {
            if (cnt[k] == 0) continue;
            for (int j = 0; j < DIM; j++)
                c[k * DIM + j] = sum[k][j] / cnt[k];
        }
    }
}

// ============================================================
// CUDA KERNEL
// ============================================================

__global__ void kmeans_kernel(
    const float* x,
    const float* c,
    int* labels,
    float* sum,
    int* cnt,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float* p = x + (size_t)i * DIM;

    float best = 1e30f;
    int best_k = 0;

    for (int k = 0; k < K; k++) {
        float d = 0;
        const float* ck = c + k * DIM;
        for (int j = 0; j < DIM; j++) {
            float diff = p[j] - ck[j];
            d += diff * diff;
        }
        if (d < best) {
            best = d;
            best_k = k;
        }
    }

    labels[i] = best_k;

    atomicAdd(&cnt[best_k], 1);
    for (int j = 0; j < DIM; j++)
        atomicAdd(&sum[best_k * DIM + j], p[j]);
}

// ============================================================
// STREAM SETUP
// ============================================================

void init_streams(StreamContext* ctx, int chunk_size) {
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaStreamCreate(&ctx->streams[i]));
        CUDA_CHECK(cudaMalloc(&ctx->d_x[i], chunk_size * DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ctx->d_labels[i], chunk_size * sizeof(int)));
    }

    CUDA_CHECK(cudaMalloc(&ctx->d_sum, K * DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_cnt, K * sizeof(int)));
}

void free_streams(StreamContext* ctx) {
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(ctx->streams[i]);
        cudaFree(ctx->d_x[i]);
        cudaFree(ctx->d_labels[i]);
    }
    cudaFree(ctx->d_sum);
    cudaFree(ctx->d_cnt);
}

// ============================================================
// STREAMED GPU KMEANS
// ============================================================

void gpu_streamed_kmeans(float* h_x,
                         float* d_c,
                         int* labels,
                         int n,
                         int block_size)
{
    StreamContext ctx;
    init_streams(&ctx, CHUNK_SIZE);

    int chunks = (n + CHUNK_SIZE - 1) / CHUNK_SIZE;
    int blocks = (CHUNK_SIZE + block_size - 1) / block_size;

    for (int it = 0; it < ITER; it++) {

        for (int c = 0; c < chunks; c++) {

            int sid = c % NUM_STREAMS;
            cudaStream_t s = ctx.streams[sid];

            int offset = c * CHUNK_SIZE;
            int size = (offset + CHUNK_SIZE > n) ? (n - offset) : CHUNK_SIZE;

            CUDA_CHECK(cudaMemsetAsync(ctx.d_sum, 0, K * DIM * sizeof(float), s));
            CUDA_CHECK(cudaMemsetAsync(ctx.d_cnt, 0, K * sizeof(int), s));

            CUDA_CHECK(cudaMemcpyAsync(
                ctx.d_x[sid],
                h_x + (size_t)offset * DIM,
                size * DIM * sizeof(float),
                cudaMemcpyHostToDevice,
                s));

            kmeans_kernel<<<blocks, block_size, 0, s>>>(
                ctx.d_x[sid],
                d_c,
                ctx.d_labels[sid],
                ctx.d_sum,
                ctx.d_cnt,
                size
            );
        }

        CUDA_CHECK(cudaDeviceSynchronize());
    }

    free_streams(&ctx);
}

// ============================================================
// MAIN
// ============================================================

int main(int argc, char** argv) {
    int n = 20000;
    int block = 256;

    float* x = (float*)malloc(n * DIM * sizeof(float));
    float* c = (float*)malloc(K * DIM * sizeof(float));
    int* labels = (int*)malloc(n * sizeof(int));

    srand(0);
    for (int i = 0; i < n * DIM; i++)
        x[i] = (float)rand() / RAND_MAX;

    for (int i = 0; i < K * DIM; i++)
        c[i] = x[i];

    float *d_c;
    CUDA_CHECK(cudaMalloc(&d_c, K * DIM * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_c, c, K * DIM * sizeof(float), cudaMemcpyHostToDevice));

    clock_t start = clock();

    cpu_kmeans(x, c, labels, n);

    clock_t cpu_time = clock() - start;

    printf("CPU done: %.2f sec\n",
        (float)cpu_time / CLOCKS_PER_SEC);

    CUDA_CHECK(cudaMemcpy(d_c, c, K * DIM * sizeof(float), cudaMemcpyHostToDevice));

    gpu_streamed_kmeans(x, d_c, labels, n, block);

    printf("GPU streamed done\n");

    cudaFree(d_c);
    free(x);
    free(c);
    free(labels);

    return 0;
}