#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <float.h>
#include <string.h>

#define K 10
#define DIM 3072
#define ITER 20

// fixed dataset size (NO runtime n argument anymore)
#define N 200000

#define TILE_DIM 128
#define STREAMS 2
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
// CPU REFERENCE
// ============================================================

void cpu_kmeans(const float* x, float* c, int* labels)
{
    float tmp[K * DIM];
    int cnt[K];

    for (int it = 0; it < ITER; it++) {

        for (int i = 0; i < N; i++) {

            float best = 1e30f;
            int bestk = 0;

            for (int k = 0; k < K; k++) {

                float dist = 0.0f;

                for (int d = 0; d < DIM; d++) {
                    float diff = x[i * DIM + d] - c[k * DIM + d];
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
                tmp[k * DIM + d] = 0;
        }

        for (int i = 0; i < N; i++) {
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
// GPU ASSIGN (TILED CENTROIDS)
// ============================================================

__global__ void assign_kernel(
    const float* __restrict__ x,
    const float* __restrict__ c,
    int* labels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const float* xi = x + (size_t)idx * DIM;

    __shared__ float c_tile[K * DIM];

    for (int i = threadIdx.x; i < K * DIM; i += blockDim.x)
        c_tile[i] = c[i];

    __syncthreads();

    float best = 1e30f;
    int bestk = 0;

    #pragma unroll
    for (int k = 0; k < K; k++) {
        float dist = 0.0f;

        for (int d = 0; d < DIM; d++) {
            float diff = xi[d] - c_tile[k * DIM + d];
            dist += diff * diff;
        }

        if (dist < best) {
            best = dist;
            bestk = k;
        }
    }

    labels[idx] = bestk;
}


// ============================================================
// BLOCK REDUCTION MINI-BATCH
// ============================================================

__global__ void accumulate_kernel(
    const float* __restrict__ x,
    const int* labels,
    float* sum,
    int* cnt)
{
    __shared__ float sh_sum[K * TILE_DIM];
    __shared__ int sh_cnt[K];

    int tid = threadIdx.x;

    for (int i = tid; i < K * TILE_DIM; i += blockDim.x)
        sh_sum[i] = 0.0f;

    for (int k = tid; k < K; k += blockDim.x)
        sh_cnt[k] = 0;

    __syncthreads();

    int idx = blockIdx.x * blockDim.x + tid;
    if (idx < N) {

        int k = labels[idx];
        const float* xi = x + (size_t)idx * DIM;

        atomicAdd(&sh_cnt[k], 1);

        for (int d0 = 0; d0 < DIM; d0 += TILE_DIM) {
            #pragma unroll
            for (int d = 0; d < TILE_DIM; d++) {
                sh_sum[k * TILE_DIM + d] += xi[d0 + d];
            }
        }
    }

    __syncthreads();

    for (int k = tid; k < K; k += blockDim.x) {

        int c_k = sh_cnt[k];
        if (c_k > 0) {
            for (int d = 0; d < TILE_DIM; d++) {
                atomicAdd(&sum[k * DIM + d],
                          sh_sum[k * TILE_DIM + d] / c_k);
            }
            atomicAdd(&cnt[k], c_k);
        }
    }
}


// ============================================================
// CENTROID UPDATE
// ============================================================

__global__ void update_centroids_kernel(
    float* c,
    const float* sum,
    const int* cnt)
{
    int k = blockIdx.x;
    if (k >= K || cnt[k] == 0) return;

    for (int d = threadIdx.x; d < DIM; d += blockDim.x) {
        float mean = sum[k * DIM + d] / cnt[k];
        c[k * DIM + d] =
            (1.0f - LR) * c[k * DIM + d] + LR * mean;
    }
}


// ============================================================
// MINI-BATCH DRIVER (ASYNC STREAMED)
// ============================================================

float gpu_minibatch(const float* x, float* c, int* labels,
                    int threads, int batch_size)
{
    float *d_x[STREAMS], *d_c, *d_sum;
    int *d_labels, *d_cnt;

    cudaStream_t stream[STREAMS];

    for (int i = 0; i < STREAMS; i++)
        CUDA_CHECK(cudaStreamCreate(&stream[i]));

    for (int i = 0; i < STREAMS; i++)
        CUDA_CHECK(cudaMalloc(&d_x[i], batch_size * DIM * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_c, K * DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sum, K * DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cnt, K * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_c, c, K * DIM * sizeof(float),
                          cudaMemcpyHostToDevice));

    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));
    CUDA_CHECK(cudaEventRecord(s));

    int num_chunks = (N + batch_size - 1) / batch_size;

    int blocks_per_chunk;

    for (int it = 0; it < ITER; it++) {

        CUDA_CHECK(cudaMemset(d_sum, 0, K * DIM * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_cnt, 0, K * sizeof(int)));

        for (int i = 0; i < num_chunks; i += STREAMS) {

            for (int sidx = 0; sidx < STREAMS; sidx++) {

                int chunk = i + sidx;
                if (chunk >= num_chunks) continue;

                int offset = chunk * batch_size;
                int current_n = (offset + batch_size > N)
                                ? (N - offset)
                                : batch_size;

                blocks_per_chunk =
                    (current_n + threads - 1) / threads;

                CUDA_CHECK(cudaMemcpyAsync(
                    d_x[sidx],
                    x + offset * DIM,
                    current_n * DIM * sizeof(float),
                    cudaMemcpyHostToDevice,
                    stream[sidx]
                ));

                assign_kernel<<<blocks_per_chunk, threads, 0, stream[sidx]>>>(
                    d_x[sidx], d_c, d_labels + offset
                );

                accumulate_kernel<<<blocks_per_chunk, threads, 0, stream[sidx]>>>(
                    d_x[sidx], d_labels + offset, d_sum, d_cnt
                );
            }
        }

        for (int i = 0; i < STREAMS; i++)
            CUDA_CHECK(cudaStreamSynchronize(stream[i]));

        update_centroids_kernel<<<K, threads>>>(d_c, d_sum, d_cnt);
    }

    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));

    CUDA_CHECK(cudaMemcpy(c, d_c, K * DIM * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(labels, d_labels, N * sizeof(int),
                          cudaMemcpyDeviceToHost));

    for (int i = 0; i < STREAMS; i++) {
        cudaFree(d_x[i]);
        cudaStreamDestroy(stream[i]);
    }

    cudaFree(d_c);
    cudaFree(d_labels);
    cudaFree(d_sum);
    cudaFree(d_cnt);

    return ms;
}


// ============================================================
// INIT + MAIN
// ============================================================

void init(float* x)
{
    for (int i = 0; i < N * DIM; i++)
        x[i] = (float)rand() / RAND_MAX;
}

int main(int argc, char** argv)
{
    if (argc < 4) {
        printf("Usage: %s <threads> <batch_size> [--mini-batch]\n", argv[0]);
        return 1;
    }

    int threads = atoi(argv[1]);
    int batch_size = atoi(argv[2]);
    int use_mb = (argc > 3 && strcmp(argv[3], "--mini-batch") == 0);

    float *x = (float*)malloc(N * DIM * sizeof(float));
    float *c = (float*)malloc(K * DIM * sizeof(float));
    int *labels = (int*)malloc(N * sizeof(int));

    srand(0);
    init(x);

    for (int i = 0; i < K * DIM; i++)
        c[i] = x[i];

    float ms;

    if (use_mb) {
        ms = gpu_minibatch(x, c, labels, threads, batch_size);
        printf("MODE: MINI-BATCH (ASYNC STREAMED)\n");
    } else {
        printf("MODE: CPU ONLY (GPU standard omitted)\n");
        cpu_kmeans(x, c, labels);
        ms = 0.0f;
    }

    printf("N=%d Threads=%d Batch=%d\n", N, threads, batch_size);
    printf("Time=%.2f ms\n", ms);

    free(x);
    free(c);
    free(labels);

    return 0;
}