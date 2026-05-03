#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <float.h>
#include <time.h>
#include <string.h>

#define K 10
#define DIM 3072
#define ITER 20

#define WARP_SIZE 32
#define MINI_BATCH_SIZE 4096
#define LR 0.1f

#define CUDA_CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        printf("CUDA ERROR: %s (%s:%d)\n", \
            cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// ---------------- CPU STANDARD ----------------

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

// ---------------- GPU ASSIGN ----------------

__global__ void assign_kernel(
    const float* __restrict__ x,
    const float* __restrict__ c,
    int* labels,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const float* xi = x + (size_t)idx * DIM;

    float best_dist = 1e30f;
    int best_k = 0;

    for (int k = 0; k < K; k++) {
        float dist = 0.0f;
        for (int d = 0; d < DIM; d++) {
            float diff = xi[d] - c[k * DIM + d];
            dist += diff * diff;
        }
        if (dist < best_dist) {
            best_dist = dist;
            best_k = k;
        }
    }

    labels[idx] = best_k;
}

// ---------------- GPU UPDATE (batch) ----------------

__global__ void update_kernel(
    const float* __restrict__ x,
    const int* __restrict__ labels,
    float* c,
    int n)
{
    __shared__ float s_sum[K][32];
    __shared__ int s_cnt[K];

    int tid = threadIdx.x;

    for (int k = tid; k < K; k += blockDim.x)
        s_cnt[k] = 0;

    for (int i = tid; i < K * 32; i += blockDim.x)
        ((float*)s_sum)[i] = 0.0f;

    __syncthreads();

    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < n) {
        int k = labels[idx];
        atomicAdd(&s_cnt[k], 1);

        for (int d = 0; d < 32 && d < DIM; d++)
            atomicAdd(&s_sum[k][d], x[idx * DIM + d]);
    }

    __syncthreads();

    for (int k = tid; k < K; k += blockDim.x) {
        if (s_cnt[k] > 0) {
            for (int d = 0; d < 32 && d < DIM; d++)
                c[k * DIM + d] = s_sum[k][d] / s_cnt[k];
        }
    }
}

// ---------------- GPU STANDARD ----------------

float gpu_kmeans_standard(const float* x, float* c, int* labels, int n, int block_size)
{
    float *d_x, *d_c;
    int *d_labels;

    CUDA_CHECK(cudaMalloc(&d_x, (size_t)n * DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, K * DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels, (size_t)n * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_x, x, (size_t)n * DIM * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, c, K * DIM * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int blocks = (n + block_size - 1) / block_size;

    for (int it = 0; it < ITER; it++) {
        assign_kernel<<<blocks, block_size>>>(d_x, d_c, d_labels, n);
        update_kernel<<<blocks, block_size>>>(d_x, d_labels, d_c, n);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    CUDA_CHECK(cudaMemcpy(labels, d_labels, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_x);
    cudaFree(d_c);
    cudaFree(d_labels);

    return ms;
}

// ---------------- FUSED MINI-BATCH ----------------

__global__ void kmeans_minibatch_fused(
    const float* __restrict__ x,
    float* __restrict__ c,
    int n)
{
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    __shared__ float s_sum[K][32];
    __shared__ int s_cnt[K];

    for (int k = tid; k < K; k += blockDim.x)
        s_cnt[k] = 0;

    for (int i = tid; i < K * 32; i += blockDim.x)
        ((float*)s_sum)[i] = 0.0f;

    __syncthreads();

    if (idx < n)
    {
        const float* xi = x + (size_t)idx * DIM;

        float best_dist = 1e30f;
        int best_k = 0;

        for (int k = 0; k < K; k++) {
            float dist = 0.0f;
            for (int d = 0; d < DIM; d++) {
                float diff = xi[d] - c[k * DIM + d];
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_k = k;
            }
        }

        atomicAdd(&s_cnt[best_k], 1);

        for (int d = 0; d < 32 && d < DIM; d++)
            atomicAdd(&s_sum[best_k][d], xi[d]);
    }

    __syncthreads();

    for (int k = tid; k < K; k += blockDim.x)
    {
        if (s_cnt[k] > 0) {
            for (int d = 0; d < 32 && d < DIM; d++) {

                float mean = s_sum[k][d] / s_cnt[k];

                c[k * DIM + d] =
                    (1.0f - LR) * c[k * DIM + d] + LR * mean;
            }
        }
    }
}

// ---------------- GPU MINI-BATCH ----------------

float gpu_kmeans_minibatch(const float* x, float* c, int n, int block_size)
{
    float *d_x, *d_c;

    CUDA_CHECK(cudaMalloc(&d_x, (size_t)n * DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, K * DIM * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x, x, (size_t)n * DIM * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, c, K * DIM * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int it = 0; it < ITER; it++) {
        for (int offset = 0; offset < n; offset += MINI_BATCH_SIZE)
        {
            int cur = (offset + MINI_BATCH_SIZE > n) ? (n - offset) : MINI_BATCH_SIZE;

            int blocks = (cur + block_size - 1) / block_size;

            kmeans_minibatch_fused<<<blocks, block_size>>>(
                d_x + (size_t)offset * DIM,
                d_c,
                cur
            );
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaFree(d_x);
    cudaFree(d_c);

    return ms;
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
    int n = atoi(argv[1]);
    int block_size = atoi(argv[2]);
    int minibatch = (argc >= 4 && strcmp(argv[3], "--mini-batch") == 0);

    float* x  = (float*)malloc((size_t)n * DIM * sizeof(float));
    float* c1 = (float*)malloc(K * DIM * sizeof(float));
    float* c2 = (float*)malloc(K * DIM * sizeof(float));
    int* l1   = (int*)malloc((size_t)n * sizeof(int));
    int* l2   = (int*)malloc((size_t)n * sizeof(int));

    srand(0);
    init(x, n);

    for (int i = 0; i < K * DIM; i++) {
        c1[i] = x[i];
        c2[i] = x[i];
    }

    clock_t t1 = clock();
    cpu_kmeans(x, c1, l1, n);
    float cpu_ms = (float)(clock() - t1) / CLOCKS_PER_SEC * 1000;

    float gpu_ms = minibatch ?
        gpu_kmeans_minibatch(x, c2, n, block_size) :
        gpu_kmeans_standard(x, c2, l2, n, block_size);

    printf("Mode: %s\n", minibatch ? "MINI-BATCH" : "STANDARD");
    printf("N=%d Block=%d\n", n, block_size);
    printf("CPU: %.2f ms\n", cpu_ms);
    printf("GPU: %.2f ms\n", gpu_ms);
    printf("Speedup: %.2fx\n", cpu_ms / gpu_ms);

    return 0;
}