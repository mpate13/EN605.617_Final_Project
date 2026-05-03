#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <float.h>
#include <time.h>

#define K 10
#define DIM 3072
#define ITER 20

#define NUM_STREAMS 4
#define CHUNK_SIZE 50000

#define CUDA_CHECK(x) \
if ((x) != cudaSuccess) { \
    printf("CUDA ERROR: %s (%s:%d)\n", cudaGetErrorString(x), __FILE__, __LINE__); \
    exit(1); \
}

// ---------------- CPU (correct baseline) ----------------

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
                tmp[k * DIM + d_i] = 0;
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

// ---------------- GPU TILED + STREAMED CORE ----------------

__global__ void kmeans_kernel_tiled(
    const float* __restrict__ x,
    const float* __restrict__ c,
    int* labels,
    float* sum,
    int* cnt,
    int n)
{
    extern __shared__ float tile[];   // [BLOCK_SIZE * DIM]

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    float* xi = tile + tid * DIM;

    if (i < n) {
        for (int d = 0; d < DIM; d++)
            xi[d] = x[i * DIM + d];
    }

    __syncthreads();
    if (i >= n) return;

    float best = 1e30f;
    int bestk = 0;

    // 🔥 register cached centroid loop
    for (int k = 0; k < K; k++) {

        float d = 0;

        const float* ck = c + k * DIM;

        #pragma unroll 4
        for (int d_i = 0; d_i < DIM; d_i++) {
            float diff = xi[d_i] - ck[d_i];
            d += diff * diff;
        }

        if (d < best) {
            best = d;
            bestk = k;
        }
    }

    labels[i] = bestk;

    atomicAdd(&cnt[bestk], 1);

    for (int d_i = 0; d_i < DIM; d_i++)
        atomicAdd(&sum[bestk * DIM + d_i], xi[d_i]);
}

// ---------------- GPU DRIVER ----------------

float gpu_kmeans_streamed(const float* x, float* c, int* labels, int n, int block_size)
{
    cudaStream_t streams[NUM_STREAMS];

    float *d_x[NUM_STREAMS];
    float *d_sum[NUM_STREAMS];
    int   *d_cnt[NUM_STREAMS];

    float *d_c;
    int *d_labels;

    const int chunk = CHUNK_SIZE;

    for (int s = 0; s < NUM_STREAMS; s++) {
        CUDA_CHECK(cudaStreamCreate(&streams[s]));

        CUDA_CHECK(cudaMalloc(&d_x[s], chunk * DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_sum[s], K * DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_cnt[s], K * sizeof(int)));
    }

    CUDA_CHECK(cudaMalloc(&d_c, K * DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels, n * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_c, c,
        K * DIM * sizeof(float),
        cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int it = 0; it < ITER; it++) {

        for (int s = 0; s < NUM_STREAMS; s++) {
            CUDA_CHECK(cudaMemsetAsync(d_sum[s], 0, K * DIM * sizeof(float), streams[s]));
            CUDA_CHECK(cudaMemsetAsync(d_cnt[s], 0, K * sizeof(int), streams[s]));
        }

        for (int offset = 0; offset < n; offset += chunk) {

            int cur = (offset + chunk > n) ? (n - offset) : chunk;
            int s = (offset / chunk) % NUM_STREAMS;

            CUDA_CHECK(cudaMemcpyAsync(
                d_x[s],
                x + (size_t)offset * DIM,
                (size_t)cur * DIM * sizeof(float),
                cudaMemcpyHostToDevice,
                streams[s]
            ));

            int blocks = (cur + block_size - 1) / block_size;

            size_t shared = block_size * DIM * sizeof(float);

            kmeans_kernel_tiled<<<blocks, block_size, shared, streams[s]>>>(
                d_x[s],
                d_c,
                d_labels + offset,
                d_sum[s],
                d_cnt[s],
                cur
            );

            CUDA_CHECK(cudaGetLastError());
        }

        for (int s = 0; s < NUM_STREAMS; s++)
            CUDA_CHECK(cudaStreamSynchronize(streams[s]));

        CUDA_CHECK(cudaMemcpy(c, d_c,
            K * DIM * sizeof(float),
            cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaMemcpy(labels, d_labels,
        n * sizeof(int),
        cudaMemcpyDeviceToHost));

    for (int s = 0; s < NUM_STREAMS; s++) {
        cudaFree(d_x[s]);
        cudaFree(d_sum[s]);
        cudaFree(d_cnt[s]);
        cudaStreamDestroy(streams[s]);
    }

    cudaFree(d_c);
    cudaFree(d_labels);

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

    float* x = (float*)malloc(n * DIM * sizeof(float));
    float* c1 = (float*)malloc(K * DIM * sizeof(float));
    float* c2 = (float*)malloc(K * DIM * sizeof(float));

    int* l1 = (int*)malloc(n * sizeof(int));
    int* l2 = (int*)malloc(n * sizeof(int));

    srand(0);
    init(x, n);

    for (int i = 0; i < K * DIM; i++) {
        c1[i] = x[i];
        c2[i] = x[i];
    }

    clock_t t1 = clock();
    cpu_kmeans(x, c1, l1, n);
    float cpu_ms = (float)(clock() - t1) / CLOCKS_PER_SEC * 1000;

    float gpu_ms = gpu_kmeans_streamed(x, c2, l2, n, block_size);

    printf("N=%d Block=%d\n", n, block_size);
    printf("CPU: %.2f ms\n", cpu_ms);
    printf("GPU: %.2f ms\n", gpu_ms);
    printf("Speedup: %.2fx\n", cpu_ms / gpu_ms);

    free(x);
    free(c1);
    free(c2);
    free(l1);
    free(l2);

    return 0;
}