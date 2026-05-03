#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <float.h>
#include <time.h>

#define K 10
#define DIM 3072
#define ITER 20

#define NUM_STREAMS 4
#define CHUNK_SIZE 50000   // safe for Tesla T4

// ---------------- ERROR CHECK ----------------

#define CUDA_CHECK(x) \
if ((x) != cudaSuccess) { \
    printf("CUDA ERROR: %s (%s:%d)\n", \
    cudaGetErrorString(x), __FILE__, __LINE__); \
    exit(1); \
}

// ---------------- CPU BASELINE ----------------

void cpu_kmeans(const float* x, float* c, int* labels, int n)
{
    float tmp[K * DIM];
    int cnt[K];

    for (int it = 0; it < ITER; it++) {

        // assignment
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

        // reset
        for (int k = 0; k < K; k++) {
            cnt[k] = 0;
            for (int d_i = 0; d_i < DIM; d_i++)
                tmp[k * DIM + d_i] = 0;
        }

        // accumulate
        for (int i = 0; i < n; i++) {
            int k = labels[i];
            cnt[k]++;
            for (int d_i = 0; d_i < DIM; d_i++) {
                tmp[k * DIM + d_i] += x[i * DIM + d_i];
            }
        }

        // normalize
        for (int k = 0; k < K; k++) {
            if (cnt[k] == 0) continue;
            for (int d_i = 0; d_i < DIM; d_i++) {
                c[k * DIM + d_i] = tmp[k * DIM + d_i] / cnt[k];
            }
        }
    }
}

// ---------------- GPU KERNEL ----------------

__global__ void assign_kernel(
    const float* x,
    const float* c,
    int* labels,
    float* sum,
    int* cnt,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float xi[DIM];

    for (int d = 0; d < DIM; d++)
        xi[d] = x[i * DIM + d];

    float best = 1e30f;
    int bestk = 0;

    for (int k = 0; k < K; k++) {
        float d = 0;

        #pragma unroll 4
        for (int j = 0; j < DIM; j++) {
            float diff = xi[j] - c[k * DIM + j];
            d += diff * diff;
        }

        if (d < best) {
            best = d;
            bestk = k;
        }
    }

    labels[i] = bestk;

    atomicAdd(&cnt[bestk], 1);
    for (int j = 0; j < DIM; j++) {
        atomicAdd(&sum[bestk * DIM + j], xi[j]);
    }
}

// ---------------- GPU DRIVER ----------------

float gpu_kmeans_streamed(const float* x, float* c, int* labels, int n, int block_size)
{
    cudaStream_t streams[NUM_STREAMS];

    float *d_x[NUM_STREAMS];   // 🔥 FIX: per-stream buffers
    float *d_c, *d_sum;
    int *d_cnt, *d_labels;

    const int chunk = CHUNK_SIZE;

    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));

        CUDA_CHECK(cudaMalloc(&d_x[i], chunk * DIM * sizeof(float)));
    }

    CUDA_CHECK(cudaMalloc(&d_c, K * DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum, K * DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cnt, K * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_labels, n * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_c, c,
        K * DIM * sizeof(float),
        cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int it = 0; it < ITER; it++) {

        CUDA_CHECK(cudaMemset(d_sum, 0, K * DIM * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_cnt, 0, K * sizeof(int)));

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

            assign_kernel<<<blocks, block_size, 0, streams[s]>>>(
                d_x[s],                 // 🔥 FIX: stream-specific buffer
                d_c,
                d_labels + offset,
                d_sum,
                d_cnt,
                cur
            );

            CUDA_CHECK(cudaGetLastError());
        }

        for (int i = 0; i < NUM_STREAMS; i++)
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));

        CUDA_CHECK(cudaDeviceSynchronize());

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

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaFree(d_x[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaFree(d_c);
    cudaFree(d_sum);
    cudaFree(d_cnt);
    cudaFree(d_labels);

    return ms;
}

// ---------------- DATA INIT ----------------

void init(float* x, int n)
{
    for (int i = 0; i < n * DIM; i++)
        x[i] = (float)rand() / RAND_MAX;
}

// ---------------- MAIN ----------------

int main(int argc, char** argv)
{
    if (argc < 3) {
        printf("Usage: %s <total_threads> <block_size>\n", argv[0]);
        return 1;
    }

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