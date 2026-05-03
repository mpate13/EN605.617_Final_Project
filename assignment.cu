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
#define WARP_SIZE 32

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
            for (int d_i = 0; d_i < DIM; d_i++) {
                tmp[k * DIM + d_i] += x[i * DIM + d_i];
            }
        }

        for (int k = 0; k < K; k++) {
            if (cnt[k] == 0) continue;
            for (int d_i = 0; d_i < DIM; d_i++) {
                c[k * DIM + d_i] = tmp[k * DIM + d_i] / cnt[k];
            }
        }
    }
}

// ---------------- WARP AGGREGATED KERNEL ----------------

__global__ void kmeans_kernel_warp(
    const float* __restrict__ x,
    const float* __restrict__ c,
    int* labels,
    float* out_sum,
    int* out_cnt,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float* xi = &x[i * DIM];

    float best = 1e30f;
    int bestk = 0;

    // ---------------- ASSIGN ----------------
    for (int k = 0; k < K; k++) {
        float d = 0;
        const float* ck = &c[k * DIM];

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

    // ---------------- WARP AGGREGATION ----------------
    int lane = threadIdx.x % WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;

    __shared__ float warp_sum[K][32]; // max 32 warps per block
    __shared__ int warp_cnt[K][32];

    if (lane == 0) {
        for (int k = 0; k < K; k++) {
            warp_cnt[k][warp] = 0;
            for (int d = 0; d < DIM; d++) {
                warp_sum[k][warp] = 0.0f;
            }
        }
    }

    __syncwarp();

    atomicAdd(&warp_cnt[bestk][warp], 1);
    for (int d = 0; d < DIM; d++) {
        atomicAdd(&warp_sum[bestk][warp], xi[d]);
    }

    __syncthreads();

    if (lane == 0) {
        for (int k = 0; k < K; k++) {
            atomicAdd(&out_cnt[k], warp_cnt[k][warp]);
        }
    }

    __syncthreads();

    if (lane == 0) {
        for (int k = 0; k < K; k++) {
            atomicAdd(&out_cnt[k], warp_cnt[k][warp]);
            for (int d = 0; d < DIM; d++) {
                atomicAdd(&out_sum[k * DIM + d], warp_sum[k][warp]);
            }
        }
    }
}

// ---------------- DRIVER ----------------

float gpu_kmeans_streamed(const float* x, float* c, int* labels, int n, int block_size)
{
    cudaStream_t streams[NUM_STREAMS];

    float *d_x[NUM_STREAMS];
    float *d_sum;
    int *d_cnt;
    float *d_c;
    int *d_labels;

    int chunk = CHUNK_SIZE;

    for (int s = 0; s < NUM_STREAMS; s++) {
        CUDA_CHECK(cudaStreamCreate(&streams[s]));
        CUDA_CHECK(cudaMalloc(&d_x[s], chunk * DIM * sizeof(float)));
    }

    CUDA_CHECK(cudaMalloc(&d_sum, K * DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cnt, K * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_c, K * DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels, (size_t)n * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_c, c, K * DIM * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int it = 0; it < ITER; it++) {

        CUDA_CHECK(cudaMemset(d_sum, 0, K * DIM * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_cnt, 0, K * sizeof(int)));

        for (int offset = 0; offset < n; offset += chunk) {

            int cur = chunk;
            if (offset + cur > n) cur = n - offset;
            if (cur <= 0) continue;

            int s = (offset / chunk) % NUM_STREAMS;

            CUDA_CHECK(cudaMemcpyAsync(
                d_x[s],
                x + (size_t)offset * DIM,
                (size_t)cur * DIM * sizeof(float),
                cudaMemcpyHostToDevice,
                streams[s]
            ));

            int blocks = (cur + block_size - 1) / block_size;

            kmeans_kernel_warp<<<blocks, block_size, 0, streams[s]>>>(
                d_x[s],
                d_c,
                d_labels + offset,
                d_sum,
                d_cnt,
                cur
            );

            CUDA_CHECK(cudaGetLastError());
        }

        for (int s = 0; s < NUM_STREAMS; s++)
            CUDA_CHECK(cudaStreamSynchronize(streams[s]));

        CUDA_CHECK(cudaMemcpy(c, d_c, K * DIM * sizeof(float), cudaMemcpyDeviceToHost));
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    CUDA_CHECK(cudaMemcpy(labels, d_labels, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost));

    for (int s = 0; s < NUM_STREAMS; s++) {
        cudaFree(d_x[s]);
        cudaStreamDestroy(streams[s]);
    }

    cudaFree(d_sum);
    cudaFree(d_cnt);
    cudaFree(d_c);
    cudaFree(d_labels);

    return ms;
}

// ---------------- INIT + MAIN ----------------

void init(float* x, int n)
{
    for (int i = 0; i < n * DIM; i++)
        x[i] = (float)rand() / RAND_MAX;
}

int main(int argc, char** argv)
{
    int n = atoi(argv[1]);
    int block_size = atoi(argv[2]);

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