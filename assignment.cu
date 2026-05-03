#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <float.h>
#include <time.h>

#define K 10
#define DIM 3072
#define ITER 20

#define WARP_SIZE 32

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
                tmp[k * DIM + d] = 0;
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

// ---------------- WARP REDUCTION ----------------

__device__ __forceinline__
float warpReduceSum(float val)
{
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// ---------------- WARP KERNEL ----------------

__global__ void kmeans_warp_kernel(
    const float* __restrict__ x,
    const float* __restrict__ c,
    int* labels,
    float* sum,
    int* cnt,
    int n,
    int warps_per_block)
{
    int lane = threadIdx.x & 31;
    int warp_in_block = threadIdx.x >> 5;

    int warp_id = blockIdx.x * warps_per_block + warp_in_block;

    if (warp_id >= n) return;

    const float* xi = x + warp_id * DIM;

    float bestDist = 1e30f;
    int bestk = 0;

    for (int k = 0; k < K; k++) {

        float partial = 0;

        for (int d = lane; d < DIM; d += WARP_SIZE) {
            float diff = xi[d] - c[k * DIM + d];
            partial += diff * diff;
        }

        partial = warpReduceSum(partial);

        if (lane == 0) {
            if (partial < bestDist) {
                bestDist = partial;
                bestk = k;
            }
        }
    }

    if (lane == 0) {
        labels[warp_id] = bestk;
        atomicAdd(&cnt[bestk], 1);
    }

    for (int d = lane; d < DIM; d += WARP_SIZE) {
        atomicAdd(&sum[bestk * DIM + d], xi[d]);
    }
}

// ---------------- GPU DRIVER ----------------

float gpu_kmeans(const float* x, float* c, int* labels, int n, int block_size)
{
    if (block_size % 32 != 0) {
        printf("ERROR: block_size must be multiple of 32\n");
        exit(1);
    }

    int warps_per_block = block_size / 32;

    float *d_x, *d_c, *d_sum;
    int *d_cnt, *d_labels;

    CUDA_CHECK(cudaMalloc(&d_x, n * DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, K * DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum, K * DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cnt, K * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_labels, n * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_x, x, n * DIM * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, c, K * DIM * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    int blocks = (n + warps_per_block - 1) / warps_per_block;

    for (int it = 0; it < ITER; it++) {

        CUDA_CHECK(cudaMemset(d_sum, 0, K * DIM * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_cnt, 0, K * sizeof(int)));

        kmeans_warp_kernel<<<blocks, block_size>>>(
            d_x,
            d_c,
            d_labels,
            d_sum,
            d_cnt,
            n,
            warps_per_block
        );

        CUDA_CHECK(cudaDeviceSynchronize());

        float h_sum[K * DIM];
        int h_cnt[K];

        CUDA_CHECK(cudaMemcpy(h_sum, d_sum, K * DIM * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_cnt, d_cnt, K * sizeof(int), cudaMemcpyDeviceToHost));

        for (int k = 0; k < K; k++) {
            if (h_cnt[k] == 0) continue;
            for (int d = 0; d < DIM; d++)
                c[k * DIM + d] = h_sum[k * DIM + d] / h_cnt[k];
        }
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaMemcpy(labels, d_labels, n * sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_x);
    cudaFree(d_c);
    cudaFree(d_sum);
    cudaFree(d_cnt);
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
    if (argc < 3) {
        printf("Usage: %s <N> <block_size>\n", argv[0]);
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

    float gpu_ms = gpu_kmeans(x, c2, l2, n, block_size);

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