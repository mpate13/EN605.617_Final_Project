#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <float.h>
#include <time.h>

#define K 10
#define DIM 3072
#define ITER 20

// ---------------- CPU BASELINE ----------------

void cpu_kmeans(const float* x, float* c, int* labels, int n) {

    float tmp[K * DIM];
    int cnt[K];

    for (int it = 0; it < ITER; it++) {

        // assign
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

        // update
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

// ---------------- GPU TILED KERNEL ----------------

__global__ void kmeans_kernel(
    const float* __restrict__ x,
    const float* __restrict__ c,
    int* labels,
    float* out_sum,
    int* out_cnt,
    int n)
{
    extern __shared__ float tile[];  // ONLY input tile

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    float* xi = tile + tid * DIM;

    // load input tile (OK for shared memory)
    if (i < n) {
        for (int d = 0; d < DIM; d++) {
            xi[d] = x[i * DIM + d];
        }
    }

    __syncthreads();

    if (i >= n) return;

    // keep centroid reads in GLOBAL (cached by L2!)
    float best = 1e30f;
    int bestk = 0;

    for (int k = 0; k < K; k++) {
        float d = 0;

        #pragma unroll 4
        for (int d_i = 0; d_i < DIM; d_i++) {
            float diff = xi[d_i] - c[k * DIM + d_i];
            d += diff * diff;
        }

        if (d < best) {
            best = d;
            bestk = k;
        }
    }

    labels[i] = bestk;

    // SAFE global accumulation (no shared memory explosion)
    atomicAdd(&out_cnt[bestk], 1);

    for (int d_i = 0; d_i < DIM; d_i++) {
        atomicAdd(&out_sum[bestk * DIM + d_i],
                  xi[d_i]);
    }
}

// ---------------- NORMALIZE ----------------

__global__ void normalize(float* c, float* sum, int* cnt) {
    int k = threadIdx.x;

    if (k < K && cnt[k] > 0) {
        for (int d = 0; d < DIM; d++) {
            c[k * DIM + d] = sum[k * DIM + d] / cnt[k];
        }
    }
}

// ---------------- GPU DRIVER ----------------

float gpu_kmeans(const float* x, float* c, int* labels,
                 int n, int block_size) {

    float *d_x, *d_c, *d_sum;
    int *d_labels, *d_cnt;

    cudaMalloc(&d_x, n * DIM * sizeof(float));
    cudaMalloc(&d_c, K * DIM * sizeof(float));
    cudaMalloc(&d_labels, n * sizeof(int));
    cudaMalloc(&d_sum, K * DIM * sizeof(float));
    cudaMalloc(&d_cnt, K * sizeof(int));

    cudaMemcpy(d_x, x, n * DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, K * DIM * sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (n + block_size - 1) / block_size;

    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);

    cudaEventRecord(s);

    for (int it = 0; it < ITER; it++) {

        cudaMemset(d_sum, 0, K * DIM * sizeof(float));
        cudaMemset(d_cnt, 0, K * sizeof(int));

        kmeans_kernel<<<blocks, block_size>>>(
            d_x, d_c, d_labels, d_sum, d_cnt, n
        );

        normalize<<<1, K>>>(d_c, d_sum, d_cnt);
    }

    cudaEventRecord(e);
    cudaEventSynchronize(e);

    float ms;
    cudaEventElapsedTime(&ms, s, e);

    cudaMemcpy(labels, d_labels, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(c, d_c, K * DIM * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_c);
    cudaFree(d_labels);
    cudaFree(d_sum);
    cudaFree(d_cnt);

    return ms;
}

// ---------------- MAIN ----------------

void init(float* x, int n) {
    for (int i = 0; i < n * DIM; i++)
        x[i] = (float)rand() / RAND_MAX;
}

int main(int argc, char** argv) {

    int n = atoi(argv[1]);          // total threads = data points
    int block_size = atoi(argv[2]); // CUDA config

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
}