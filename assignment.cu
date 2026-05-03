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
#define TILE_D 32

#define MINI_BATCH_SIZE 256   // <-- you can tune this

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

// ---------------- GPU KERNEL (unchanged) ----------------

__global__ void kmeans_kernel_research(
    const float* __restrict__ x,
    const float* __restrict__ c,
    int* labels,
    float* out_sum,
    int* out_cnt,
    int n)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x & 31;

    if (warp_id >= n) return;

    const float* xi = x + warp_id * DIM;

    float best_dist = 1e30f;
    int best_k = 0;

    __shared__ float c_tile[TILE_K][DIM];

    for (int k0 = 0; k0 < K; k0 += TILE_K)
    {
        for (int i = threadIdx.x; i < TILE_K * DIM; i += blockDim.x)
        {
            int ck = i / DIM;
            int cd = i % DIM;

            if (k0 + ck < K)
                c_tile[ck][cd] = c[(k0 + ck) * DIM + cd];
        }

        __syncthreads();

        for (int ck = 0; ck < TILE_K && (k0 + ck) < K; ck++)
        {
            float dist = 0.0f;

            for (int d = lane; d < DIM; d += WARP_SIZE * TILE_D)
            {
                float acc = 0.0f;

                #pragma unroll
                for (int t = 0; t < TILE_D && (d + t) < DIM; t++)
                {
                    float diff = xi[d + t] - c_tile[ck][d + t];
                    acc += diff * diff;
                }

                for (int offset = 16; offset > 0; offset >>= 1)
                    acc += __shfl_down_sync(0xffffffff, acc, offset);

                if (lane == 0)
                    dist += acc;
            }

            if (dist < best_dist)
            {
                best_dist = dist;
                best_k = k0 + ck;
            }
        }

        __syncthreads();
    }

    if (lane == 0)
        labels[warp_id] = best_k;

    atomicAdd(&out_cnt[best_k], 1);

    for (int d = lane; d < DIM; d += WARP_SIZE)
        atomicAdd(&out_sum[best_k * DIM + d], xi[d]);
}

// ---------------- MINI-BATCH CPU (reference logic) ----------------

void cpu_kmeans_minibatch(const float* x, float* c, int n)
{
    float lr = 0.1f;
    int batch = MINI_BATCH_SIZE;

    for (int it = 0; it < ITER; it++) {

        for (int b = 0; b < n; b += batch) {

            int cur = (b + batch > n) ? (n - b) : batch;

            for (int i = 0; i < cur; i++) {
                int idx = b + i;

                float best = 1e30f;
                int bestk = 0;

                for (int k = 0; k < K; k++) {
                    float d = 0;
                    for (int d_i = 0; d_i < DIM; d_i++) {
                        float diff = x[idx * DIM + d_i] - c[k * DIM + d_i];
                        d += diff * diff;
                    }
                    if (d < best) {
                        best = d;
                        bestk = k;
                    }
                }

                // online update
                for (int d_i = 0; d_i < DIM; d_i++) {
                    float xval = x[idx * DIM + d_i];
                    c[bestk * DIM + d_i] =
                        (1 - lr) * c[bestk * DIM + d_i] + lr * xval;
                }
            }
        }
    }
}

// ---------------- GPU DRIVER (unchanged) ----------------

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

        for (int offset = 0; offset < n; offset += chunk)
        {
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

            kmeans_kernel_research<<<blocks, block_size, 0, streams[s]>>>(
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

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaMemcpy(labels, d_labels, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost));

    return ms;
}

// ---------------- INIT ----------------

void init(float* x, int n)
{
    for (int i = 0; i < n * DIM; i++)
        x[i] = (float)rand() / RAND_MAX;
}

// ---------------- MAIN (UPDATED FLAG SUPPORT) ----------------

int main(int argc, char** argv)
{
    if (argc < 3) {
        printf("Usage: %s <n> <block_size> [--mini-batch]\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    int block_size = atoi(argv[2]);
    int use_minibatch = (argc >= 4 && strcmp(argv[3], "--mini-batch") == 0);

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

    if (use_minibatch)
        cpu_kmeans_minibatch(x, c1, n);
    else
        cpu_kmeans(x, c1, l1, n);

    float cpu_ms = (float)(clock() - t1) / CLOCKS_PER_SEC * 1000;

    float gpu_ms = gpu_kmeans_streamed(x, c2, l2, n, block_size);

    printf("Mode: %s\n", use_minibatch ? "MINI-BATCH" : "STANDARD");
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