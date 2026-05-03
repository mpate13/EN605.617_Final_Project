#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <float.h>
#include <time.h>
#include <string.h>

#define MAX_CLUSTERS 10
#define IMAGE_DIMENSIONS 3072
#define MAX_ITERATIONS 10
#define MINI_BATCH_SIZE 1024

#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

/* =========================
   GLOBAL MEMORY (GPU STATE)
   ========================= */
__device__ float d_centroids[MAX_CLUSTERS * IMAGE_DIMENSIONS];
__device__ int d_cluster_count[MAX_CLUSTERS];

/* =========================
   ASSIGNMENT KERNEL
   ========================= */
__global__ void assign_kernel(
    const float* images,
    int* assignments,
    int n,
    int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float best_dist = FLT_MAX;
    int best_k = 0;

    for (int c = 0; c < k; c++) {
        float dist = 0.0f;

        #pragma unroll 4
        for (int d = 0; d < IMAGE_DIMENSIONS; d++) {
            float diff = images[i * IMAGE_DIMENSIONS + d] -
                         d_centroids[c * IMAGE_DIMENSIONS + d];
            dist += diff * diff;
        }

        if (dist < best_dist) {
            best_dist = dist;
            best_k = c;
        }
    }

    assignments[i] = best_k;
}

/* =========================
   REDUCTION KERNEL (NO ATOMICS PER PIXEL)
   ========================= */
__global__ void accumulate_kernel(
    const float* images,
    const int* assignments,
    float* centroid_sums,
    int* counts,
    int n,
    int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int c = assignments[i];

    atomicAdd(&counts[c], 1);

    const float* img = &images[i * IMAGE_DIMENSIONS];

    for (int d = 0; d < IMAGE_DIMENSIONS; d++) {
        atomicAdd(&centroid_sums[c * IMAGE_DIMENSIONS + d], img[d]);
    }
}

/* =========================
   CENTROID UPDATE
   ========================= */
__global__ void update_centroids(
    float* centroid_sums,
    int* counts,
    int k)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= k) return;

    int count = counts[c];
    if (count == 0) return;

    for (int d = 0; d < IMAGE_DIMENSIONS; d++) {
        centroid_sums[c * IMAGE_DIMENSIONS + d] /= count;
        d_centroids[c * IMAGE_DIMENSIONS + d] =
            centroid_sums[c * IMAGE_DIMENSIONS + d];
    }
}

/* =========================
   INIT CENTROIDS
   ========================= */
__global__ void init_centroids(const float* images) {
    int i = threadIdx.x;
    if (i < MAX_CLUSTERS * IMAGE_DIMENSIONS) {
        d_centroids[i] = images[i];
    }
}

/* =========================
   CPU BASELINE
   ========================= */
void cpu_kmeans(const float* images, int* labels, int n, int k) {
    for (int i = 0; i < n; i++) {
        float best = FLT_MAX;
        int best_k = 0;

        for (int c = 0; c < k; c++) {
            float dist = 0;

            for (int d = 0; d < IMAGE_DIMENSIONS; d++) {
                float diff = images[i * IMAGE_DIMENSIONS + d] -
                             images[c * IMAGE_DIMENSIONS + d];
                dist += diff * diff;
            }

            if (dist < best) {
                best = dist;
                best_k = c;
            }
        }
        labels[i] = best_k;
    }
}

/* =========================
   GPU DRIVER
   ========================= */
float gpu_kmeans(float* h_images, int* h_labels, int n, int k, int block_size)
{
    float *d_images;
    int *d_labels;
    float *d_sums;
    int *d_counts;

    checkCuda(cudaMalloc(&d_images, n * IMAGE_DIMENSIONS * sizeof(float)));
    checkCuda(cudaMalloc(&d_labels, n * sizeof(int)));

    checkCuda(cudaMalloc(&d_sums, k * IMAGE_DIMENSIONS * sizeof(float)));
    checkCuda(cudaMalloc(&d_counts, k * sizeof(int)));

    checkCuda(cudaMemcpy(d_images, h_images,
                         n * IMAGE_DIMENSIONS * sizeof(float),
                         cudaMemcpyHostToDevice));

    init_centroids<<<1, MAX_CLUSTERS * IMAGE_DIMENSIONS>>>(d_images);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {

        checkCuda(cudaMemset(d_sums, 0, k * IMAGE_DIMENSIONS * sizeof(float)));
        checkCuda(cudaMemset(d_counts, 0, k * sizeof(int)));

        int grid = (n + block_size - 1) / block_size;

        assign_kernel<<<grid, block_size>>>(d_images, d_labels, n, k);

        cudaDeviceSynchronize();

        accumulate_kernel<<<grid, block_size>>>(
            d_images, d_labels, d_sums, d_counts, n, k);

        cudaDeviceSynchronize();

        update_centroids<<<(k + 255)/256, 256>>>(d_sums, d_counts, k);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_labels, d_labels, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_images);
    cudaFree(d_labels);
    cudaFree(d_sums);
    cudaFree(d_counts);

    return ms;
}

/* =========================
   MAIN
   ========================= */
int main(int argc, char** argv)
{
    if (argc < 3) {
        printf("Usage: ./assignment <n> <block_size> [--mini-batch]\n");
        return 0;
    }

    int n = atoi(argv[1]);
    int block_size = atoi(argv[2]);

    float* h_images = (float*)malloc(n * IMAGE_DIMENSIONS * sizeof(float));
    int* cpu_labels = (int*)malloc(n * sizeof(int));
    int* gpu_labels = (int*)malloc(n * sizeof(int));

    for (int i = 0; i < n * IMAGE_DIMENSIONS; i++) {
        h_images[i] = (float)rand() / RAND_MAX;
    }

    clock_t start = clock();
    cpu_kmeans(h_images, cpu_labels, n, MAX_CLUSTERS);
    double cpu_ms = (double)(clock() - start) / CLOCKS_PER_SEC * 1000;

    float gpu_ms = gpu_kmeans(h_images, gpu_labels, n, MAX_CLUSTERS, block_size);

    printf("CPU: %.2f ms\n", cpu_ms);
    printf("GPU: %.2f ms\n", gpu_ms);
    printf("Speedup: %.2fx\n", cpu_ms / gpu_ms);

    free(h_images);
    free(cpu_labels);
    free(gpu_labels);

    return 0;
}