#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <float.h>
#include <time.h>
#include <string.h>

#define K 10
#define DIM 3072
#define ITER 50
#define NUM_STREAMS 4
#define CHUNK_SIZE 50000
#define WARP_SIZE 32
#define TILE_K 2
#define CIFAR_BINARY_ROW_SIZE 3073
#define DATASET_SIZE_LIMIT 50000
#define FIRST_DATA_CHANNEL_OFFSET 1
#define NORMALIZE_PIXEL_VALUE 255.0f

#define CUDA_CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        printf("CUDA ERROR: %s (%s:%d)\n", \
            cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

/**
 * CIFAR-10 K-MEANS CLUSTERING: GPU ACCELERATION, MEMORY HIERARCHY OPTIMIZATION,
 * AND MINI-BATCH PERFORMANCE TRADEOFFS
 * ----------------------------------------------------------------------------
 *
 * MOTIVATION:
 * CIFAR-10 (3072-dimensional image vectors) makes classical K-Means extremely
 * expensive due to high dimensionality and repeated distance computations.
 * This project benchmarks CPU vs GPU implementations and explores how CUDA
 * optimizations and algorithmic approximations (mini-batching) improve runtime
 * while preserving clustering quality.
 *
 * The primary bottleneck is the O(N * K * DIM) distance computation combined
 * with high memory bandwidth pressure when repeatedly accessing centroid data.
 *
 * ----------------------------------------------------------------------------
 * CORE DESIGN GOALS:
 * 1. Minimize global memory bandwidth pressure
 * 2. Maximize data reuse via tiling and shared memory
 * 3. Reduce synchronization overhead
 * 4. Explore stochastic convergence via mini-batch K-Means
 *
 * ----------------------------------------------------------------------------
 * GPU OPTIMIZATION STRATEGIES:
 *
 * (1) TILE-BASED CENTROID LOADING (TILE_K):
 *     - Centroids are processed in small chunks (tiles) to improve cache reuse
 *     - Avoids repeated global memory fetches of all K centroids per thread
 *     - Trades recomputation for reduced memory bandwidth pressure
 *
 * (2) SHARED MEMORY ACCELERATION:
 *     - Centroid tiles are staged in __shared__ memory
 *     - Enables low-latency reuse during distance calculations
 *     - Critical for high-dimensional (DIM=3072) vector operations
 *
 * (3) WARP AND THREAD PARALLELISM:
 *     - Each thread processes one data point
 *     - DIM loop is distributed across threads (strided access pattern)
 *     - Atomic operations used for safe concurrent aggregation
 *
 * (4) MINI-BATCH K-MEANS:
 *     - Processes random subsets of data instead of full dataset per iteration
 *     - Reduces computational load from O(N) → O(batch_size)
 *     - Introduces stochastic updates using learning rate smoothing
 *     - Trades convergence stability for major runtime improvements
 *
 * (5) ATOMIC REDUCTION STRATEGY:
 *     - cluster_sums and cluster_counts use atomicAdd operations
 *     - Necessary due to many-thread write contention
 *     - Chosen over block-level reductions for simplicity and scalability
 *
 * ----------------------------------------------------------------------------
 * KEY BOTTLENECKS AND TRADEOFFS:
 *
 * - HIGH DIMENSIONALITY (DIM = 3072):
 *   Dominates arithmetic workload; unrolling or vectorization is limited
 *
 * - GLOBAL MEMORY LATENCY:
 *   Repeated centroid access is expensive → mitigated via shared memory tiling
 *
 * - ATOMIC CONTENTION:
 *   cluster_sums updates can serialize under heavy load
 *
 * - SYNCHRONIZATION OVERHEAD:
 *   __syncthreads() required between tile loads and computation phases
 *
 * - CONSTANTS USED (K, DIM, TILE_K, ITER):
 *   These are compile-time constants to:
 *   - Enable loop unrolling and compiler optimization
 *   - Avoid dynamic indexing overhead in kernels
 *   - Ensure shared memory allocation is static and efficient
 *
 * ----------------------------------------------------------------------------
 * EXPECTED OUTPUT:
 *
 * 1. CONSOLE:
 *    - CPU baseline runtime
 *    - GPU standard K-Means runtime
 *    - GPU mini-batch runtime
 *    - Speedup factor comparison
 *
 * 2. CSV OUTPUTS:
 *    - cpu_n*_b*_*.csv  → CPU cluster assignments
 *    - gpu_n*_b*_*.csv  → GPU cluster assignments
 *
 * ----------------------------------------------------------------------------
 * SUMMARY:
 * This implementation demonstrates how classical clustering algorithms can be
 * significantly accelerated using CUDA through:
 * - Memory hierarchy optimization (global => shared)
 * - Data tiling strategies for large centroid sets
 * - Parallel per-point assignment model
 * - Stochastic mini-batch convergence acceleration
 *
 * The design highlights the tradeoff between deterministic convergence (CPU /
 * full-batch GPU) and high-throughput approximate learning (mini-batch GPU).
 */

// ============================================================
// UTILITY FUNCTIONS
// ============================================================

/*
 * Writes cluster assignment results to a CSV file.
 * Each row contains an image ID and its assigned cluster ID.
 */
void export_to_csv(const char* file_path, const int* cluster_assignments, 
                                                            int num_images) {
    FILE* f = fopen(file_path, "w");
    if (!f) {
        printf("Failed to open %s\n", file_path);
        return;
    }

    fprintf(f, "ImageID,ClusterID\n");

    for (int i = 0; i < num_images; i++) {
        fprintf(f, "%d,%d\n", i, cluster_assignments[i]);
    }

    fclose(f);
}

/*
 * Loads image data from a CIFAR-style binary dataset file into a host buffer.
 * If the file cannot be opened, it fills the buffer with random noise.
 * If fewer images are available than requested, it cycles through loaded data.
 */
void load_cifar_dataset(const char* file_path, float* host_pixels, 
                                                            int num_images) {
    // Temporary buffer to hold the actual data (50k images max)
    size_t base_size = DATASET_SIZE_LIMIT * DIM * sizeof(float);
    float* base_buffer = (float*)malloc(base_size);

    FILE* file_pointer = fopen(file_path, "rb");
    if (!file_pointer) {
        printf("Error: Could not open %s. Using noise.\n", file_path);
        for(int i=0; i < num_images * DIM; i++) 
            host_pixels[i] = (float)rand()/RAND_MAX;
        free(base_buffer);
        return;
    }

    unsigned char row_buffer[CIFAR_BINARY_ROW_SIZE];
    int loaded_count = 0;
    while (loaded_count < DATASET_SIZE_LIMIT && 
           fread(row_buffer, 1, CIFAR_BINARY_ROW_SIZE, file_pointer) == 
                                                        CIFAR_BINARY_ROW_SIZE) {
        for (int d = 0; d < DIM; d++) {
            base_buffer[loaded_count * DIM + d] = 
                (float)row_buffer[d + FIRST_DATA_CHANNEL_OFFSET] / 
                                                        NORMALIZE_PIXEL_VALUE;
        }
        loaded_count++;
    }
    fclose(file_pointer);
    // printf("Loaded %d real images from disk.\n", loaded_count);

    for (int i = 0; i < num_images; i++) {
        int source_idx = i % loaded_count;
        memcpy(&host_pixels[i * DIM], &base_buffer[source_idx * DIM], 
                                                        DIM * sizeof(float));
    }

    free(base_buffer);
    // printf("Host buffer filled with %d images (using cyclic data).\n", 
                                                                // num_images);
}

void init(float* x, int n) {
    for (int i = 0; i < n * DIM; i++)
        x[i] = (float)rand() / RAND_MAX;
}

/*
 * Prints command-line usage instructions for running the program,
 * including required arguments and optional flags.
 */
void print_usage(char* prog_name) {
    printf("Usage: %s <n> <block_size> [--mini-batch]\n", prog_name);
}

/*
 * Prints a formatted benchmark report comparing CPU and GPU K-means execution,
 * including runtime and speedup.
 */
void print_report(int n, int block, int mb, float cpu, float gpu) {
    printf("\n==================================================\n");
    printf("K-MEANS BENCHMARK\n");
    printf("--------------------------------------------------\n");
    printf("Mode        : %s\n", mb ? "MINI-BATCH" : "STANDARD");
    printf("N           : %d\n", n);
    printf("Block Size  : %d\n", block);
    printf("--------------------------------------------------\n");
    printf("CPU Time    : %.2f ms\n", cpu);
    printf("GPU Time    : %.2f ms\n", gpu);
    printf("Speedup     : %.2fx\n", cpu / gpu);
    printf("==================================================\n\n");
}


// ============================================================
// CPU BASELINE
// ============================================================

/*
 * Assigns each data point to the nearest cluster centroid using squared 
 * Euclidean distance.
 * Updates the labels array with the index of closest centroid for each point.
 */
void assign_points_to_clusters(const float* points, const float* centroids, 
                               int* labels, int num_points) {
    for (int i = 0; i < num_points; i++) {
        float min_dist_sq = 1e30f;
        int nearest_cluster = 0;

        for (int k = 0; k < K; k++) {
            float dist_sq = 0.0f;
            const float* cluster_ptr = centroids + k * DIM;

            for (int d = 0; d < DIM; d++) {
                float diff = points[i * DIM + d] - cluster_ptr[d];
                dist_sq += diff * diff;
            }

            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
                nearest_cluster = k;
            }
        }
        labels[i] = nearest_cluster;
    }
}

/*
 * Recomputes cluster centroids by averaging all points assigned to each cluster
 * Uses accumulated sums and counts per cluster to update centroid positions.
 */
void update_centroids(const float* points, float* centroids, 
                      const int* labels, int num_points) {
    float sum_buffers[K * DIM] = {0.0f};
    int cluster_counts[K] = {0};

    for (int i = 0; i < num_points; i++) {
        int k = labels[i];
        cluster_counts[k]++;
        for (int d = 0; d < DIM; d++) {
            sum_buffers[k * DIM + d] += points[i * DIM + d];
        }
    }

    for (int k = 0; k < K; k++) {
        if (cluster_counts[k] == 0) continue;
        for (int d = 0; d < DIM; d++) {
            centroids[k * DIM + d] = sum_buffers[k * DIM + d] / 
                                     cluster_counts[k];
        }
    }
}

/*
 * Runs the K-means clustering algorithm on the CPU for a fixed number of 
 * iterations.
 * Alternates between assigning points to the nearest centroid and updating 
 * centroids.
 */
void cpu_kmeans(const float* points, float* centroids, int* labels, 
                                     int num_points) {
    for (int iter = 0; iter < ITER; iter++) {
        assign_points_to_clusters(points, centroids, labels, num_points);
        update_centroids(points, centroids, labels, num_points);
    }
}

/*
 * Runs and times the CPU implementation of K-means clustering.
 * Uses a copy of centroids to avoid modifying the input values and returns
 * the execution time in milliseconds.
 */
float cpu_kmeans_timed(const float* data_points, 
                                        float* centroids, 
                                        int* cluster_labels, 
                                        int num_points) {
    // Create a working copy of centroids to avoid modifying the originals
    float* centroids_backup = (float*)malloc(K * DIM * sizeof(float));

    for (int i = 0; i < K * DIM; i++) {
        centroids_backup[i] = centroids[i];
    }

    clock_t start_time = clock();
    
    // Execute the core algorithm
    cpu_kmeans(data_points, centroids_backup, cluster_labels, num_points);
    
    clock_t end_time = clock();

    // Calculate duration in milliseconds
    float elapsed_time_ms = 1000.0f * (float)(end_time - start_time) / 
                                                                CLOCKS_PER_SEC;

    free(centroids_backup);
    
    return elapsed_time_ms;
}


// ============================================================
// GPU Device Functions (building blocks)
// ============================================================

/*
 * Finds the nearest cluster centroid for a given data point using a tiled
 * shared-memory approach to improve memory efficiency.
 * Computes squared Euclidean distance and returns the closest cluster index.
 */
__device__ int find_nearest_cluster(const float* data_point,
    const float* centroids, float* shared_centroids_tile) {
    float min_dist_sq = 1e30f;
    int nearest_cluster_idx = 0;

    for (int tile_start = 0; tile_start < K; tile_start += TILE_K) {
        // Load chunk of centroids into shared memory
        for (int i = threadIdx.x; i < TILE_K * DIM; i += blockDim.x) {
            int tile_offset = i / DIM;
            int dim_idx = i % DIM;

            if (tile_start + tile_offset < K) {
                shared_centroids_tile[tile_offset * DIM + dim_idx] = 
                    centroids[(tile_start + tile_offset) * DIM + dim_idx];
            }
        }
        __syncthreads();
        // Calculate squared Euclidean distance against the current tile
        for (int tile_offset = 0; tile_offset < TILE_K && 
                                (tile_start + tile_offset) < K; tile_offset++) {
            const float* centroid_ptr = 
                                    &shared_centroids_tile[tile_offset * DIM];
            float current_dist_sq = 0.0f;

            for (int d = 0; d < DIM; d++) {
                float diff = data_point[d] - centroid_ptr[d];
                current_dist_sq += diff * diff;
            }

            if (current_dist_sq < min_dist_sq) {
                min_dist_sq = current_dist_sq;
                nearest_cluster_idx = tile_start + tile_offset;
            }
        }

        __syncthreads();
    }

    return nearest_cluster_idx;
}

/*
 * Atomically updates per-cluster statistics for a given data point.
 * Accumulates feature sums and increments the count for the assigned cluster.
 */
__device__ void update_cluster_statistics(
    float* cluster_sums,
    int* cluster_counts,
    const float* data_point,
    int cluster_idx) {
    // Accumulate values into the global sum array
    for (int d = threadIdx.x; d < DIM; d += blockDim.x) {
        atomicAdd(&cluster_sums[cluster_idx * DIM + d], data_point[d]);
    }

    // Increment count (only performed by thread 0 of the block)
    if (threadIdx.x == 0) {
        atomicAdd(&cluster_counts[cluster_idx], 1);
    }
}


// ============================================================
// GPU Kernels
// ============================================================

/*
 * CUDA kernel that performs a minibatch update of cluster centroids.
 * Each block handles one cluster, updating its centroid using a weighted
 * move toward the batch mean based on a learning rate.
 */
__global__ void update_centroids_minibatch(
    float* centroids,
    const float* batch_sums,
    const int* cluster_counts,
    float learning_rate)
{
    int cluster_idx = blockIdx.x;
    
    // Bounds check and ensure the cluster had assigned points
    if (cluster_idx >= K) return;

    int count = cluster_counts[cluster_idx];
    if (count == 0) count = 1;

    // Update each dimension for the given cluster
    for (int dim_idx = threadIdx.x; dim_idx < DIM; dim_idx += blockDim.x) {
        int offset = cluster_idx * DIM + dim_idx;
        
        float batch_mean = batch_sums[offset] / count;
        float prev_val = centroids[offset];

        // Apply learning rate update
        centroids[offset] = prev_val + learning_rate * (batch_mean - prev_val);
    }
}

/*
 * Performs a minibatch step of K-means clustering.
 * Each thread processes one data point, assigns it to the nearest centroid,
 * and updates global cluster statistics (sums and counts).
 */
__global__ void kmeans_minibatch_kernel(
    const float* __restrict__ data_points,
    const float* __restrict__ centroids,
    int* labels,
    float* cluster_sums,
    int* cluster_counts,
    int num_points) {
    int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (global_thread_idx >= num_points) return;

    const float* current_data_point = 
                                data_points + (size_t)global_thread_idx * DIM;

    // Shared memory for centroid tiles
    __shared__ float shared_centroids_tile[TILE_K * DIM];

    // Identify nearest centroid
    int nearest_cluster_idx = find_nearest_cluster(
        current_data_point, 
        centroids, 
        shared_centroids_tile
    );

    // Save label
    labels[global_thread_idx] = nearest_cluster_idx;

    // Update global sums and counts
    update_cluster_statistics(
        cluster_sums, 
        cluster_counts, 
        current_data_point, 
        nearest_cluster_idx
    );
}

/*
 * CUDA kernel that recomputes full centroids by dividing accumulated sums
 * by the number of points assigned to each cluster.
 */
__global__ void update_centroids_full(
    float* centroids,
    const float* sums,
    const int* counts)
{
    int k = blockIdx.x;

    if (k >= K || counts[k] == 0) return;

    for (int d = threadIdx.x; d < DIM; d += blockDim.x) {
        centroids[k * DIM + d] =
            sums[k * DIM + d] / counts[k];
    }
}


// ============================================================
// GPU Standard Implementation Pipeline
// ============================================================
// Struct to encapsulate device memory management
struct KMeansDeviceBuffers {
    float *d_x, *d_c, *d_sum;
    int *d_labels, *d_cnt;
};

/*
 * Allocates all required GPU memory buffers for K-means clustering,
 * including data points, centroids, labels, and per-cluster statistics.
 */
void allocate_kmeans_buffers(KMeansDeviceBuffers& buffers, int n) {
    CUDA_CHECK(cudaMalloc(&buffers.d_x, (size_t)n * DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers.d_c, K * DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers.d_labels, (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&buffers.d_sum, K * DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers.d_cnt, K * sizeof(int)));
}

/*
 * Copies input data points and initial centroids from host memory to GPU 
 * device memory.
 */
void upload_kmeans_data(const KMeansDeviceBuffers& buffers, 
                                        const float* x, const float* c, int n) {
    CUDA_CHECK(cudaMemcpy(buffers.d_x, x, 
                    (size_t)n * DIM * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buffers.d_c, c, 
                            K * DIM * sizeof(float), cudaMemcpyHostToDevice));
}

/*
 * Executes the main K-means training loop on the GPU.
 * Repeats minibatch assignment and centroid updates for a fixed number of 
 * iterations.
 */
void run_kmeans_loop(const KMeansDeviceBuffers& buffers, int n, int block_size){
    int blocks = (n + block_size - 1) / block_size;

    for (int it = 0; it < ITER; it++) {
        CUDA_CHECK(cudaMemset(buffers.d_sum, 0, K * DIM * sizeof(float)));
        CUDA_CHECK(cudaMemset(buffers.d_cnt, 0, K * sizeof(int)));

        kmeans_minibatch_kernel<<<blocks, block_size>>>(buffers.d_x, 
            buffers.d_c, buffers.d_labels, buffers.d_sum, buffers.d_cnt, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        update_centroids_full<<<K, block_size>>>(buffers.d_c, 
                                                buffers.d_sum, buffers.d_cnt);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

/*
 * Frees all GPU memory buffers used by the K-means implementation,
 * preventing memory leaks after computation completes.
 */
void free_kmeans_buffers(KMeansDeviceBuffers& buffers) {
    cudaFree(buffers.d_x);
    cudaFree(buffers.d_c);
    cudaFree(buffers.d_sum);
    cudaFree(buffers.d_cnt);
    cudaFree(buffers.d_labels);
}

/*
 * Runs the standard GPU K-means pipeline including memory allocation,
 * data transfer, iterative clustering, and cleanup, while measuring runtime.
 */
float gpu_kmeans_standard(const float* x, float* c, int* labels, int n, 
                                                            int block_size) {
    KMeansDeviceBuffers buffers;
    allocate_kmeans_buffers(buffers, n);
    upload_kmeans_data(buffers, x, c, n);

    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));
    CUDA_CHECK(cudaEventRecord(s));

    run_kmeans_loop(buffers, n, block_size);

    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));

    free_kmeans_buffers(buffers);

    cudaEventDestroy(s);
    cudaEventDestroy(e);

    return ms;
}


// ============================================================
// GPU Minibatch Pipeline
// ============================================================

/*
 * Allocates all GPU memory resources required for K-means clustering,
 * including data points, centroids, labels, and per-cluster statistics.
 */
void allocate_kmeans_resources(float** d_data, float** d_centroids, 
                                int** d_labels, float** d_centroid_sums, 
                                int** d_centroid_counts, int num_samples) {
    CUDA_CHECK(cudaMalloc(d_data, (size_t)num_samples * DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(d_centroids, K * DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(d_labels, (size_t)num_samples * sizeof(int)));
    CUDA_CHECK(cudaMalloc(d_centroid_sums, K * DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(d_centroid_counts, K * sizeof(int)));
}

/*
 * Frees all GPU memory resources used by the K-means clustering implementation.
 * This includes data points, centroids, labels, and cluster statistics arrays.
 */
void deallocate_kmeans_resources(float* d_data, float* d_centroids, 
                int* d_labels, float* d_centroid_sums, int* d_centroid_counts) {
    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_centroid_sums);
    cudaFree(d_centroid_counts);
    cudaFree(d_labels);
}

/*
 * Executes the minibatch K-means training loop on the GPU.
 * Each iteration samples a random batch, runs clustering, and updates centroids
 * using a learning-rate-based update rule.
 */
void execute_kmeans_loop(float* d_data, float* d_centroids, int* d_labels,
                         float* d_centroid_sums, int* d_centroid_counts,
                         int num_samples, int block_size) {
    int batch_size = num_samples / 10;
    float learning_rate = 0.1f;
    int blocks = (batch_size + block_size - 1) / block_size;

    for (int iteration = 0; iteration < ITER; iteration++) {
        int offset = (iteration * batch_size) % (num_samples - batch_size);

        // Reset accumulators
        CUDA_CHECK(cudaMemset(d_centroid_sums, 0, K * DIM * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_centroid_counts, 0, K * sizeof(int)));

        // Run Kernel
        kmeans_minibatch_kernel<<<blocks, block_size>>>(
            d_data + (size_t)offset * DIM,
            d_centroids,
            d_labels + offset,
            d_centroid_sums,
            d_centroid_counts,
            batch_size
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        // Update Centroids
        update_centroids_minibatch<<<K, block_size>>>(d_centroids, 
            d_centroid_sums, d_centroid_counts, learning_rate);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

/*
 * Runs the GPU minibatch K-means pipeline including memory allocation,
 * data transfer, iterative training, and cleanup, while measuring runtime.
 */
float gpu_kmeans_minibatch(const float* x, float* c, int* labels, int n, 
                                                            int block_size) {
    float *d_data, *d_centroids, *d_centroid_sums;
    int *d_labels, *d_centroid_counts;

    allocate_kmeans_resources(&d_data, &d_centroids, 
                            &d_labels, &d_centroid_sums, &d_centroid_counts, n);

    // Initial Data Transfer
    CUDA_CHECK(cudaMemcpy(d_data, x, (size_t)n * DIM * sizeof(float), 
                                                    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centroids, c, K * DIM * sizeof(float), 
                                                    cudaMemcpyHostToDevice));

    // Timing Setup
    cudaEvent_t start_event, end_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&end_event));
    CUDA_CHECK(cudaEventRecord(start_event));

    // Execution
    execute_kmeans_loop(d_data, d_centroids, d_labels, d_centroid_sums, 
                                            d_centroid_counts, n, block_size);

    // Timing Teardown
    CUDA_CHECK(cudaEventRecord(end_event));
    CUDA_CHECK(cudaEventSynchronize(end_event));

    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, end_event));

    CUDA_CHECK(cudaMemcpy(labels, d_labels, n * sizeof(int),
                                                cudaMemcpyDeviceToHost));
    // Cleanup
    deallocate_kmeans_resources(d_data, d_centroids, d_labels, d_centroid_sums, 
                                                            d_centroid_counts);
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(end_event));

    return elapsed_ms;
}

// ============================================================
// Benchmark Wrappers
// ============================================================
/*
 * Runs the CPU K-means benchmark, records execution time, and exports
 * clustering results to a CSV file.
 */
void run_cpu_bench(float* x, float* c, int* labels, int n, int b, int mb, 
                   float* out_ms) {
    *out_ms = cpu_kmeans_timed(x, c, labels, n);
    // printf("CPU Time: %.2f ms\n", *out_ms);

    char file[128];
    snprintf(file, sizeof(file), "cpu_n%d_b%d_%s.csv", n, b, 
             mb ? "minibatch" : "standard");
    export_to_csv(file, labels, n);
}

/*
 * Runs the GPU K-means benchmark (standard or minibatch), records execution
 * time, and exports clustering results to a CSV file.
 */
void run_gpu_bench(float* x, float* c, int* labels, int n, int b, int mb, 
                   float* out_ms) {
    *out_ms = mb ? gpu_kmeans_minibatch(x, c, labels, n, b)
                 : gpu_kmeans_standard(x, c, labels, n, b);

    char file[128];
    snprintf(file, sizeof(file), "gpu_n%d_b%d_%s.csv", n, b, 
             mb ? "minibatch" : "standard");
    export_to_csv(file, labels, n);
}

// ============================================================
// Main
// ============================================================
/*
 * Program entry point for the K-means benchmarking application.
 * Parses arguments, loads data, runs CPU and GPU benchmarks, and prints results.
 */
int main(int argc, char** argv) {
    if (argc < 3) { print_usage(argv[0]); return 1; }

    int n = atoi(argv[1]);
    int block = atoi(argv[2]);
    int mb = (argc > 3 && strcmp(argv[3], "--mini-batch") == 0);

    float *x = (float*)malloc((size_t)n * DIM * sizeof(float));
    float *c = (float*)malloc(K * DIM * sizeof(float));
    int *cpu_labels = (int*)malloc(n * sizeof(int));
    int *gpu_labels = (int*)malloc(n * sizeof(int));

    // srand(0);
    // init(x, n);
    load_cifar_dataset("cifar10_full.bin", x, n);
    for (int i = 0; i < K * DIM; i++) c[i] = x[i];

    float cpu_ms, gpu_ms;
    run_cpu_bench(x, c, cpu_labels, n, block, mb, &cpu_ms);

    for (int i = 0; i < K * DIM; i++) c[i] = x[i];
    run_gpu_bench(x, c, gpu_labels, n, block, mb, &gpu_ms);

    print_report(n, block, mb, cpu_ms, gpu_ms);  

    free(x); free(c);
    free(cpu_labels); free(gpu_labels);
    return 0;
}