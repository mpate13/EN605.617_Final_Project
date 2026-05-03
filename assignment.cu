#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <float.h>
#include <time.h>

/**
 * CIFAR-10 K-MEANS CLUSTERING: PERFORMANCE BENCHMARK & ARCHITECTURAL EVOLUTION
 * ----------------------------------------------------------------------------
 * MOTIVATION:
 * Standard K-Means on high-dimensional datasets presents computational 
 * bottlenecks. This program evaluates the trade-offs between global convergence 
 * and the stochastic efficiency of Mini-Batch K-Means, aiming to reduce 
 * processing latency in time-sensitive high-performance systems.
 *
 * CONTINUATION FROM MODULE 5:
 * This implementation evolves the basic GPU K-Means from Module 5 by 
 * transitioning to a multi-tiered memory hierarchy:
 * 1. READ-ONLY CACHE: Replaces __constant__ to bypass 64KB hardware limits.
 * 2. SHARED MEMORY: L1-speed staging for 3072-dimension vectors.
 * 3. MANAGED MEMORY: Unified coherence for atomic updates and centroids.
 *
 * EXPECTED OUTPUT:
 * 1. CONSOLE: Reports execution time for CPU (baseline) and GPU, followed by 
 * the Speedup Factor (CPU_Time / GPU_Time).
 * 2. FILES: Generates two CSV files ("cpu_n...csv" and "gpu_n...csv") 
 * containing comma-separated ImageID and ClusterID pairs for  
 * validation.
 */

#define MAX_CLUSTERS 10             
#define IMAGE_DIMENSIONS 3072       
#define MAX_ITERATIONS 20           
#define CIFAR_BINARY_ROW_SIZE 3073 
#define NORMALIZE_PIXEL_VALUE 255.0f 
#define SUCCESS_EXIT_CODE 0
#define FAILURE_EXIT_CODE 1
#define MILLISECONDS_CONVERSION 1000.0
#define MINI_BATCH_SIZE_DEFAULT 1024 
#define MINIBATCH_ENABLED_VAL 1      
#define FIRST_DATA_CHANNEL_OFFSET 1  
#define FILENAME_BUFFER_SIZE 256
#define NUM_STREAMS 4

__managed__ float g_accumulated_centroids[MAX_CLUSTERS * IMAGE_DIMENSIONS];
__managed__ int g_cluster_population[MAX_CLUSTERS];

/**
 * KERNEL: reset_accumulators_kernel
 * Resets accumulator state for the new training iteration.
 */
__global__ void reset_accumulators_kernel(int num_clusters) {
    int cluster_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cluster_idx >= num_clusters) return;

    g_cluster_population[cluster_idx] = 0;
    for (int dim_idx = 0; dim_idx < IMAGE_DIMENSIONS; dim_idx++) {
        g_accumulated_centroids[cluster_idx * IMAGE_DIMENSIONS + dim_idx] = 0.0f;
    }
}

/**
 * DEVICE FUNCTION: find_nearest_cluster
 * Performs an L2-norm (Euclidean) distance comparison between a single image 
 * vector and all available cluster centroids using the Read-Only Cache.
 * Returns the index of the cluster with the minimum distance.
 */
__device__ int find_nearest_cluster(const float* image_pixels, 
                                    const float* __restrict__ centroids, 
                                    int num_clusters) {
    float minimum_distance = FLT_MAX;
    int closest_cluster_id = 0;

    for (int cluster_idx = 0; cluster_idx < num_clusters; cluster_idx++) {
        float current_distance = 0.0f;
        for (int dim_idx = 0; dim_idx < IMAGE_DIMENSIONS; dim_idx++) {
            float difference = image_pixels[dim_idx] - centroids[cluster_idx 
                * IMAGE_DIMENSIONS + dim_idx];
            current_distance += (difference * difference);
        }
        if (current_distance < minimum_distance) {
            minimum_distance = current_distance;
            closest_cluster_id = cluster_idx;
        }
    }
    return closest_cluster_id;
}

/**
 * KERNEL: assignment_kernel
 * Assignment Phase: Assigns images to clusters. 
 * Minimizes Global Memory transactions by staging the high-dimensional 
 * image data into Shared Memory.
 */
__global__ void assignment_kernel(const float* __restrict__ device_pixels, 
                                  const float* __restrict__ device_centroids,
                                  int* device_assignments, 
                                  int image_count, 
                                  int num_clusters, 
                                  int global_offset) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= image_count) return;

    __shared__ float s_image[IMAGE_DIMENSIONS];

    // Cooperatively tile/load image pixels into shared memory
    for (int d = threadIdx.x; d < IMAGE_DIMENSIONS; d += blockDim.x) {
        s_image[d] = device_pixels[thread_id * IMAGE_DIMENSIONS + d];
    }
    __syncthreads();

    // The assignment step uses the cached shared memory vector
    device_assignments[thread_id + global_offset] = 
        find_nearest_cluster(s_image, device_centroids, num_clusters);
}

/**
 * KERNEL: update_kernel
 * Update Phase: Aggregates image pixel values for each cluster. 
 * Uses atomicAdd on __managed__ memory buffers to safely accumulate dimensions 
 * and population counts across multiple thread blocks without race conditions.
 */
__global__ void update_kernel(const float* device_pixels, 
                              const int* device_assignments, 
                              int image_count, int global_offset) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= image_count) return;

    int global_image_idx = thread_id + global_offset;
    int assigned_cluster = device_assignments[global_image_idx];

    atomicAdd(&g_cluster_population[assigned_cluster], 1);
    for (int dim_idx = 0; dim_idx < IMAGE_DIMENSIONS; dim_idx++) {
        atomicAdd(&g_accumulated_centroids[assigned_cluster * IMAGE_DIMENSIONS 
            + dim_idx], 
                  device_pixels[thread_id * IMAGE_DIMENSIONS + dim_idx]);
    }
}

/**
 * KERNEL: finalize_centroids_kernel
 * Normalization Phase: Calculates the new mean (centroid) for each cluster by 
 * dividing the accumulated sums by the total cluster population.
 */
__global__ void finalize_centroids_kernel(int num_clusters) {
    int cluster_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cluster_idx >= num_clusters) return;

    int current_population = g_cluster_population[cluster_idx];
    if (current_population > 0) {
        for (int dim_idx = 0; dim_idx < IMAGE_DIMENSIONS; dim_idx++) {
            int offset = cluster_idx * IMAGE_DIMENSIONS + dim_idx;
            g_accumulated_centroids[offset] /= (float)current_population;
        }
    }
}

/**
 * HOST FUNCTION: export_to_csv
 * Formats and writes the cluster assignment results (ImageID, ClusterID) 
 * to a CSV file. 
 */
void export_to_csv(const char* filename, const int* assignments, 
                   int total_images) {
    FILE* file_pointer = fopen(filename, "w");
    if (!file_pointer) return;
    fprintf(file_pointer, "ImageID,ClusterID\n");
    for (int i = 0; i < total_images; i++) {
        fprintf(file_pointer, "%d,%d\n", i, assignments[i]);
    }
    fclose(file_pointer);
    printf("Exported: %s\n", filename);
}

/**
 * HOST FUNCTION: execute_cpu_baseline
 * Implements a standard, single-threaded K-Means assignment loop on the CPU.
 */
void execute_cpu_baseline(const float* host_pixels, int* cpu_results, 
                          const float* centroids, int n, int k) {
    for (int i = 0; i < n; i++) {
        float min_dist = FLT_MAX;
        int best_id = 0;
        for (int j = 0; j < k; j++) {
            float cur_dist = 0.0f;
            for (int d = 0; d < IMAGE_DIMENSIONS; d++) {
                float diff = host_pixels[i * IMAGE_DIMENSIONS + d] - 
                    centroids[j * IMAGE_DIMENSIONS + d];
                cur_dist += (diff * diff);
            }
            if (cur_dist < min_dist) { min_dist = cur_dist; best_id = j; }
        }
        cpu_results[i] = best_id;
    }
}

/**
 * HOST FUNCTION: setup_gpu_centroids
 * Initializes the clustering process by selecting the first K images 
 * from the dataset to serve as the starting centroids. 
 */
void setup_gpu_centroids(float* host_pixel_buffer) {
    size_t total_elements = MAX_CLUSTERS * IMAGE_DIMENSIONS;
    for (size_t i = 0; i < total_elements; i++) {
        g_accumulated_centroids[i] = host_pixel_buffer[i];
    }
}

/**
 * HOST FUNCTION: load_cifar_dataset
 */
void load_cifar_dataset(const char* file_path, float* host_pixels, 
                        int num_images) {
    FILE* file_pointer = fopen(file_path, "rb");
    
    if (!file_pointer) {
        printf("Warning: %s not found. Using random data.\n", file_path);
        for (int i = 0; i < num_images * IMAGE_DIMENSIONS; i++) 
            host_pixels[i] = (float)rand() / (float)RAND_MAX;
        return;
    }

    unsigned char row_buffer[CIFAR_BINARY_ROW_SIZE];
    for (int i = 0; i < num_images; i++) {
        size_t bytes_read = fread(row_buffer, 1, CIFAR_BINARY_ROW_SIZE, file_pointer);
        if (bytes_read < CIFAR_BINARY_ROW_SIZE) break;

        for (int d = 0; d < IMAGE_DIMENSIONS; d++) {
            host_pixels[i * IMAGE_DIMENSIONS + d] = 
                (float)row_buffer[d + FIRST_DATA_CHANNEL_OFFSET] / 
                NORMALIZE_PIXEL_VALUE;
        }
    }
    fclose(file_pointer);
}

/**
 * HOST FUNCTION: dispatch_gpu_step
 * Executes the assignment and update phases within an iteration.
 */
void dispatch_gpu_step(float* device_pixels, 
                       int* device_assignments, int batch_size, int clusters, 
                       int threads, int total_n, cudaStream_t stream, int offset) {
    int blocks = (batch_size + threads - 1) / threads;
    
    assignment_kernel<<<blocks, threads, 0, stream>>>(device_pixels, 
        g_accumulated_centroids, device_assignments, batch_size, clusters, offset);
        
    update_kernel<<<blocks, threads, 0, stream>>>(device_pixels, device_assignments, 
        batch_size, offset);
}

/**
 * HOST FUNCTION: run_gpu_benchmark
 * Orchestrates the GPU training process across both chunks and iterations.
 */
float run_gpu_benchmark(float* host_pixels, int* gpu_results, int total_n, 
                        int k, int batch, int threads) {
    float *device_pixel_buffer, elapsed_ms;
    int *device_assignment_buffer;
    cudaEvent_t start, stop;
    cudaStream_t streams[NUM_STREAMS];

    int chunk_size = (total_n > 100000) ? 100000 : total_n;

    for (int i = 0; i < NUM_STREAMS; i++) cudaStreamCreate(&streams[i]);

    cudaMalloc(&device_pixel_buffer, (size_t)chunk_size * IMAGE_DIMENSIONS * sizeof(float));
    cudaMalloc(&device_assignment_buffer, (size_t)total_n * sizeof(int));
    
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Iteration loop sits at the outermost level
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        
        // Reset accumulation buffers per iteration over the entire dataset
        int reset_blocks = (k + threads - 1) / threads;
        reset_accumulators_kernel<<<reset_blocks, threads>>>(k);
        cudaDeviceSynchronize();

        for (int offset = 0; offset < total_n; offset += chunk_size) {
            int current_chunk = (offset + chunk_size > total_n) ? (total_n - offset) : chunk_size;
            int s_idx = (offset / chunk_size) % NUM_STREAMS;

            cudaMemcpyAsync(device_pixel_buffer, host_pixels + (size_t)offset * IMAGE_DIMENSIONS, 
                (size_t)current_chunk * IMAGE_DIMENSIONS * sizeof(float), 
                cudaMemcpyHostToDevice, streams[s_idx]);

            dispatch_gpu_step(device_pixel_buffer, device_assignment_buffer, 
                current_chunk, k, threads, total_n, streams[s_idx], offset);
        }

        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamSynchronize(streams[i]);
        }

        // Finalize centroids for the current iteration
        int finalize_blocks = (k + threads - 1) / threads;
        finalize_centroids_kernel<<<finalize_blocks, threads>>>(k);
        cudaDeviceSynchronize();
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    
    cudaMemcpy(gpu_results, device_assignment_buffer, (size_t)total_n * sizeof(int), 
        cudaMemcpyDeviceToHost);

    for (int i = 0; i < NUM_STREAMS; i++) cudaStreamDestroy(streams[i]);
    cudaFree(device_pixel_buffer); cudaFree(device_assignment_buffer);
    return elapsed_ms;
}

/**
 * HOST FUNCTION: parse_arguments
 */
void parse_arguments(int argc, char** argv, int* total_image_count, 
                     int* threads_per_block, int* batch_size, 
                     const char** mode_string) {
    *total_image_count = atoi(argv[1]);
    *threads_per_block = atoi(argv[2]);
    *batch_size = *total_image_count;
    *mode_string = "standard";

    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--mini-batch") == 0) {
            *batch_size = MINI_BATCH_SIZE_DEFAULT;
            *mode_string = "minibatch";
        }
    }
}

/**
 * HOST FUNCTION: display_usage
 */
void display_usage(const char* program_name) {
    printf("USAGE:\n");
    printf("  %s <total_threads> <block_size> [--mini-batch]\n\n", 
           program_name);
    printf("ARGUMENTS:\n");
    printf("  total_threads : Total number of images in the dataset\n");
    printf("  block_size    : Number of CUDA threads per block\n");
    printf("  --mini-batch  : (Optional) Use stochastic sampling (size: %d)\n", 
           MINI_BATCH_SIZE_DEFAULT);
}

/**
 * HOST FUNCTION: run_performance_comparison
 */
void run_performance_comparison(float* host_pixel_buffer,
                                int* cpu_results,
                                int* gpu_results,
                                int total_image_count,
                                int threads_per_block,
                                int current_batch_size,
                                const char* execution_mode) {
    clock_t cpu_start_timer = clock();
    execute_cpu_baseline(host_pixel_buffer, cpu_results, 
                         host_pixel_buffer, total_image_count, 
                         MAX_CLUSTERS);
    double cpu_ms = (double)(clock() - cpu_start_timer) / 
                    CLOCKS_PER_SEC * MILLISECONDS_CONVERSION;

    float gpu_ms = run_gpu_benchmark(host_pixel_buffer, 
                                     gpu_results, 
                                     total_image_count, 
                                     MAX_CLUSTERS, 
                                     current_batch_size, 
                                     threads_per_block);

    printf("Execution Mode: %s\n", execution_mode);
    printf("CPU Execution Time: %.2f ms\n", cpu_ms);
    printf("GPU Execution Time: %.2f ms\n", gpu_ms);
    printf("Speedup Factor:     %.2fx\n", (float)cpu_ms / gpu_ms);
}

/**
 * HOST FUNCTION: export_benchmark_results
 */
void export_benchmark_results(int* cpu_results,
                              int* gpu_results,
                              int total_count,
                              int block_size,
                              const char* mode) {
    char cpu_filename[FILENAME_BUFFER_SIZE];
    char gpu_filename[FILENAME_BUFFER_SIZE];

    sprintf(cpu_filename, "cpu_n%d_b%d_%s.csv", 
            total_count, block_size, mode);
    sprintf(gpu_filename, "gpu_n%d_b%d_%s.csv", 
            total_count, block_size, mode);

    export_to_csv(cpu_filename, cpu_results, total_count);
    export_to_csv(gpu_filename, gpu_results, total_count);
}

/**
 * HOST FUNCTION: allocate_host_resources
 */
void allocate_host_resources(int total_image_count, 
                             float** pixels, 
                             int** gpu_res, 
                             int** cpu_res) {
    size_t pixel_size = (size_t)total_image_count * IMAGE_DIMENSIONS * sizeof(float);
    size_t result_size = (size_t)total_image_count * sizeof(int);

    cudaHostAlloc(pixels, pixel_size, cudaHostAllocDefault);
    *gpu_res = (int*)malloc(result_size);
    *cpu_res = (int*)malloc(result_size);

    if (*pixels == NULL || *gpu_res == NULL || *cpu_res == NULL) {
        fprintf(stderr, "Critical: Host allocation failed.\n");
        exit(FAILURE_EXIT_CODE);
    }
}

/**
 * HOST FUNCTION: initialize_dataset
 */
void initialize_dataset(float* host_pixel_buffer, int total_image_count) {
    load_cifar_dataset("data_batch_1.bin", 
                       host_pixel_buffer, 
                       total_image_count);
    setup_gpu_centroids(host_pixel_buffer);
}

/**
 * HOST FUNCTION: cleanup_host_resources
 */
void cleanup_host_resources(float* pixels, int* gpu_res, int* cpu_res) {
    if (pixels) cudaFreeHost(pixels);
    if (gpu_res) free(gpu_res);
    if (cpu_res) free(cpu_res);
}

/**
 * MAIN: Application controller.
 */
int main(int argc, char** argv) {
    if (argc < 3) {
        display_usage(argv[0]);
        return FAILURE_EXIT_CODE;
    }

    srand(time(NULL));
    int total_image_count, threads_per_block, current_batch_size;
    const char* execution_mode;

    parse_arguments(argc, argv, &total_image_count, &threads_per_block, 
                    &current_batch_size, &execution_mode);

    float *host_pixel_buffer = NULL;
    int *gpu_results = NULL;
    int *cpu_results = NULL;

    allocate_host_resources(total_image_count, &host_pixel_buffer, 
                            &gpu_results, &cpu_results);

    initialize_dataset(host_pixel_buffer, total_image_count);

    run_performance_comparison(host_pixel_buffer, cpu_results, 
                               gpu_results, total_image_count, 
                               threads_per_block, current_batch_size, 
                               execution_mode);

    export_benchmark_results(cpu_results, gpu_results, 
                             total_image_count, threads_per_block, 
                             execution_mode);

    cleanup_host_resources(host_pixel_buffer, gpu_results, cpu_results);

    return SUCCESS_EXIT_CODE;
}