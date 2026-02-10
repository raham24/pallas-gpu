// MAKE NOVA WORK WITH EXSISTING CODE
// PLAY WITH BATCH SIZES
// different versions of gpu arch and compare with current arch - how good is the performance
// work more on the writing

// Include all CUDA implementations in one compilation unit
#include "pallas_field.cu"
#include "pallas_curve.cu"
#include <stdio.h>

namespace pallas {

// ============================================================================
// MSM Configuration
// ============================================================================

#define MSM_WINDOW_SIZE 8
#define MSM_NUM_BUCKETS ((1 << MSM_WINDOW_SIZE) - 1)  // 255
#define MSM_NUM_WINDOWS ((256 + MSM_WINDOW_SIZE - 1) / MSM_WINDOW_SIZE)  // 32

#define THREADS_PER_BLOCK 256
#define POINTS_PER_CHUNK 4096  // Points processed per block

// ============================================================================
// Helper: Extract window from scalar
// ============================================================================

__device__ __forceinline__ uint32_t get_window(const uint32_t* scalar, int start_bit) {
    int limb_idx = start_bit / 32;
    int bit_offset = start_bit % 32;

    if (limb_idx >= LIMBS) return 0;

    uint32_t window = scalar[limb_idx] >> bit_offset;

    if (bit_offset + MSM_WINDOW_SIZE > 32 && limb_idx + 1 < LIMBS) {
        window |= (scalar[limb_idx + 1] << (32 - bit_offset));
    }

    window &= ((1u << MSM_WINDOW_SIZE) - 1);
    return window;
}

// ============================================================================
// Atomic bucket addition using spin-lock
// ============================================================================

__device__ void atomic_bucket_add(
    projective_point_t* bucket,
    int* lock,
    const affine_point_t* point
) {
    // Acquire lock (spin)
    while (atomicCAS(lock, 0, 1) != 0) {
        // Spin - could add __nanosleep for less contention
    }
    __threadfence();  // Ensure we see latest bucket value

    // Critical section
    point_add_mixed(bucket, bucket, point);

    __threadfence();  // Ensure write is visible
    atomicExch(lock, 0);  // Release lock
}

// ============================================================================
// Phase 1: Parallel bucket accumulation with atomic locks
// ============================================================================
// Grid: (num_chunks, num_windows)
// Each block processes one chunk of points for one window

__global__ void msm_accumulate_atomic_kernel(
    projective_point_t* g_buckets,    // [num_windows * num_buckets]
    int* g_locks,                      // [num_windows * num_buckets]
    const affine_point_t* points,
    const uint32_t* scalars,
    int num_points
) {
    int chunk_idx = blockIdx.x;
    int window_idx = blockIdx.y;
    int tid = threadIdx.x;

    int chunk_start = chunk_idx * POINTS_PER_CHUNK;
    int chunk_end = min(chunk_start + POINTS_PER_CHUNK, num_points);

    if (chunk_start >= num_points) return;

    int start_bit = window_idx * MSM_WINDOW_SIZE;
    int bucket_base = window_idx * MSM_NUM_BUCKETS;

    // Each thread processes multiple points with striding
    for (int i = chunk_start + tid; i < chunk_end; i += blockDim.x) {
        const affine_point_t* point = &points[i];

        // Skip identity
        if (point_is_identity_affine(point)) continue;

        const uint32_t* scalar = &scalars[i * LIMBS];
        uint32_t window_val = get_window(scalar, start_bit);

        if (window_val == 0) continue;

        int bucket_idx = bucket_base + (window_val - 1);

        // Atomic add to global bucket
        atomic_bucket_add(&g_buckets[bucket_idx], &g_locks[bucket_idx], point);
    }
}

// ============================================================================
// Phase 2: Bucket reduction (summation by parts)
// ============================================================================
// One block per window, reduces 255 buckets to one window sum

__global__ void msm_bucket_reduce_kernel(
    projective_point_t* window_sums,     // [num_windows]
    const projective_point_t* buckets,   // [num_windows * num_buckets]
    int num_windows
) {
    int window_idx = blockIdx.x;
    if (window_idx >= num_windows) return;
    if (threadIdx.x != 0) return;  // Single thread per window

    const projective_point_t* window_buckets = &buckets[window_idx * MSM_NUM_BUCKETS];

    projective_point_t running_sum;
    projective_point_t total;

    point_set_identity_projective(&running_sum);
    point_set_identity_projective(&total);

    // Summation by parts
    for (int i = MSM_NUM_BUCKETS - 1; i >= 0; i--) {
        point_add(&running_sum, &running_sum, &window_buckets[i]);
        point_add(&total, &total, &running_sum);
    }

    point_copy_projective(&window_sums[window_idx], &total);
}

// ============================================================================
// Phase 3: Combine windows
// ============================================================================

__global__ void msm_combine_windows_kernel(
    projective_point_t* result,
    const projective_point_t* window_sums,
    int num_windows
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    projective_point_t acc;
    point_set_identity_projective(&acc);

    // High to low windows
    for (int w = num_windows - 1; w >= 0; w--) {
        // Double by window_size bits
        for (int d = 0; d < MSM_WINDOW_SIZE; d++) {
            point_double(&acc, &acc);
        }
        point_add(&acc, &acc, &window_sums[w]);
    }

    point_copy_projective(result, &acc);
}

// ============================================================================
// Alternative: Chunk-local accumulation (no atomics, uses more memory)
// ============================================================================
// Each block accumulates into its own local buckets, then we reduce

__global__ void msm_chunk_local_kernel(
    projective_point_t* chunk_buckets,   // [num_chunks * num_windows * num_buckets]
    const affine_point_t* points,
    const uint32_t* scalars,
    int num_points,
    int num_chunks
) {
    int chunk_idx = blockIdx.x;
    int window_idx = blockIdx.y;
    int tid = threadIdx.x;

    // Shared memory for this chunk's buckets
    __shared__ projective_point_t s_buckets[MSM_NUM_BUCKETS];

    // Initialize shared buckets
    for (int b = tid; b < MSM_NUM_BUCKETS; b += blockDim.x) {
        point_set_identity_projective(&s_buckets[b]);
    }
    __syncthreads();

    int chunk_start = chunk_idx * POINTS_PER_CHUNK;
    int chunk_end = min(chunk_start + POINTS_PER_CHUNK, num_points);
    int start_bit = window_idx * MSM_WINDOW_SIZE;

    // Process points - threads take turns to avoid races
    // Batch points for better efficiency
    #define BATCH_SIZE 32

    for (int batch_start = chunk_start; batch_start < chunk_end; batch_start += BATCH_SIZE) {
        int batch_end = min(batch_start + BATCH_SIZE, chunk_end);

        // Each iteration, one thread from each warp does work
        for (int t = 0; t < BATCH_SIZE && batch_start + t < chunk_end; t++) {
            int i = batch_start + t;
            int responsible_thread = t % blockDim.x;

            if (tid == responsible_thread) {
                const affine_point_t* point = &points[i];
                if (!point_is_identity_affine(point)) {
                    const uint32_t* scalar = &scalars[i * LIMBS];
                    uint32_t window_val = get_window(scalar, start_bit);

                    if (window_val > 0) {
                        int bucket_idx = window_val - 1;
                        point_add_mixed(&s_buckets[bucket_idx], &s_buckets[bucket_idx], point);
                    }
                }
            }
            __syncthreads();
        }
    }

    #undef BATCH_SIZE
    __syncthreads();

    // Write to global chunk buckets
    int out_offset = (chunk_idx * MSM_NUM_WINDOWS + window_idx) * MSM_NUM_BUCKETS;
    for (int b = tid; b < MSM_NUM_BUCKETS; b += blockDim.x) {
        point_copy_projective(&chunk_buckets[out_offset + b], &s_buckets[b]);
    }
}

// Reduce chunk buckets across all chunks
__global__ void msm_reduce_chunks_kernel(
    projective_point_t* buckets,         // [num_windows * num_buckets]
    const projective_point_t* chunk_buckets,  // [num_chunks * num_windows * num_buckets]
    int num_chunks,
    int num_windows
) {
    int window_idx = blockIdx.x;
    int bucket_idx = threadIdx.x;

    if (window_idx >= num_windows || bucket_idx >= MSM_NUM_BUCKETS) return;

    projective_point_t sum;
    point_set_identity_projective(&sum);

    // Sum across all chunks
    for (int c = 0; c < num_chunks; c++) {
        int offset = (c * num_windows + window_idx) * MSM_NUM_BUCKETS + bucket_idx;
        point_add(&sum, &sum, &chunk_buckets[offset]);
    }

    buckets[window_idx * MSM_NUM_BUCKETS + bucket_idx] = sum;
}

// ============================================================================
// Serial fallback (for small inputs)
// ============================================================================

__device__ void msm_serial(
    projective_point_t* result,
    const affine_point_t* points,
    const uint32_t* scalars,
    int num_points
) {
    projective_point_t acc;
    point_set_identity_projective(&acc);

    for (int w = MSM_NUM_WINDOWS - 1; w >= 0; w--) {
        for (int d = 0; d < MSM_WINDOW_SIZE; d++) {
            point_double(&acc, &acc);
        }

        projective_point_t buckets[MSM_NUM_BUCKETS];
        for (int b = 0; b < MSM_NUM_BUCKETS; b++) {
            point_set_identity_projective(&buckets[b]);
        }

        int start_bit = w * MSM_WINDOW_SIZE;
        for (int i = 0; i < num_points; i++) {
            const uint32_t* scalar = &scalars[i * LIMBS];
            uint32_t window_val = get_window(scalar, start_bit);

            if (window_val > 0) {
                point_add_mixed(&buckets[window_val - 1], &buckets[window_val - 1], &points[i]);
            }
        }

        projective_point_t running_sum, window_sum;
        point_set_identity_projective(&running_sum);
        point_set_identity_projective(&window_sum);

        for (int i = MSM_NUM_BUCKETS - 1; i >= 0; i--) {
            point_add(&running_sum, &running_sum, &buckets[i]);
            point_add(&window_sum, &window_sum, &running_sum);
        }

        point_add(&acc, &acc, &window_sum);
    }

    point_copy_projective(result, &acc);
}

__global__ void msm_serial_kernel(
    const affine_point_t* points,
    const uint32_t* scalars,
    projective_point_t* result,
    int num_points
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        msm_serial(result, points, scalars, num_points);
    }
}

// ============================================================================
// Montgomery conversion kernels (for standard ↔ Montgomery form)
// ============================================================================

// Convert input points from standard form to Montgomery form
__global__ void convert_points_to_montgomery_kernel(
    affine_point_t* points,
    int num_points
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;
    fq_to_montgomery(&points[idx].x, &points[idx].x);
    fq_to_montgomery(&points[idx].y, &points[idx].y);
}

// Convert result from Montgomery form to standard form
__global__ void convert_result_from_montgomery_kernel(
    projective_point_t* result
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    fq_from_montgomery(&result->x, &result->x);
    fq_from_montgomery(&result->y, &result->y);
    fq_from_montgomery(&result->z, &result->z);
}

} // namespace pallas

// ============================================================================
// Host API
// ============================================================================

extern "C" {

static bool g_device_selected = false;
static int g_selected_device = 0;

void select_best_gpu() {
    if (g_device_selected) return;

    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    if (device_count == 0) {
        fprintf(stderr, "No CUDA devices found!\n");
        return;
    }

    int best_device = 0;
    int best_sm_count = 0;

    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        int sm_count = prop.multiProcessorCount;

        printf("GPU %d: %s (SMs: %d, Compute: %d.%d, Memory: %lu MB)\n",
               i, prop.name, sm_count, prop.major, prop.minor,
               prop.totalGlobalMem / (1024 * 1024));

        if (sm_count > best_sm_count) {
            best_sm_count = sm_count;
            best_device = i;
        }
    }

    printf("Selecting GPU %d\n", best_device);
    cudaSetDevice(best_device);
    g_selected_device = best_device;
    g_device_selected = true;
}

// Main MSM entry point
cudaError_t pallas_msm_gpu(
    const pallas::affine_point_t* h_points,
    const uint32_t* h_scalars,
    pallas::projective_point_t* h_result,
    size_t num_points
) {
    select_best_gpu();

    if (num_points == 0) {
        memset(h_result, 0, sizeof(pallas::projective_point_t));
        h_result->y.limbs[0] = 1;  // Identity: (0:1:0)
        return cudaSuccess;
    }

    cudaError_t err;

    // For small inputs, use serial kernel
    if (num_points < 64) {
        pallas::affine_point_t* d_points;
        uint32_t* d_scalars;
        pallas::projective_point_t* d_result;

        size_t points_size = num_points * sizeof(pallas::affine_point_t);
        size_t scalars_size = num_points * pallas::LIMBS * sizeof(uint32_t);

        cudaMalloc(&d_points, points_size);
        cudaMalloc(&d_scalars, scalars_size);
        cudaMalloc(&d_result, sizeof(pallas::projective_point_t));

        cudaMemcpy(d_points, h_points, points_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_scalars, h_scalars, scalars_size, cudaMemcpyHostToDevice);

        // Convert points from standard form to Montgomery form
        {
            int threads = 256;
            int blocks = ((int)num_points + threads - 1) / threads;
            pallas::convert_points_to_montgomery_kernel<<<blocks, threads>>>(d_points, num_points);
        }

        pallas::msm_serial_kernel<<<1, 1>>>(d_points, d_scalars, d_result, num_points);

        // Convert result from Montgomery form to standard form
        pallas::convert_result_from_montgomery_kernel<<<1, 1>>>(d_result);

        cudaDeviceSynchronize();
        cudaMemcpy(h_result, d_result, sizeof(pallas::projective_point_t), cudaMemcpyDeviceToHost);

        cudaFree(d_points);
        cudaFree(d_scalars);
        cudaFree(d_result);

        return cudaGetLastError();
    }

    // Parallel implementation for larger inputs

    // Calculate grid dimensions
    int num_chunks = (num_points + POINTS_PER_CHUNK - 1) / POINTS_PER_CHUNK;

    // Device allocations
    pallas::affine_point_t* d_points = nullptr;
    uint32_t* d_scalars = nullptr;
    pallas::projective_point_t* d_buckets = nullptr;
    pallas::projective_point_t* d_window_sums = nullptr;
    pallas::projective_point_t* d_result = nullptr;
    int* d_locks = nullptr;

    size_t points_size = num_points * sizeof(pallas::affine_point_t);
    size_t scalars_size = num_points * pallas::LIMBS * sizeof(uint32_t);
    size_t buckets_size = MSM_NUM_WINDOWS * MSM_NUM_BUCKETS * sizeof(pallas::projective_point_t);
    size_t locks_size = MSM_NUM_WINDOWS * MSM_NUM_BUCKETS * sizeof(int);
    size_t window_sums_size = MSM_NUM_WINDOWS * sizeof(pallas::projective_point_t);

    err = cudaMalloc(&d_points, points_size);
    if (err != cudaSuccess) goto cleanup;

    err = cudaMalloc(&d_scalars, scalars_size);
    if (err != cudaSuccess) goto cleanup;

    err = cudaMalloc(&d_buckets, buckets_size);
    if (err != cudaSuccess) goto cleanup;

    err = cudaMalloc(&d_locks, locks_size);
    if (err != cudaSuccess) goto cleanup;

    err = cudaMalloc(&d_window_sums, window_sums_size);
    if (err != cudaSuccess) goto cleanup;

    err = cudaMalloc(&d_result, sizeof(pallas::projective_point_t));
    if (err != cudaSuccess) goto cleanup;

    // Initialize buckets and locks to zero
    cudaMemset(d_buckets, 0, buckets_size);
    cudaMemset(d_locks, 0, locks_size);

    // Copy inputs
    err = cudaMemcpy(d_points, h_points, points_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;

    err = cudaMemcpy(d_scalars, h_scalars, scalars_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;

    // Convert points from standard form to Montgomery form
    {
        int conv_threads = 256;
        int conv_blocks = ((int)num_points + conv_threads - 1) / conv_threads;
        pallas::convert_points_to_montgomery_kernel<<<conv_blocks, conv_threads>>>(d_points, num_points);
        err = cudaGetLastError();
        if (err != cudaSuccess) goto cleanup;
    }

    // Phase 1: Accumulate into buckets
    {
        dim3 grid(num_chunks, MSM_NUM_WINDOWS);
        dim3 block(THREADS_PER_BLOCK);

        pallas::msm_accumulate_atomic_kernel<<<grid, block>>>(
            d_buckets, d_locks, d_points, d_scalars, num_points
        );
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup;

    // Phase 2: Reduce buckets to window sums
    pallas::msm_bucket_reduce_kernel<<<MSM_NUM_WINDOWS, 1>>>(
        d_window_sums, d_buckets, MSM_NUM_WINDOWS
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup;

    // Phase 3: Combine windows
    pallas::msm_combine_windows_kernel<<<1, 1>>>(d_result, d_window_sums, MSM_NUM_WINDOWS);

    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup;

    // Convert result from Montgomery form to standard form
    pallas::convert_result_from_montgomery_kernel<<<1, 1>>>(d_result);

    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup;

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) goto cleanup;

    // Copy result
    err = cudaMemcpy(h_result, d_result, sizeof(pallas::projective_point_t), cudaMemcpyDeviceToHost);

cleanup:
    if (d_points) cudaFree(d_points);
    if (d_scalars) cudaFree(d_scalars);
    if (d_buckets) cudaFree(d_buckets);
    if (d_locks) cudaFree(d_locks);
    if (d_window_sums) cudaFree(d_window_sums);
    if (d_result) cudaFree(d_result);

    return err;
}

void pallas_field_mul_batch(uint32_t* h_results, const uint32_t* h_a, const uint32_t* h_b, int count) {
    // Placeholder
}

} // extern "C"
