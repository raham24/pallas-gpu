// Include all CUDA implementations in one compilation unit
// This avoids device code linking issues
#include "pallas_field.cu"
#include "pallas_curve.cu"
#include <stdio.h>

namespace pallas {

// ============================================================================
// MSM Configuration
// ============================================================================

// Window size for Pippenger's algorithm
// Optimal c ≈ ln(n), but 8 works well for most sizes
#define MSM_WINDOW_SIZE 8
#define MSM_NUM_BUCKETS ((1 << MSM_WINDOW_SIZE) - 1)  // 2^c - 1 = 255
#define MSM_NUM_WINDOWS ((256 + MSM_WINDOW_SIZE - 1) / MSM_WINDOW_SIZE)  // 32 windows

// Parallelization config
#define THREADS_PER_BLOCK 256
#define POINTS_PER_THREAD 4  // Each thread handles multiple points

// ============================================================================
// Helper: Extract window from scalar
// ============================================================================

__device__ inline uint32_t get_window(const uint32_t* scalar, int start_bit, int window_size) {
    int limb_idx = start_bit / 32;
    int bit_offset = start_bit % 32;

    if (limb_idx >= LIMBS) return 0;

    uint32_t window = scalar[limb_idx] >> bit_offset;

    // Handle case where window spans two limbs
    if (bit_offset + window_size > 32 && limb_idx + 1 < LIMBS) {
        window |= (scalar[limb_idx + 1] << (32 - bit_offset));
    }

    // Mask to window size
    window &= ((1u << window_size) - 1);
    return window;
}

// ============================================================================
// Parallel Bucket Accumulation Kernel
// ============================================================================
// Each block processes a chunk of points for ONE window.
// Uses shared memory for local bucket accumulation, then writes to global.

__global__ void msm_bucket_accumulate_kernel(
    projective_point_t* g_buckets,       // [num_windows * num_buckets] global buckets
    const affine_point_t* points,        // Input points
    const uint32_t* scalars,             // Input scalars
    int num_points,
    int window_idx                       // Which window this kernel processes
) {
    // Shared memory for this block's local buckets
    __shared__ projective_point_t s_buckets[MSM_NUM_BUCKETS];

    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize shared buckets (each thread initializes some buckets)
    for (int b = tid; b < MSM_NUM_BUCKETS; b += blockDim.x) {
        point_set_identity_projective(&s_buckets[b]);
    }
    __syncthreads();

    // Calculate which points this thread processes
    int start_bit = window_idx * MSM_WINDOW_SIZE;

    // Each thread processes multiple points with striding
    for (int i = global_idx; i < num_points; i += gridDim.x * blockDim.x) {
        const affine_point_t* point = &points[i];

        // Skip identity points
        if (point_is_identity_affine(point)) continue;

        const uint32_t* scalar = &scalars[i * LIMBS];
        uint32_t window_val = get_window(scalar, start_bit, MSM_WINDOW_SIZE);

        if (window_val == 0) continue;

        int bucket_idx = window_val - 1;

        // Atomic-free accumulation using critical section
        // Each thread adds to bucket sequentially within the block
        // This is slow but correct - optimize with atomics or sorting later
        for (int t = 0; t < blockDim.x; t++) {
            if (tid == t) {
                point_add_mixed(&s_buckets[bucket_idx], &s_buckets[bucket_idx], point);
            }
            __syncthreads();
        }
    }
    __syncthreads();

    // Write shared buckets to global memory (accumulate with existing)
    int bucket_offset = window_idx * MSM_NUM_BUCKETS;
    for (int b = tid; b < MSM_NUM_BUCKETS; b += blockDim.x) {
        // Atomic add to global buckets (using point addition)
        // Since multiple blocks may write to same window, we need atomics
        // For now, we'll use a single block per window to avoid this
        point_add(&g_buckets[bucket_offset + b], &g_buckets[bucket_offset + b], &s_buckets[b]);
    }
}

// ============================================================================
// Bucket Reduction Kernel
// ============================================================================
// Reduces buckets for each window using "summation by parts"
// Each block handles one window

__global__ void msm_bucket_reduce_kernel(
    projective_point_t* window_sums,     // Output: one sum per window
    const projective_point_t* buckets,   // Input: all buckets [num_windows * num_buckets]
    int num_windows
) {
    int window_idx = blockIdx.x;
    if (window_idx >= num_windows) return;

    // Only thread 0 does the reduction (sequential within window)
    if (threadIdx.x != 0) return;

    const projective_point_t* window_buckets = &buckets[window_idx * MSM_NUM_BUCKETS];

    projective_point_t running_sum;
    projective_point_t total;

    point_set_identity_projective(&running_sum);
    point_set_identity_projective(&total);

    // Summation by parts: sum(i * bucket[i])
    // = bucket[n] + (bucket[n] + bucket[n-1]) + ...
    for (int i = MSM_NUM_BUCKETS - 1; i >= 0; i--) {
        point_add(&running_sum, &running_sum, &window_buckets[i]);
        point_add(&total, &total, &running_sum);
    }

    point_copy_projective(&window_sums[window_idx], &total);
}

// ============================================================================
// Window Combination Kernel
// ============================================================================
// Combines window sums: result = sum(window_sum[i] * 2^(i*window_size))
// Single thread - could parallelize but window combination is fast

__global__ void msm_combine_windows_kernel(
    projective_point_t* result,
    const projective_point_t* window_sums,
    int num_windows
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    projective_point_t acc;
    point_set_identity_projective(&acc);

    // Process from highest window to lowest
    for (int w = num_windows - 1; w >= 0; w--) {
        // Double accumulator by window_size bits (shift left)
        for (int d = 0; d < MSM_WINDOW_SIZE; d++) {
            point_double(&acc, &acc);
        }
        // Add this window's contribution
        point_add(&acc, &acc, &window_sums[w]);
    }

    point_copy_projective(result, &acc);
}

// ============================================================================
// Alternative: Single-Kernel Parallel MSM (Better for smaller inputs)
// ============================================================================
// Each block handles all points for one window

__global__ void msm_single_kernel(
    projective_point_t* window_sums,     // Output: one sum per window [num_windows]
    const affine_point_t* points,
    const uint32_t* scalars,
    int num_points
) {
    int window_idx = blockIdx.x;
    if (window_idx >= MSM_NUM_WINDOWS) return;

    int tid = threadIdx.x;

    // Shared memory buckets for this window
    __shared__ projective_point_t s_buckets[MSM_NUM_BUCKETS];

    // Initialize buckets
    for (int b = tid; b < MSM_NUM_BUCKETS; b += blockDim.x) {
        point_set_identity_projective(&s_buckets[b]);
    }
    __syncthreads();

    int start_bit = window_idx * MSM_WINDOW_SIZE;

    // Each thread accumulates its assigned points into local temp
    // Then we do a sequential merge (to avoid race conditions)

    // Process points in strided fashion
    for (int base = 0; base < num_points; base += blockDim.x) {
        int i = base + tid;

        uint32_t window_val = 0;
        affine_point_t local_point;
        bool valid = false;

        if (i < num_points) {
            const affine_point_t* point = &points[i];
            if (!point_is_identity_affine(point)) {
                const uint32_t* scalar = &scalars[i * LIMBS];
                window_val = get_window(scalar, start_bit, MSM_WINDOW_SIZE);
                if (window_val > 0) {
                    // Copy point to local
                    for (int l = 0; l < LIMBS; l++) {
                        local_point.x.limbs[l] = point->x.limbs[l];
                        local_point.y.limbs[l] = point->y.limbs[l];
                    }
                    valid = true;
                }
            }
        }

        // Sequential accumulation - each thread takes a turn
        for (int t = 0; t < blockDim.x; t++) {
            if (tid == t && valid) {
                int bucket_idx = window_val - 1;
                point_add_mixed(&s_buckets[bucket_idx], &s_buckets[bucket_idx], &local_point);
            }
            __syncthreads();
        }
    }
    __syncthreads();

    // Thread 0 does bucket reduction
    if (tid == 0) {
        projective_point_t running_sum;
        projective_point_t total;

        point_set_identity_projective(&running_sum);
        point_set_identity_projective(&total);

        for (int i = MSM_NUM_BUCKETS - 1; i >= 0; i--) {
            point_add(&running_sum, &running_sum, &s_buckets[i]);
            point_add(&total, &total, &running_sum);
        }

        point_copy_projective(&window_sums[window_idx], &total);
    }
}

// ============================================================================
// MSM Serial (kept for reference/small inputs)
// ============================================================================

__device__ void msm_serial(
    projective_point_t* result,
    const affine_point_t* points,
    const uint32_t* scalars,
    int num_points
) {
    const int window_size = MSM_WINDOW_SIZE;
    const int num_buckets = MSM_NUM_BUCKETS;
    const int num_windows = MSM_NUM_WINDOWS;

    projective_point_t acc;
    point_set_identity_projective(&acc);

    for (int w = num_windows - 1; w >= 0; w--) {
        for (int d = 0; d < window_size; d++) {
            point_double(&acc, &acc);
        }

        projective_point_t buckets[MSM_NUM_BUCKETS];
        for (int b = 0; b < num_buckets; b++) {
            point_set_identity_projective(&buckets[b]);
        }

        int start_bit = w * window_size;
        for (int i = 0; i < num_points; i++) {
            const uint32_t* scalar = &scalars[i * LIMBS];
            uint32_t window_val = get_window(scalar, start_bit, window_size);

            if (window_val > 0) {
                int bucket_idx = window_val - 1;
                point_add_mixed(&buckets[bucket_idx], &buckets[bucket_idx], &points[i]);
            }
        }

        // Reduce buckets
        projective_point_t running_sum, window_sum;
        point_set_identity_projective(&running_sum);
        point_set_identity_projective(&window_sum);

        for (int i = num_buckets - 1; i >= 0; i--) {
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

} // namespace pallas

// ============================================================================
// Host API
// ============================================================================

extern "C" {

static bool g_device_selected = false;

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
    g_device_selected = true;
}

// GPU MSM entry point - uses parallel implementation
cudaError_t pallas_msm_gpu(
    const pallas::affine_point_t* h_points,
    const uint32_t* h_scalars,
    pallas::projective_point_t* h_result,
    size_t num_points
) {
    select_best_gpu();

    if (num_points == 0) {
        for (int i = 0; i < pallas::LIMBS; i++) {
            h_result->x.limbs[i] = 0;
            h_result->y.limbs[i] = 0;
            h_result->z.limbs[i] = 0;
        }
        // Set Y = 1 for identity in projective (0:1:0)
        h_result->y.limbs[0] = 1;
        return cudaSuccess;
    }

    cudaError_t err;

    // Device memory
    pallas::affine_point_t* d_points = nullptr;
    uint32_t* d_scalars = nullptr;
    pallas::projective_point_t* d_window_sums = nullptr;
    pallas::projective_point_t* d_result = nullptr;

    size_t points_size = num_points * sizeof(pallas::affine_point_t);
    size_t scalars_size = num_points * pallas::LIMBS * sizeof(uint32_t);
    size_t window_sums_size = MSM_NUM_WINDOWS * sizeof(pallas::projective_point_t);
    size_t result_size = sizeof(pallas::projective_point_t);

    // Allocate
    err = cudaMalloc(&d_points, points_size);
    if (err != cudaSuccess) goto cleanup;

    err = cudaMalloc(&d_scalars, scalars_size);
    if (err != cudaSuccess) goto cleanup;

    err = cudaMalloc(&d_window_sums, window_sums_size);
    if (err != cudaSuccess) goto cleanup;

    err = cudaMalloc(&d_result, result_size);
    if (err != cudaSuccess) goto cleanup;

    // Copy inputs
    err = cudaMemcpy(d_points, h_points, points_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;

    err = cudaMemcpy(d_scalars, h_scalars, scalars_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;

    // Choose implementation based on input size
    if (num_points < 64) {
        // Small inputs: use serial kernel (overhead of parallel not worth it)
        pallas::msm_serial_kernel<<<1, 1>>>(d_points, d_scalars, d_result, num_points);
    } else {
        // Parallel implementation:
        // Phase 1: Each block processes one window (32 blocks, 256 threads each)
        pallas::msm_single_kernel<<<MSM_NUM_WINDOWS, THREADS_PER_BLOCK>>>(
            d_window_sums, d_points, d_scalars, num_points
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) goto cleanup;

        // Phase 2: Combine windows (single thread)
        pallas::msm_combine_windows_kernel<<<1, 1>>>(d_result, d_window_sums, MSM_NUM_WINDOWS);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup;

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) goto cleanup;

    // Copy result
    err = cudaMemcpy(h_result, d_result, result_size, cudaMemcpyDeviceToHost);

cleanup:
    if (d_points) cudaFree(d_points);
    if (d_scalars) cudaFree(d_scalars);
    if (d_window_sums) cudaFree(d_window_sums);
    if (d_result) cudaFree(d_result);

    return err;
}

// Keep for testing
void pallas_field_mul_batch(
    uint32_t* h_results,
    const uint32_t* h_a,
    const uint32_t* h_b,
    int count
) {
    // Placeholder
}

} // extern "C"
