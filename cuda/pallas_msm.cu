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
// Optimal c ≈ ln(n), but we use a fixed value for simplicity
// Can be made dynamic based on input size
#define MSM_WINDOW_SIZE 8
#define MSM_NUM_BUCKETS ((1 << MSM_WINDOW_SIZE) - 1)  // 2^c - 1 = 255

// ============================================================================
// Helper: Extract window from scalar
// ============================================================================
// Extracts a c-bit window from scalar starting at bit position `start_bit`

__device__ inline uint32_t get_window(const uint32_t* scalar, int start_bit, int window_size) {
    int limb_idx = start_bit / 32;
    int bit_offset = start_bit % 32;

    if (limb_idx >= LIMBS) return 0;

    uint32_t window = scalar[limb_idx] >> bit_offset;

    // Handle case where window spans two limbs
    if (bit_offset + window_size > 32 && limb_idx + 1 < LIMBS) {
        int remaining_bits = bit_offset + window_size - 32;
        window |= (scalar[limb_idx + 1] << (32 - bit_offset));
    }

    // Mask to window size
    window &= ((1u << window_size) - 1);
    return window;
}

// ============================================================================
// Bucket Reduction Kernel
// ============================================================================
// Reduces buckets using "summation by parts":
// sum(i * bucket[i]) = bucket[n] + (bucket[n] + bucket[n-1]) + ... =
// running sum from high to low, accumulated

__device__ void reduce_buckets(
    projective_point_t* result,
    projective_point_t* buckets,
    int num_buckets
) {
    projective_point_t running_sum;
    projective_point_t total;

    point_set_identity_projective(&running_sum);
    point_set_identity_projective(&total);

    // Iterate from highest bucket to lowest
    for (int i = num_buckets - 1; i >= 0; i--) {
        // running_sum += bucket[i]
        point_add(&running_sum, &running_sum, &buckets[i]);
        // total += running_sum
        point_add(&total, &total, &running_sum);
    }

    point_copy_projective(result, &total);
}

// ============================================================================
// Simple Parallel MSM Kernel (Phase 1: Accumulate into buckets)
// ============================================================================
// Each thread handles one scalar-point pair, atomically accumulates into buckets
// Note: This is a simplified version. Production would use more sophisticated
// bucket management to avoid atomic contention.

__global__ void msm_accumulate_kernel(
    projective_point_t* buckets,       // [num_windows * num_buckets] buckets
    const affine_point_t* points,      // Input points
    const uint32_t* scalars,           // Input scalars (LIMBS * num_points)
    int num_points,
    int window_size,
    int num_windows,
    int num_buckets
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    const affine_point_t* point = &points[idx];
    const uint32_t* scalar = &scalars[idx * LIMBS];

    // Skip if point is identity
    if (point_is_identity_affine(point)) return;

    // For each window
    for (int w = 0; w < num_windows; w++) {
        int start_bit = w * window_size;
        uint32_t window_val = get_window(scalar, start_bit, window_size);

        // Skip if window value is 0 (no contribution to any bucket)
        if (window_val == 0) continue;

        // Bucket index (window_val - 1 because we don't have a zero bucket)
        int bucket_idx = w * num_buckets + (window_val - 1);

        // Add point to bucket
        // Note: This is a critical section. In production, use better synchronization
        // or per-thread local accumulation followed by reduction
        projective_point_t temp;
        affine_to_projective(&temp, point);

        // Simple atomic-free version: each thread writes to shared memory
        // then reduce. For now, just direct addition (race condition!)
        // TODO: Implement proper bucket accumulation with atomics or reduction
        point_add_mixed(&buckets[bucket_idx], &buckets[bucket_idx], point);
    }
}

// ============================================================================
// MSM Serial (for reference/testing)
// ============================================================================
// Single-threaded Pippenger implementation

__device__ void msm_serial(
    projective_point_t* result,
    const affine_point_t* points,
    const uint32_t* scalars,
    int num_points
) {
    const int window_size = MSM_WINDOW_SIZE;
    const int num_buckets = MSM_NUM_BUCKETS;
    const int num_windows = (256 + window_size - 1) / window_size;

    projective_point_t acc;
    point_set_identity_projective(&acc);

    // Process windows from high to low
    for (int w = num_windows - 1; w >= 0; w--) {
        // Double accumulator by window_size bits
        for (int d = 0; d < window_size; d++) {
            point_double(&acc, &acc);
        }

        // Initialize buckets for this window
        projective_point_t buckets[MSM_NUM_BUCKETS];
        for (int b = 0; b < num_buckets; b++) {
            point_set_identity_projective(&buckets[b]);
        }

        // Accumulate points into buckets
        int start_bit = w * window_size;
        for (int i = 0; i < num_points; i++) {
            const uint32_t* scalar = &scalars[i * LIMBS];
            uint32_t window_val = get_window(scalar, start_bit, window_size);

            if (window_val > 0) {
                int bucket_idx = window_val - 1;
                point_add_mixed(&buckets[bucket_idx], &buckets[bucket_idx], &points[i]);
            }
        }

        // Reduce buckets using summation by parts
        projective_point_t window_sum;
        reduce_buckets(&window_sum, buckets, num_buckets);

        // Add window sum to accumulator
        point_add(&acc, &acc, &window_sum);
    }

    point_copy_projective(result, &acc);
}

// ============================================================================
// MSM Kernel (Wrapper)
// ============================================================================

__global__ void msm_kernel(
    const affine_point_t* points,
    const uint32_t* scalars,
    projective_point_t* result,
    int num_points
) {
    // For now, use serial implementation on single thread
    // TODO: Implement parallel version with proper bucket management
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        msm_serial(result, points, scalars, num_points);
    }
}

} // namespace pallas

// ============================================================================
// Host API
// ============================================================================

extern "C" {

// Select the best available GPU (discrete GPU preferred over integrated)
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

        // Calculate SM count (multiProcessorCount)
        int sm_count = prop.multiProcessorCount;

        printf("GPU %d: %s (SMs: %d, Compute: %d.%d, Memory: %lu MB)\n",
               i, prop.name, sm_count, prop.major, prop.minor,
               prop.totalGlobalMem / (1024 * 1024));

        // Prefer device with more SMs (discrete GPUs have more)
        if (sm_count > best_sm_count) {
            best_sm_count = sm_count;
            best_device = i;
        }
    }

    printf("Selecting GPU %d\n", best_device);
    cudaSetDevice(best_device);
    g_device_selected = true;
}

// GPU MSM entry point
cudaError_t pallas_msm_gpu(
    const pallas::affine_point_t* h_points,
    const uint32_t* h_scalars,
    pallas::projective_point_t* h_result,
    size_t num_points
) {
    // Select best GPU on first call
    select_best_gpu();

    if (num_points == 0) {
        // Return identity
        for (int i = 0; i < pallas::LIMBS; i++) {
            h_result->x.limbs[i] = 0;
            h_result->y.limbs[i] = pallas::host::R_SQUARED[i]; // 1 in Montgomery form
            h_result->z.limbs[i] = 0;
        }
        return cudaSuccess;
    }

    pallas::affine_point_t* d_points;
    uint32_t* d_scalars;
    pallas::projective_point_t* d_result;

    size_t points_size = num_points * sizeof(pallas::affine_point_t);
    size_t scalars_size = num_points * pallas::LIMBS * sizeof(uint32_t);
    size_t result_size = sizeof(pallas::projective_point_t);

    // Allocate device memory
    cudaError_t err;
    err = cudaMalloc(&d_points, points_size);
    if (err != cudaSuccess) return err;

    err = cudaMalloc(&d_scalars, scalars_size);
    if (err != cudaSuccess) {
        cudaFree(d_points);
        return err;
    }

    err = cudaMalloc(&d_result, result_size);
    if (err != cudaSuccess) {
        cudaFree(d_points);
        cudaFree(d_scalars);
        return err;
    }

    // Copy data to device
    err = cudaMemcpy(d_points, h_points, points_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;

    err = cudaMemcpy(d_scalars, h_scalars, scalars_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;

    // Launch kernel
    // Using single block/thread for serial implementation
    // TODO: Launch parallel kernel with appropriate grid/block size
    pallas::msm_kernel<<<1, 1>>>(d_points, d_scalars, d_result, num_points);

    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup;

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) goto cleanup;

    // Copy result back
    err = cudaMemcpy(h_result, d_result, result_size, cudaMemcpyDeviceToHost);

cleanup:
    cudaFree(d_points);
    cudaFree(d_scalars);
    cudaFree(d_result);

    return err;
}

// Batch field multiplication (for testing)
void pallas_field_mul_batch(
    uint32_t* h_results,
    const uint32_t* h_a,
    const uint32_t* h_b,
    int count
) {
    // TODO: Implement batch field multiplication kernel
    // For now, this is a placeholder
}

} // extern "C"
