// GPU-accelerated Pippenger MSM for Pallas curve
// Uses shared-memory bucket accumulation with per-block local buckets

#include "pallas_field.cu"
#include "pallas_curve.cu"
#include <stdio.h>
#include <pthread.h>

namespace pallas {

// ============================================================================
// MSM Configuration
// ============================================================================

#define MSM_WINDOW_SIZE 8
#define MSM_NUM_BUCKETS ((1 << MSM_WINDOW_SIZE) - 1)  // 255
#define MSM_NUM_WINDOWS ((256 + MSM_WINDOW_SIZE - 1) / MSM_WINDOW_SIZE)  // 32

#define THREADS_PER_BLOCK 256

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
// Phase 1: Shared-memory bucket accumulation
// ============================================================================
// Grid: (num_chunks, MSM_NUM_WINDOWS)
// Each block accumulates one chunk of points for one window into shared memory,
// then writes to per-chunk global memory. Shared-memory spin-locks give L1-speed
// atomics with contention limited to 256 threads over 255 buckets (~1 per bucket).

__global__ void msm_bucket_acc_smem_kernel(
    projective_point_t* chunk_buckets,  // [num_chunks * MSM_NUM_WINDOWS * MSM_NUM_BUCKETS]
    const affine_point_t* points,
    const uint32_t* scalars,
    int num_points,
    int points_per_chunk
) {
    __shared__ projective_point_t s_buckets[MSM_NUM_BUCKETS];  // 255 * 96 = ~24.5 KB
    __shared__ int s_locks[MSM_NUM_BUCKETS];                   // 255 * 4  = ~1 KB

    int chunk_idx = blockIdx.x;
    int window_idx = blockIdx.y;
    int tid = threadIdx.x;

    // Initialize shared memory buckets and locks
    for (int b = tid; b < MSM_NUM_BUCKETS; b += blockDim.x) {
        point_set_identity_projective(&s_buckets[b]);
        s_locks[b] = 0;
    }
    __syncthreads();

    int chunk_start = chunk_idx * points_per_chunk;
    int chunk_end = min(chunk_start + points_per_chunk, num_points);
    int start_bit = window_idx * MSM_WINDOW_SIZE;

    // Each thread processes its stride of points
    for (int i = chunk_start + tid; i < chunk_end; i += blockDim.x) {
        const affine_point_t* pt = &points[i];
        if (point_is_identity_affine(pt)) continue;

        uint32_t wv = get_window(&scalars[i * LIMBS], start_bit);
        if (wv == 0) continue;

        int bi = wv - 1;

        // Acquire shared-memory spin-lock (L1 latency, ~10x faster than global)
        while (atomicCAS(&s_locks[bi], 0, 1) != 0) {}
        __threadfence_block();  // Acquire fence: ensure we see previous writer's updates
        point_add_mixed(&s_buckets[bi], &s_buckets[bi], pt);
        __threadfence_block();  // Release fence: ensure our updates are visible
        atomicExch(&s_locks[bi], 0);
    }
    __syncthreads();

    // Write shared buckets to per-chunk global memory
    int num_windows = gridDim.y;
    int out_base = (chunk_idx * num_windows + window_idx) * MSM_NUM_BUCKETS;
    for (int b = tid; b < MSM_NUM_BUCKETS; b += blockDim.x) {
        chunk_buckets[out_base + b] = s_buckets[b];
    }
}

// ============================================================================
// Phase 2: Reduce per-chunk buckets across all chunks
// ============================================================================
// Grid: (MSM_NUM_WINDOWS), Block: (THREADS_PER_BLOCK)
// Each thread handles one bucket index, sums across all chunks

__global__ void msm_reduce_chunks_kernel(
    projective_point_t* buckets,              // [MSM_NUM_WINDOWS * MSM_NUM_BUCKETS]
    const projective_point_t* chunk_buckets,  // [num_chunks * MSM_NUM_WINDOWS * MSM_NUM_BUCKETS]
    int num_chunks,
    int num_windows
) {
    int window_idx = blockIdx.x;
    int bucket_idx = threadIdx.x;

    if (window_idx >= num_windows || bucket_idx >= MSM_NUM_BUCKETS) return;

    projective_point_t sum;
    point_set_identity_projective(&sum);

    for (int c = 0; c < num_chunks; c++) {
        int offset = (c * num_windows + window_idx) * MSM_NUM_BUCKETS + bucket_idx;
        point_add(&sum, &sum, &chunk_buckets[offset]);
    }

    buckets[window_idx * MSM_NUM_BUCKETS + bucket_idx] = sum;
}

// ============================================================================
// Phase 3: Bucket reduction (summation by parts)
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
// Phase 4: Combine windows
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
        for (int d = 0; d < MSM_WINDOW_SIZE; d++) {
            point_double(&acc, &acc);
        }
        point_add(&acc, &acc, &window_sums[w]);
    }

    point_copy_projective(result, &acc);
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
// Montgomery conversion kernels (for standard <-> Montgomery form)
// ============================================================================

__global__ void convert_points_to_montgomery_kernel(
    affine_point_t* points,
    int num_points
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;
    fq_to_montgomery(&points[idx].x, &points[idx].x);
    fq_to_montgomery(&points[idx].y, &points[idx].y);
}

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

static pthread_mutex_t g_msm_mutex = PTHREAD_MUTEX_INITIALIZER;

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

// ============================================================================
// Persistent GPU memory pool (avoids cudaMalloc/cudaFree per call)
// ============================================================================

static pallas::affine_point_t*      g_d_points = nullptr;
static uint32_t*                    g_d_scalars = nullptr;
static pallas::projective_point_t*  g_d_chunk_buckets = nullptr;
static pallas::projective_point_t*  g_d_buckets = nullptr;
static pallas::projective_point_t*  g_d_window_sums = nullptr;
static pallas::projective_point_t*  g_d_result = nullptr;

static size_t g_cap_points = 0;        // current capacity in number of points
static size_t g_cap_chunk_buckets = 0;  // current capacity in bytes

static void ensure_device_memory(size_t num_points, size_t chunk_buckets_bytes) {
    // Reallocate points and scalars if needed
    if (num_points > g_cap_points) {
        if (g_d_points)  cudaFree(g_d_points);
        if (g_d_scalars) cudaFree(g_d_scalars);
        cudaMalloc(&g_d_points,  num_points * sizeof(pallas::affine_point_t));
        cudaMalloc(&g_d_scalars, num_points * pallas::LIMBS * sizeof(uint32_t));
        g_cap_points = num_points;
    }

    // Reallocate chunk_buckets if needed
    if (chunk_buckets_bytes > g_cap_chunk_buckets) {
        if (g_d_chunk_buckets) cudaFree(g_d_chunk_buckets);
        cudaMalloc(&g_d_chunk_buckets, chunk_buckets_bytes);
        g_cap_chunk_buckets = chunk_buckets_bytes;
    }

    // These are fixed-size, allocate once
    if (!g_d_buckets) {
        size_t buckets_size = MSM_NUM_WINDOWS * MSM_NUM_BUCKETS * sizeof(pallas::projective_point_t);
        size_t window_sums_size = MSM_NUM_WINDOWS * sizeof(pallas::projective_point_t);
        cudaMalloc(&g_d_buckets, buckets_size);
        cudaMalloc(&g_d_window_sums, window_sums_size);
        cudaMalloc(&g_d_result, sizeof(pallas::projective_point_t));
    }
}

// ============================================================================
// Main MSM entry point
// ============================================================================

cudaError_t pallas_msm_gpu(
    const pallas::affine_point_t* h_points,
    const uint32_t* h_scalars,
    pallas::projective_point_t* h_result,
    size_t num_points
) {
    pthread_mutex_lock(&g_msm_mutex);

    select_best_gpu();

    if (num_points == 0) {
        memset(h_result, 0, sizeof(pallas::projective_point_t));
        h_result->y.limbs[0] = 1;  // Identity: (0:1:0)
        pthread_mutex_unlock(&g_msm_mutex);
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

        {
            int threads = 256;
            int blocks = ((int)num_points + threads - 1) / threads;
            pallas::convert_points_to_montgomery_kernel<<<blocks, threads>>>(d_points, num_points);
        }

        pallas::msm_serial_kernel<<<1, 1>>>(d_points, d_scalars, d_result, num_points);
        pallas::convert_result_from_montgomery_kernel<<<1, 1>>>(d_result);

        cudaDeviceSynchronize();
        cudaMemcpy(h_result, d_result, sizeof(pallas::projective_point_t), cudaMemcpyDeviceToHost);

        cudaFree(d_points);
        cudaFree(d_scalars);
        cudaFree(d_result);

        err = cudaGetLastError();
        goto done;
    }

    // ====================================================================
    // Parallel Pippenger MSM
    // ====================================================================
    {
        // Adaptive chunk size: larger chunks = fewer chunks = less reduce work + memory
        int points_per_chunk;
        if ((int)num_points <= (1 << 14))       points_per_chunk = 4096;
        else if ((int)num_points <= (1 << 18))  points_per_chunk = 16384;
        else                                     points_per_chunk = 32768;

        int num_chunks = ((int)num_points + points_per_chunk - 1) / points_per_chunk;

        size_t points_size = num_points * sizeof(pallas::affine_point_t);
        size_t scalars_size = num_points * pallas::LIMBS * sizeof(uint32_t);
        size_t chunk_buckets_size = (size_t)num_chunks * MSM_NUM_WINDOWS * MSM_NUM_BUCKETS
                                    * sizeof(pallas::projective_point_t);

        // Ensure persistent device memory is large enough
        ensure_device_memory(num_points, chunk_buckets_size);

        // Copy inputs to device
        err = cudaMemcpy(g_d_points, h_points, points_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto done;

        err = cudaMemcpy(g_d_scalars, h_scalars, scalars_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto done;

        // Convert points from standard form to Montgomery form
        {
            int conv_threads = 256;
            int conv_blocks = ((int)num_points + conv_threads - 1) / conv_threads;
            pallas::convert_points_to_montgomery_kernel<<<conv_blocks, conv_threads>>>(
                g_d_points, (int)num_points
            );
            err = cudaGetLastError();
            if (err != cudaSuccess) goto done;
        }

        // Phase 1: Accumulate into per-chunk shared-memory buckets
        {
            dim3 grid(num_chunks, MSM_NUM_WINDOWS);
            pallas::msm_bucket_acc_smem_kernel<<<grid, THREADS_PER_BLOCK>>>(
                g_d_chunk_buckets, g_d_points, g_d_scalars,
                (int)num_points, points_per_chunk
            );
            err = cudaGetLastError();
            if (err != cudaSuccess) goto done;
        }

        // Phase 2: Reduce per-chunk buckets to global buckets
        pallas::msm_reduce_chunks_kernel<<<MSM_NUM_WINDOWS, THREADS_PER_BLOCK>>>(
            g_d_buckets, g_d_chunk_buckets, num_chunks, MSM_NUM_WINDOWS
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) goto done;

        // Phase 3: Summation by parts (bucket reduction)
        pallas::msm_bucket_reduce_kernel<<<MSM_NUM_WINDOWS, 1>>>(
            g_d_window_sums, g_d_buckets, MSM_NUM_WINDOWS
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) goto done;

        // Phase 4: Combine windows
        pallas::msm_combine_windows_kernel<<<1, 1>>>(
            g_d_result, g_d_window_sums, MSM_NUM_WINDOWS
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) goto done;

        // Convert result from Montgomery form to standard form
        pallas::convert_result_from_montgomery_kernel<<<1, 1>>>(g_d_result);
        err = cudaGetLastError();
        if (err != cudaSuccess) goto done;

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) goto done;

        // Copy result back
        err = cudaMemcpy(h_result, g_d_result, sizeof(pallas::projective_point_t),
                         cudaMemcpyDeviceToHost);
    }

done:
    pthread_mutex_unlock(&g_msm_mutex);
    return err;
}

void pallas_field_mul_batch(uint32_t* h_results, const uint32_t* h_a, const uint32_t* h_b, int count) {
    // Placeholder
}

} // extern "C"
