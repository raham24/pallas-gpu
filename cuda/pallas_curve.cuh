#ifndef PALLAS_CURVE_CUH
#define PALLAS_CURVE_CUH

#include "pallas_field.cuh"

namespace pallas {

// ============================================================================
// Point Types
// ============================================================================

// Point in affine coordinates (x, y)
// Used for input/output and storage
typedef struct {
    fq_t x;
    fq_t y;
} affine_point_t;

// Point in projective coordinates (X : Y : Z)
// Affine (x, y) = (X/Z, Y/Z)
// Used for intermediate computations (avoids expensive inversions)
typedef struct {
    fq_t x;
    fq_t y;
    fq_t z;
} projective_point_t;

// ============================================================================
// Point Operation Declarations
// ============================================================================

// Point doubling: result = 2 * p (projective coordinates)
__device__ void point_double(projective_point_t* result, const projective_point_t* p);

// Point addition: result = p + q (both projective)
__device__ void point_add(projective_point_t* result, const projective_point_t* p, const projective_point_t* q);

// Mixed addition: result = p (projective) + q (affine)
// More efficient when one point is in affine form
__device__ void point_add_mixed(projective_point_t* result, const projective_point_t* p, const affine_point_t* q);

// Coordinate conversion
__device__ void affine_to_projective(projective_point_t* result, const affine_point_t* p);
__device__ void projective_to_affine(affine_point_t* result, const projective_point_t* p);

// Point at infinity checks
__device__ bool point_is_identity_affine(const affine_point_t* p);
__device__ bool point_is_identity_projective(const projective_point_t* p);

// Set point to identity
__device__ void point_set_identity_projective(projective_point_t* p);

// Copy operations
__device__ void point_copy_projective(projective_point_t* dest, const projective_point_t* src);
__device__ void point_copy_affine(affine_point_t* dest, const affine_point_t* src);

// Scalar multiplication (double-and-add)
__device__ void point_scalar_mul(projective_point_t* result, const affine_point_t* p, const uint32_t* scalar);

} // namespace pallas

#endif // PALLAS_CURVE_CUH
