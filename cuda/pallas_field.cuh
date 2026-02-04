#ifndef PALLAS_FIELD_CUH
#define PALLAS_FIELD_CUH

#include <cuda_runtime.h>
#include <stdint.h>

// ============================================================================
// Pallas Base Field (Fq) Parameters
// ============================================================================
// Modulus p = 0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001
//           = 28948022309329048855892746252171976963363056481941560715954676764349967630337
//
// This is the base field for the Pallas curve (used in Nova, Halo2, Zcash Orchard)
// Pallas and Vesta form a 2-cycle: Pallas.Fq = Vesta.Fr and vice versa

namespace pallas {

// Number of 32-bit limbs to represent a field element (256 bits / 32 = 8 limbs)
constexpr int LIMBS = 8;

// The Pallas base field modulus (little-endian 32-bit limbs)
__device__ __constant__ uint32_t MODULUS[LIMBS] = {
    0x00000001, 0x992d30ed, 0x094cf91b, 0x224698fc,
    0x00000000, 0x00000000, 0x00000000, 0x40000000
};

// Montgomery constant R = 2^256 mod p
__device__ __constant__ uint32_t R[LIMBS] = {
    0xfffffffd, 0x34786d38, 0xe41914ad, 0x992c350b,
    0xffffffff, 0xffffffff, 0xffffffff, 0x3fffffff
};

// Montgomery constant R^2 = 2^512 mod p (for to_montgomery conversion)
__device__ __constant__ uint32_t R_SQUARED[LIMBS] = {
    0x0000000f, 0x8c78ecb3, 0x8b0de0e7, 0xd7d30dbd,
    0xc3c95d18, 0x7797a99b, 0x7b9cb714, 0x096d41af
};

// Montgomery constant INV = -p^(-1) mod 2^32 (for Montgomery reduction)
__device__ __constant__ uint32_t INV = 0xffffffff;

// Pallas curve parameter: y^2 = x^3 + 5 (b = 5)
__device__ __constant__ uint32_t CURVE_B[LIMBS] = {
    0x00000005, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000
};

// Curve B in Montgomery form (5 * R mod p)
// Computed as: montgomery_mul(5, R_SQUARED)
__device__ __constant__ uint32_t CURVE_B_MONT[LIMBS] = {
    0xfffffff1, 0x03c3ed4f, 0xd9b5fec5, 0xdd1450dd,
    0xffffffff, 0xffffffff, 0xffffffff, 0x3fffffff
};

// ============================================================================
// Field Element Type
// ============================================================================

// Field element represented as 8 x 32-bit limbs in Montgomery form
typedef struct {
    uint32_t limbs[LIMBS];
} fq_t;

// ============================================================================
// Field Arithmetic Function Declarations
// ============================================================================

// Basic arithmetic
__device__ void fq_add(fq_t* result, const fq_t* a, const fq_t* b);
__device__ void fq_sub(fq_t* result, const fq_t* a, const fq_t* b);
__device__ void fq_neg(fq_t* result, const fq_t* a);
__device__ void fq_mul(fq_t* result, const fq_t* a, const fq_t* b);
__device__ void fq_sqr(fq_t* result, const fq_t* a);
__device__ void fq_inv(fq_t* result, const fq_t* a);

// Comparison
__device__ bool fq_is_zero(const fq_t* a);
__device__ bool fq_eq(const fq_t* a, const fq_t* b);

// Montgomery conversion
__device__ void fq_to_montgomery(fq_t* result, const fq_t* a);
__device__ void fq_from_montgomery(fq_t* result, const fq_t* a);

// Copy
__device__ void fq_copy(fq_t* dest, const fq_t* src);

// Set to constant
__device__ void fq_set_zero(fq_t* a);
__device__ void fq_set_one(fq_t* a);  // Sets to 1 in Montgomery form (= R mod p)

// ============================================================================
// Host-side constants (for CPU code)
// ============================================================================

#ifdef __cplusplus
namespace host {
    constexpr uint32_t MODULUS[8] = {
        0x00000001, 0x992d30ed, 0x094cf91b, 0x224698fc,
        0x00000000, 0x00000000, 0x00000000, 0x40000000
    };

    constexpr uint32_t R_SQUARED[8] = {
        0x0000000f, 0x8c78ecb3, 0x8b0de0e7, 0xd7d30dbd,
        0xc3c95d18, 0x7797a99b, 0x7b9cb714, 0x096d41af
    };

    constexpr uint32_t INV = 0xffffffff;
}
#endif

} // namespace pallas

#endif // PALLAS_FIELD_CUH
