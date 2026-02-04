#include "pallas_field.cuh"
#include <stdio.h>

namespace pallas {

// ============================================================================
// Multi-Precision Arithmetic Helpers
// ============================================================================

// Add two field elements: result = a + b (no modular reduction)
// Returns carry bit
__device__ inline uint32_t add_limbs(uint32_t* result, const uint32_t* a, const uint32_t* b) {
    uint64_t carry = 0;
    #pragma unroll
    for (int i = 0; i < LIMBS; i++) {
        uint64_t sum = (uint64_t)a[i] + (uint64_t)b[i] + carry;
        result[i] = (uint32_t)sum;
        carry = sum >> 32;
    }
    return (uint32_t)carry;
}

// Subtract two field elements: result = a - b (no modular reduction)
// Returns borrow bit
__device__ inline uint32_t sub_limbs(uint32_t* result, const uint32_t* a, const uint32_t* b) {
    int64_t borrow = 0;
    #pragma unroll
    for (int i = 0; i < LIMBS; i++) {
        int64_t diff = (int64_t)a[i] - (int64_t)b[i] - borrow;
        result[i] = (uint32_t)diff;
        borrow = (diff < 0) ? 1 : 0;
    }
    return (uint32_t)borrow;
}

// Modular reduction: if a >= p, compute a - p
__device__ inline void reduce_once(uint32_t* result, const uint32_t* a) {
    uint32_t temp[LIMBS];
    uint32_t borrow = sub_limbs(temp, a, MODULUS);

    // If borrow, then a < p, so use a; otherwise use temp (a - p)
    #pragma unroll
    for (int i = 0; i < LIMBS; i++) {
        result[i] = borrow ? a[i] : temp[i];
    }
}

// ============================================================================
// Montgomery Multiplication (CIOS Method)
// ============================================================================
// Coarsely Integrated Operand Scanning (CIOS) for Montgomery multiplication
// Reference: "Montgomery Arithmetic from a Software Perspective"

__device__ void montgomery_mul(uint32_t* result, const uint32_t* a, const uint32_t* b) {
    // T: accumulator for intermediate result (need extra limbs for overflow)
    uint32_t T[LIMBS + 2];

    // Initialize T to zero
    #pragma unroll
    for (int i = 0; i < LIMBS + 2; i++) {
        T[i] = 0;
    }

    // CIOS multiplication and reduction interleaved
    #pragma unroll
    for (int i = 0; i < LIMBS; i++) {
        // Step 1: T = T + a[i] * b
        uint64_t carry = 0;
        #pragma unroll
        for (int j = 0; j < LIMBS; j++) {
            uint64_t prod = (uint64_t)a[i] * (uint64_t)b[j];
            uint64_t sum = (uint64_t)T[j] + prod + carry;
            T[j] = (uint32_t)sum;
            carry = sum >> 32;
        }
        uint64_t sum = (uint64_t)T[LIMBS] + carry;
        T[LIMBS] = (uint32_t)sum;
        T[LIMBS + 1] = (uint32_t)(sum >> 32);

        // Step 2: m = T[0] * INV mod 2^32
        uint32_t m = T[0] * INV;

        // Step 3: T = (T + m * MODULUS) >> 32
        carry = 0;
        uint64_t prod0 = (uint64_t)m * (uint64_t)MODULUS[0];
        sum = (uint64_t)T[0] + prod0;
        carry = sum >> 32;

        #pragma unroll
        for (int j = 1; j < LIMBS; j++) {
            uint64_t prod = (uint64_t)m * (uint64_t)MODULUS[j];
            sum = (uint64_t)T[j] + prod + carry;
            T[j - 1] = (uint32_t)sum;
            carry = sum >> 32;
        }

        sum = (uint64_t)T[LIMBS] + carry;
        T[LIMBS - 1] = (uint32_t)sum;
        carry = sum >> 32;

        sum = (uint64_t)T[LIMBS + 1] + carry;
        T[LIMBS] = (uint32_t)sum;
        T[LIMBS + 1] = 0;
    }

    // Final reduction: T might be >= MODULUS
    reduce_once(result, T);
}

// Montgomery squaring (slightly optimized version of multiplication)
__device__ void montgomery_sqr(uint32_t* result, const uint32_t* a) {
    // For now, use multiplication. Can be optimized later.
    montgomery_mul(result, a, a);
}

// ============================================================================
// Field Arithmetic Operations
// ============================================================================

// Field addition: result = (a + b) mod p
__device__ void fq_add(fq_t* result, const fq_t* a, const fq_t* b) {
    uint32_t temp[LIMBS];
    uint32_t carry = add_limbs(temp, a->limbs, b->limbs);

    // Reduce if sum >= p
    uint32_t reduced[LIMBS];
    uint32_t borrow = sub_limbs(reduced, temp, MODULUS);

    // If carry or no borrow, use reduced; otherwise use temp
    bool needs_reduction = (carry != 0) || (borrow == 0);
    #pragma unroll
    for (int i = 0; i < LIMBS; i++) {
        result->limbs[i] = needs_reduction ? reduced[i] : temp[i];
    }
}

// Field subtraction: result = (a - b) mod p
__device__ void fq_sub(fq_t* result, const fq_t* a, const fq_t* b) {
    uint32_t temp[LIMBS];
    uint32_t borrow = sub_limbs(temp, a->limbs, b->limbs);

    // If borrow, add modulus back
    if (borrow) {
        add_limbs(result->limbs, temp, MODULUS);
    } else {
        #pragma unroll
        for (int i = 0; i < LIMBS; i++) {
            result->limbs[i] = temp[i];
        }
    }
}

// Field negation: result = -a mod p
__device__ void fq_neg(fq_t* result, const fq_t* a) {
    // Check if a is zero
    bool is_zero = true;
    #pragma unroll
    for (int i = 0; i < LIMBS; i++) {
        if (a->limbs[i] != 0) is_zero = false;
    }

    if (is_zero) {
        #pragma unroll
        for (int i = 0; i < LIMBS; i++) {
            result->limbs[i] = 0;
        }
    } else {
        sub_limbs(result->limbs, MODULUS, a->limbs);
    }
}

// Field multiplication: result = a * b (in Montgomery form)
__device__ void fq_mul(fq_t* result, const fq_t* a, const fq_t* b) {
    montgomery_mul(result->limbs, a->limbs, b->limbs);
}

// Field squaring: result = a^2 (in Montgomery form)
__device__ void fq_sqr(fq_t* result, const fq_t* a) {
    montgomery_sqr(result->limbs, a->limbs);
}

// Field inversion using Fermat's little theorem: a^(-1) = a^(p-2) mod p
// Uses binary exponentiation
__device__ void fq_inv(fq_t* result, const fq_t* a) {
    // p - 2 for Pallas field
    // p = 0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001
    // p - 2 = 0x40000000000000000000000000000000224698fc094cf91b992d30ecffffffff

    const uint32_t exp[LIMBS] = {
        0xffffffff, 0x992d30ec, 0x094cf91b, 0x224698fc,
        0x00000000, 0x00000000, 0x00000000, 0x40000000
    };

    fq_t base, acc;

    // Copy a to base
    #pragma unroll
    for (int i = 0; i < LIMBS; i++) {
        base.limbs[i] = a->limbs[i];
    }

    // Initialize accumulator to 1 in Montgomery form (= R mod p)
    #pragma unroll
    for (int i = 0; i < LIMBS; i++) {
        acc.limbs[i] = R[i];
    }

    // Binary exponentiation
    for (int limb = 0; limb < LIMBS; limb++) {
        uint32_t e = exp[limb];
        for (int bit = 0; bit < 32; bit++) {
            if (e & 1) {
                fq_mul(&acc, &acc, &base);
            }
            fq_sqr(&base, &base);
            e >>= 1;
        }
    }

    // Copy result
    #pragma unroll
    for (int i = 0; i < LIMBS; i++) {
        result->limbs[i] = acc.limbs[i];
    }
}

// ============================================================================
// Comparison Operations
// ============================================================================

// Check if field element is zero
__device__ bool fq_is_zero(const fq_t* a) {
    #pragma unroll
    for (int i = 0; i < LIMBS; i++) {
        if (a->limbs[i] != 0) return false;
    }
    return true;
}

// Check if two field elements are equal
__device__ bool fq_eq(const fq_t* a, const fq_t* b) {
    #pragma unroll
    for (int i = 0; i < LIMBS; i++) {
        if (a->limbs[i] != b->limbs[i]) return false;
    }
    return true;
}

// ============================================================================
// Montgomery Conversion
// ============================================================================

// Convert to Montgomery form: result = a * R mod p
__device__ void fq_to_montgomery(fq_t* result, const fq_t* a) {
    montgomery_mul(result->limbs, a->limbs, R_SQUARED);
}

// Convert from Montgomery form: result = a * R^(-1) mod p
__device__ void fq_from_montgomery(fq_t* result, const fq_t* a) {
    uint32_t one[LIMBS] = {1, 0, 0, 0, 0, 0, 0, 0};
    montgomery_mul(result->limbs, a->limbs, one);
}

// ============================================================================
// Utility Functions
// ============================================================================

// Copy field element
__device__ void fq_copy(fq_t* dest, const fq_t* src) {
    #pragma unroll
    for (int i = 0; i < LIMBS; i++) {
        dest->limbs[i] = src->limbs[i];
    }
}

// Set to zero
__device__ void fq_set_zero(fq_t* a) {
    #pragma unroll
    for (int i = 0; i < LIMBS; i++) {
        a->limbs[i] = 0;
    }
}

// Set to one in Montgomery form (= R mod p)
__device__ void fq_set_one(fq_t* a) {
    #pragma unroll
    for (int i = 0; i < LIMBS; i++) {
        a->limbs[i] = R[i];
    }
}

} // namespace pallas
