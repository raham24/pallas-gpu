#include "pallas_curve.cuh"

namespace pallas {

// ============================================================================
// Point Identity Checks
// ============================================================================

// Check if affine point is at infinity (represented as (0, 0))
__device__ bool point_is_identity_affine(const affine_point_t* p) {
    return fq_is_zero(&p->x) && fq_is_zero(&p->y);
}

// Check if projective point is at infinity (Z = 0)
__device__ bool point_is_identity_projective(const projective_point_t* p) {
    return fq_is_zero(&p->z);
}

// Set projective point to identity (0 : 1 : 0)
__device__ void point_set_identity_projective(projective_point_t* p) {
    fq_set_zero(&p->x);
    fq_set_one(&p->y);
    fq_set_zero(&p->z);
}

// ============================================================================
// Copy Operations
// ============================================================================

__device__ void point_copy_projective(projective_point_t* dest, const projective_point_t* src) {
    fq_copy(&dest->x, &src->x);
    fq_copy(&dest->y, &src->y);
    fq_copy(&dest->z, &src->z);
}

__device__ void point_copy_affine(affine_point_t* dest, const affine_point_t* src) {
    fq_copy(&dest->x, &src->x);
    fq_copy(&dest->y, &src->y);
}

// ============================================================================
// Coordinate Conversion
// ============================================================================

// Convert affine to projective: (x, y) -> (x : y : 1)
__device__ void affine_to_projective(projective_point_t* result, const affine_point_t* p) {
    if (point_is_identity_affine(p)) {
        point_set_identity_projective(result);
    } else {
        fq_copy(&result->x, &p->x);
        fq_copy(&result->y, &p->y);
        fq_set_one(&result->z);
    }
}

// Convert projective to affine: (X : Y : Z) -> (X/Z, Y/Z)
// Requires inversion
__device__ void projective_to_affine(affine_point_t* result, const projective_point_t* p) {
    if (point_is_identity_projective(p)) {
        fq_set_zero(&result->x);
        fq_set_zero(&result->y);
    } else {
        fq_t z_inv;
        fq_inv(&z_inv, &p->z);

        fq_mul(&result->x, &p->x, &z_inv);
        fq_mul(&result->y, &p->y, &z_inv);
    }
}

// ============================================================================
// Point Doubling
// ============================================================================
// Formula for short Weierstrass curve y^2 = x^3 + b (a = 0)
// Using projective coordinates (X : Y : Z) where affine = (X/Z, Y/Z)
//
// 2(X1, Y1, Z1) = (X3, Y3, Z3)
// Standard doubling formula for a = 0:
// XX = X1^2
// YY = Y1^2
// ZZ = Z1^2
// S = 4 * X1 * YY
// M = 3 * XX
// X3 = M^2 - 2*S
// Y3 = M * (S - X3) - 8 * YY^2
// Z3 = 2 * Y1 * Z1

__device__ void point_double(projective_point_t* result, const projective_point_t* p) {
    // Handle identity
    if (point_is_identity_projective(p)) {
        point_set_identity_projective(result);
        return;
    }

    // Check if Y = 0 (result is identity)
    if (fq_is_zero(&p->y)) {
        point_set_identity_projective(result);
        return;
    }

    fq_t XX, YY, ZZ, S, M, t1, t2, t3;

    // XX = X1^2
    fq_sqr(&XX, &p->x);

    // YY = Y1^2
    fq_sqr(&YY, &p->y);

    // ZZ = Z1^2 (not used in this simplified formula but kept for reference)
    fq_sqr(&ZZ, &p->z);

    // S = 4 * X1 * YY = 2 * (2 * X1 * YY)
    fq_mul(&t1, &p->x, &YY);    // t1 = X1 * YY
    fq_add(&t1, &t1, &t1);       // t1 = 2 * X1 * YY
    fq_add(&S, &t1, &t1);        // S = 4 * X1 * YY

    // M = 3 * XX
    fq_add(&t1, &XX, &XX);       // t1 = 2 * XX
    fq_add(&M, &t1, &XX);        // M = 3 * XX

    // X3 = M^2 - 2*S
    fq_sqr(&t1, &M);             // t1 = M^2
    fq_add(&t2, &S, &S);         // t2 = 2*S
    fq_sub(&result->x, &t1, &t2); // X3 = M^2 - 2*S

    // Y3 = M * (S - X3) - 8 * YY^2
    fq_sub(&t1, &S, &result->x); // t1 = S - X3
    fq_mul(&t2, &M, &t1);        // t2 = M * (S - X3)
    fq_sqr(&t1, &YY);            // t1 = YY^2
    fq_add(&t1, &t1, &t1);       // t1 = 2 * YY^2
    fq_add(&t1, &t1, &t1);       // t1 = 4 * YY^2
    fq_add(&t1, &t1, &t1);       // t1 = 8 * YY^2
    fq_sub(&result->y, &t2, &t1); // Y3 = M * (S - X3) - 8 * YY^2

    // Z3 = 2 * Y1 * Z1
    fq_mul(&t1, &p->y, &p->z);   // t1 = Y1 * Z1
    fq_add(&result->z, &t1, &t1); // Z3 = 2 * Y1 * Z1
}

// ============================================================================
// Point Addition (Projective + Projective)
// ============================================================================
// Standard addition formula for short Weierstrass curves
// P + Q where P = (X1:Y1:Z1), Q = (X2:Y2:Z2)

__device__ void point_add(projective_point_t* result, const projective_point_t* p, const projective_point_t* q) {
    // Handle identity cases
    if (point_is_identity_projective(p)) {
        point_copy_projective(result, q);
        return;
    }
    if (point_is_identity_projective(q)) {
        point_copy_projective(result, p);
        return;
    }

    fq_t U1, U2, S1, S2, H, R, HH, HHH, V, t1, t2;

    // U1 = X1 * Z2
    fq_mul(&U1, &p->x, &q->z);
    // U2 = X2 * Z1
    fq_mul(&U2, &q->x, &p->z);
    // S1 = Y1 * Z2
    fq_mul(&S1, &p->y, &q->z);
    // S2 = Y2 * Z1
    fq_mul(&S2, &q->y, &p->z);

    // H = U2 - U1
    fq_sub(&H, &U2, &U1);
    // R = S2 - S1
    fq_sub(&R, &S2, &S1);

    // Check if points are the same (H = 0 and R = 0 -> doubling case)
    if (fq_is_zero(&H) && fq_is_zero(&R)) {
        point_double(result, p);
        return;
    }

    // Check if points are negatives (H = 0 and R != 0 -> result is identity)
    if (fq_is_zero(&H)) {
        point_set_identity_projective(result);
        return;
    }

    // HH = H^2
    fq_sqr(&HH, &H);
    // HHH = H * HH
    fq_mul(&HHH, &H, &HH);
    // V = U1 * HH
    fq_mul(&V, &U1, &HH);

    // X3 = R^2 - HHH - 2*V
    fq_sqr(&t1, &R);             // t1 = R^2
    fq_sub(&t1, &t1, &HHH);      // t1 = R^2 - HHH
    fq_add(&t2, &V, &V);         // t2 = 2*V
    fq_sub(&result->x, &t1, &t2); // X3 = R^2 - HHH - 2*V

    // Y3 = R * (V - X3) - S1 * HHH
    fq_sub(&t1, &V, &result->x); // t1 = V - X3
    fq_mul(&t1, &R, &t1);        // t1 = R * (V - X3)
    fq_mul(&t2, &S1, &HHH);      // t2 = S1 * HHH
    fq_sub(&result->y, &t1, &t2); // Y3 = R * (V - X3) - S1 * HHH

    // Z3 = Z1 * Z2 * H
    fq_mul(&t1, &p->z, &q->z);   // t1 = Z1 * Z2
    fq_mul(&result->z, &t1, &H); // Z3 = Z1 * Z2 * H
}

// ============================================================================
// Mixed Addition (Projective + Affine)
// ============================================================================
// More efficient when one point is in affine form (Z = 1)
// P (projective) + Q (affine)

__device__ void point_add_mixed(projective_point_t* result, const projective_point_t* p, const affine_point_t* q) {
    // Handle identity cases
    if (point_is_identity_projective(p)) {
        affine_to_projective(result, q);
        return;
    }
    if (point_is_identity_affine(q)) {
        point_copy_projective(result, p);
        return;
    }

    fq_t U1, S1, H, R, HH, HHH, V, t1, t2;

    // U1 = X1 (already in correct form since we're doing P + Q where Q.z = 1)
    // U2 = X2 * Z1
    fq_mul(&t1, &q->x, &p->z);    // t1 = U2 = X2 * Z1

    // S1 = Y1
    // S2 = Y2 * Z1
    fq_mul(&t2, &q->y, &p->z);    // t2 = S2 = Y2 * Z1

    // H = U2 - U1 = X2*Z1 - X1
    fq_sub(&H, &t1, &p->x);

    // R = S2 - S1 = Y2*Z1 - Y1
    fq_sub(&R, &t2, &p->y);

    // Check if points are the same
    if (fq_is_zero(&H) && fq_is_zero(&R)) {
        // Need to double p
        point_double(result, p);
        return;
    }

    // Check if points are negatives
    if (fq_is_zero(&H)) {
        point_set_identity_projective(result);
        return;
    }

    // HH = H^2
    fq_sqr(&HH, &H);
    // HHH = H * HH
    fq_mul(&HHH, &H, &HH);
    // V = X1 * HH
    fq_mul(&V, &p->x, &HH);

    // X3 = R^2 - HHH - 2*V
    fq_sqr(&t1, &R);
    fq_sub(&t1, &t1, &HHH);
    fq_add(&t2, &V, &V);
    fq_sub(&result->x, &t1, &t2);

    // Y3 = R * (V - X3) - Y1 * HHH
    fq_sub(&t1, &V, &result->x);
    fq_mul(&t1, &R, &t1);
    fq_mul(&t2, &p->y, &HHH);
    fq_sub(&result->y, &t1, &t2);

    // Z3 = Z1 * H
    fq_mul(&result->z, &p->z, &H);
}

// ============================================================================
// Scalar Multiplication (Double-and-Add)
// ============================================================================
// Computes result = scalar * p

__device__ void point_scalar_mul(projective_point_t* result, const affine_point_t* p, const uint32_t* scalar) {
    projective_point_t acc;
    projective_point_t base;

    // Initialize accumulator to identity
    point_set_identity_projective(&acc);

    // Convert base point to projective
    affine_to_projective(&base, p);

    // Double-and-add from least significant bit
    for (int limb = 0; limb < LIMBS; limb++) {
        uint32_t s = scalar[limb];
        for (int bit = 0; bit < 32; bit++) {
            if (s & 1) {
                point_add(&acc, &acc, &base);
            }
            point_double(&base, &base);
            s >>= 1;
        }
    }

    point_copy_projective(result, &acc);
}

} // namespace pallas
