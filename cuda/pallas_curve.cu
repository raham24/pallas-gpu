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
// Using homogeneous projective coordinates (X : Y : Z) where affine = (X/Z, Y/Z)
//
// 2(X1, Y1, Z1) = (X3, Y3, Z3)
// Homogeneous projective doubling formula for a = 0:
// W = 3 * X1^2
// S = Y1 * Z1
// B = X1 * Y1 * S
// H = W^2 - 8 * B
// X3 = 2 * H * S
// Y3 = W * (4*B - H) - 8 * (Y1*S)^2
// Z3 = 8 * S^3

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

    fq_t W, S, B, H, t1, t2;

    // W = 3 * X1^2  (a = 0 for Pallas, so no a*Z1^2 term)
    fq_sqr(&t1, &p->x);              // t1 = X1^2
    fq_add(&W, &t1, &t1);            // W = 2 * X1^2
    fq_add(&W, &W, &t1);             // W = 3 * X1^2

    // S = Y1 * Z1
    fq_mul(&S, &p->y, &p->z);        // S = Y1 * Z1

    // B = X1 * Y1 * S
    fq_mul(&t1, &p->x, &p->y);       // t1 = X1 * Y1
    fq_mul(&B, &t1, &S);             // B = X1 * Y1 * S

    // H = W^2 - 8 * B
    fq_sqr(&H, &W);                  // H = W^2
    fq_add(&t1, &B, &B);             // t1 = 2*B
    fq_add(&t1, &t1, &t1);           // t1 = 4*B
    fq_add(&t1, &t1, &t1);           // t1 = 8*B
    fq_sub(&H, &H, &t1);             // H = W^2 - 8*B

    // X3 = 2 * H * S
    fq_mul(&t1, &H, &S);             // t1 = H * S
    fq_add(&result->x, &t1, &t1);    // X3 = 2 * H * S

    // Y3 = W * (4*B - H) - 8 * (Y1*S)^2
    fq_add(&t1, &B, &B);             // t1 = 2*B
    fq_add(&t1, &t1, &t1);           // t1 = 4*B
    fq_sub(&t1, &t1, &H);            // t1 = 4*B - H
    fq_mul(&t1, &W, &t1);            // t1 = W * (4*B - H)
    fq_mul(&t2, &p->y, &S);          // t2 = Y1 * S
    fq_sqr(&t2, &t2);                // t2 = (Y1*S)^2
    fq_add(&t2, &t2, &t2);           // t2 = 2*(Y1*S)^2
    fq_add(&t2, &t2, &t2);           // t2 = 4*(Y1*S)^2
    fq_add(&t2, &t2, &t2);           // t2 = 8*(Y1*S)^2
    fq_sub(&result->y, &t1, &t2);    // Y3 = W*(4*B - H) - 8*(Y1*S)^2

    // Z3 = 8 * S^3
    fq_sqr(&t1, &S);                 // t1 = S^2
    fq_mul(&t1, &t1, &S);            // t1 = S^3
    fq_add(&t1, &t1, &t1);           // t1 = 2*S^3
    fq_add(&t1, &t1, &t1);           // t1 = 4*S^3
    fq_add(&result->z, &t1, &t1);    // Z3 = 8*S^3
}

// ============================================================================
// Point Addition (Projective + Projective)
// ============================================================================
// Homogeneous projective addition (add-1998-cmo-2 from EFD)
// P + Q where P = (X1:Y1:Z1), Q = (X2:Y2:Z2), affine = (X/Z, Y/Z)
//
// u = Y2*Z1 - Y1*Z2
// v = X2*Z1 - X1*Z2
// vv = v^2, vvv = v^3
// R = vv * X1 * Z2
// A = u^2 * Z1*Z2 - vvv - 2*R
// X3 = v * A
// Y3 = u * (R - A) - vvv * Y1*Z2
// Z3 = vvv * Z1*Z2

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

    fq_t u, v, uu, vv, vvv, R_val, A, Z1Z2, S1, t1, t2;

    // u = Y2*Z1 - Y1*Z2
    fq_mul(&t1, &q->y, &p->z);       // t1 = Y2*Z1
    fq_mul(&t2, &p->y, &q->z);       // t2 = Y1*Z2 (= S1, save for later)
    fq_copy(&S1, &t2);               // S1 = Y1*Z2 (needed for Y3)
    fq_sub(&u, &t1, &t2);            // u = Y2*Z1 - Y1*Z2

    // v = X2*Z1 - X1*Z2
    fq_mul(&t1, &q->x, &p->z);       // t1 = X2*Z1
    fq_mul(&t2, &p->x, &q->z);       // t2 = X1*Z2 (= U1, save for R_val)
    fq_sub(&v, &t1, &t2);            // v = X2*Z1 - X1*Z2

    // Check if points are the same (v = 0 and u = 0 -> doubling case)
    if (fq_is_zero(&v) && fq_is_zero(&u)) {
        point_double(result, p);
        return;
    }

    // Check if points are negatives (v = 0 and u != 0 -> result is identity)
    if (fq_is_zero(&v)) {
        point_set_identity_projective(result);
        return;
    }

    // Z1Z2 = Z1 * Z2
    fq_mul(&Z1Z2, &p->z, &q->z);

    // vv = v^2
    fq_sqr(&vv, &v);
    // vvv = v^3
    fq_mul(&vvv, &v, &vv);
    // R_val = vv * X1*Z2 = vv * t2 (t2 still holds X1*Z2)
    fq_mul(&R_val, &vv, &t2);

    // A = u^2 * Z1Z2 - vvv - 2*R_val
    fq_sqr(&uu, &u);                 // uu = u^2
    fq_mul(&A, &uu, &Z1Z2);          // A = u^2 * Z1Z2
    fq_sub(&A, &A, &vvv);            // A = u^2*Z1Z2 - vvv
    fq_add(&t1, &R_val, &R_val);     // t1 = 2*R_val
    fq_sub(&A, &A, &t1);             // A = u^2*Z1Z2 - vvv - 2*R_val

    // X3 = v * A
    fq_mul(&result->x, &v, &A);

    // Y3 = u * (R_val - A) - vvv * S1
    fq_sub(&t1, &R_val, &A);         // t1 = R_val - A
    fq_mul(&t1, &u, &t1);            // t1 = u * (R_val - A)
    fq_mul(&t2, &vvv, &S1);          // t2 = vvv * Y1*Z2
    fq_sub(&result->y, &t1, &t2);    // Y3 = u*(R_val-A) - vvv*S1

    // Z3 = vvv * Z1Z2
    fq_mul(&result->z, &vvv, &Z1Z2);
}

// ============================================================================
// Mixed Addition (Projective + Affine)
// ============================================================================
// Homogeneous projective mixed addition (add-1998-cmo-2 specialized for Z2=1)
// P (projective) + Q (affine), affine = (X/Z, Y/Z)
//
// u = Y2*Z1 - Y1
// v = X2*Z1 - X1
// vv = v^2, vvv = v^3
// R = vv * X1
// A = u^2 * Z1 - vvv - 2*R
// X3 = v * A
// Y3 = u * (R - A) - vvv * Y1
// Z3 = vvv * Z1

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

    fq_t u, v, vv, vvv, R_val, A, t1, t2;

    // u = Y2*Z1 - Y1 (since Z2=1)
    fq_mul(&t1, &q->y, &p->z);       // t1 = Y2*Z1
    fq_sub(&u, &t1, &p->y);          // u = Y2*Z1 - Y1

    // v = X2*Z1 - X1 (since Z2=1)
    fq_mul(&t1, &q->x, &p->z);       // t1 = X2*Z1
    fq_sub(&v, &t1, &p->x);          // v = X2*Z1 - X1

    // Check if points are the same (v=0 and u=0 -> doubling case)
    if (fq_is_zero(&v) && fq_is_zero(&u)) {
        point_double(result, p);
        return;
    }

    // Check if points are negatives (v=0 and u!=0 -> identity)
    if (fq_is_zero(&v)) {
        point_set_identity_projective(result);
        return;
    }

    // vv = v^2
    fq_sqr(&vv, &v);
    // vvv = v^3
    fq_mul(&vvv, &v, &vv);
    // R_val = vv * X1 (since Z2=1, X1*Z2 = X1)
    fq_mul(&R_val, &vv, &p->x);

    // A = u^2 * Z1 - vvv - 2*R_val (since Z1*Z2 = Z1)
    fq_sqr(&t1, &u);                 // t1 = u^2
    fq_mul(&A, &t1, &p->z);          // A = u^2 * Z1
    fq_sub(&A, &A, &vvv);            // A = u^2*Z1 - vvv
    fq_add(&t1, &R_val, &R_val);     // t1 = 2*R_val
    fq_sub(&A, &A, &t1);             // A = u^2*Z1 - vvv - 2*R_val

    // X3 = v * A
    fq_mul(&result->x, &v, &A);

    // Y3 = u * (R_val - A) - vvv * Y1 (since Z2=1, Y1*Z2 = Y1)
    fq_sub(&t1, &R_val, &A);         // t1 = R_val - A
    fq_mul(&t1, &u, &t1);            // t1 = u * (R_val - A)
    fq_mul(&t2, &vvv, &p->y);        // t2 = vvv * Y1
    fq_sub(&result->y, &t1, &t2);    // Y3 = u*(R_val-A) - vvv*Y1

    // Z3 = vvv * Z1 (since Z1*Z2 = Z1)
    fq_mul(&result->z, &vvv, &p->z);
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
