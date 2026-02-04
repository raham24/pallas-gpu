//! GPU-accelerated MSM for Pallas curve
//!
//! This crate provides CUDA-accelerated multi-scalar multiplication (MSM)
//! for the Pallas elliptic curve, designed for integration with Nova-gpu.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │  MSM Algorithm (Pippenger)                              │
//! │  Complexity: O(n / log(n))                              │
//! ├─────────────────────────────────────────────────────────┤
//! │  Group Operations (Point add, double, mixed)            │
//! ├─────────────────────────────────────────────────────────┤
//! │  Field Arithmetic (Montgomery CIOS)                     │
//! └─────────────────────────────────────────────────────────┘
//! ```

use std::vec::Vec;

// Number of 32-bit limbs for a field element
const LIMBS: usize = 8;

// ============================================================================
// GPU Types (matching CUDA structs)
// ============================================================================

/// Field element in Montgomery form (8 x 32-bit limbs)
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct FqGpu {
    pub limbs: [u32; LIMBS],
}

/// Point in affine coordinates
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct AffinePointGpu {
    pub x: FqGpu,
    pub y: FqGpu,
}

/// Point in projective coordinates
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct ProjectivePointGpu {
    pub x: FqGpu,
    pub y: FqGpu,
    pub z: FqGpu,
}

// ============================================================================
// FFI Declarations
// ============================================================================

extern "C" {
    fn pallas_msm_gpu(
        points: *const AffinePointGpu,
        scalars: *const u32,
        result: *mut ProjectivePointGpu,
        num_points: usize,
    ) -> i32;
}

// ============================================================================
// Pallas Field Constants (for conversion)
// ============================================================================

/// Pallas base field modulus
pub const PALLAS_MODULUS: [u32; LIMBS] = [
    0x00000001, 0x992d30ed, 0x094cf91b, 0x224698fc,
    0x00000000, 0x00000000, 0x00000000, 0x40000000,
];

/// R^2 mod p (for Montgomery conversion)
pub const PALLAS_R_SQUARED: [u32; LIMBS] = [
    0x0000000f, 0x8c78ecb3, 0x8b0de0e7, 0xd7d30dbd,
    0xc3c95d18, 0x7797a99b, 0x7b9cb714, 0x096d41af,
];

/// R mod p (Montgomery 1)
pub const PALLAS_R: [u32; LIMBS] = [
    0xfffffffd, 0x34786d38, 0xe41914ad, 0x992c350b,
    0xffffffff, 0xffffffff, 0xffffffff, 0x3fffffff,
];

// ============================================================================
// Conversion Traits
// ============================================================================

impl FqGpu {
    /// Create a new field element from raw limbs
    pub fn new(limbs: [u32; LIMBS]) -> Self {
        Self { limbs }
    }

    /// Create zero element
    pub fn zero() -> Self {
        Self { limbs: [0; LIMBS] }
    }

    /// Create one in Montgomery form
    pub fn one() -> Self {
        Self { limbs: PALLAS_R }
    }

    /// Check if element is zero
    pub fn is_zero(&self) -> bool {
        self.limbs.iter().all(|&x| x == 0)
    }
}

impl AffinePointGpu {
    /// Create a new affine point
    pub fn new(x: FqGpu, y: FqGpu) -> Self {
        Self { x, y }
    }

    /// Create the identity point (point at infinity)
    pub fn identity() -> Self {
        Self {
            x: FqGpu::zero(),
            y: FqGpu::zero(),
        }
    }

    /// Check if point is identity
    pub fn is_identity(&self) -> bool {
        self.x.is_zero() && self.y.is_zero()
    }
}

impl ProjectivePointGpu {
    /// Create a new projective point
    pub fn new(x: FqGpu, y: FqGpu, z: FqGpu) -> Self {
        Self { x, y, z }
    }

    /// Create the identity point
    pub fn identity() -> Self {
        Self {
            x: FqGpu::zero(),
            y: FqGpu::one(),
            z: FqGpu::zero(),
        }
    }

    /// Check if point is identity (Z = 0)
    pub fn is_identity(&self) -> bool {
        self.z.is_zero()
    }
}

// ============================================================================
// MSM API
// ============================================================================

/// GPU-accelerated multi-scalar multiplication for Pallas curve
///
/// Computes: result = sum(scalars[i] * points[i])
///
/// # Arguments
/// * `points` - Slice of affine points in GPU format
/// * `scalars` - Slice of scalars as raw u32 limbs (LIMBS per scalar)
///
/// # Returns
/// * `Ok(ProjectivePointGpu)` - Result in projective coordinates
/// * `Err(String)` - Error message if CUDA operation failed
pub fn msm_gpu(
    points: &[AffinePointGpu],
    scalars: &[u32],
) -> Result<ProjectivePointGpu, String> {
    let num_points = points.len();

    // Validate input
    if scalars.len() != num_points * LIMBS {
        return Err(format!(
            "Scalars length mismatch: expected {} ({}*{}), got {}",
            num_points * LIMBS,
            num_points,
            LIMBS,
            scalars.len()
        ));
    }

    if num_points == 0 {
        return Ok(ProjectivePointGpu::identity());
    }

    let mut result = ProjectivePointGpu::default();

    unsafe {
        let ret = pallas_msm_gpu(
            points.as_ptr(),
            scalars.as_ptr(),
            &mut result,
            num_points,
        );

        if ret != 0 {
            return Err(format!("CUDA MSM failed with error code: {}", ret));
        }
    }

    Ok(result)
}

/// Convenience function that converts from Vec of scalars
pub fn msm_gpu_with_scalar_vecs(
    points: &[AffinePointGpu],
    scalars: &[[u32; LIMBS]],
) -> Result<ProjectivePointGpu, String> {
    if points.len() != scalars.len() {
        return Err(format!(
            "Points and scalars length mismatch: {} vs {}",
            points.len(),
            scalars.len()
        ));
    }

    // Flatten scalars
    let flat_scalars: Vec<u32> = scalars.iter().flat_map(|s| s.iter().copied()).collect();

    msm_gpu(points, &flat_scalars)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_element_zero() {
        let zero = FqGpu::zero();
        assert!(zero.is_zero());
    }

    #[test]
    fn test_field_element_one() {
        let one = FqGpu::one();
        assert!(!one.is_zero());
        assert_eq!(one.limbs, PALLAS_R);
    }

    #[test]
    fn test_affine_point_identity() {
        let identity = AffinePointGpu::identity();
        assert!(identity.is_identity());
    }

    #[test]
    fn test_projective_point_identity() {
        let identity = ProjectivePointGpu::identity();
        assert!(identity.is_identity());
    }

    #[test]
    fn test_msm_empty() {
        let points: Vec<AffinePointGpu> = vec![];
        let scalars: Vec<u32> = vec![];

        let result = msm_gpu(&points, &scalars).unwrap();
        assert!(result.is_identity());
    }

    #[test]
    fn test_scalars_length_validation() {
        let points = vec![AffinePointGpu::identity()];
        let scalars = vec![0u32; 4]; // Wrong length, should be LIMBS=8

        let result = msm_gpu(&points, &scalars);
        assert!(result.is_err());
    }

    #[test]
    #[ignore] // Only run when CUDA is available
    fn test_msm_single_point() {
        // Create a simple test point (not necessarily on curve for basic test)
        let point = AffinePointGpu {
            x: FqGpu { limbs: [1, 0, 0, 0, 0, 0, 0, 0] },
            y: FqGpu { limbs: [2, 0, 0, 0, 0, 0, 0, 0] },
        };
        let scalar = [1u32, 0, 0, 0, 0, 0, 0, 0]; // scalar = 1

        let result = msm_gpu(&[point], &scalar);
        assert!(result.is_ok());
    }
}
