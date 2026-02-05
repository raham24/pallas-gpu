//! GPU-accelerated MSM for Pallas curve
//!
//! This crate provides CUDA-accelerated multi-scalar multiplication (MSM)
//! for the Pallas elliptic curve, designed for integration with Nova-gpu.
//!
//! # Usage with Nova
//!
//! ```ignore
//! use pallas_gpu::msm_pallas;
//! use halo2curves::pallas;
//!
//! let scalars: Vec<pallas::Scalar> = ...;
//! let bases: Vec<pallas::Affine> = ...;
//! let result: pallas::Point = msm_pallas(&scalars, &bases);
//! ```

use std::vec::Vec;

// Number of 32-bit limbs for a 256-bit field element
pub const LIMBS: usize = 8;

// ============================================================================
// GPU Types (matching CUDA structs)
// ============================================================================

/// Field element in Montgomery form (8 x 32-bit limbs, little-endian)
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct FqGpu {
    pub limbs: [u32; LIMBS],
}

/// Point in affine coordinates (x, y)
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct AffinePointGpu {
    pub x: FqGpu,
    pub y: FqGpu,
}

/// Point in projective coordinates (X:Y:Z) where x = X/Z, y = Y/Z
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
// Pallas Field Constants
// ============================================================================

/// Pallas base field modulus (little-endian 32-bit limbs)
/// p = 0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001
pub const PALLAS_MODULUS: [u32; LIMBS] = [
    0x00000001, 0x992d30ed, 0x094cf91b, 0x224698fc,
    0x00000000, 0x00000000, 0x00000000, 0x40000000,
];

/// R^2 mod p (for Montgomery conversion)
pub const PALLAS_R_SQUARED: [u32; LIMBS] = [
    0x0000000f, 0x8c78ecb3, 0x8b0de0e7, 0xd7d30dbd,
    0xc3c95d18, 0x7797a99b, 0x7b9cb714, 0x096d41af,
];

/// R mod p (Montgomery representation of 1)
pub const PALLAS_R: [u32; LIMBS] = [
    0xfffffffd, 0x34786d38, 0xe41914ad, 0x992c350b,
    0xffffffff, 0xffffffff, 0xffffffff, 0x3fffffff,
];

// ============================================================================
// Type Implementations
// ============================================================================

impl FqGpu {
    pub fn new(limbs: [u32; LIMBS]) -> Self {
        Self { limbs }
    }

    pub fn zero() -> Self {
        Self { limbs: [0; LIMBS] }
    }

    pub fn one() -> Self {
        Self { limbs: PALLAS_R }
    }

    pub fn is_zero(&self) -> bool {
        self.limbs.iter().all(|&x| x == 0)
    }

    /// Convert from little-endian byte array (32 bytes)
    pub fn from_bytes_le(bytes: &[u8; 32]) -> Self {
        let mut limbs = [0u32; LIMBS];
        for i in 0..LIMBS {
            limbs[i] = u32::from_le_bytes([
                bytes[i * 4],
                bytes[i * 4 + 1],
                bytes[i * 4 + 2],
                bytes[i * 4 + 3],
            ]);
        }
        Self { limbs }
    }

    /// Convert to little-endian byte array (32 bytes)
    pub fn to_bytes_le(&self) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        for i in 0..LIMBS {
            let limb_bytes = self.limbs[i].to_le_bytes();
            bytes[i * 4..i * 4 + 4].copy_from_slice(&limb_bytes);
        }
        bytes
    }

    /// Convert from 64-bit limbs (4 limbs) to 32-bit limbs (8 limbs)
    pub fn from_u64_limbs(limbs64: &[u64; 4]) -> Self {
        let mut limbs = [0u32; LIMBS];
        for i in 0..4 {
            limbs[i * 2] = limbs64[i] as u32;
            limbs[i * 2 + 1] = (limbs64[i] >> 32) as u32;
        }
        Self { limbs }
    }

    /// Convert to 64-bit limbs (4 limbs)
    pub fn to_u64_limbs(&self) -> [u64; 4] {
        let mut limbs64 = [0u64; 4];
        for i in 0..4 {
            limbs64[i] = self.limbs[i * 2] as u64 | ((self.limbs[i * 2 + 1] as u64) << 32);
        }
        limbs64
    }
}

impl AffinePointGpu {
    pub fn new(x: FqGpu, y: FqGpu) -> Self {
        Self { x, y }
    }

    pub fn identity() -> Self {
        Self {
            x: FqGpu::zero(),
            y: FqGpu::zero(),
        }
    }

    pub fn is_identity(&self) -> bool {
        self.x.is_zero() && self.y.is_zero()
    }
}

impl ProjectivePointGpu {
    pub fn new(x: FqGpu, y: FqGpu, z: FqGpu) -> Self {
        Self { x, y, z }
    }

    pub fn identity() -> Self {
        Self {
            x: FqGpu::zero(),
            y: FqGpu::one(),
            z: FqGpu::zero(),
        }
    }

    pub fn is_identity(&self) -> bool {
        self.z.is_zero()
    }
}

// ============================================================================
// Low-level MSM API
// ============================================================================

/// GPU-accelerated multi-scalar multiplication (low-level API)
///
/// Computes: result = sum(scalars[i] * points[i])
///
/// # Arguments
/// * `points` - Slice of affine points in GPU format
/// * `scalars` - Flat slice of scalars as u32 limbs (LIMBS=8 per scalar)
///
/// # Returns
/// * `Ok(ProjectivePointGpu)` - Result in projective coordinates
/// * `Err(String)` - Error message if CUDA operation failed
pub fn msm_gpu(
    points: &[AffinePointGpu],
    scalars: &[u32],
) -> Result<ProjectivePointGpu, String> {
    let num_points = points.len();

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

/// Convenience function that takes array scalars
pub fn msm_gpu_with_scalar_arrays(
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

    let flat_scalars: Vec<u32> = scalars.iter().flat_map(|s| s.iter().copied()).collect();
    msm_gpu(points, &flat_scalars)
}

// ============================================================================
// Nova Integration API (for halo2curves types)
// ============================================================================

/// Trait for converting halo2curves field elements to GPU format
pub trait ToGpuField {
    fn to_gpu(&self) -> FqGpu;
}

/// Trait for converting halo2curves points to GPU format
pub trait ToGpuPoint {
    fn to_gpu_affine(&self) -> AffinePointGpu;
}

/// Trait for converting GPU results back to halo2curves types
pub trait FromGpuPoint {
    fn from_gpu_projective(p: &ProjectivePointGpu) -> Self;
}

// ============================================================================
// Conversion helpers for Nova integration
// ============================================================================

/// Convert a slice of 64-bit limb representations to GPU format scalars
/// This is the format used by halo2curves internally
pub fn scalars_to_gpu(scalars_64: &[[u64; 4]]) -> Vec<u32> {
    let mut result = Vec::with_capacity(scalars_64.len() * LIMBS);
    for scalar in scalars_64 {
        // Convert each 64-bit limb to two 32-bit limbs (little-endian)
        for limb64 in scalar {
            result.push(*limb64 as u32);
            result.push((*limb64 >> 32) as u32);
        }
    }
    result
}

/// Convert a slice of affine points (as 64-bit limb coords) to GPU format
pub fn points_to_gpu(points_64: &[([u64; 4], [u64; 4])]) -> Vec<AffinePointGpu> {
    points_64
        .iter()
        .map(|(x, y)| AffinePointGpu {
            x: FqGpu::from_u64_limbs(x),
            y: FqGpu::from_u64_limbs(y),
        })
        .collect()
}

/// High-level MSM that takes 64-bit limb format (halo2curves internal format)
///
/// This is the main entry point for Nova integration.
///
/// # Arguments
/// * `scalars` - Scalars as arrays of 4 x u64 limbs (little-endian)
/// * `points` - Points as (x, y) tuples of 4 x u64 limbs each
///
/// # Returns
/// * Projective point as (X, Y, Z) tuples of 4 x u64 limbs each
pub fn msm_u64(
    scalars: &[[u64; 4]],
    points: &[([u64; 4], [u64; 4])],
) -> Result<([u64; 4], [u64; 4], [u64; 4]), String> {
    if scalars.len() != points.len() {
        return Err(format!(
            "Length mismatch: {} scalars, {} points",
            scalars.len(),
            points.len()
        ));
    }

    let gpu_scalars = scalars_to_gpu(scalars);
    let gpu_points = points_to_gpu(points);

    let result = msm_gpu(&gpu_points, &gpu_scalars)?;

    Ok((
        result.x.to_u64_limbs(),
        result.y.to_u64_limbs(),
        result.z.to_u64_limbs(),
    ))
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
    fn test_u64_limb_conversion() {
        let limbs64: [u64; 4] = [0x123456789abcdef0, 0xfedcba9876543210, 0x1111222233334444, 0x5555666677778888];
        let fq = FqGpu::from_u64_limbs(&limbs64);
        let back = fq.to_u64_limbs();
        assert_eq!(limbs64, back);
    }

    #[test]
    fn test_bytes_conversion() {
        let mut bytes = [0u8; 32];
        for i in 0..32 {
            bytes[i] = i as u8;
        }
        let fq = FqGpu::from_bytes_le(&bytes);
        let back = fq.to_bytes_le();
        assert_eq!(bytes, back);
    }

    #[test]
    #[ignore] // Only run when CUDA is available
    fn test_msm_single_point() {
        let point = AffinePointGpu {
            x: FqGpu { limbs: [1, 0, 0, 0, 0, 0, 0, 0] },
            y: FqGpu { limbs: [2, 0, 0, 0, 0, 0, 0, 0] },
        };
        let scalar = [1u32, 0, 0, 0, 0, 0, 0, 0];

        let result = msm_gpu(&[point], &scalar);
        assert!(result.is_ok());
    }
}
