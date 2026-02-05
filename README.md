# pallas-gpu

[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)

GPU-accelerated Multi-Scalar Multiplication (MSM) for the Pallas elliptic curve, optimized for [Nova](https://github.com/microsoft/Nova) recursive SNARKs.

## Overview

This library provides a high-performance CUDA implementation of MSM for the Pallas curve, a key component of the Pasta curve cycle used in Nova's Incrementally Verifiable Computation (IVC).

### Key Features

- **Pippenger's Algorithm**: O(n / log n) complexity MSM implementation
- **Montgomery Arithmetic**: CIOS (Coarsely Integrated Operand Scanning) for fast modular multiplication
- **Parallel Bucket Accumulation**: Multi-block GPU parallelism with atomic synchronization
- **Nova-Ready API**: Direct integration with halo2curves types

### Performance

Benchmarked on NVIDIA RTX 4080 (76 SMs, 16GB VRAM):

| MSM Size | Time | Throughput |
|----------|------|------------|
| 1,000 | 9.6 ms | 104k points/sec |
| 10,000 | 12.2 ms | 820k points/sec |
| 100,000 | 54.2 ms | 1.8M points/sec |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Pippenger MSM                                 [HIGH LEVEL] │
│  ├── Window decomposition (8-bit windows)                   │
│  ├── Parallel bucket accumulation (atomic locks)            │
│  └── Bucket reduction (summation by parts)                  │
├─────────────────────────────────────────────────────────────┤
│  Elliptic Curve Operations                   [MIDDLE LEVEL] │
│  ├── Point addition (Projective coordinates)                │
│  ├── Point doubling                                         │
│  └── Mixed addition (Projective + Affine)                   │
├─────────────────────────────────────────────────────────────┤
│  Field Arithmetic                              [LOW LEVEL]  │
│  ├── Montgomery multiplication (CIOS, 32-bit limbs)         │
│  ├── Field addition/subtraction                             │
│  └── Field inversion (Fermat's little theorem)              │
└─────────────────────────────────────────────────────────────┘
```

## Pallas Curve

Pallas is part of the Pasta curve cycle, designed for efficient recursive proof composition:

| Parameter | Value |
|-----------|-------|
| Field | Fq (base field) |
| Modulus | `0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001` |
| Curve Equation | y² = x³ + 5 |
| Cycle Partner | Vesta (Pallas.Fq = Vesta.Fr) |

This 2-cycle property enables Nova's IVC without expensive non-native field arithmetic.

## Requirements

- **CUDA Toolkit** 11.0+ (tested with 12.x)
- **NVIDIA GPU** with Compute Capability 8.0+ (Ampere/Ada)
- **Rust** 1.70+
- **Linux** (tested on Ubuntu)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
pallas-gpu = { git = "https://github.com/raham24/pallas-gpu.git" }
```

Or for local development:

```toml
[dependencies]
pallas-gpu = { path = "../pallas-gpu" }
```

## Building

```bash
# Ensure CUDA is in PATH
export CUDA_PATH=/usr/local/cuda
export PATH=$CUDA_PATH/bin:$PATH

# Build release
cargo build --release

# Run tests
cargo test --release

# Run benchmarks
cargo bench
```

### Build Options

Set GPU architecture (default: sm_80):

```bash
CUDA_ARCH=sm_89 cargo build --release  # For RTX 40-series (Ada)
CUDA_ARCH=sm_86 cargo build --release  # For RTX 30-series (Ampere)
```

## Usage

### Low-Level API

```rust
use pallas_gpu::{AffinePointGpu, FqGpu, msm_gpu, LIMBS};

// Create points and scalars
let points: Vec<AffinePointGpu> = vec![/* ... */];
let scalars: Vec<u32> = vec![/* LIMBS (8) u32 values per scalar */];

// Compute MSM on GPU
let result = msm_gpu(&points, &scalars)?;
// result is ProjectivePointGpu (X:Y:Z coordinates)
```

### High-Level API (for Nova integration)

```rust
use pallas_gpu::msm_u64;

// Using 64-bit limb format (halo2curves internal representation)
let scalars: Vec<[u64; 4]> = vec![/* ... */];
let points: Vec<([u64; 4], [u64; 4])> = vec![/* (x, y) coordinates */];

// Compute MSM
let (x, y, z) = msm_u64(&scalars, &points)?;
```

### Type Conversions

```rust
use pallas_gpu::FqGpu;

// From 64-bit limbs (halo2curves format)
let fq = FqGpu::from_u64_limbs(&[u64; 4]);

// From bytes
let fq = FqGpu::from_bytes_le(&[u8; 32]);

// Back to 64-bit limbs
let limbs: [u64; 4] = fq.to_u64_limbs();
```

## Project Structure

```
pallas-gpu/
├── Cargo.toml              # Crate manifest
├── build.rs                # CUDA compilation script
├── cuda/
│   ├── pallas_field.cuh    # Field element types and constants
│   ├── pallas_field.cu     # Montgomery field arithmetic
│   ├── pallas_curve.cuh    # Point types (Affine, Projective)
│   ├── pallas_curve.cu     # Elliptic curve operations
│   └── pallas_msm.cu       # Pippenger MSM kernels
├── src/
│   └── lib.rs              # Rust FFI bindings and API
└── benches/
    └── msm_benchmark.rs    # Criterion benchmarks
```

## Integration with Nova

This library is designed for use with [Nova-gpu](https://github.com/microsoft/Nova). See the integration guide for:

1. Creating `src/provider/pallas_gpu.rs` wrapper
2. Modifying `src/provider/pasta.rs` with conditional compilation
3. Adding feature flags to `Cargo.toml`

## Algorithm Details

### Pippenger's Algorithm

1. **Window Decomposition**: Split 256-bit scalars into 32 windows of 8 bits each
2. **Bucket Accumulation**: For each window, accumulate points into 255 buckets based on window value
3. **Bucket Reduction**: Use "summation by parts" to compute weighted sum: `sum(i * bucket[i])`
4. **Window Combination**: Combine window results with appropriate bit shifts

### Parallelization Strategy

- **Grid**: `(num_chunks, 32)` blocks where `num_chunks = ceil(n / 4096)`
- **Block**: 256 threads per block
- **Synchronization**: Per-bucket spin-locks for atomic accumulation
- **Memory**: Global memory for buckets, shared memory for intermediate results

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Acknowledgments

- [Nova](https://github.com/microsoft/Nova) - Recursive SNARKs without trusted setup
- [halo2curves](https://github.com/privacy-scaling-explorations/halo2curves) - Pallas/Vesta curve implementations
- [ICICLE](https://github.com/ingonyama-zk/icicle) - Reference for GPU cryptography patterns
