# pallas-gpu

GPU-accelerated Multi-Scalar Multiplication (MSM) for the Pallas elliptic curve.

## Overview

This crate provides CUDA-accelerated MSM for the Pallas curve, designed for integration with Nova-gpu for recursive proof composition (IVC).

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  MSM Algorithm (Pippenger)                    ← HIGH LEVEL  │
│  Complexity: O(n / log(n))                                  │
│  ├── Window decomposition: split scalars into c-bit chunks  │
│  ├── Bucket accumulation: points sorted by window value     │
│  └── Summation by parts: 3a+2b+c = a+(a+b)+(a+b+c)          │
├─────────────────────────────────────────────────────────────┤
│  Group Operations                           ← MIDDLE LEVEL  │
│  ├── Point addition: P + Q                                  │
│  ├── Point doubling: 2P                                     │
│  ├── Mixed addition: Projective + Affine                    │
│  └── Affine conversion                                      │
├─────────────────────────────────────────────────────────────┤
│  Field Arithmetic (Montgomery CIOS)          ← LOW LEVEL    │
│  ├── Field multiplication: a × b mod p                      │
│  ├── Field addition/subtraction                             │
│  ├── Field inversion (Fermat's little theorem)              │
│  └── Montgomery conversion: to/from Montgomery form         │
└─────────────────────────────────────────────────────────────┘
```

## Pallas Curve Parameters

```
Field: Fq (base field)
Modulus: 0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001

Curve: y² = x³ + 5 (short Weierstrass, a=0, b=5)

Pallas and Vesta form a 2-cycle:
- Pallas.Fq = Vesta.Fr
- Vesta.Fq = Pallas.Fr
```

## Requirements

- CUDA Toolkit (nvcc)
- NVIDIA GPU (tested on RTX 4080)
- Rust 1.70+

## Building

```bash
# Set CUDA path if not in standard location
export CUDA_PATH=/usr/local/cuda

# Build
cargo build --release

# Run tests (requires CUDA-capable GPU)
cargo test --release

# Run benchmarks
cargo bench
```

## Usage

```rust
use pallas_gpu::{AffinePointGpu, FqGpu, msm_gpu};

// Create points and scalars
let points: Vec<AffinePointGpu> = /* ... */;
let scalars: Vec<u32> = /* ... */;  // LIMBS=8 u32 per scalar

// Compute MSM on GPU
let result = msm_gpu(&points, &scalars)?;
```

## Integration with Nova-gpu

Add to `Cargo.toml`:

```toml
[dependencies]
pallas-gpu = { path = "../pallas-gpu" }

[features]
pallas-gpu = ["dep:pallas-gpu"]
```

See the main implementation plan for full integration details.

## File Structure

```
pallas-gpu/
├── Cargo.toml
├── build.rs              # CUDA compilation
├── cuda/
│   ├── pallas_field.cuh  # Field types and constants
│   ├── pallas_field.cu   # Field arithmetic (CIOS Montgomery)
│   ├── pallas_curve.cuh  # Point types
│   ├── pallas_curve.cu   # Point operations
│   └── pallas_msm.cu     # Pippenger MSM kernel
├── src/
│   └── lib.rs            # Rust API
└── benches/
    └── msm_benchmark.rs  # Performance benchmarks
```

