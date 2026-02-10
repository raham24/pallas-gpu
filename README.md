# pallas-gpu

GPU-accelerated Multi-Scalar Multiplication (MSM) for the Pallas elliptic curve, built for integration with [Nova-GPU](https://github.com/raham24/Nova-gpu).

## Overview

This library provides a CUDA implementation of MSM for the Pallas curve using Pippenger's algorithm. It serves as the GPU backend for Nova's Pedersen commitment engine, replacing the CPU MSM to accelerate IVC proof generation.

### Features

- **Pippenger's algorithm**: O(n / log n) MSM with 8-bit window decomposition
- **Montgomery field arithmetic**: CIOS multiplication with 8x32-bit limbs
- **Homogeneous projective coordinates**: Consistent EFD add-1998-cmo-2 formulas across all point operations
- **Parallel bucket accumulation**: Multi-block GPU parallelism with per-bucket atomic locks
- **Automatic Montgomery conversion**: Points enter in standard form, GPU handles Montgomery conversion internally

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Pippenger MSM (pallas_msm.cu)                              │
│  ├── Window decomposition (8-bit, 32 windows)               │
│  ├── Parallel bucket accumulation (atomic locks)            │
│  ├── Bucket reduction (summation by parts)                  │
│  ├── Window combination (double-and-add)                    │
│  └── Serial fallback for < 64 points                        │
├─────────────────────────────────────────────────────────────┤
│  Elliptic Curve Operations (pallas_curve.cu)                │
│  ├── point_double    (homogeneous projective, a=0)          │
│  ├── point_add       (EFD add-1998-cmo-2)                   │
│  ├── point_add_mixed (add-1998-cmo-2, Z2=1)                │
│  └── projective_to_affine (X/Z, Y/Z)                       │
├─────────────────────────────────────────────────────────────┤
│  Field Arithmetic (pallas_field.cu)                         │
│  ├── Montgomery multiplication (CIOS, 8x32-bit limbs)      │
│  ├── Field addition/subtraction (mod p)                     │
│  └── Field inversion (Fermat's little theorem)              │
└─────────────────────────────────────────────────────────────┘
```

## Pallas Curve

Pallas is part of the Pasta curve cycle, designed for efficient recursive proof composition:

| Parameter | Value |
|-----------|-------|
| Curve equation | y² = x³ + 5 |
| Base field modulus | `0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001` |
| Cycle partner | Vesta (Pallas.Fq = Vesta.Fr) |

This 2-cycle property enables Nova's IVC without expensive non-native field arithmetic.

## Data Flow

```
Rust (halo2curves)          GPU (CUDA)                    Rust
───────────────────    ──────────────────────────    ───────────────
points in standard  →  convert to Montgomery form
form via to_repr()     ↓
                       Pippenger MSM (all arithmetic
scalars in standard →  in Montgomery form; scalars
form via to_repr()     used for bit extraction only)
                       ↓
                       convert result from Montgomery
                       form to standard form         →  reconstruct via
                                                        from_repr()
```

## Project Structure

```
pallas-gpu/
├── Cargo.toml              # Crate manifest
├── build.rs                # CUDA compilation via nvcc
├── cuda/
│   ├── pallas_field.cuh    # Field constants (modulus, R, R², INV)
│   ├── pallas_field.cu     # Montgomery field arithmetic (CIOS)
│   ├── pallas_curve.cuh    # Point type declarations
│   ├── pallas_curve.cu     # Point double, add, add_mixed, scalar_mul
│   └── pallas_msm.cu       # Pippenger MSM kernels + host API
├── src/
│   └── lib.rs              # Rust FFI bindings and type conversions
└── benches/
    └── msm_benchmark.rs    # Criterion benchmarks
```

## Requirements

- **NVIDIA GPU** with Compute Capability 8.0+ (Ampere, Ada Lovelace, or newer)
- **CUDA Toolkit** 11.0+ (tested with 12.x)
- **Rust** 1.70+
- **Linux** (tested on Ubuntu with NVIDIA GeForce RTX 4080)

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
# Build release
cargo build --release

# Run tests
cargo test --release

# Run benchmarks
cargo bench
```

### GPU Architecture

Default target is `sm_80` (Ampere). Override with:

```bash
CUDA_ARCH=sm_89 cargo build --release  # RTX 40-series (Ada Lovelace)
CUDA_ARCH=sm_86 cargo build --release  # RTX 30-series (Ampere)
CUDA_ARCH=sm_80 cargo build --release  # A100, RTX 30-series (default)
```

## Usage

### High-Level API (for Nova integration)

```rust
use pallas_gpu::msm_u64;

// Scalars and points as 64-bit limb arrays (standard form, little-endian)
let scalars: Vec<[u64; 4]> = vec![/* ... */];
let points: Vec<([u64; 4], [u64; 4])> = vec![/* (x, y) coordinates */];

// Returns projective result (X, Y, Z) in standard form
let (x, y, z) = msm_u64(&scalars, &points)?;
```

### Low-Level API

```rust
use pallas_gpu::{AffinePointGpu, msm_gpu, LIMBS};

let points: Vec<AffinePointGpu> = vec![/* ... */];
let scalars: Vec<u32> = vec![/* 8 u32 limbs per scalar */];

let result = msm_gpu(&points, &scalars)?;
// result is ProjectivePointGpu { x, y, z }
```

## Integration with Nova

This library is used by [Nova-GPU](https://github.com/raham24/Nova-gpu), a fork of Microsoft's Nova that adds GPU-accelerated MSM. In that fork:

1. `src/provider/pallas_gpu.rs` wraps this crate's `msm_u64` API, converting between `halo2curves` types and GPU types
2. `src/provider/pasta.rs` conditionally dispatches MSM to the GPU when the `pallas-gpu` feature is enabled
3. `Cargo.toml` declares `pallas-gpu` as an optional dependency behind a feature flag

## Algorithm Details

### Pippenger's Algorithm

1. **Window decomposition**: Split each 256-bit scalar into 32 windows of 8 bits
2. **Bucket accumulation**: For each window, add points into 255 buckets (bucket index = window value - 1)
3. **Bucket reduction**: Summation by parts -- `255*b[255] + 254*b[254] + ... + 1*b[1]` computed as a running sum
4. **Window combination**: Process windows high-to-low, doubling 8 times between each

### Parallelization

- **Large inputs (>= 64 points)**: Grid of `(num_chunks, 32)` blocks with 256 threads each. Points are chunked into groups of 4096. Per-bucket spin-locks synchronize atomic accumulation across chunks.
- **Small inputs (< 64 points)**: Single-thread serial Pippenger kernel to avoid synchronization overhead.

## Acknowledgments

- [Microsoft Nova](https://github.com/microsoft/Nova) -- the recursive SNARK framework this library accelerates
- [halo2curves](https://github.com/privacy-scaling-explorations/halo2curves) -- Pallas/Vesta curve implementations
- [Explicit-Formulas Database](https://hyperelliptic.org/EFD/) -- source for the add-1998-cmo-2 point addition formulas
