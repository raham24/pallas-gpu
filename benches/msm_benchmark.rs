use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, SamplingMode};
use pallas_gpu::{AffinePointGpu, FqGpu, msm_gpu};
use rand::Rng;
use std::time::Duration;

const LIMBS: usize = 8;

/// Generate random field element (for testing, not cryptographically secure)
fn random_fq() -> FqGpu {
    let mut rng = rand::thread_rng();
    let mut limbs = [0u32; LIMBS];
    for i in 0..LIMBS {
        limbs[i] = rng.gen();
    }
    // Simple reduction: just clear top bit to ensure < modulus
    limbs[LIMBS - 1] &= 0x3fffffff;
    FqGpu::new(limbs)
}

/// Generate random affine point (for testing)
fn random_point() -> AffinePointGpu {
    AffinePointGpu::new(random_fq(), random_fq())
}

/// Generate random scalar
fn random_scalar() -> [u32; LIMBS] {
    let mut rng = rand::thread_rng();
    let mut scalar = [0u32; LIMBS];
    for i in 0..LIMBS {
        scalar[i] = rng.gen();
    }
    // Simple reduction
    scalar[LIMBS - 1] &= 0x3fffffff;
    scalar
}

fn bench_msm_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("MSM GPU");

    // Reduce sample size for faster benchmarking (serial impl is slow)
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    for size in [10, 100, 1000].iter() {  // Removed 10000 for now (too slow)
        let points: Vec<AffinePointGpu> = (0..*size).map(|_| random_point()).collect();
        let scalars: Vec<[u32; LIMBS]> = (0..*size).map(|_| random_scalar()).collect();
        let flat_scalars: Vec<u32> = scalars.iter().flat_map(|s| s.iter().copied()).collect();

        group.bench_with_input(
            BenchmarkId::new("msm_gpu", size),
            size,
            |b, _| {
                b.iter(|| {
                    let _ = msm_gpu(black_box(&points), black_box(&flat_scalars));
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_msm_gpu);
criterion_main!(benches);
