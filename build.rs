use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Find CUDA installation
    let cuda_path = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    let cuda_lib_path = PathBuf::from(&cuda_path).join("lib64");
    let cuda_include_path = PathBuf::from(&cuda_path).join("include");

    // Tell cargo where to find CUDA libraries
    println!("cargo:rustc-link-search=native={}", cuda_lib_path.display());
    println!("cargo:rustc-link-lib=cudart");

    // Rerun build if CUDA source files change
    println!("cargo:rerun-if-changed=cuda/pallas_field.cuh");
    println!("cargo:rerun-if-changed=cuda/pallas_field.cu");
    println!("cargo:rerun-if-changed=cuda/pallas_curve.cuh");
    println!("cargo:rerun-if-changed=cuda/pallas_curve.cu");
    println!("cargo:rerun-if-changed=cuda/pallas_msm.cu");

    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");

    // Detect GPU architecture
    // RTX 4080 is Ada Lovelace (sm_89), but we'll use sm_80 for broader compatibility
    // Common architectures:
    // - sm_70: Volta (V100)
    // - sm_75: Turing (RTX 20xx)
    // - sm_80: Ampere (RTX 30xx, A100)
    // - sm_86: Ampere (RTX 30xx laptop)
    // - sm_89: Ada Lovelace (RTX 40xx)
    let cuda_arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_80".to_string());

    println!("cargo:warning=Compiling CUDA code for architecture: {}", cuda_arch);

    // Compile CUDA source files
    // Note: pallas_msm.cu includes the other .cu files to avoid device linking issues
    let cuda_sources = [
        "cuda/pallas_msm.cu",
    ];

    let mut object_files = Vec::new();

    for source in &cuda_sources {
        let source_path = PathBuf::from(source);
        let obj_name = source_path
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string()
            + ".o";
        let obj_path = format!("{}/{}", out_dir, obj_name);

        println!("cargo:warning=Compiling {} -> {}", source, obj_path);

        let status = Command::new("nvcc")
            .args(&[
                "-c",
                source,
                "-o",
                &obj_path,
                &format!("-arch={}", cuda_arch),
                "--compiler-options",
                "-fPIC",
                "-I",
                &cuda_include_path.to_string_lossy(),
                "-I",
                "cuda",  // Include our own headers
                "-O3",   // Optimization
                "--expt-relaxed-constexpr",  // Allow constexpr in device code
            ])
            .status()
            .expect(&format!("Failed to compile {}", source));

        if !status.success() {
            panic!("CUDA compilation failed for {}", source);
        }

        object_files.push(obj_path);
    }

    // Link all object files into a static library
    println!("cargo:warning=Linking CUDA objects into static library");

    // Use ar to create static library
    let lib_path = format!("{}/libpallas_cuda.a", out_dir);

    let status = Command::new("ar")
        .args(&["rcs", &lib_path])
        .args(&object_files)
        .status()
        .expect("Failed to create static library");

    if !status.success() {
        panic!("Failed to create static library");
    }

    // Tell cargo to link our CUDA library
    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=pallas_cuda");

    // Also need to link against CUDA device runtime for device-side code
    println!("cargo:rustc-link-lib=cudadevrt");

    // Link C++ standard library (needed by CUDA)
    println!("cargo:rustc-link-lib=stdc++");

    println!("cargo:warning=CUDA compilation complete");
}
