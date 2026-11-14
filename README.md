# CUDA Kernel Lab

An experimental study of GPU performance engineering using CUDA.

The goal of this repository is to understand how GPU kernels behave, identify performance bottlenecks, and progressively optimize implementations through measurement and profiling.

## Current focus: Matrix Multiplication

Starting from a naive FP32 implementation, I investigate:

1. Thread mapping and memory coalescing
2. Global memory access patterns
3. Shared memory tiling
4. Register tiling and instruction-level parallelism
5. Warp-level optimizations
6. Tensor Core acceleration

## Methodology

Each optimization follows an experimental workflow:

- Establish a baseline.
- Form a performance hypothesis.
- Change one major variable at a time.
- Profile using NVIDIA Nsight Compute.
- Measure using CUDA events.
- Document observations, bottlenecks, and conclusions.

## Hardware

- GPU: NVIDIA RTX 3050 Ti Laptop GPU
- Programming model: CUDA
- Profiling: NVIDIA Nsight Compute
- Benchmarking: CUDA Events
