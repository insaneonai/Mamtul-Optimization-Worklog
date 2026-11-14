# CUDA GEMM — Toward cuBLAS Performance

Building and optimizing a high-performance matrix multiplication (GEMM) kernel from scratch using CUDA C++.

The goal is to progressively optimize a custom GPU matrix multiplication kernel and get as close as possible to cuBLAS performance on an NVIDIA RTX 3050 Ti Laptop GPU.

## Objective

Implement a sequence of CUDA GEMM kernels, understand the performance bottlenecks at each stage, and optimize them through profiling and experimentation.

This is not just an implementation exercise. Every optimization should be backed by measurements and profiling evidence.

## Optimization Roadmap

| Version | Optimization               | Main Focus                               |
| ------- | -------------------------- | ---------------------------------------- |
| V0      | Naive GEMM                 | Correctness and baseline                 |
| V1      | Thread mapping experiments | Memory access patterns                   |
| V2      | Coalesced GEMM             | Global memory efficiency                 |
| V3      | Shared memory tiling       | Data reuse and reduced global loads      |
| V4      | Register tiling            | Thread-level data reuse                  |
| V5      | Warp-level optimization    | Communication and instruction efficiency |
| V6      | Pipelining                 | Overlapping memory and computation       |
| V7      | Tensor Core GEMM           | Hardware matrix operations               |
| V8      | Final tuning               | Performance close to cuBLAS              |

Each version will be benchmarked against the previous implementation and cuBLAS.
