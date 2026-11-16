#include <cuda_runtime.h>
#include <iostream>
#include <random>

// Macro for tracking CUDA errors
#define CHECK_CUDA(call)                                      \
    do {                                                      \
        std::cout << "Executing: " << #call << std::endl;      \
        cudaError_t err = call;                               \
        std::cout << "Error code: " << (int)err << std::endl; \
        std::cout << "Error name: "                              \
                  << cudaGetErrorName(err) << std::endl;      \
        std::cout << "Error string: ["                         \
                  << cudaGetErrorString(err) << "]"            \
                  << std::endl;                                \
        if (err != cudaSuccess) exit(1);                       \
    } while (0)

__global__ void matmul_coalesed(const float* A, const float* B, float* C, int M, int N, int K){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;


    if (row < M && col < N){
        float sum = 0.0f;
        for (int i = 0; i < K; i++){
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

void validateResult(const float* C, int rows, int cols) {
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            printf("%8.4f ", C[row * cols + col]);
        }
        printf("\n");
    }
}

int main() {
    int size = 65536;
    float *A = (float*)malloc(sizeof(float) * size);
    float *B = (float*)malloc(sizeof(float) * size);
    float *C = (float*)malloc(sizeof(float) * size);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    CHECK_CUDA(cudaMalloc((void**)&d_A, sizeof(float) * size));
    CHECK_CUDA(cudaMalloc((void**)&d_B, sizeof(float) * size));
    CHECK_CUDA(cudaMalloc((void**)&d_C, sizeof(float) * size));

    std::mt19937 gen(42); 
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < size; i++) {
        A[i] = dist(gen);
        B[i] = dist(gen);
    }

    CHECK_CUDA(cudaMemcpy(d_A, A, sizeof(float) * size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B, sizeof(float) * size, cudaMemcpyHostToDevice));
    // REMOVED: cudaMemcpy(d_C, C...) because C contains uninitialized garbage data.

    dim3 threadsPerBlock(16, 16); 
    dim3 numBlocks(16, 16); 

    matmul_coalesed<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, 256, 256, 256);
    
    // Catch kernel launch errors execution failures
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(C, d_C, sizeof(float) * size, cudaMemcpyDeviceToHost));

    validateResult(C, 256, 256);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(A);
    free(B);
    free(C);

    return 0;
}