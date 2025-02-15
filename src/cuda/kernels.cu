#include "kernels.cuh"
#include <stdio.h>

__global__ void add_one_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] += 1.0f;
    }
}

void add_one_kernel_wrapper(float* input, int size) {
    float* d_array;
    cudaError_t err;

    err = cudaMalloc(&d_array, size * sizeof(float));
    if (err != cudaSuccess) printf("Malloc error: %s\n", cudaGetErrorString(err));

    err = cudaMemcpy(d_array, input, size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) printf("H2D Memcpy error: %s\n", cudaGetErrorString(err));
    
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    add_one_kernel<<<num_blocks, block_size>>>(d_array, size);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) printf("Kernel error: %s\n", cudaGetErrorString(err));

    err = cudaMemcpy(input, d_array, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) printf("D2H Memcpy error: %s\n", cudaGetErrorString(err));

    cudaFree(d_array);
}
