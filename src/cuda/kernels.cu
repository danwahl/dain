#include "kernels.cuh"

__global__ void add_one_kernel(float *data, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        data[idx] += 1.0f;
    }
}

int add_one_kernel_wrapper(float *h_array, int size)
{
    float *d_array = nullptr;
    cudaError_t ret = cudaMalloc(&d_array, size * sizeof(float));
    if (ret == cudaSuccess)
    {
        ret = cudaMemcpy(d_array, h_array, size * sizeof(float), cudaMemcpyHostToDevice);
    }

    if (ret == cudaSuccess)
    {
        const int block_size = 256;
        const int num_blocks = (size + block_size - 1) / block_size;
        add_one_kernel<<<num_blocks, block_size>>>(d_array, size);

        ret = cudaGetLastError();
    }

    if (ret == cudaSuccess)
    {
        ret = cudaMemcpy(h_array, d_array, size * sizeof(float), cudaMemcpyDeviceToHost);
    }

    cudaFree(d_array);
    return ret == cudaSuccess ? 0 : 1;
}
