#include "kernels.cuh"

__global__ void add_kernel(float *a, float *b, float *c, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void matmul_kernel(float *a, float *b, float *c, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n)
    {
        float sum = 0.0f;
        for (int i = 0; i < k; i++)
        {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

__global__ void relu_kernel(float *x, float *y, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        y[idx] = fmaxf(0.0f, x[idx]);
    }
}

__global__ void relu_grad_kernel(float *x, float *grad_in, float *grad_out, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        grad_out[idx] = x[idx] > 0.0f ? grad_in[idx] : 0.0f;
    }
}

int dain_add(float *h_a, float *h_b, float *h_c, int size)
{
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

    cudaError_t ret = cudaMalloc(&d_a, size * sizeof(float));
    if (ret == cudaSuccess)
    {
        ret = cudaMalloc(&d_b, size * sizeof(float));
    }
    if (ret == cudaSuccess)
    {
        ret = cudaMalloc(&d_c, size * sizeof(float));
    }

    if (ret == cudaSuccess)
    {
        ret = cudaMemcpy(d_a, h_a, size * sizeof(float), cudaMemcpyHostToDevice);
    }
    if (ret == cudaSuccess)
    {
        ret = cudaMemcpy(d_b, h_b, size * sizeof(float), cudaMemcpyHostToDevice);
    }

    if (ret == cudaSuccess)
    {
        const int block_size = 256;
        const int num_blocks = (size + block_size - 1) / block_size;
        add_kernel<<<num_blocks, block_size>>>(d_a, d_b, d_c, size);
        ret = cudaGetLastError();
    }

    if (ret == cudaSuccess)
    {
        ret = cudaMemcpy(h_c, d_c, size * sizeof(float), cudaMemcpyDeviceToHost);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return ret == cudaSuccess ? 0 : 1;
}

int dain_matmul(float *h_a, float *h_b, float *h_c, int m, int n, int k)
{
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

    cudaError_t ret = cudaMalloc(&d_a, m * k * sizeof(float));
    if (ret == cudaSuccess)
    {
        ret = cudaMalloc(&d_b, k * n * sizeof(float));
    }
    if (ret == cudaSuccess)
    {
        ret = cudaMalloc(&d_c, m * n * sizeof(float));
    }

    if (ret == cudaSuccess)
    {
        ret = cudaMemcpy(d_a, h_a, m * k * sizeof(float), cudaMemcpyHostToDevice);
    }
    if (ret == cudaSuccess)
    {
        ret = cudaMemcpy(d_b, h_b, k * n * sizeof(float), cudaMemcpyHostToDevice);
    }

    if (ret == cudaSuccess)
    {
        const dim3 block_dim(16, 16);
        const dim3 grid_dim((n + block_dim.x - 1) / block_dim.x,
                            (m + block_dim.y - 1) / block_dim.y);

        matmul_kernel<<<grid_dim, block_dim>>>(d_a, d_b, d_c, m, n, k);
        ret = cudaGetLastError();
    }

    if (ret == cudaSuccess)
    {
        ret = cudaMemcpy(h_c, d_c, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return ret == cudaSuccess ? 0 : 1;
}

int dain_relu(float *h_x, float *h_y, int size)
{
    float *d_x = nullptr, *d_y = nullptr;

    cudaError_t ret = cudaMalloc(&d_x, size * sizeof(float));
    if (ret == cudaSuccess)
    {
        ret = cudaMalloc(&d_y, size * sizeof(float));
    }

    if (ret == cudaSuccess)
    {
        ret = cudaMemcpy(d_x, h_x, size * sizeof(float), cudaMemcpyHostToDevice);
    }

    if (ret == cudaSuccess)
    {
        const int block_size = 256;
        const int num_blocks = (size + block_size - 1) / block_size;
        relu_kernel<<<num_blocks, block_size>>>(d_x, d_y, size);
        ret = cudaGetLastError();
    }

    if (ret == cudaSuccess)
    {
        ret = cudaMemcpy(h_y, d_y, size * sizeof(float), cudaMemcpyDeviceToHost);
    }

    cudaFree(d_x);
    cudaFree(d_y);

    return ret == cudaSuccess ? 0 : 1;
}

int dain_relu_grad(float *x, float *h_grad_in, float *h_grad_out, int size)
{
    float *d_x = nullptr, *d_grad_in = nullptr, *d_grad_out = nullptr;

    cudaError_t ret = cudaMalloc(&d_x, size * sizeof(float));
    if (ret == cudaSuccess)
    {
        ret = cudaMalloc(&d_grad_in, size * sizeof(float));
    }
    if (ret == cudaSuccess)
    {
        ret = cudaMalloc(&d_grad_out, size * sizeof(float));
    }

    if (ret == cudaSuccess)
    {
        ret = cudaMemcpy(d_x, x, size * sizeof(float), cudaMemcpyHostToDevice);
    }
    if (ret == cudaSuccess)
    {
        ret = cudaMemcpy(d_grad_in, h_grad_in, size * sizeof(float), cudaMemcpyHostToDevice);
    }

    if (ret == cudaSuccess)
    {
        const int block_size = 256;
        const int num_blocks = (size + block_size - 1) / block_size;
        relu_grad_kernel<<<num_blocks, block_size>>>(d_x, d_grad_in, d_grad_out, size);
        ret = cudaGetLastError();
    }

    if (ret == cudaSuccess)
    {
        ret = cudaMemcpy(h_grad_out, d_grad_out, size * sizeof(float), cudaMemcpyDeviceToHost);
    }

    cudaFree(d_x);
    cudaFree(d_grad_in);
    cudaFree(d_grad_out);

    return ret == cudaSuccess ? 0 : 1;
}
