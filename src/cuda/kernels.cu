#include "kernels.cuh"

__global__ void add_kernel(const float *a, const float *b, float *c, int a_rows, int a_cols, int b_rows, int b_cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = a_rows * a_cols;

    if (idx < total_elements)
    {
        int row = idx / a_cols;
        int col = idx % a_cols;

        float a_val = a[idx];

        float b_val;
        if (b_rows == 1)
        {
            b_val = b[col];
        }
        else if (b_cols == 1)
        {
            b_val = b[row];
        }
        else
        {
            b_val = b[idx];
        }

        c[idx] = a_val + b_val;
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

__global__ void mse_kernel(float *pred, float *target, float *sum, int size)
{
    __shared__ float shared_sum[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float local_sum = 0.0f;
    while (idx < size)
    {
        float diff = pred[idx] - target[idx];
        local_sum += diff * diff;
        idx += gridDim.x * blockDim.x;
    }
    shared_sum[tid] = local_sum;

    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        atomicAdd(sum, shared_sum[0]);
    }
}

__global__ void mse_grad_kernel(float *pred, float *target, float *grad, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        grad[idx] = 2.0f * (pred[idx] - target[idx]) / size;
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

int dain_add(const float *h_a, const float *h_b, float *h_c, int a_rows, int a_cols, int b_rows, int b_cols)
{
    if ((b_rows != 1 && b_rows != a_rows) ||
        (b_cols != 1 && b_cols != a_cols))
    {
        return 1;
    }

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

    cudaError_t ret = cudaMalloc(&d_a, a_rows * a_cols * sizeof(float));
    if (ret == cudaSuccess)
    {
        ret = cudaMalloc(&d_b, b_rows * b_cols * sizeof(float));
    }
    if (ret == cudaSuccess)
    {
        ret = cudaMalloc(&d_c, a_rows * a_cols * sizeof(float));
    }

    if (ret == cudaSuccess)
    {
        ret = cudaMemcpy(d_a, h_a, a_rows * a_cols * sizeof(float), cudaMemcpyHostToDevice);
    }
    if (ret == cudaSuccess)
    {
        ret = cudaMemcpy(d_b, h_b, b_rows * b_cols * sizeof(float), cudaMemcpyHostToDevice);
    }

    if (ret == cudaSuccess)
    {
        const int block_size = 256;
        const int num_blocks = (a_rows * a_cols + block_size - 1) / block_size;
        add_kernel<<<num_blocks, block_size>>>(d_a, d_b, d_c,
                                               a_rows, a_cols,
                                               b_rows, b_cols);
        ret = cudaGetLastError();
    }

    if (ret == cudaSuccess)
    {
        ret = cudaMemcpy(h_c, d_c, a_rows * a_cols * sizeof(float), cudaMemcpyDeviceToHost);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return ret == cudaSuccess ? 0 : 1;
}

int dain_matmul(const float *h_a, const float *h_b, float *h_c, int m, int n, int k)
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

int dain_mse(const float *h_pred, const float *h_target, float *h_loss, int size)
{
    float *d_pred = nullptr, *d_target = nullptr, *d_sum = nullptr;

    cudaError_t ret = cudaMalloc(&d_pred, size * sizeof(float));
    if (ret == cudaSuccess)
    {
        ret = cudaMalloc(&d_target, size * sizeof(float));
    }
    if (ret == cudaSuccess)
    {
        ret = cudaMalloc(&d_sum, sizeof(float));
    }

    if (ret == cudaSuccess)
    {
        ret = cudaMemcpy(d_pred, h_pred, size * sizeof(float), cudaMemcpyHostToDevice);
    }
    if (ret == cudaSuccess)
    {
        ret = cudaMemcpy(d_target, h_target, size * sizeof(float), cudaMemcpyHostToDevice);
    }
    if (ret == cudaSuccess)
    {
        ret = cudaMemset(d_sum, 0, sizeof(float));
    }

    if (ret == cudaSuccess)
    {
        const int block_size = 256;
        const int num_blocks = min(256, (size + block_size - 1) / block_size);
        mse_kernel<<<num_blocks, block_size>>>(d_pred, d_target, d_sum, size);
        ret = cudaGetLastError();
    }

    if (ret == cudaSuccess)
    {
        float sum;
        ret = cudaMemcpy(&sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
        if (ret == cudaSuccess)
        {
            *h_loss = sum / size;
        }
    }

    cudaFree(d_pred);
    cudaFree(d_target);
    cudaFree(d_sum);

    return ret == cudaSuccess ? 0 : 1;
}

int dain_mse_grad(const float *h_pred, const float *h_target, float *h_grad, int size)
{
    float *d_pred = nullptr, *d_target = nullptr, *d_grad = nullptr;

    cudaError_t ret = cudaMalloc(&d_pred, size * sizeof(float));
    if (ret == cudaSuccess)
    {
        ret = cudaMalloc(&d_target, size * sizeof(float));
    }
    if (ret == cudaSuccess)
    {
        ret = cudaMalloc(&d_grad, size * sizeof(float));
    }

    if (ret == cudaSuccess)
    {
        ret = cudaMemcpy(d_pred, h_pred, size * sizeof(float), cudaMemcpyHostToDevice);
    }
    if (ret == cudaSuccess)
    {
        ret = cudaMemcpy(d_target, h_target, size * sizeof(float), cudaMemcpyHostToDevice);
    }

    if (ret == cudaSuccess)
    {
        const int block_size = 256;
        const int num_blocks = (size + block_size - 1) / block_size;
        mse_grad_kernel<<<num_blocks, block_size>>>(d_pred, d_target, d_grad, size);
        ret = cudaGetLastError();
    }

    if (ret == cudaSuccess)
    {
        ret = cudaMemcpy(h_grad, d_grad, size * sizeof(float), cudaMemcpyDeviceToHost);
    }

    cudaFree(d_pred);
    cudaFree(d_target);
    cudaFree(d_grad);

    return ret == cudaSuccess ? 0 : 1;
}

int dain_relu(const float *h_x, float *h_y, int size)
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

int dain_relu_grad(const float *x, const float *h_grad_in, float *h_grad_out, int size)
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
