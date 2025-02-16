#pragma once

extern "C"
{
    int dain_add(float *h_a, float *h_b, float *h_c, int size);
    int dain_matmul(float *h_a, float *h_b, float *h_c, int m, int n, int k);
    int dain_relu(float *h_x, float *h_y, int size);
    int dain_relu_grad(float *h_x, float *h_grad_in, float *h_grad_out, int size);
}
