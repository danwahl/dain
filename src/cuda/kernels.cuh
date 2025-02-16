#pragma once

extern "C"
{
    int dain_add(const float *h_a, const float *h_b, float *h_c, int a_rows, int a_cols, int b_rows, int b_cols);
    int dain_matmul(const float *h_a, const float *h_b, float *h_c, int m, int n, int k);
    int dain_mse(const float *h_pred, const float *h_target, float *h_loss, int size);
    int dain_mse_grad(const float *h_pred, const float *h_target, float *h_grad, int size);
    int dain_relu(const float *h_x, float *h_y, int size);
    int dain_relu_grad(const float *h_x, const float *h_grad_in, float *h_grad_out, int size);
}
