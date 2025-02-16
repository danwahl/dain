#pragma once

extern "C"
{
    int dain_add(float *a, float *b, float *c, int size);
    int dain_matmul(float *a, float *b, float *c, int m, int n, int k);
}
