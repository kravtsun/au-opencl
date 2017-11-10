int index(int i, int j, int n)
{
    return i * n + j;
}

__kernel void gpu_convolution_2d_gmem(__global double* a, __global double* b, __global double* output, int m, int n)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i >= n || j >= n)
        return;

    int hm = (m - 1) / 2;
    double res = 0;

    for (int k = max(-i, -hm); k <= hm && i + k < n; ++k)
    {
        for (int l = max(-j, -hm); l <= hm && j + l < n; ++l)
        {
            res += a[index(i + k, j + l, n)] * b[index(k + hm, l + hm, m)];
        }
    }

    int idx = index(i, j, n);
    output[idx] = res;
}
