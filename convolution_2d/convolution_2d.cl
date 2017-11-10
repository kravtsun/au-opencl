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

    for (int k = -hm; k <= hm; ++k)
    {
        if (i + k < 0 || i + k >= n) continue;
        for (int l = -hm; l <= hm; ++l)
        {
            if (j + l < 0 || j + l >= n) continue;
            res += a[index(i + k, j + l, n)] * b[index(k + hm, l + hm, m)];
        }
    }

    //for (int k = -i; k <= hm && i + k < n; ++k)
    //{
    //    for (int l = -j; l <= hm && j + l < n; ++l)
    //    {
    //        int aindex = index(i + k, j + l, n);
    //        int bindex = index(k + hm, l + hm, m);
    //        res += a[aindex] * b[bindex];
    //    }
    //}
    int idx = index(i, j, n);
    output[idx] = res;
}
