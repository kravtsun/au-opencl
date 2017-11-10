__kernel void gpu_convolution_2d_gmem(__global int * input, __global int * mask,
    __global int * output, int mask_width, int width)
{
    int idx = get_global_id(0);
    int res = 0;
    for (int j = 0; j < mask_width; ++j)
    {
        int input_idx = (idx + j - mask_width / 2);
        if (input_idx >= 0 && input_idx < width)
            res += input[input_idx] * mask[j];
    }
    output[idx] = res;
}
