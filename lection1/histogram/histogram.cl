int calc_bin(int x)
{
   return x % 4;
}

__kernel void gpu_histogram_naive(__global int * input, __global int * output)
{
   size_t idx = get_global_id(0);
   int bin = calc_bin(input[idx]);
   output[bin]++;
}

__kernel void gpu_histogram_atomic(__global int * input, __global int * output)
{
   size_t idx = get_global_id(0);
   int bin = calc_bin(input[idx]);
   atomic_inc(&output[bin]);
}