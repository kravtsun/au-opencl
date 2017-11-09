__kernel void gpu_convolution_gmem(__global int * input, __global int * mask, 
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

__kernel void gpu_convolution_lmem(__global int * input, __global int * mask, 
                                   __global int * output, int mask_width, int width,
                                   __local int * sdata)
{
   int n = mask_width / 2;
   size_t block_dim  = get_local_size(0);
   size_t block_idx  = get_group_id  (0);
   size_t thread_idx = get_local_id  (0);

   int halo_left = (block_idx - 1) * block_dim + thread_idx;
   if (thread_idx >= block_dim - n)
      sdata[thread_idx - (block_dim - n)] = (halo_left < 0) ? 0 : input[halo_left];

   sdata[n + thread_idx] = input[block_idx * block_dim + thread_idx];

   int halo_right = (block_idx + 1) * block_dim + thread_idx;
   if (thread_idx < n)
      sdata[thread_idx + block_dim + n] = (halo_right < width) ? input[halo_right] : 0;

   barrier(CLK_LOCAL_MEM_FENCE);

   int idx = get_global_id(0);
   int res = 0;
   for (int j = 0; j < mask_width; ++j)
      res += sdata[thread_idx + j] * mask[j];

   output[idx] = res;
}