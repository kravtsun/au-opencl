__kernel void gpu_reduce_gmem(__global int * g_in, __global int * g_out)
{
   size_t idx        = get_global_id (0);
   size_t thread_idx = get_local_id  (0);
   size_t block_size = get_local_size(0);

   for(size_t s = 1; s < block_size; s *= 2)
   {
      if (thread_idx % (2 * s) == 0)
         g_in[idx] += g_in[idx + s];
      barrier(CLK_GLOBAL_MEM_FENCE);
   }
   if(thread_idx == 0) g_out[get_group_id(0)] = g_in[idx];
}

__kernel void gpu_reduce_lmem(__global int * g_in, __global int * g_out, __local int * sdata)
{
   size_t idx        = get_global_id (0);
   size_t thread_idx = get_local_id  (0);
   size_t block_size = get_local_size(0);

   // аждый поток загружает один элемент из глобальной пам€ти в раздел€емую
   sdata[thread_idx] = g_in[idx];
   barrier(CLK_LOCAL_MEM_FENCE);

   for(size_t s = 1; s < block_size; s *= 2)
   {
      if (thread_idx % (2 * s) == 0)
         sdata[thread_idx] += sdata[thread_idx + s];
      barrier(CLK_LOCAL_MEM_FENCE);
   }
   if(thread_idx == 0) g_out[get_group_id(0)] = sdata[0];
}