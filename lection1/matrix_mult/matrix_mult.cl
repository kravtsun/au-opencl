__kernel void matrix_mult(__global int * a, __global int * b, __global int * c, int n)
{
   int row = get_global_id(0);
   int col = get_global_id(1);

   if (row >= n || col >= n)
      return;

   int sum = 0;

   for (int k = 0; k < n; ++k)
      sum += a[row * n + k] * b[k * n + col];

   c[row * n + col] = sum;
}

__kernel void matrix_mult_shared(__global int * a, __global int * b, __global int * c, int n)
{
   __local float tileAs[BLOCK_SIZE][BLOCK_SIZE];
   __local float tileBs[BLOCK_SIZE][BLOCK_SIZE];

   int tx = get_local_id(0); int ty = get_local_id(1);
   int row = get_global_id(0);
   int col = get_global_id(1);

   int tiles_to_mul = (n - 1) / BLOCK_SIZE + 1;

   int row_width = row * n;

   float dot_product = 0; // результирующее значение для (row, col) 

   bool allow_write = (row < n) && (col < n);

   for (int t = 0; t < tiles_to_mul; ++t)
   {
      if (row < n && t * BLOCK_SIZE + tx < n)
        tileAs[tx][ty] = a[row_width + t * BLOCK_SIZE + ty];

      if (col < n && t * BLOCK_SIZE + ty < n)
        tileBs[tx][ty] = b[(t * BLOCK_SIZE + tx) * n + col]; 

      barrier(CLK_LOCAL_MEM_FENCE);

      int dot_len = BLOCK_SIZE;

      if (allow_write)
         for (int k = 0; k < dot_len; ++k)
            dot_product += tileAs[tx][k] * tileBs[k][ty];

      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
   }

   if (allow_write)
      c[row * n + col] = dot_product;   
}