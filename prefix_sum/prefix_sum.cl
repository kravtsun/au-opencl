#define SWAP(a,b) {__local int * tmp=a; a=b; b=tmp;}
#define TYPE int
#define BLOCK_SIZE 256

uint get_worker_base(uint group_base, uint perworker_size, uint worker_index) {
    return group_base + worker_index * perworker_size + 0;
}

// TODO a, b block_size!!!
void scan_hillis_steele(__local int * a, __local int * b, uint lid)
{
    uint block_size = get_local_size(0);
    for (uint s = 1; s < block_size; s <<= 1)
    {
        if (lid >= s)
        {
            b[lid] = a[lid] + a[lid - s];
        }
        else
        {
            b[lid] = a[lid];
        }
        SWAP(a, b);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void bottom_up(__global int * input, __global int * output, __local int * a, __local int * b)
{
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    a[lid] = b[lid] = input[gid];
    barrier(CLK_LOCAL_MEM_FENCE);
    scan_hillis_steele(a, b, lid);
    output[gid] = a[lid];
}

// For one work group.
__kernel void reduce_blocks(__global int * output, __global int * block_output, __local int * a, __local int * b, uint n)
{
    uint big_block_id = get_local_id(0);
    uint n_small_blocks = n / BLOCK_SIZE;
    uint perworker_size = n_small_blocks / BLOCK_SIZE;
    if (perworker_size == 0)
    {
        perworker_size = 1;
    }

    int s = 0;
    for (uint i = 0; i < perworker_size; ++i)
    {
        int gid = BLOCK_SIZE * perworker_size * big_block_id + i * BLOCK_SIZE - 1;
        if (gid > 0)
        {
            s += output[gid];
        }
        int block_output_id = perworker_size * big_block_id + i;
        block_output[block_output_id] = s;
    }

    uint lid = big_block_id;
    a[lid] = b[lid] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    scan_hillis_steele(a, b, lid);
    int previous_blocks_sum = 0;
    if (lid > 0)
    {
        previous_blocks_sum = a[lid - 1];
    }
    for (uint i = 0; i < perworker_size; ++i)
    {
        int block_output_id = perworker_size * big_block_id + i;
        block_output[block_output_id] += previous_blocks_sum;
    }
}

__kernel void reduce_all(__global int * output, __global int * block_output)
{
    uint gid = get_global_id(0);
    uint block_id = gid / get_local_size(0);
    output[gid] += block_output[block_id];
}
