#define SWAP(a,b) {__local int * tmp=a; a=b; b=tmp;}
#define TYPE int

uint get_worker_base(uint group_base, uint perworker_size, uint worker_index) {
    return group_base + worker_index * perworker_size + 0;
}

// TODO a, b block_size!!!
void scan_hillis_steele(__local int * a, __local int * b, uint lid, uint block_size)
{
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
    uint block_size = get_local_size(0);

    a[lid] = b[lid] = input[gid];
    barrier(CLK_LOCAL_MEM_FENCE);
    scan_hillis_steele(a, b, lid, block_size);
    output[gid] = a[lid];
}

__kernel void reduce_blocks(__global int * output, __global int * block_output, __local int * a, __local int * b)
{
    uint block_id = get_global_id(0);
    uint block_size = get_local_size(0);
    uint base_block_size = block_size;

    uint blocks_count = get_global_size(0);

    uint lid = get_local_id(0);
    if (block_id > 0)
    {
        uint gid = block_id * base_block_size - 1;
        a[lid] = b[lid] = output[gid];
    }
    else
    {
        a[lid] = b[lid] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    scan_hillis_steele(a, b, lid, block_size);
    block_output[block_id] = a[lid];
}

__kernel void reduce_all(__global int * output, __global int * block_output)
{
    uint gid = get_global_id(0);
    uint block_id = gid / get_local_size(0);
    output[gid] += block_output[block_id];
}

//void top_down(__global int * input, __global int * output)
//{
//    uint gid = get_global_id(0);
//    uint lid = get_local_id(0);
//    //uint block_size = get_local_size(0);
//    output[gid] = input[gid];
//}
//
//__kernel void block_scan(__global TYPE * input,
//                         __global TYPE * a, // inside-block scan.
//                         __global TYPE * b // for utility purposes.
//)
//{
//    uint n = get_global_size(0); // 8092
//    uint gid = get_global_id(0);
//    uint group_size = get_local_size(0); // 256
//    uint worker_index = get_local_id(0);
//    b[gid] = 1;
//    return;
//
//    // 32
//    uint block_size = n / group_size; // 1..2^12
//    uint perworker_size;
//    if (block_size >= group_size)
//    {
//        perworker_size = block_size / group_size; // 2^4
//    }
//    else
//    {
//        perworker_size = 1;
//    }
//
//    uint group_index = gid / block_size; // 0..(2^20/256)
//    uint group_base = group_index * block_size;
//
//    uint worker_base = get_worker_base(group_base, perworker_size, worker_index);
//    for (uint i = 0; i < perworker_size; ++i)
//    {
//        uint current_index = worker_base + i;
//        a[current_index] = input[current_index];
//        if (i > 0)
//        {
//            a[current_index] += a[current_index - 1];
//        }
//    }
//    barrier(CLK_LOCAL_MEM_FENCE);
//
//    //// let one worker of group do this part?
//    //TYPE other_workers_sum = 0;
//    //for (uint j = 0; j < worker_index; ++j)
//    //{
//    //    uint jbase = get_worker_base(group_base, perworker_size, j);
//    //    other_workers_sum += b[jbase + perworker_size - 1];
//    //}
//    //barrier(CLK_LOCAL_MEM_FENCE);
//
//    //for (uint i = 0; i < perworker_size; ++i)
//    //{
//    //    current_index = worker_base + i;
//    //    a[current_index] = b[current_index] + other_workers_sum;
//    //}
//    //barrier(CLK_LOCAL_MEM_FENCE);
//}
//
//__kernel void block_reduce(__global int * a,
//                           __global int * output)
//{
//    //uint n = get_global_size(0);
//    //uint gid = get_global_id(0);
//    //uint block_size = get_local_size(0); // all groups' sizes should be constant!
//
//    //uint block_index = gid / block_size;
//
//    //__local TYPE other_blocks_sum;
//    //other_blocks_sum = 0; // why not allowed?? https://www.khronos.org/registry/OpenCL/sdk/1.1/docs/man/xhtml/local.html
//
//    //// only one thread working.
//    //if (gid % block_size == 0)
//    //{
//    //    // count sum to be added from other blocks.
//    //    for (uint i = 0; i < block_index; ++i)
//    //    {
//    //        other_blocks_sum += a[i * block_size + block_size - 1];
//    //    }
//    //}
//    //barrier(CLK_LOCAL_MEM_FENCE);
//    //output[gid] = a[gid] + other_blocks_sum;
//}
