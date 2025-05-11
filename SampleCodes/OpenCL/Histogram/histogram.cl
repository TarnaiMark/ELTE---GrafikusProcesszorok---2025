__kernel void gpu_histo_global_atomics( __global unsigned int* output, __global uchar4* input, int w, int h )
{
    // Linear block index within 2D grid
    const int B = get_group_id(0) + get_group_id(1) * get_num_groups(0);

    // Output index start for this block's histogram:
    const int I = B*(3*256);
    __global unsigned int* H = output + I;
    
    // Process pixel blocks horizontally
    // Updates our block's partial histogram in global memory
    const int y = get_group_id(1) * get_local_size(1) + get_local_id(1);
    if(y >= h){ return; }
    for (int x = get_local_id(0); x < w; x += get_local_size(0))
    {
        uchar4 pixels = input[y * w + x];
        atomic_add(&H[0 * 256 + pixels.x], 1);
        atomic_add(&H[1 * 256 + pixels.y], 1);
        atomic_add(&H[2 * 256 + pixels.z], 1);
    }
}

__kernel void gpu_histo_shared_atomics( __global unsigned int* output, __global uchar4* input, int w, int h )
{
    // Histograms are in shared (local) memory:
    __local unsigned int histo[3 * 256];

    // Number of threads in the block:
    const int Nthreads = get_local_size(0) * get_local_size(1);
    // Linear thread idx:
    const int LinID = get_local_id(0) + get_local_id(1) * get_local_size(0);
    // Zero histogram:
    for (int i = LinID; i < 3*256; i += Nthreads){ histo[i] = 0; }
    
    barrier(CLK_LOCAL_MEM_FENCE);

    // Linear block index within 2D grid
    const int B = get_group_id(0) + get_group_id(1) * get_num_groups(0);

    // Process pixel blocks horizontally
    // Updates the partial histogram in shared memory
    const int y = get_group_id(1) * get_local_size(1) + get_local_id(1);

    if(y < h)
    {
        for (int x = get_local_id(0); x < w; x += get_local_size(0))
        {
            uchar4 pixels = input[y * w + x];
            atomic_add(&histo[0 * 256 + pixels.x], 1);
            atomic_add(&histo[1 * 256 + pixels.y], 1);
            atomic_add(&histo[2 * 256 + pixels.z], 1);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Output index start for this block's histogram:
    const int I = B*(3*256);
    __global unsigned int* H = output + I;

    // Copy shared memory histograms to globl memory:
    for (int i = LinID; i < 256; i += Nthreads)
    {
        H[0*256 + i] = histo[0*256 + i];
        H[1*256 + i] = histo[1*256 + i];
        H[2*256 + i] = histo[2*256 + i];
    }
}

__kernel void gpu_histo_accumulate( __global unsigned int* out, __global const unsigned int* in, int nBlocks)
{
    // Each thread sums one shade of the r, g, b histograms
    const int i = get_global_id(0);
    if(i < 3 * 256)
    {
        unsigned int sum = 0;
        for(int j = 0; j < nBlocks; j++)
        {
            sum += in[i + (3*256) * j];
        }            
        out[i] = sum;
    }
}