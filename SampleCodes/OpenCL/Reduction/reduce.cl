#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void reduce(const __global double* A, __global double* B, int N)
{
    const int b = 256;
    const int t = get_local_id(0);
    const int i = get_group_id(0) * get_local_size(0) * 2 + t;
    double x = i   < N ? A[i  ] : 0.0;
    double y = i+b < N ? A[i+b] : 0.0;
    __local double tmp[256];
    tmp[t] = x + y;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(t < 128){ tmp[t] = tmp[t] + tmp[t + 128]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(t < 64){ tmp[t] = tmp[t] + tmp[t + 64]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(t < 32){ tmp[t] = tmp[t] + tmp[t + 32]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(t < 16){ tmp[t] = tmp[t] + tmp[t + 16]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(t < 8){ tmp[t] = tmp[t] + tmp[t + 8]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(t < 4){ tmp[t] = tmp[t] + tmp[t + 4]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(t < 2){ tmp[t] = tmp[t] + tmp[t + 2]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(t == 0){ B[get_group_id(0)] = tmp[0] + tmp[1]; }
}