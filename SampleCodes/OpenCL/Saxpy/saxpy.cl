#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void saxpy(double a, __global double* x, __global double* y, int N)
{
    const int gidx = get_global_id(0);
    if(gidx < N)
    {
        x[gidx] = a * x[gidx] + y[gidx];
    }
}