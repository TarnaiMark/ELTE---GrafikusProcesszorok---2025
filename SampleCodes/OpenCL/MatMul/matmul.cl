#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// Typename T will be defined on the host side at kernel compilation time!

__kernel void matmul0(__global T* A, 
                      __global T* B, 
                      __global T* C, 
                             int  N)
{  
   int gx = get_global_id(0); 
   int gy = get_global_id(1);

   T acc = 0;
   for (int i = 0; i < N; i++)
   {
      acc += A[gy * N + i] * B[i * N + gx];
   }
 
   C[gy * N + gx] = acc;
}

__kernel void matmul1(__global T* A,
                      __global T* B,
                      __global T* C,
					         int  N)
{
	int lx = get_local_id(0);
	int ly = get_local_id(1);

	int gx = get_global_id(0);
	int gy = get_global_id(1);

	__local T Ablock[BS*BS];
	__local T Bblock[BS*BS];

	T acc = 0;
	for( int s=0; s<N/BS; s++)
	{
		Ablock[ly * BS + lx] = A[gy * N + s * BS + lx];
		Bblock[ly * BS + lx] = B[(s * BS + ly) * N + gx];

		barrier(CLK_LOCAL_MEM_FENCE);

		for (int i = 0; i < BS; i++)
		{
			acc += Ablock[ly*BS+i] * Bblock[i*BS+lx];
		}
        
        barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	C[gy * N + gx] = acc;
}