#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>

__global__ void reduce(const double* A, double* B, int N)
{
    static const int b = 1024;
    const int t = threadIdx.x;
    const int i = blockIdx.x * blockDim.x * 2 + t;
    double x = i   < N ? A[i  ] : double{};
    double y = i+b < N ? A[i+b] : double{};
    __shared__ double tmp[b];
    tmp[t] = x + y;
    __syncthreads();
    if(t < 512){ tmp[t] = tmp[t] + tmp[t + 512]; }
    __syncthreads();
    if(t < 256){ tmp[t] = tmp[t] + tmp[t + 256]; }
    __syncthreads();
    if(t < 128){ tmp[t] = tmp[t] + tmp[t + 128]; }
    __syncthreads();
    if(t < 64){ tmp[t] = tmp[t] + tmp[t + 64]; }
    __syncthreads();
    if(t < 32){ tmp[t] = tmp[t] + tmp[t + 32]; }
    __syncthreads();
    if(t < 16){ tmp[t] = tmp[t] + tmp[t + 16]; }
    __syncthreads();
    if(t < 8){ tmp[t] = tmp[t] + tmp[t + 8]; }
    __syncthreads();
    if(t < 4){ tmp[t] = tmp[t] + tmp[t + 4]; }
    __syncthreads();
    if(t < 2){ tmp[t] = tmp[t] + tmp[t + 2]; }
    __syncthreads();
    if(t == 0){ B[blockIdx.x] = tmp[0] + tmp[1]; }
}

int main()
{
    cudaError_t err = cudaSuccess;
    
    // Using the implicitely selected first cuda device:
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if(err != cudaSuccess){ std::cout << "Error getting device properties for device 0: " << cudaGetErrorString(err) << "\n"; return -1; }
    std::cout << "Selected device name: " << prop.name << "\n";
    
    // Actual program logic:
    // Allocate and setup data buffers:
    const int N = 1024*1024*2;
    double result = 0.0;
    std::vector<double> X(N, 0.0);
    for(int i=0; i<N; ++i)
    {
        X[i] = i*1.0/N;
    }

    double* buffer_x = nullptr;
	double* buffer_y = nullptr;
    double* buffer_z = nullptr;

    err = cudaMalloc( (void**)&buffer_x, N*sizeof(double) );
	if(err != cudaSuccess){ std::cout << "Error allocating CUDA memory (X): " << cudaGetErrorString(err) << "\n"; return -1; }
	
	err = cudaMalloc( (void**)&buffer_y, N*sizeof(double) );
	if(err != cudaSuccess){ std::cout << "Error allocating CUDA memory (Y): " << cudaGetErrorString(err) << "\n"; return -1; }

    err = cudaMalloc( (void**)&buffer_z, 1*sizeof(double) );
	if(err != cudaSuccess){ std::cout << "Error allocating CUDA memory (Z): " << cudaGetErrorString(err) << "\n"; return -1; }

    err = cudaMemset( buffer_y, 0, N*sizeof(double) );
    if(err != cudaSuccess){ std::cout << "Error zeroing CUDA memory (Y): " << cudaGetErrorString(err) << "\n"; return -1; }
	
	err = cudaMemcpy( buffer_x, X.data(), N*sizeof(double), cudaMemcpyHostToDevice );
	if(err != cudaSuccess){ std::cout << "Error copying memory to device (X): " << cudaGetErrorString(err) << "\n"; return -1; }
    
    const unsigned int num_blocks = static_cast<unsigned int>(std::ceil(N*1.0f/1024/2));
	
    // Launch first kernel:
    {
        dim3 grid_size( num_blocks );
	    dim3 block_size( 1024 );
	    reduce<<<grid_size, block_size>>>(buffer_x, buffer_y, N);

	    err = cudaGetLastError();
	    if(err != cudaSuccess){ std::cout << "CUDA error in kernel call (1): " << cudaGetErrorString(err) << "\n"; return -1; }
    }

    // second kernel:
	{
        dim3 grid_size( 1 );
	    dim3 block_size( 1024 );
	    reduce<<<grid_size, block_size>>>(buffer_y, buffer_z, num_blocks);

	    err = cudaGetLastError();
	    if(err != cudaSuccess){ std::cout << "CUDA error in kernel call (2): " << cudaGetErrorString(err) << "\n"; return -1; }
    }

    // Copy back results (implicitely synchronizes on the default stream that we are using):
	err = cudaMemcpy( &result, buffer_z, 1*sizeof(double), cudaMemcpyDeviceToHost );
	if(err != cudaSuccess){ std::cout << "Error copying memory to host: " << cudaGetErrorString(err) << "\n"; return -1; }

    // Host reference computation:
    const double ref = std::accumulate(X.begin(), X.end(), 0.0, std::plus<double>());
    
    // Verify results:
    std::cout << std::setprecision(16) << "Reference: " << ref << "\n";
    std::cout << std::setprecision(16) << "Result   : " << result << "\n";
    const double error = std::abs(ref-result)/std::abs(ref);
    if(error > 1e-9)
    {
        std::cout << "FAILURE: GPU result does not match CPU reference\n";
    }
    else
    {
        std::cout << "SUCCESS: GPU result matches CPU reference\n";
    }
    
    // Clean-up:
	err = cudaFree( buffer_x );
	if(err != cudaSuccess){ std::cout << "Error freeing allocation (X): " << cudaGetErrorString(err) << "\n"; return -1; }

	err = cudaFree( buffer_y );
	if(err != cudaSuccess){ std::cout << "Error freeing allocation (Y): " << cudaGetErrorString(err) << "\n"; return -1; }

    err = cudaFree( buffer_z );
	if(err != cudaSuccess){ std::cout << "Error freeing allocation (Z): " << cudaGetErrorString(err) << "\n"; return -1; }

	return 0;
}