#include <iostream>
#include <iomanip>
#include <vector>

__global__ void saxpy(double a, double* x, double* y, int N)
{
    const int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    if(gidx < N)
    {
        x[gidx] = a * x[gidx] + y[gidx];
    }
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
    double scalar = 5.0;

    const int N = 32;
    std::vector<double> X(N);
    std::vector<double> Y(N);
    std::vector<double> Result(N);
    for(int i=0; i<N; ++i)
    {
        X[i] = i;
        Y[i] = i*i/100.0;
        Result[i] = 0.0;
    }

    double* buffer_x = nullptr;
	double* buffer_y = nullptr;

    err = cudaMalloc( (void**)&buffer_x, N*sizeof(double) );
	if(err != cudaSuccess){ std::cout << "Error allocating CUDA memory (X): " << cudaGetErrorString(err) << "\n"; return -1; }
	
	err = cudaMalloc( (void**)&buffer_y, N*sizeof(double) );
	if(err != cudaSuccess){ std::cout << "Error allocating CUDA memory (Y): " << cudaGetErrorString(err) << "\n"; return -1; }
	
	err = cudaMemcpy( buffer_x, X.data(), N*sizeof(double), cudaMemcpyHostToDevice );
	if(err != cudaSuccess){ std::cout << "Error copying memory to device (X): " << cudaGetErrorString(err) << "\n"; return -1; }
    
    err = cudaMemcpy( buffer_y, Y.data(), N*sizeof(double), cudaMemcpyHostToDevice );
	if(err != cudaSuccess){ std::cout << "Error copying memory to device (Y): " << cudaGetErrorString(err) << "\n"; return -1; }
    
    // Launch kernel:
	dim3 grid_size( 1 );
	dim3 block_size( N );
	saxpy<<<grid_size, block_size>>>(scalar, buffer_x, buffer_y, N);

	err = cudaGetLastError();
	if(err != cudaSuccess){ std::cout << "CUDA error in kernel call: " << cudaGetErrorString(err) << "\n"; return -1; }

    // Copy back results (implicitely synchronizes on the default stream that we are using):
	err = cudaMemcpy( Result.data(), buffer_x, N*sizeof(double), cudaMemcpyDeviceToHost );
	if(err != cudaSuccess){ std::cout << "Error copying memory to host: " << cudaGetErrorString(err) << "\n"; return -1; }

    // Verify results:
    bool check = true;
    std::cout << std::setw(12) << "Result:" << std::setw(12) << "Reference:" << std::setw(12) << "Error:" << "\n";
    for(int i=0; i<N; ++i)
    {
        const double res = Result[i];
        const double ref = scalar * X[i] + Y[i];
        const double error = std::abs(ref-res);
        if(error > 1e-10){ check = false; }
        std::cout << std::fixed << std::setprecision(2) << std::setw(12) << res;
        std::cout << std::fixed << std::setprecision(2) << std::setw(12) << ref;
        std::cout << std::fixed << std::setprecision(2) << std::setw(12) << error << "\n";
    }

    if(check)
    {
        std::cout << "SUCCESS: GPU result matches CPU reference\n";
    }
    else
    {
        std::cout << "FAILURE: GPU result does not match CPU reference\n";
    }

    // Clean-up:
	err = cudaFree( buffer_x );
	if(err != cudaSuccess){ std::cout << "Error freeing allocation (X): " << cudaGetErrorString(err) << "\n"; return -1; }

	err = cudaFree( buffer_y );
	if(err != cudaSuccess){ std::cout << "Error freeing allocation (Y): " << cudaGetErrorString(err) << "\n"; return -1; }

	return 0;
}