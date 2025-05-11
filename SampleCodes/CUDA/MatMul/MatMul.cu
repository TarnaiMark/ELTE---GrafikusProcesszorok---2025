#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>
#include <array>
#include <iostream>
#include "cpu_matmul.h"

using T = double;

constexpr int N = 1024;
constexpr int block_size = 16;
constexpr int n_blocks = N / block_size;
constexpr int BS = block_size;

__global__ void matmul0(T* A, T* B, T* C, int N)
{  
    int gx = blockIdx.x*blockDim.x + threadIdx.x; 
    int gy = blockIdx.y*blockDim.y + threadIdx.y;
    
    T acc = 0;
    for (int i = 0; i < N; i++)
    {
        acc += A[gy * N + i] * B[i * N + gx];
    }
    C[gy * N + gx] = acc;
}

__global__ void matmul1(T* A, T* B, T* C, int N)
{
    int lx = threadIdx.x;
    int ly = threadIdx.y;

    int gx = blockIdx.x*blockDim.x + lx; 
    int gy = blockIdx.y*blockDim.y + ly;

    __shared__ T Ablock[BS*BS];
    __shared__ T Bblock[BS*BS];
    
    T acc = 0;
    for( int s=0; s<N/BS; s++)
    {
        Ablock[ly * BS + lx] = A[gy * N + s * BS + lx];
        Bblock[ly * BS + lx] = B[(s * BS + ly) * N + gx];

        __syncthreads();

        for (int i = 0; i < BS; i++)
        {
            acc += Ablock[ly*BS+i] * Bblock[i*BS+lx];
        }
        
        __syncthreads();
    }
    C[gy * N + gx] = acc;
}

int main()
{
    // Allocate and setup data buffers:
    if( std::is_same_v<T, float> )
    {
        std::cout << "This code multiplies " << N << " x " << N << " float matrices with block size: " << block_size << "\n";
    }
    else if( std::is_same_v<T, double> )
    {
        std::cout << "This code multiplies " << N << " x " << N << " double matrices with block size: " << block_size << "\n";
    }
    else{ std::cout << "This code only supports float or double elements\n"; return -1; }

    if(N % block_size != 0){ std::cout << "This code requires the block size to evenly divide the number of elements\n"; return -1; }

    std::vector<T> A(N*N), B(N*N); // input matrices
    std::vector<T> C0(N*N); // CPU naive;
    std::vector<T> C1(N*N); // CPU opt;
    std::vector<T> C2(N*N); // GPU naive;
    std::vector<T> C3(N*N); // GPU opt;
    
    std::mt19937 mersenne_engine{42};  // Generates random integers
    std::uniform_real_distribution<T> dist{-0.1, 0.1};

    auto gen = [&dist, &mersenne_engine](){ return dist(mersenne_engine); };
    generate(A.begin(), A.end(), gen);
    generate(B.begin(), B.end(), gen);
    std::fill(C0.begin(), C0.end(), (T)(0.0));
    std::fill(C1.begin(), C1.end(), (T)(0.0));
    std::fill(C2.begin(), C2.end(), (T)(0.0));
    std::fill(C3.begin(), C3.end(), (T)(0.0));

    // Run CPU computations:
    auto t0 = std::chrono::high_resolution_clock::now();
    cpu_matmul_naive<T>(C0, A, B, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto t_cpu_naive = std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count()/1e6;

    auto t2 = std::chrono::high_resolution_clock::now();
    cpu_matmul_improved<T, block_size>(C1, A, B, N);
    auto t3 = std::chrono::high_resolution_clock::now();
    auto t_cpu_improved = std::chrono::duration_cast<std::chrono::nanoseconds>(t3-t2).count()/1e6;

    // CUDA:
    cudaError_t err = cudaSuccess;
    
    // Using the implicitely selected first cuda device:
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if(err != cudaSuccess){ std::cout << "CUDA error getting device properties for device 0: " << cudaGetErrorString(err) << "\n"; return -1; }
    std::cout << "Selected device name: " << prop.name << "\n";
	
	T* pA = nullptr;
    T* pB = nullptr;
    T* pC2 = nullptr;
    T* pC3 = nullptr;

    // Allocate and upload buffers:
	err = cudaMalloc( (void**)&pA, N*N*sizeof(T) );
	if( err != cudaSuccess){ std::cout << "CUDA error allocating memory: " << cudaGetErrorString(err) << "\n"; return -1; }
	
	err = cudaMalloc( (void**)&pB, N*N*sizeof(T) );
	if( err != cudaSuccess){ std::cout << "CUDA error allocating memory: " << cudaGetErrorString(err) << "\n"; return -1; }
    
    err = cudaMalloc( (void**)&pC2, N*N*sizeof(T) );
	if( err != cudaSuccess){ std::cout << "CUDA error allocating memory: " << cudaGetErrorString(err) << "\n"; return -1; }

    err = cudaMalloc( (void**)&pC3, N*N*sizeof(T) );
	if( err != cudaSuccess){ std::cout << "CUDA error allocating memory: " << cudaGetErrorString(err) << "\n"; return -1; }
	
	err = cudaMemcpy( pA, A.data(), N*N*sizeof(T), cudaMemcpyHostToDevice );
    if( err != cudaSuccess){ std::cout << "CUDA error copying memory to device: " << cudaGetErrorString(err) << "\n"; return -1; }
    
    err = cudaMemcpy( pB, B.data(), N*N*sizeof(T), cudaMemcpyHostToDevice );
	if( err != cudaSuccess){ std::cout << "CUDA error copying memory to device: " << cudaGetErrorString(err) << "\n"; return -1; }
    
    // Timer events:
    std::array<cudaEvent_t, 4> evts;
    for(auto& e : evts)
    {
        err = cudaEventCreate(&e);
        if( err != cudaSuccess){ std::cout << "CUDA error creating event: " << cudaGetErrorString(err) << "\n"; return -1; }
    }

    dim3 dim_grid( n_blocks, n_blocks );
    dim3 dim_block( block_size, block_size );

    // Warmup kernel launches:
    matmul0<<<dim_grid, dim_block>>>(pA, pB, pC2, N);
    err = cudaGetLastError();
    if (err != cudaSuccess){ std::cout << "CUDA error in kernel call 0: " << cudaGetErrorString(err) << "\n"; return -1; }

    matmul1<<<dim_grid, dim_block>>>(pA, pB, pC3, N);
    err = cudaGetLastError();
    if (err != cudaSuccess){ std::cout << "CUDA error in kernel call 1: " << cudaGetErrorString(err) << "\n"; return -1; }

    // Measured kernel launches:
    err = cudaEventRecord(evts[0]);
    if (err != cudaSuccess){ std::cout << "CUDA error in cudaEventRecord 0: " << cudaGetErrorString(err) << "\n"; return -1; }

    matmul0<<<dim_grid, dim_block>>>(pA, pB, pC2, N);
    err = cudaGetLastError();
    if (err != cudaSuccess){ std::cout << "CUDA error in kernel call 2: " << cudaGetErrorString(err) << "\n"; return -1; }

    err = cudaEventRecord(evts[1]);
    if (err != cudaSuccess){ std::cout << "CUDA error in cudaEventRecord 1: " << cudaGetErrorString(err) << "\n"; return -1; }

    err = cudaEventRecord(evts[2]);
    if (err != cudaSuccess){ std::cout << "CUDA error in cudaEventRecord 2: " << cudaGetErrorString(err) << "\n"; return -1; }

    matmul1<<<dim_grid, dim_block>>>(pA, pB, pC3, N);
    err = cudaGetLastError();
    if (err != cudaSuccess){ std::cout << "CUDA error in kernel call 3: " << cudaGetErrorString(err) << "\n"; return -1; }
    
    err = cudaEventRecord(evts[3]);
    if (err != cudaSuccess){ std::cout << "CUDA error in cudaEventRecord 3: " << cudaGetErrorString(err) << "\n"; return -1; }

    // Wait for the last event to finish all previous commands:
    err = cudaEventSynchronize(evts[3]);
    if (err != cudaSuccess){ std::cout << "CUDA error in cudaEventSynchronize: " << cudaGetErrorString(err) << "\n"; return -1; }

    // Copy data back to host:
	err = cudaMemcpy( C2.data(), pC2, N*N*sizeof(T), cudaMemcpyDeviceToHost );
	if( err != cudaSuccess){ std::cout << "CUDA error copying memory C2 to host: " << cudaGetErrorString(err) << "\n"; return -1; }

    err = cudaMemcpy( C3.data(), pC3, N*N*sizeof(T), cudaMemcpyDeviceToHost );
	if( err != cudaSuccess){ std::cout << "CUDA error copying memory C3 to host: " << cudaGetErrorString(err) << "\n"; return -1; }

    // Query elapsed times:
    float t_gpu_naive = 0.0f; // milliseconds!
    float t_gpu_improved = 0.0f; // milliseconds!
    err = cudaEventElapsedTime(&t_gpu_naive, evts[0], evts[1]);
    if (err != cudaSuccess){ std::cout << "CUDA error in cudaEventElapsedTime 0: " << cudaGetErrorString(err) << "\n"; return -1; }

    err = cudaEventElapsedTime(&t_gpu_improved, evts[2], evts[3]);
    if (err != cudaSuccess){ std::cout << "CUDA error in cudaEventElapsedTime 1: " << cudaGetErrorString(err) << "\n"; return -1; }

    // Free events and allocations:
    for(auto& e : evts)
    {
        err = cudaEventDestroy(e);
        if( err != cudaSuccess){ std::cout << "CUDA error destroying even: " << cudaGetErrorString(err) << "\n"; return -1; }
    }

	err = cudaFree( pA );
	if( err != cudaSuccess){ std::cout << "CUDA error freeing allocation: " << cudaGetErrorString(err) << "\n"; return -1; }

	err = cudaFree( pB );
	if( err != cudaSuccess){ std::cout << "CUDA error freeing allocation: " << cudaGetErrorString(err) << "\n"; return -1; }

    err = cudaFree( pC2 );
	if( err != cudaSuccess){ std::cout << "CUDA error freeing allocation: " << cudaGetErrorString(err) << "\n"; return -1; }

    err = cudaFree( pC3 );
	if( err != cudaSuccess){ std::cout << "CUDA error freeing allocation: " << cudaGetErrorString(err) << "\n"; return -1; }

    // Verify results:
    const T max_err = 1e-5f;
    auto comparator = [max_err](T l, T r){ return std::abs(l-r) < max_err; };
    auto compare = [comparator](const char* name, std::vector<T> const& V, std::vector<T> const& U)
    {
        bool mismatch = false;
        for(int i=0; i<N*N; ++i)
        {
            if( !comparator(V[i], U[i]) )
            {
                std::cout << name << " [" << i << "] : " << V[i] << "   " << U[i] << " absolute error: " << std::abs(V[i]-U[i]) << "\n";
                mismatch = true;
                break;
            }
        }
        return !mismatch;
    };

    bool success_1 = compare("cpu naive - cpu improved", C0, C1);
    if(!success_1){ std::cout << "Mismatch between CPU naive and improved results.\n"; }
    bool success_2 = compare("cpu naive - gpu naive   ", C0, C2);
    if(!success_2){ std::cout << "Mismatch between CPU naive and GPU naive results.\n"; }
    bool success_3 = compare("cpu naive - gpu improved", C0, C3);
    if(!success_3){ std::cout << "Mismatch between CPU naive and GPU improved results.\n"; }

    if(success_1 && success_2 && success_3){ std::cout << "All results match\n"; }

    std::cout << "Naive    CPU time: " << t_cpu_naive << " ms\n";
    std::cout << "Improved CPU time: " << t_cpu_improved << " ms\n";
    std::cout << "Naive    GPU time: " << t_gpu_naive << " ms\n";
    std::cout << "Improved GPU time: " << t_gpu_improved << " ms\n";
	return 0;
}
