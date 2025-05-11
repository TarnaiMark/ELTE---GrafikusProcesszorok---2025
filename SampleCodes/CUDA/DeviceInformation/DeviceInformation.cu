#include <iostream>

int main()
{
    cudaError_t err = cudaSuccess;

	int device_count = 0;

    err = cudaGetDeviceCount(&device_count);
    if( err != cudaSuccess){ std::cout << "Error getting device count: " << cudaGetErrorString(err) << "\n"; return -1; }
    
    std::cout << "There are " << device_count << " device(s)\n";
    
    for(int d = 0; d < device_count; ++d)
    {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, d);
        if( err != cudaSuccess){ std::cout << "Error getting device properties for device " << d << ": " << cudaGetErrorString(err) << "\n"; continue; }
        std::cout << "Device " << d << " name:               " << prop.name << "\n";
        std::cout << "Device " << d << " global memory size: " << prop.totalGlobalMem/1024/1024 << " MiB\n";
    }

	return 0;
}