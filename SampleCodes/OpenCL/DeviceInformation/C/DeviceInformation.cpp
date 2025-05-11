#ifdef __APPLE__ //Mac OSX has a different name for the header file
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#include <iostream>
#include <vector>
#include <string>

int main()
{
    cl_int err = CL_SUCCESS;

    cl_uint platform_count = 0;

    err = clGetPlatformIDs(0, nullptr, &platform_count);
    if( err != CL_SUCCESS){ std::cout << "Error getting platform count (clGetPlatformIDs)\n"; return -1; }

    if(platform_count == 0){ std::cout << "No OpenCL platform detected.\n"; return -1; }
    
    std::cout << "There are " << platform_count << " OpenCL platform(s)\n\n";
    
    std::vector<cl_platform_id> platform_ids(platform_count);
    err = clGetPlatformIDs(platform_count, platform_ids.data(), nullptr);
    if( err != CL_SUCCESS){ std::cout << "Error getting platform ids (clGetPlatformIDs)\n"; return -1; }

    for(cl_uint i = 0; i < platform_count; ++i)
    {
        {
            size_t size{};
            err = clGetPlatformInfo(platform_ids[i], CL_PLATFORM_VENDOR, UINTMAX_MAX, nullptr, &size);
            if( err != CL_SUCCESS){ std::cout << "Error getting platform vendor name length (clGetPlatformInfo)\n"; return -1; }

            std::string platform_vendor_name(size, '\0');
            err = clGetPlatformInfo(platform_ids[i], CL_PLATFORM_VENDOR, size, const_cast<char*>(platform_vendor_name.data()), nullptr);
            if( err != CL_SUCCESS){ std::cout << "Error getting platform vendor name (clGetPlatformInfo)\n"; return -1; }

            std::cout << "Platform " << i << " vendor: " << platform_vendor_name << "\n";
        }

        {
            size_t size{};
            err = clGetPlatformInfo(platform_ids[i], CL_PLATFORM_NAME, UINTMAX_MAX, nullptr, &size);
            if( err != CL_SUCCESS){ std::cout << "Error getting platform name length (clGetPlatformInfo)\n"; return -1; }

            std::string platform_name(size, '\0');
            err = clGetPlatformInfo(platform_ids[i], CL_PLATFORM_NAME, size, const_cast<char*>(platform_name.data()), nullptr);
            if( err != CL_SUCCESS){ std::cout << "Error getting platform name (clGetPlatformInfo)\n"; return -1; }

            std::cout << "Platform " << i << " name:   " << platform_name << "\n";
        }

        {
            cl_uint device_count = 0;
            err = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &device_count);
            if( err != CL_SUCCESS){ std::cout << "Error getting device count for platform " << i << " (clGetDeviceIDs)\n"; return -1; }

            std::cout << "Platform " << i << " has:    " << device_count << " device(s):\n";

            std::vector<cl_device_id> devices(device_count);
            err = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ALL, device_count, devices.data(), nullptr);
            if( err != CL_SUCCESS){ std::cout << "Error getting device ids for platform " << i << " (clGetDeviceIDs)\n"; return -1; }

            for (cl_uint j = 0; j < device_count; ++j)
            {
                {
                    size_t size{};
                    err = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, nullptr, &size);
                    if( err != CL_SUCCESS){ std::cout << "Error getting device " << j << " name length in platform " << i << " (clGetDeviceInfo)\n"; return -1; }

                    std::string device_name(size, '\0');
                    err = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, size, const_cast<char*>(device_name.data()), nullptr);
                    if( err != CL_SUCCESS){ std::cout << "Error getting device " << j << " name in platform " << i << " (clGetDeviceInfo)\n"; return -1; }

                    std::cout << "Platform " << i << " device " << j << " name:               " << device_name << "\n";
                }

                {
                    size_t size{};
                    err = clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(size), &size, nullptr);
                    if( err != CL_SUCCESS){ std::cout << "Error getting device " << j << " global memory size in platform " << i << " (clGetDeviceInfo)\n"; return -1; }
                    std::cout << "Platform " << i << " device " << j << " global memory size: " << size/1024/1024 << " MiB\n";
                }

                

                err = clReleaseDevice(devices[j]);
                if( err != CL_SUCCESS){ std::cout << "Error releasing device " << j << " in platform " << i << " (clReleaseDevice)\n"; return -1; }
            }

            std::cout << "\n";
        }
    }

    return 0;
}