#include <iostream>
#include <string>
#include <vector>

#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 200
#define CL_HPP_ENABLE_EXCEPTIONS 1
#include <CL/opencl.hpp>

int main()
{
    try
    {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        std::cout << "There are " << platforms.size() << " OpenCL platforms\n\n";
        if(platforms.size() == 0){ std::cout << "No OpenCL platform detected\n"; return -1; }

        for(std::size_t i = 0; i<platforms.size(); ++i)
        {
            std::cout << "Platform " << i << " vendor: " << platforms[i].getInfo<CL_PLATFORM_VENDOR>() << "\n";
            std::cout << "Platform " << i << " name:   " << platforms[i].getInfo<CL_PLATFORM_NAME>() << "\n";

            std::vector<cl::Device> devices;
            platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices);

            std::cout << "Platform " << i << " has:    " << devices.size() << " device(s)\n";

            for(std::size_t j = 0; j<devices.size(); ++j)
            {
                std::cout << "Platform " << i << " device " << j << " name:               " << devices[j].getInfo<CL_DEVICE_NAME>() << "\n";
                std::cout << "Platform " << i << " device " << j << " global memory size: " << devices[j].getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>()/1024/1024 << " MiB\n";
            }
            std::cout << "\n";
        }
    }
    catch(cl::Error& e)
    {
        std::cout << "OpenCL error: " << e.what() << "\n";
        return -1;
    }
    catch(std::exception& e)
    {
        std::cerr << "C++ STL Error: " << e.what() << "\n";
        return -1;
    }
    
    return 0;
}