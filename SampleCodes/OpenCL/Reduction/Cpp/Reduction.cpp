#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <fstream>
#include <numeric>
#include <cmath>

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
        if(platforms.size() == 0){ std::cout << "No OpenCL platform detected\n"; return -1; }

        cl::Platform selected_platform{};
        cl::Device selected_device{};
        bool found_gpu_device = false;
        for(std::size_t i = 0; i<platforms.size(); ++i)
        {
            std::vector<cl::Device> devices;
            platforms[i].getDevices(CL_DEVICE_TYPE_GPU, &devices);

            if(devices.size() == 0){ continue; }

            for(std::size_t j = 0; j<devices.size(); ++j)
            {
                // pick first device:
                if(j == 0)
                {
                    selected_platform = platforms[i];
                    selected_device = devices[j];
                    found_gpu_device = true;
                }
            }

            // skip other platforms if a GPU device is found:
            if(found_gpu_device){ break; }
        }

        std::cout << "Selected platform vendor: " << selected_platform.getInfo<CL_PLATFORM_VENDOR>() << "\n";
        std::cout << "Selected platform name:   " << selected_platform.getInfo<CL_PLATFORM_NAME>() << "\n";
        std::cout << "Selected device name:     " << selected_device.getInfo<CL_DEVICE_NAME>() << "\n";

        // Actual program logic: create context and command queue:
        std::vector<cl_context_properties>cps{ CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(selected_platform()), 0};
        cl::Context context{ selected_device, cps.data() };
        cl::CommandQueue queue{ context, selected_device };

        // Load and compile kernel program:
        std::ifstream source{"./reduce.cl"};
        if( !source.is_open() ){ throw std::runtime_error{ std::string{"Error opening kernel file: reduce.cl"} }; }
        std::string source_string{ std::istreambuf_iterator<char>{ source },
                                   std::istreambuf_iterator<char>{} };
        cl::Program program{ context, source_string };
        program.build({selected_device});
        
        auto reduce_kernel = cl::KernelFunctor<cl::Buffer, cl::Buffer, int>(program, "reduce");

        // Allocate and setup data buffers:
        const int N = 256*256*2;
        double result = 0.0;
        std::vector<double> X(N, 0.0);
        for(int i=0; i<N; ++i)
        {
            X[i] = i*1.0/N;
        }

        cl::Buffer buffer_x{ context, std::begin(X), std::end(X), true }; // true: read-only
        cl::Buffer buffer_y{ context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, N * sizeof(double) };
        cl::Buffer buffer_z{ context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, 1 * sizeof(double)};

        queue.enqueueFillBuffer<double>(buffer_y, 0.0, 0, N);
        queue.finish();

        // Launch first kernel:
        size_t thread_count = static_cast<size_t>( std::ceil(N*1.0f/256/2)*256 );
        size_t workgroup_size = 256;
        reduce_kernel(cl::EnqueueArgs{queue, cl::NDRange{thread_count}, cl::NDRange{workgroup_size}}, buffer_x, buffer_y, N);

        // second kernel:
        thread_count = 256;
        workgroup_size = 256;
        const int N2 = static_cast<int>( std::ceil(N*1.0f/workgroup_size/2) );
        reduce_kernel(cl::EnqueueArgs{queue, cl::NDRange{thread_count}, cl::NDRange{workgroup_size}}, buffer_y, buffer_z, N2);
        
        // Copy back results:
        cl::copy(queue, buffer_z, &result, &result+1);

        // Synchronize:
        queue.finish();

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
    }
    catch(cl::BuildError& error) // If kernel failed to build
    {
        std::cout << "Build failed. Log:\n";
        for (const auto& log : error.getBuildLog())
        {
            std::cout << log.second;
        }
        return -1;
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