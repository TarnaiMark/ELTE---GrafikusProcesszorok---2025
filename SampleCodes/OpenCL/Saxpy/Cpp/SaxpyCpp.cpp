#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <fstream>

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
        std::ifstream source{"./saxpy.cl"};
        if( !source.is_open() ){ throw std::runtime_error{ std::string{"Error opening kernel file: saxpy.cl"} }; }
        std::string source_string{ std::istreambuf_iterator<char>{ source },
                                   std::istreambuf_iterator<char>{} };
        cl::Program program{ context, source_string };
        program.build({selected_device});
        
        auto saxpy_kernel = cl::KernelFunctor<cl_double, cl::Buffer, cl::Buffer, int>(program, "saxpy");

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

        cl::Buffer buffer_x{ context, std::begin(X), std::end(X), false }; // false: read-write
        cl::Buffer buffer_y{ context, std::begin(Y), std::end(Y), true };  // true: read-only

        // Launch kernel:
        saxpy_kernel(cl::EnqueueArgs{queue, cl::NDRange{N}}, scalar, buffer_x, buffer_y, N);
        
        // Copy back results:
        cl::copy(queue, buffer_x, std::begin(Result), std::end(Result));

        // Synchronize:
        queue.finish();

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