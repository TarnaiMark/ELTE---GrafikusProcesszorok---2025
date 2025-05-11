#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <array>
#include <random>
#include <chrono>
#include <fstream>

#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 200
#define CL_HPP_ENABLE_EXCEPTIONS 1
#include <CL/opencl.hpp>

#include "../cpu_matmul.h"

using T = float;

constexpr int N = 1024;
constexpr int block_size = 16;

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
        
        // Enable profiling on the queue:
        cl::QueueProperties qps{cl::QueueProperties::Profiling};
        cl::CommandQueue queue{ context, selected_device, qps};

        // Load and compile kernel program:
        std::ifstream source{"./matmul.cl"};
        if( !source.is_open() ){ throw std::runtime_error{ std::string{"Error opening kernel file: matmul.cl"} }; }
        std::string source_string{ std::istreambuf_iterator<char>{ source },
                                   std::istreambuf_iterator<char>{} };
        cl::Program program{ context, source_string };

        std::string build_option;
        if     ( std::is_same_v<T, float> ){ build_option = "-DT=float -DBS="; build_option += std::to_string(block_size); }
        else if( std::is_same_v<T, double> ){ build_option = "-DT=double -DBS="; build_option += std::to_string(block_size); }
        program.build({selected_device}, build_option.c_str());
        
        auto matmul_kernel0 = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, int>(program, "matmul0");
        auto matmul_kernel1 = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, int>(program, "matmul1");

        // Allocate and setup data buffers:
        cl::Buffer buffer_A{ context, std::begin(A), std::end(A), true }; // true: read-only
        cl::Buffer buffer_B{ context, std::begin(B), std::end(B), true }; // true: read-only
        cl::Buffer buffer_C2{ context, CL_MEM_WRITE_ONLY, C2.size() * sizeof(T) };
        cl::Buffer buffer_C3{ context, CL_MEM_WRITE_ONLY, C3.size() * sizeof(T) };

        // Warmup kernel launches:
        cl::NDRange grid_size = cl::NDRange{N, N};
        cl::NDRange workgroup_size = cl::NDRange{block_size, block_size};

        matmul_kernel0(cl::EnqueueArgs{queue, grid_size                }, buffer_A, buffer_B, buffer_C2, N);
        matmul_kernel1(cl::EnqueueArgs{queue, grid_size, workgroup_size}, buffer_A, buffer_B, buffer_C3, N);

        // Synchronize:
        queue.finish();

        // Measured kernel launches:
        cl::Event ev0 = matmul_kernel0(cl::EnqueueArgs{queue, grid_size                }, buffer_A, buffer_B, buffer_C2, N);
        cl::Event ev1 = matmul_kernel1(cl::EnqueueArgs{queue, grid_size, workgroup_size}, buffer_A, buffer_B, buffer_C3, N);
        
        // Copy back results:
        std::vector<cl::Event> vev0{ev0};
        queue.enqueueReadBuffer(buffer_C2, CL_FALSE, 0, C2.size() * sizeof(T), C2.data(), &vev0);
        std::vector<cl::Event> vev1{ev1};
        queue.enqueueReadBuffer(buffer_C3, CL_FALSE, 0, C3.size() * sizeof(T), C3.data(), &vev1);

        // Synchronize:
        queue.finish();

        // Time counters:
        cl_ulong t0_0 = ev0.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        cl_ulong t0_1 = ev0.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        cl_ulong t1_0 = ev1.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        cl_ulong t1_1 = ev1.getProfilingInfo<CL_PROFILING_COMMAND_END>();

        auto t_gpu_naive = (t0_1 - t0_0)*1e-6;
        auto t_gpu_improved = (t1_1 - t1_0)*1e-6;

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