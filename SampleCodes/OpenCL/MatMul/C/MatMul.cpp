#ifdef __APPLE__ //Mac OSX has a different name for the header file
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include <string>
#include <random>
#include <chrono>
#include <fstream>

#include "../cpu_matmul.h"

using T = double;

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

    // Setup OpenCL:

    cl_int err = CL_SUCCESS;

    cl_uint platform_count = 0;

    err = clGetPlatformIDs(0, nullptr, &platform_count);
    if(err != CL_SUCCESS){ std::cout << "Error getting platform count (clGetPlatformIDs)\n"; return -1; }

    if(platform_count == 0){ std::cout << "No OpenCL platform detected.\n"; return -1; }
    
    std::vector<cl_platform_id> platform_ids(platform_count);
    err = clGetPlatformIDs(platform_count, platform_ids.data(), nullptr);
    if(err != CL_SUCCESS){ std::cout << "Error getting platform ids (clGetPlatformIDs)\n"; return -1; }

    cl_platform_id selected_platform_id{};
    cl_device_id selected_device_id{};
    bool found_gpu_device = false;
    for(cl_uint i = 0; i < platform_count; ++i)
    {
        cl_uint device_count = 0;
        err = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count);
        if(err != CL_SUCCESS){ std::cout << "Error getting device count for platform " << i << " (clGetDeviceIDs)\n"; return -1; }

        if(device_count == 0){ continue; }

        std::vector<cl_device_id> devices(device_count);
        err = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_GPU, device_count, devices.data(), nullptr);
        if(err != CL_SUCCESS){ std::cout << "Error getting device ids for platform " << i << " (clGetDeviceIDs)\n"; return -1; }

        for (cl_uint j = 0; j < device_count; ++j)
        {
            // pick first device, release all others:
            if(j == 0)
            {
                selected_platform_id = platform_ids[i];
                selected_device_id = devices[j];
                found_gpu_device = true;
            }
            else
            {
                err = clReleaseDevice(devices[j]);
                if(err != CL_SUCCESS){ std::cout << "Error releasing device " << j << " in platform " << i << " (clReleaseDevice)\n"; return -1; }
            }
        }

        // skip other platforms if a GPU device is found:
        if(found_gpu_device){ break; }
    }

    if(!found_gpu_device)
    {
        std::cout << "No OpenCL GPU device found. Change CL_DEVICE_TYPE_GPU to CL_DEVICE_TYPE_CPU to check for CPU devices.\n";
        return -1;
    }

    // Print selected platform and device name:
    {
        size_t size{};
        err = clGetPlatformInfo(selected_platform_id, CL_PLATFORM_VENDOR, UINTMAX_MAX, nullptr, &size);
        if(err != CL_SUCCESS){ std::cout << "Error getting platform vendor name length (clGetPlatformInfo)\n"; return -1; }

        std::string platform_vendor_name(size, '\0');
        err = clGetPlatformInfo(selected_platform_id, CL_PLATFORM_VENDOR, size, const_cast<char*>(platform_vendor_name.data()), nullptr);
        if(err != CL_SUCCESS){ std::cout << "Error getting platform vendor name (clGetPlatformInfo)\n"; return -1; }

        std::cout << "Selected platform vendor: " << platform_vendor_name << "\n";
    }

    {
        size_t size{};
        err = clGetPlatformInfo(selected_platform_id, CL_PLATFORM_NAME, UINTMAX_MAX, nullptr, &size);
        if(err != CL_SUCCESS){ std::cout << "Error getting platform name length (clGetPlatformInfo)\n"; return -1; }

        std::string platform_name(size, '\0');
        err = clGetPlatformInfo(selected_platform_id, CL_PLATFORM_NAME, size, const_cast<char*>(platform_name.data()), nullptr);
        if(err != CL_SUCCESS){ std::cout << "Error getting platform name (clGetPlatformInfo)\n"; return -1; }

        std::cout << "Selected platform name:   " << platform_name << "\n";
    }

    {
        size_t size{};
        err = clGetDeviceInfo(selected_device_id, CL_DEVICE_NAME, 0, nullptr, &size);
        if(err != CL_SUCCESS){ std::cout << "Error getting device name length in platform (clGetDeviceInfo)\n"; return -1; }

        std::string device_name(size, '\0');
        err = clGetDeviceInfo(selected_device_id, CL_DEVICE_NAME, size, const_cast<char*>(device_name.data()), nullptr);
        if(err != CL_SUCCESS){ std::cout << "Error getting device name in platform (clGetDeviceInfo)\n"; return -1; }

        std::cout << "Selected device name:     " << device_name << "\n";
    }

    // Actual program logic: create context and command queue:

    cl_context_properties cps[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)selected_platform_id, 0};
    cl_context context = clCreateContext(cps, 1, &selected_device_id, 0, 0, &err);
    if(err != CL_SUCCESS){ std::cout << "Error creating context (clCreateContext)\n"; return -1; }

    // Enable profiling on the queue:
    cl_command_queue_properties cqps = CL_QUEUE_PROFILING_ENABLE;
	std::array<cl_queue_properties, 3> qps = { CL_QUEUE_PROPERTIES, cqps, 0 };
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, selected_device_id, qps.data(), &err);
    if(err != CL_SUCCESS){ std::cout << "Error creating comand queue (clCreateCommandQueueWithProperties)\n"; return -1; }

    // Load and compile kernel program:
    std::ifstream file("./matmul.cl");
    if(!file.is_open()){ std::cout << "Error opening kernel file: matmul.cl\n"; return -1; }

    std::string source{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    size_t      source_size = source.size();
    const char* source_ptr  = source.c_str();
    cl_program program = clCreateProgramWithSource(context, 1, &source_ptr, &source_size, &err);
    if(err != CL_SUCCESS){ std::cout << "Error creating program object (clCreateProgramWithSource)\n"; return -1; }

    std::string build_option;
    if     ( std::is_same_v<T, float> ){ build_option = "-DT=float -DBS="; build_option += std::to_string(block_size); }
    else if( std::is_same_v<T, double> ){ build_option = "-DT=double -DBS="; build_option += std::to_string(block_size); }

    err = clBuildProgram(program, 1, &selected_device_id, build_option.c_str(), nullptr, nullptr);
    if(err != CL_SUCCESS)
    {
        size_t size = 0;
        err = clGetProgramBuildInfo(program, selected_device_id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size);
        std::string log(size, '\0');
    
        err = clGetProgramBuildInfo(program, selected_device_id, CL_PROGRAM_BUILD_LOG, size, const_cast<char*>(log.data()), nullptr);
        std::cout << "Build failed. Log:\n" << log.c_str();
        return -1;
    }
    
    cl_kernel kernel0 = clCreateKernel(program, "matmul0", &err);
    if(err != CL_SUCCESS){ std::cout << "Error creating kernel 0 (clCreateKernel)\n"; return -1; }

    cl_kernel kernel1 = clCreateKernel(program, "matmul1", &err);
    if(err != CL_SUCCESS){ std::cout << "Error creating kernel 1 (clCreateKernel)\n"; return -1; }

    cl_mem buffer_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, A.size() * sizeof(T), A.data(), &err);
    if(err != CL_SUCCESS){ std::cout << "Error creating buffer A (clCreateBuffer)\n"; return -1; }

    cl_mem buffer_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, B.size() * sizeof(T), B.data(), &err);
    if(err != CL_SUCCESS){ std::cout << "Error creating buffer B (clCreateBuffer)\n"; return -1; }
    
    cl_mem buffer_C2 = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, C2.size() * sizeof(T), nullptr, &err);
    if(err != CL_SUCCESS){ std::cout << "Error creating buffer C2 (clCreateBuffer)\n"; return -1; }

    cl_mem buffer_C3 = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, C3.size() * sizeof(T), nullptr, &err);
    if(err != CL_SUCCESS){ std::cout << "Error creating buffer C3 (clCreateBuffer)\n"; return -1; }
    
    // Set kernel arguments:
    err = clSetKernelArg(kernel0, 0, sizeof(buffer_A), &buffer_A);
    if(err != CL_SUCCESS){ std::cout << "Error setting kernel 0 argument 0 (clSetKernelArg)\n"; return -1; }
    err = clSetKernelArg(kernel0, 1, sizeof(buffer_B), &buffer_B);
    if(err != CL_SUCCESS){ std::cout << "Error setting kernel 0 argument 1 (clSetKernelArg)\n"; return -1; }
    err = clSetKernelArg(kernel0, 2, sizeof(buffer_C2), &buffer_C2);
    if(err != CL_SUCCESS){ std::cout << "Error setting kernel 0 argument 2 (clSetKernelArg)\n"; return -1; }
    err = clSetKernelArg(kernel0, 3, sizeof(N), &N);
    if(err != CL_SUCCESS){ std::cout << "Error setting kernel 0 argument 3 (clSetKernelArg)\n"; return -1; }

    err = clSetKernelArg(kernel1, 0, sizeof(buffer_A), &buffer_A);
    if(err != CL_SUCCESS){ std::cout << "Error setting kernel 1 argument 0 (clSetKernelArg)\n"; return -1; }
    err = clSetKernelArg(kernel1, 1, sizeof(buffer_B), &buffer_B);
    if(err != CL_SUCCESS){ std::cout << "Error setting kernel 1 argument 1 (clSetKernelArg)\n"; return -1; }
    err = clSetKernelArg(kernel1, 2, sizeof(buffer_C3), &buffer_C3);
    if(err != CL_SUCCESS){ std::cout << "Error setting kernel 1 argument 2 (clSetKernelArg)\n"; return -1; }
    err = clSetKernelArg(kernel1, 3, sizeof(N), &N);
    if(err != CL_SUCCESS){ std::cout << "Error setting kernel 1 argument 3 (clSetKernelArg)\n"; return -1; }

    // Warmup kernel launches:
    std::array<size_t, 2> grid_size = {N, N};
    std::array<size_t, 2> workgroup_size = {block_size, block_size};

    err = clEnqueueNDRangeKernel(queue, kernel0, 2, nullptr, grid_size.data(), nullptr, 0, nullptr, nullptr);
    if(err != CL_SUCCESS){ std::cout << "Error enqueueing kernel0 (clEnqueueNDRangeKernel)\n"; return -1; }
    err = clEnqueueNDRangeKernel(queue, kernel1, 2, nullptr, grid_size.data(), workgroup_size.data(), 0, nullptr, nullptr);
    if(err != CL_SUCCESS){ std::cout << "Error enqueueing kernel1 (clEnqueueNDRangeKernel)\n"; return -1; }

    // Synchronize:
    err = clFinish(queue);
    if(err != CL_SUCCESS){ std::cout << "Error synchronizing with device (clFinish)\n"; return -1; }

    // Storage for GPU events:
    std::array<cl_event, 2> evts;

    // Actual measured kernel launches:
    err = clEnqueueNDRangeKernel(queue, kernel0, 2, nullptr, grid_size.data(), nullptr, 0, nullptr, &evts[0]);
    if(err != CL_SUCCESS){ std::cout << "Error enqueueing kernel0 (clEnqueueNDRangeKernel)\n"; return -1; }
    err = clEnqueueNDRangeKernel(queue, kernel1, 2, nullptr, grid_size.data(), workgroup_size.data(), 0, nullptr, &evts[1]);
    if(err != CL_SUCCESS){ std::cout << "Error enqueueing kernel1 (clEnqueueNDRangeKernel)\n"; return -1; }
    
    // Copy back results:
    err = clEnqueueReadBuffer(queue, buffer_C2, false, 0, C2.size() * sizeof(T), C2.data(), 1, &evts[0], nullptr);
    if(err != CL_SUCCESS){ std::cout << "Error reading buffer C2 (clEnqueueReadBuffer): " << err << "\n"; return -1; }
    err = clEnqueueReadBuffer(queue, buffer_C3, false, 0, C3.size() * sizeof(T), C3.data(), 1, &evts[1], nullptr);
    if(err != CL_SUCCESS){ std::cout << "Error reading buffer C3 (clEnqueueReadBuffer): " << err << "\n"; return -1; }

    // Synchronize:
    err = clFinish(queue);
    if(err != CL_SUCCESS){ std::cout << "Error synchronizing with device (clFinish)\n"; return -1; }

    // Time counters:
    cl_ulong t0_0, t0_1, t1_0, t1_1;

    err = clGetEventProfilingInfo(evts[0], CL_PROFILING_COMMAND_START, sizeof(t0_0), &t0_0, nullptr);
    if(err != CL_SUCCESS){ std::cout << "Error getting kernel 0 start time (clGetEventProfilingInfo)\n"; return -1; }
	err = clGetEventProfilingInfo(evts[0], CL_PROFILING_COMMAND_END, sizeof(t0_1), &t0_1, nullptr);
    if(err != CL_SUCCESS){ std::cout << "Error getting kernel 0 end time (clGetEventProfilingInfo)\n"; return -1; }

    err = clGetEventProfilingInfo(evts[1], CL_PROFILING_COMMAND_START, sizeof(t1_0), &t1_0, nullptr);
    if(err != CL_SUCCESS){ std::cout << "Error getting kernel 1 start time (clGetEventProfilingInfo)\n"; return -1; }
	err = clGetEventProfilingInfo(evts[1], CL_PROFILING_COMMAND_END, sizeof(t1_1), &t1_1, nullptr);
    if(err != CL_SUCCESS){ std::cout << "Error getting kernel 1 end time (clGetEventProfilingInfo)\n"; return -1; }

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
    
    // cleanup:
    {
        err = clReleaseEvent(evts[0]);
        if( err != CL_SUCCESS){ std::cout << "Error releasing event 0 (clReleaseEvent)\n"; return -1; }

        err = clReleaseEvent(evts[1]);
        if( err != CL_SUCCESS){ std::cout << "Error releasing event 1 (clReleaseEvent)\n"; return -1; }

        err = clReleaseMemObject(buffer_C3);
        if( err != CL_SUCCESS){ std::cout << "Error releasing buffer C3 (clReleaseMemObject)\n"; return -1; }

        err = clReleaseMemObject(buffer_C2);
        if( err != CL_SUCCESS){ std::cout << "Error releasing buffer C2 (clReleaseMemObject)\n"; return -1; }

        err = clReleaseMemObject(buffer_B);
        if( err != CL_SUCCESS){ std::cout << "Error releasing buffer B (clReleaseMemObject)\n"; return -1; }

        err = clReleaseMemObject(buffer_A);
        if( err != CL_SUCCESS){ std::cout << "Error releasing buffer A (clReleaseMemObject)\n"; return -1; }

        err = clReleaseKernel(kernel1 );
        if( err != CL_SUCCESS){ std::cout << "Error releasing kernel 1 (clReleaseKernel)\n"; return -1; }

        err = clReleaseKernel(kernel0);
        if( err != CL_SUCCESS){ std::cout << "Error releasing kernel 0 (clReleaseKernel)\n"; return -1; }

        err = clReleaseProgram(program);
        if( err != CL_SUCCESS){ std::cout << "Error releasing program (clReleaseProgram)\n"; return -1; }

        err = clReleaseCommandQueue(queue);
        if( err != CL_SUCCESS){ std::cout << "Error releasing queue (clReleaseCommandQueue)\n"; return -1; }

        err = clReleaseContext(context);
        if( err != CL_SUCCESS){ std::cout << "Error releasing context (clReleaseContext)\n"; return -1; }

        err = clReleaseDevice(selected_device_id);
        if( err != CL_SUCCESS){ std::cout << "Error releasing device (clReleaseDevice)\n"; return -1; }
    }

    return 0;
}