#ifdef __APPLE__ //Mac OSX has a different name for the header file
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
#include <numeric>
#include <cmath>

int main()
{
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
    
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, selected_device_id, nullptr, &err);
    if(err != CL_SUCCESS){ std::cout << "Error creating comand queue (clCreateCommandQueueWithProperties)\n"; return -1; }

    // Load and compile kernel program:
    std::ifstream file("./reduce.cl");
    if(!file.is_open()){ std::cout << "Error opening kernel file: reduce.cl\n"; return -1; }

    std::string source{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    size_t      source_size = source.size();
    const char* source_ptr  = source.c_str();
    cl_program program = clCreateProgramWithSource(context, 1, &source_ptr, &source_size, &err);
    if(err != CL_SUCCESS){ std::cout << "Error creating program object (clCreateProgramWithSource)\n"; return -1; }

    err = clBuildProgram(program, 1, &selected_device_id, "", nullptr, nullptr);
    if(err != CL_SUCCESS)
    {
        size_t size = 0;
        err = clGetProgramBuildInfo(program, selected_device_id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size);
        std::string log(size, '\0');
    
        err = clGetProgramBuildInfo(program, selected_device_id, CL_PROGRAM_BUILD_LOG, size, const_cast<char*>(log.data()), nullptr);
        std::cout << "Build failed. Log:\n" << log.c_str();
        return -1;
    }
    
    cl_kernel kernel = clCreateKernel(program, "reduce", &err);
    if(err != CL_SUCCESS){ std::cout << "Error creating kernel (clCreateKernel)\n"; return -1; }

    // Allocate and setup data buffers:
    const int N = 256*256*2;
    double result = 0.0;
    std::vector<double> X(N, 0.0);
    for(int i=0; i<N; ++i)
    {
        X[i] = i*1.0/N;
    }
    
    cl_mem buffer_x = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, X.size() * sizeof(double), X.data(), &err);
    if(err != CL_SUCCESS){ std::cout << "Error creating buffer X (clCreateBuffer)\n"; return -1; }
    cl_mem buffer_y = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, X.size() * sizeof(double), nullptr, &err);
    if(err != CL_SUCCESS){ std::cout << "Error creating buffer Y (clCreateBuffer)\n"; return -1; }
    cl_mem buffer_z = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, 1 * sizeof(double), nullptr, &err);
    if(err != CL_SUCCESS){ std::cout << "Error creating buffer Z (clCreateBuffer)\n"; return -1; }

    double zero = 0.0;
    err = clEnqueueFillBuffer(queue, buffer_y, &zero, sizeof(double), 0, X.size() * sizeof(double), 0, nullptr, nullptr);
    if(err != CL_SUCCESS){ std::cout << "Error enqueueing FillBuffer (clEnqueueFillBuffer)\n"; return -1; }

    // Set first kernel arguments:
    err = clSetKernelArg(kernel, 0, sizeof(buffer_x), &buffer_x);
    if(err != CL_SUCCESS){ std::cout << "Error setting kernel argument 0 (clSetKernelArg)\n"; return -1; }
    err = clSetKernelArg(kernel, 1, sizeof(buffer_y), &buffer_y);
    if(err != CL_SUCCESS){ std::cout << "Error setting kernel argument 1 (clSetKernelArg)\n"; return -1; }
    err = clSetKernelArg(kernel, 2, sizeof(int), &N);
    if(err != CL_SUCCESS){ std::cout << "Error setting kernel argument 2 (clSetKernelArg)\n"; return -1; }

    // Launch first kernel:
    size_t thread_count = static_cast<size_t>( std::ceil(N*1.0f/256/2)*256 );
    size_t workgroup_size = 256;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &thread_count, &workgroup_size, 0, nullptr, nullptr);
    if(err != CL_SUCCESS){ std::cout << "Error enqueueing kernel (clEnqueueNDRangeKernel)\n"; return -1; }

    // Set second kernel arguments:
    err = clSetKernelArg(kernel, 0, sizeof(buffer_y), &buffer_y);
    if(err != CL_SUCCESS){ std::cout << "Error setting kernel argument 0 (clSetKernelArg)\n"; return -1; }
    err = clSetKernelArg(kernel, 1, sizeof(buffer_z), &buffer_z);
    if(err != CL_SUCCESS){ std::cout << "Error setting kernel argument 1 (clSetKernelArg)\n"; return -1; }
    const int N2 = static_cast<int>( std::ceil(N*1.0f/workgroup_size/2) );
    err = clSetKernelArg(kernel, 2, sizeof(int), &N2);
    if(err != CL_SUCCESS){ std::cout << "Error setting kernel argument 2 (clSetKernelArg)\n"; return -1; }

    // Launch second kernel:
    thread_count = 256;
    workgroup_size = 256;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &thread_count, &workgroup_size, 0, nullptr, nullptr);
    if(err != CL_SUCCESS){ std::cout << "Error enqueueing kernel (clEnqueueNDRangeKernel)\n"; return -1; }

    // Copy back results:
    err = clEnqueueReadBuffer(queue, buffer_z, false, 0, 1 * sizeof(double), &result, 0, nullptr, nullptr);
    if(err != CL_SUCCESS){ std::cout << "Error reading buffer (clEnqueueReadBuffer)\n"; return -1; }

    // Synchronize:
    err = clFinish(queue);
    if(err != CL_SUCCESS){ std::cout << "Error synchronizing with device (clFinish)\n"; return -1; }

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

    // cleanup:
    {
        err = clReleaseMemObject(buffer_x);
        if( err != CL_SUCCESS){ std::cout << "Error releasing buffer X (clReleaseMemObject)\n"; return -1; }

        err = clReleaseMemObject(buffer_y);
        if( err != CL_SUCCESS){ std::cout << "Error releasing buffer Y (clReleaseMemObject)\n"; return -1; }

        err = clReleaseMemObject(buffer_z);
        if( err != CL_SUCCESS){ std::cout << "Error releasing buffer Z (clReleaseMemObject)\n"; return -1; }

        err = clReleaseKernel(kernel);
        if( err != CL_SUCCESS){ std::cout << "Error releasing kernel (clReleaseKernel)\n"; return -1; }

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