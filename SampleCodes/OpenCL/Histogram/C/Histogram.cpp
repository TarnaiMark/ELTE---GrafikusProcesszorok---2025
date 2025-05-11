#ifdef __APPLE__ //Mac OSX has a different name for the header file
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <string>
#include <cmath>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

struct color{ unsigned char r, g, b, a; };

// A 3 channel histogram helper object:
struct three_histograms
{
    std::array<unsigned int, 256> rh, gh, bh;
    void zero()
    {
        for(int i=0; i<256; ++i)
        {
            rh[i] = 0; gh[i] = 0; bh[i] = 0;
        }
    }

    void fromLinearMemory( std::vector<unsigned int>& input )
    {
        for(int i=0; i<256; ++i)
        {
            rh[i] = input[0*256+i];
            gh[i] = input[1*256+i];
            bh[i] = input[2*256+i];
        }
    }
};

int compare(three_histograms const& h1, three_histograms const& h2)
{
    int mismatches = 0;
    for(int i=0; i<256; ++i)
    {
        if(h1.rh[i] != h2.rh[i]){ std::cout << "Mismatch: red   at " << i << " : " << h1.rh[i] << " != " << h2.rh[i] << "\n"; mismatches += 1; }
        if(h1.gh[i] != h2.gh[i]){ std::cout << "Mismatch: green at " << i << " : " << h1.gh[i] << " != " << h2.gh[i] << "\n"; mismatches += 1; }
        if(h1.bh[i] != h2.bh[i]){ std::cout << "Mismatch: blue  at " << i << " : " << h1.bh[i] << " != " << h2.bh[i] << "\n"; mismatches += 1; }
    }
    return mismatches;
}

void draw_histogram(std::string const& filename, three_histograms const& data )
{
    const int w = 800;
    const int h = 800;
    std::vector<color> image(w*h);
    const color white{255, 255, 255, 255};
    std::fill(image.begin(), image.end(), white);
    auto max_r = *std::max_element(data.rh.begin(), data.rh.end());
    auto max_g = *std::max_element(data.gh.begin(), data.gh.end());
    auto max_b = *std::max_element(data.bh.begin(), data.bh.end());
    auto div = std::max(std::max(max_r, max_g), max_b);

    // draw a filled rectangle:
    auto fill_rect = [&](int x0, int y0, int width, int height, color const& c)
    {
        for(int y=y0; y>y0-height; --y)
        {
            for(int x=x0; x<x0+width; ++x)
            {
                image[y*w+x] = c;
            }
        }
    };

    // draw the three histograms next to each other with different colors:
    for(int i=0; i<256; ++i)
    {
        fill_rect(i,       780, 1, data.rh[i]*700/div, color{(unsigned char)i, 0, 0, 255});
        fill_rect(i+256,   780, 1, data.gh[i]*700/div, color{0, (unsigned char)i, 0, 255});
        fill_rect(i+256*2, 780, 1, data.bh[i]*700/div, color{0, 0, (unsigned char)i, 255});
    }
    
    // write image to file:
    int res = stbi_write_jpg(filename.c_str(), w, h, 4, image.data(), 100);
    if(res == 0)
    {
        std::cout << "Error writing output to file " << filename << "\n";
    }else{ std::cout << "Output written to file " << filename << "\n"; }
};

void cpu_histo( three_histograms& output, color* const& input, int w, int h )
{
    for(int y=0; y<h; ++y)
    {
        for(int x=0; x<w; ++x)
        {
            color c = input[y*w+x];
            output.rh[c.r] += 1;
            output.gh[c.g] += 1;
            output.bh[c.b] += 1;
        }
    }
}

const std::string input_filename   = "NZ.jpg";
const std::string output_filename1 = "cpu_out.jpg";
const std::string output_filename2 = "gpu_out1.jpg";
const std::string output_filename3 = "gpu_out2.jpg";
constexpr int block_size = 16;

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
    for(cl_uint i = 1; i < platform_count; ++i)
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
    std::ifstream file("./histogram.cl");
    if(!file.is_open()){ std::cout << "Error opening kernel file: histogram.cl\n"; return -1; }

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
    
    // Craete kernel objects:
    cl_kernel kernel_global = clCreateKernel(program, "gpu_histo_global_atomics", &err);
    if(err != CL_SUCCESS){ std::cout << "Error creating gpu_histo_global_atomics kernel (clCreateKernel)\n"; return -1; }

    cl_kernel kernel_shared = clCreateKernel(program, "gpu_histo_shared_atomics", &err);
    if(err != CL_SUCCESS){ std::cout << "Error creating gpu_histo_shared_atomics kernel (clCreateKernel)\n"; return -1; }

    cl_kernel kernel_acc = clCreateKernel(program, "gpu_histo_accumulate", &err);
    if(err != CL_SUCCESS){ std::cout << "Error creating gpu_histo_accumulate kernel (clCreateKernel)\n"; return -1; }

    // Actual program logic:
    // Read input image:
    int w = 0; // width
    int h = 0; // height
    int ch = 0; // number of components

    color* data0 = reinterpret_cast<color*>(stbi_load(input_filename.c_str(), &w, &h, &ch, 4 /* we expect 4 components */));
    if(!data0)
    {
        std::cout << "Error: could not open input file: " << input_filename << "\n";
        return -1;
    }
    else
    {
        std::cout << "Image (" << input_filename << ") opened successfully. Width x Height x Components = " << w << " x " << h << " x " << ch << "\n";
    }

    const int nBlocks = static_cast<int>( std::ceil(h*1.0f / block_size) );
    const size_t nBlocks_sz = static_cast<size_t>( nBlocks );

    three_histograms cpu;  cpu.zero();
    three_histograms gpu1; gpu1.zero();
    three_histograms gpu2; gpu2.zero();

    // CPU serial version:
    float dt0 = 0.0f;
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        cpu_histo(cpu, data0, w, h);
        auto t1 = std::chrono::high_resolution_clock::now();
        dt0 = std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count()/1000.0f;
    }

    // Allocate GPU buffers:
    cl_mem input_image = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, w*h*sizeof(color), data0, &err);
    if(err != CL_SUCCESS){ std::cout << "Cannot create input buffer: " << err << "\n"; return -1; }
        
    cl_mem partials = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,  nBlocks * 3 * 256 * sizeof(unsigned int), nullptr, &err);
    if(err != CL_SUCCESS){ std::cout << "Cannot create partial buffer: " << err << "\n"; return -1; }
        
    cl_mem output  = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,  3 * 256 * sizeof(unsigned int), nullptr, &err);
    if(err != CL_SUCCESS){ std::cout << "Cannot create output buffer: " << err << "\n"; return -1; }

    // GPU version using global atomics:
    float dt1 = 0.0f;
    {
        unsigned int zero = 0;
        err = clEnqueueFillBuffer(queue, partials, &zero, sizeof(unsigned int), 0, nBlocks * 3 * 256 * sizeof(unsigned int), 0, nullptr, nullptr);
        if(err != CL_SUCCESS){ std::cout << "Cannot zero partial buffer (1): " << err << "\n"; return -1; }

        err = clEnqueueFillBuffer(queue, output, &zero, sizeof(unsigned int), 0, 3 * 256 * sizeof(unsigned int), 0, nullptr, nullptr);
        if(err != CL_SUCCESS){ std::cout << "Cannot zero output buffer: (1):" << err << "\n"; return -1; }

        // Wait for the fills to finish:
        err = clFinish(queue);
        if(err != CL_SUCCESS){ std::cout << "Cannot finish queue: " << err << "\n"; return -1; }

        // Storage for GPU events:
        std::array<cl_event, 2> evts;

        // First kernel of global histograms:
        {
            err = clSetKernelArg(kernel_global, 0, sizeof(partials),    &partials);
            if(err != CL_SUCCESS){ std::cout << "Cannot set kernel_global's arg 0: " << err << "\n"; return -1; }
            err = clSetKernelArg(kernel_global, 1, sizeof(input_image), &input_image);
            if(err != CL_SUCCESS){ std::cout << "Cannot set kernel_global's arg 1: " << err << "\n"; return -1; }
            err = clSetKernelArg(kernel_global, 2, sizeof(int),         &w);
            if(err != CL_SUCCESS){ std::cout << "Cannot set kernel_global's arg 2: " << err << "\n"; return -1; }
            err = clSetKernelArg(kernel_global, 3, sizeof(int),         &h);
            if(err != CL_SUCCESS){ std::cout << "Cannot set kernel_global's arg 3: " << err << "\n"; return -1; }

            std::array<size_t, 2> global_threads = {block_size, nBlocks_sz * block_size};
            std::array<size_t, 2> workgroup_size = {block_size, block_size};
	        err = clEnqueueNDRangeKernel(queue, kernel_global, 2, nullptr, global_threads.data(), workgroup_size.data(), 0, nullptr, &evts[0]);
            if(err != CL_SUCCESS){ std::cout << "Cannot enqueue 'kernel_global' kernel: " << err << "\n"; return -1; }
        }

        // Second kernel: accumulate partial results:
        {
            err = clSetKernelArg(kernel_acc, 0, sizeof(output),   &output);
            if(err != CL_SUCCESS){ std::cout << "Cannot set kernel_acc's arg 0: " << err << "\n"; return -1; }
            err = clSetKernelArg(kernel_acc, 1, sizeof(partials), &partials);
            if(err != CL_SUCCESS){ std::cout << "Cannot set kernel_acc's arg 1: " << err << "\n"; return -1; }
            err = clSetKernelArg(kernel_acc, 2, sizeof(int),      &nBlocks);
            if(err != CL_SUCCESS){ std::cout << "Cannot set kernel_acc's arg 2: " << err << "\n"; return -1; }

            size_t global_threads = 3 * 256;
	        err = clEnqueueNDRangeKernel(queue, kernel_acc, 1, nullptr, &global_threads, nullptr, 1, &evts[0], &evts[1]);
            if(err != CL_SUCCESS){ std::cout << "Cannot enqueue 'kernel_acc' kernel (1): " << err << "\n"; return -1; }
        }

        // Transfer resulting histogram to host:
        std::vector<unsigned int> tmp(3*256);
        err = clEnqueueReadBuffer(queue, output, true, 0, 3*256*sizeof(unsigned int), tmp.data(), 1, &evts[1], nullptr);
        if(err != CL_SUCCESS){ std::cout << "Error copying memory to host (1): " << err << "\n"; return -1; }
        gpu1.fromLinearMemory(tmp);

        // Calculate kernel execution time:
        cl_ulong t1_0, t1_1;
        err = clGetEventProfilingInfo(evts[0], CL_PROFILING_COMMAND_START, sizeof(t1_0), &t1_0, nullptr);
        if(err != CL_SUCCESS){ std::cout << "Error getting profiling info from event (1/0): " << err << "\n"; return -1; }
        err = clGetEventProfilingInfo(evts[1], CL_PROFILING_COMMAND_END,   sizeof(t1_1), &t1_1, nullptr);
        if(err != CL_SUCCESS){ std::cout << "Error getting profiling info from event (1/1): " << err << "\n"; return -1; }
        dt1 = (t1_1 - t1_0)*0.001f*0.001f;

        err = clReleaseEvent(evts[0]);
        if(err != CL_SUCCESS){ std::cout << "Error releasing event (1/0): " << err << "\n"; return -1; }
        err = clReleaseEvent(evts[1]);
        if(err != CL_SUCCESS){ std::cout << "Error releasing event (1/1): " << err << "\n"; return -1; }
    }

    // GPU version using shared atomics:
    float dt2 = 0.0f;
    {
        unsigned int zero = 0;
        err = clEnqueueFillBuffer(queue, partials, &zero, sizeof(unsigned int), 0, nBlocks * 3 * 256 * sizeof(unsigned int), 0, nullptr, nullptr);
        if(err != CL_SUCCESS){ std::cout << "Cannot zero partial buffer (2): " << err << "\n"; return -1; }

        err = clEnqueueFillBuffer(queue, output, &zero, sizeof(unsigned int), 0, 3 * 256 * sizeof(unsigned int), 0, nullptr, nullptr);
        if(err != CL_SUCCESS){ std::cout << "Cannot zero partial buffer (2): " << err << "\n"; return -1; }

        // Wait for the fills to finish:
        err = clFinish(queue);
        if(err != CL_SUCCESS){ std::cout << "Cannot finish queue: " << err << "\n"; return -1; }

        // Storage for GPU events:
        std::array<cl_event, 2> evts;

        // First kernel of shared histograms:
        {
            err = clSetKernelArg(kernel_shared, 0, sizeof(partials),    &partials);
            if(err != CL_SUCCESS){ std::cout << "Cannot set kernel_shared's arg 0: " << err << "\n"; return -1; }
            err = clSetKernelArg(kernel_shared, 1, sizeof(input_image), &input_image);
            if(err != CL_SUCCESS){ std::cout << "Cannot set kernel_shared's arg 1: " << err << "\n"; return -1; }
            err = clSetKernelArg(kernel_shared, 2, sizeof(int),         &w);
            if(err != CL_SUCCESS){ std::cout << "Cannot set kernel_shared's arg 2: " << err << "\n"; return -1; }
            err = clSetKernelArg(kernel_shared, 3, sizeof(int),         &h);
            if(err != CL_SUCCESS){ std::cout << "Cannot set kernel_shared's arg 3: " << err << "\n"; return -1; }

            std::array<size_t, 2> global_threads = {block_size, nBlocks_sz * block_size};
            std::array<size_t, 2> workgroup_size = {block_size, block_size};
	        err = clEnqueueNDRangeKernel(queue, kernel_shared, 2, nullptr, global_threads.data(), workgroup_size.data(), 0, nullptr, &evts[0]);
            if(err != CL_SUCCESS){ std::cout << "Cannot enqueue 'kernel_shared' kernel: " << err << "\n"; return -1; }
        }

        // Second kernel: accumulate partial results:
        {
            // Note: we do not need to set again the kernel arguments, as they have not changed!

            size_t global_threads = 3 * 256;
	        err = clEnqueueNDRangeKernel(queue, kernel_acc, 1, nullptr, &global_threads, nullptr, 1, &evts[0], &evts[1]);
            if(err != CL_SUCCESS){ std::cout << "Cannot enqueue 'kernel_acc' kernel (2): " << err << "\n"; return -1; }
        }

        // Transfer resulting histogram to host:
        std::vector<unsigned int> tmp(3*256);
        err = clEnqueueReadBuffer(queue, output, true, 0, 3*256*sizeof(unsigned int), tmp.data(), 1, &evts[1], nullptr);
        if(err != CL_SUCCESS){ std::cout << "Error copying memory to host (2): " << err << "\n"; return -1; }
        gpu2.fromLinearMemory(tmp);

        // Calculate kernel execution time:
        cl_ulong t1_0, t1_1;
        err = clGetEventProfilingInfo(evts[0], CL_PROFILING_COMMAND_START, sizeof(t1_0), &t1_0, nullptr);
        if(err != CL_SUCCESS){ std::cout << "Error getting profiling info from event (2/0): " << err << "\n"; return -1; }
        err = clGetEventProfilingInfo(evts[1], CL_PROFILING_COMMAND_END,   sizeof(t1_1), &t1_1, nullptr);
        if(err != CL_SUCCESS){ std::cout << "Error getting profiling info from event (2/1): " << err << "\n"; return -1; }
        dt2 = (t1_1 - t1_0)*0.001f*0.001f;

        err = clReleaseEvent(evts[0]);
        if(err != CL_SUCCESS){ std::cout << "Error releasing event (2/0): " << err << "\n"; return -1; }
        err = clReleaseEvent(evts[1]);
        if(err != CL_SUCCESS){ std::cout << "Error releasing event (2/1): " << err << "\n"; return -1; }
    }

    // Compare GPU results to CPU reference:
    int mismatches1 = compare(gpu1, cpu);
    if     (mismatches1 == 0){ std::cout << "CPU result matches GPU global atomics result.\n"; }
    else if(mismatches1 == 1){ std::cout << "There was 1 mismatch between the CPU and GPU global atomics result.\n"; }
    else                     { std::cout << "There were " << mismatches1 << " mismatches between the CPU and GPU global atomics result.\n"; }

    int mismatches2 = compare(gpu2, cpu);
    if     (mismatches2 == 0){ std::cout << "CPU result matches GPU shared atomics result.\n"; }
    else if(mismatches2 == 1){ std::cout << "There was 1 mismatch between the CPU and GPU shared atomics result.\n"; }
    else                     { std::cout << "There were " << mismatches2 << " mismatches between the CPU and GPU shared atomics result.\n"; }

    // Print times:
    std::cout << "CPU Computation took:                " << dt0 << " ms\n";
    std::cout << "GPU global atomics computation took: " << dt1 << " ms\n";
    std::cout << "GPU shared atomics computation took: " << dt2 << " ms\n";

    // Write out histograms to file:
    draw_histogram(output_filename1, cpu);
    draw_histogram(output_filename2, gpu1);
    draw_histogram(output_filename3, gpu2);

    // Clean-up:
    err = clReleaseMemObject(output);
    if( err != CL_SUCCESS){ std::cout << "Error releasing buffer output (clReleaseMemObject):" << err << "\n"; return -1; }

    err = clReleaseMemObject(partials);
    if( err != CL_SUCCESS){ std::cout << "Error releasing buffer partials (clReleaseMemObject):" << err << "\n"; return -1; }

    err = clReleaseMemObject(input_image);
    if( err != CL_SUCCESS){ std::cout << "Error releasing buffer input_image (clReleaseMemObject):" << err << "\n"; return -1; }

    err = clReleaseKernel(kernel_acc);
    if( err != CL_SUCCESS){ std::cout << "Error releasing kernel 'kernel_acc' (clReleaseKernel):" << err << "\n"; return -1; }

    err = clReleaseKernel(kernel_shared);
    if( err != CL_SUCCESS){ std::cout << "Error releasing kernel 'kernel_shared' (clReleaseKernel):" << err << "\n"; return -1; }

    err = clReleaseKernel(kernel_global);
    if( err != CL_SUCCESS){ std::cout << "Error releasing kernel 'kernel_global' (clReleaseKernel):" << err << "\n"; return -1; }

    err = clReleaseProgram(program);
    if( err != CL_SUCCESS){ std::cout << "Error releasing program (clReleaseProgram):" << err << "\n"; return -1; }

    err = clReleaseCommandQueue(queue);
    if( err != CL_SUCCESS){ std::cout << "Error releasing queue (clReleaseCommandQueue):" << err << "\n"; return -1; }

    err = clReleaseContext(context);
    if( err != CL_SUCCESS){ std::cout << "Error releasing context (clReleaseContext):" << err << "\n"; return -1; }

    err = clReleaseDevice(selected_device_id);
    if( err != CL_SUCCESS){ std::cout << "Error releasing device (clReleaseDevice):" << err << "\n"; return -1; }

    stbi_image_free(data0);

    return 0;
}