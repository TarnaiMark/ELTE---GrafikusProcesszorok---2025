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
#include <fstream>
#include <numeric>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

struct rawcolor{ unsigned char r, g, b; };

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
    std::ifstream file("./sobel.cl");
    if(!file.is_open()){ std::cout << "Error opening kernel file: sobel.cl\n"; return -1; }

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
    
    cl_kernel kernel = clCreateKernel(program, "sobel", &err);
    if(err != CL_SUCCESS){ std::cout << "Error creating kernel (clCreateKernel)\n"; return -1; }

    // Read input image:
    int w = 0; // width
    int h = 0; // height
    int ch = 0; // number of components

    std::string input_filename = "Valve_original.png";
    rawcolor* data0 = reinterpret_cast<rawcolor*>(stbi_load(input_filename.c_str(), &w, &h, &ch, 3 /* we expect 3 components */));
    if(!data0)
    {
        std::cout << "Error: could not open input file: " << input_filename << "\n";
        return -1;
    }
    else
    {
        std::cout << "Image (" << input_filename << ") opened successfully. Width x Height x Components = " << w << " x " << h << " x " << ch << "\n";
    }

    std::vector<cl_float4> input (w*h);
    std::vector<cl_float4> output(w*h);

    std::transform(data0, data0+w*h, input.begin(), [](rawcolor c){ return cl_float4{c.r/255.0f, c.g/255.0f, c.b/255.0f, 1.0f}; } );
    
    // OpenCL needs unsigned sizes so cast them once:
    const size_t szw = static_cast<size_t>(w);
    const size_t szh = static_cast<size_t>(h);

    cl_image_format format = { CL_RGBA, CL_FLOAT };
	cl_image_desc desc = {};
	desc.image_type = CL_MEM_OBJECT_IMAGE2D;
	desc.image_width =  szw;
	desc.image_height = szh;
	desc.image_depth =  0;
	
	cl_mem img_input = clCreateImage(context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS, &format, &desc, input.data(), &err);
    if(err != CL_SUCCESS){ std::cout << "Error creating input image (clCreateImage)\n"; return -1; }
    cl_mem img_output = clCreateImage(context, CL_MEM_WRITE_ONLY,                                               &format, &desc, nullptr, &err);
	if(err != CL_SUCCESS){ std::cout << "Error creating output image (clCreateImage)\n"; return -1; }

    // Set first kernel arguments:
    err = clSetKernelArg(kernel, 0, sizeof(img_output), &img_output);
    if(err != CL_SUCCESS){ std::cout << "Error setting kernel argument 0 (clSetKernelArg)\n"; return -1; }
    err = clSetKernelArg(kernel, 1, sizeof(img_input), &img_input);
    if(err != CL_SUCCESS){ std::cout << "Error setting kernel argument 1 (clSetKernelArg)\n"; return -1; }

    // Launch kernel:
    std::array<size_t, 2> thread_count = {szw, szh};
    std::array<size_t, 2> workgroup_size = {16, 16};
    err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, thread_count.data(), workgroup_size.data(), 0, nullptr, nullptr);
    if(err != CL_SUCCESS){ std::cout << "Error enqueueing kernel (clEnqueueNDRangeKernel)\n"; return -1; }

    // Copy back results:
    std::array<size_t, 3> origin = {0, 0, 0};
    std::array<size_t, 3> dims = {szw, szh, 1};
	err = clEnqueueReadImage(queue, img_output, false, origin.data(), dims.data(), 0, 0, output.data(), 0, nullptr, nullptr);
    if(err != CL_SUCCESS){ std::cout << "Error reading back image to host (clEnqueueReadImage)" << err << "\n"; return -1; }

    // Synchronize:
    err = clFinish(queue);
    if(err != CL_SUCCESS){ std::cout << "Error synchronizing with device (clFinish)\n"; return -1; }

    // Write out image:
    {
        std::vector<rawcolor> tmp(w*h);
        std::transform(output.cbegin(), output.cend(), tmp.begin(),
            [](cl_float4 c){ return rawcolor{  (unsigned char)(c.x*255.0f),
                                               (unsigned char)(c.y*255.0f),
                                               (unsigned char)(c.z*255.0f)}; } );

        const std::string output_filename = "result.jpg";
        int res = stbi_write_jpg(output_filename.c_str(), w, h, ch, tmp.data(), 100);
        if(res == 0)
        {
            std::cout << "Error writing output to file " << output_filename << "\n";
        }else{ std::cout << "Output written to file " << output_filename << "\n"; }
    }

    // cleanup:
    {
        stbi_image_free(data0);

        err = clReleaseMemObject(img_output);
        if( err != CL_SUCCESS){ std::cout << "Error releasing buffer X (clReleaseMemObject)\n"; return -1; }

        err = clReleaseMemObject(img_input);
        if( err != CL_SUCCESS){ std::cout << "Error releasing buffer Y (clReleaseMemObject)\n"; return -1; }

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