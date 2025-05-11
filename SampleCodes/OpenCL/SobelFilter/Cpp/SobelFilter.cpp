#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <fstream>
#include <numeric>
#include <algorithm>

#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 200
#define CL_HPP_ENABLE_EXCEPTIONS 1
#include <CL/opencl.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

struct rawcolor{ unsigned char r, g, b; };

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
        std::ifstream source{"./sobel.cl"};
        if( !source.is_open() ){ throw std::runtime_error{ std::string{"Error opening kernel file: sobel.cl"} }; }
        std::string source_string{ std::istreambuf_iterator<char>{ source },
                                   std::istreambuf_iterator<char>{} };
        cl::Program program{ context, source_string };
        program.build({selected_device});
        
        auto sobel_kernel = cl::KernelFunctor<cl::Image2D, cl::Image2D>(program, "sobel");

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

        cl::ImageFormat format{ CL_RGBA, CL_FLOAT };

        cl::Image2D img_input{context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS, format, szw, szh, 0, input.data()};
        cl::Image2D img_output{context, CL_MEM_WRITE_ONLY, format, szw, szh};

        // Launch kernel:
        cl::NDRange thread_count = {szw, szh};
        cl::NDRange workgroup_size = {16, 16};
        sobel_kernel(cl::EnqueueArgs{queue, thread_count, workgroup_size}, img_output, img_input);

        // Copy back results:
        std::array<size_t, 3> origin = {0, 0, 0};
        std::array<size_t, 3> dims = {szw, szh, 1};
        queue.enqueueReadImage(img_output, false, origin, dims, 0, 0, output.data());
        
        // Synchronize:
        queue.finish();

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
        stbi_image_free(data0);
        
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
        std::cout << "OpenCL error: " << e.what() << " " << e.err() << "\n";
        return -1;
    }
    catch(std::exception& e)
    {
        std::cerr << "C++ STL Error: " << e.what() << "\n";
        return -1;
    }
    
    return 0;
}