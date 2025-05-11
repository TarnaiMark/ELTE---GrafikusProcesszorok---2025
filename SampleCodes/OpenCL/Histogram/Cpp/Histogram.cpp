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

#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 200
#define CL_HPP_ENABLE_EXCEPTIONS 1
#include <CL/opencl.hpp>

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
        std::ifstream source{"./histogram.cl"};
        if( !source.is_open() ){ throw std::runtime_error{ std::string{"Error opening kernel file: histogram.cl"} }; }
        std::string source_string{ std::istreambuf_iterator<char>{ source },
                                   std::istreambuf_iterator<char>{} };
        cl::Program program{ context, source_string };
        program.build({selected_device});
        
        auto kernel_global = cl::KernelFunctor<cl::Buffer, cl::Buffer, int, int>(program, "gpu_histo_global_atomics");
        auto kernel_shared = cl::KernelFunctor<cl::Buffer, cl::Buffer, int, int>(program, "gpu_histo_shared_atomics");
        auto kernel_acc    = cl::KernelFunctor<cl::Buffer, cl::Buffer, int     >(program, "gpu_histo_accumulate");

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
        cl::Buffer input_image{ context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, w*h*sizeof(color), data0 };
        cl::Buffer partials{ context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,  nBlocks * 3 * 256 * sizeof(unsigned int) };
        cl::Buffer output{ context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,    3 * 256 * sizeof(unsigned int), };

        // GPU version using global atomics:
        float dt1 = 0.0f;
        {
            queue.enqueueFillBuffer<unsigned int>(partials, 0, 0, nBlocks * 3 * 256 * sizeof(unsigned int));
            queue.enqueueFillBuffer<unsigned int>(output, 0, 0,             3 * 256 * sizeof(unsigned int));
            queue.finish();
            
            // First kernel of global histograms:
            cl::NDRange global_threads = {block_size, nBlocks_sz * block_size};
            cl::NDRange workgroup_size = {block_size, block_size};
            cl::Event ev0 = kernel_global(cl::EnqueueArgs{queue, global_threads, workgroup_size}, partials, input_image, w, h);

            // Second kernel: accumulate partial results:
            size_t global_threads2 = 3 * 256;
            cl::Event ev1 = kernel_acc(cl::EnqueueArgs{queue, ev0, cl::NDRange{global_threads2}, }, output, partials, nBlocks);

            // Transfer resulting histogram to host:
            std::vector<unsigned int> tmp(3*256);
            cl::copy(queue, output, tmp.begin(), tmp.end());
            gpu1.fromLinearMemory(tmp);

            // Calculate kernel execution time:
            cl_ulong t1_0 = ev0.getProfilingInfo<CL_PROFILING_COMMAND_END>() - ev0.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            cl_ulong t1_1 = ev1.getProfilingInfo<CL_PROFILING_COMMAND_END>() - ev1.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            dt1 = (t1_1 + t1_0)*0.001f*0.001f;
        }

        // GPU version using shared atomics:
        float dt2 = 0.0f;
        {
            queue.enqueueFillBuffer<unsigned int>(partials, 0, 0, nBlocks * 3 * 256 * sizeof(unsigned int));
            queue.enqueueFillBuffer<unsigned int>(output, 0, 0,             3 * 256 * sizeof(unsigned int));
            queue.finish();
            
            // First kernel of global histograms:
            cl::NDRange global_threads = {block_size, nBlocks_sz * block_size};
            cl::NDRange workgroup_size = {block_size, block_size};
            cl::Event ev0 = kernel_shared(cl::EnqueueArgs{queue, global_threads, workgroup_size}, partials, input_image, w, h);

            // Second kernel: accumulate partial results:
            size_t global_threads2 = 3 * 256;
            cl::Event ev1 = kernel_acc(cl::EnqueueArgs{queue, ev0, cl::NDRange{global_threads2}, }, output, partials, nBlocks);

            // Transfer resulting histogram to host:
            std::vector<unsigned int> tmp(3*256);
            cl::copy(queue, output, tmp.begin(), tmp.end());
            gpu2.fromLinearMemory(tmp);

            // Calculate kernel execution time:
            cl_ulong t1_0 = ev0.getProfilingInfo<CL_PROFILING_COMMAND_END>() - ev0.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            cl_ulong t1_1 = ev1.getProfilingInfo<CL_PROFILING_COMMAND_END>() - ev1.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            dt2 = (t1_1 + t1_0)*0.001f*0.001f;
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