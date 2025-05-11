#include <iostream>
#include <iomanip>
#include <array>
#include <vector>
#include <numeric>
#include <algorithm>
#include <string>
#include <cmath>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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

__global__ void gpu_histo_global_atomics( unsigned int* output, uchar4* input, int w, int h )
{
    // Linear block index within 2D grid
    const int B = blockIdx.x + blockIdx.y * gridDim.x;

    // Output index start for this block's histogram:
    const int I = B*(3*256);
    unsigned int* H = output + I;

    // Process pixel blocks horizontally
    // Updates our block's partial histogram in global memory
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(y >= h){ return; }
    for (int x = threadIdx.x; x < w; x += blockDim.x)
    {
        uchar4 pixels = input[y * w + x];
        atomicAdd(&H[0 * 256 + pixels.x], 1);
        atomicAdd(&H[1 * 256 + pixels.y], 1);
        atomicAdd(&H[2 * 256 + pixels.z], 1);
    }
}

__global__ void gpu_histo_shared_atomics( unsigned int* output, uchar4* input, int w, int h )
{
    // Histograms are in shared memory:
    __shared__ unsigned int histo[3 * 256];

    // Number of threads in the block:
    const int Nthreads = blockDim.x * blockDim.y;
    // Linear thread idx:
    const int LinID = threadIdx.x + threadIdx.y * blockDim.x;
    // Zero histogram:
    for (int i = LinID; i < 3 * 256; i += Nthreads){ histo[i] = 0; }
    
    __syncthreads();

    // Linear block index within 2D grid
    const int B = blockIdx.x + blockIdx.y * gridDim.x;

    // Process pixel blocks horizontally
    // Updates the partial histogram in shared memory
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(y < h)
    {
        for (int x = threadIdx.x; x < w; x += blockDim.x)
        {
            uchar4 pixels = input[y * w + x];
            atomicAdd(&histo[0 * 256 + pixels.x], 1);
            atomicAdd(&histo[1 * 256 + pixels.y], 1);
            atomicAdd(&histo[2 * 256 + pixels.z], 1);
        }
    }

    __syncthreads();

    // Output index start for this block's histogram:
    const int I = B*(3*256);
    unsigned int* H = output + I;

    // Copy shared memory histograms to globl memory:
    for (int i = LinID; i < 256; i += Nthreads)
    {
        H[0*256 + i] = histo[0*256 + i];
        H[1*256 + i] = histo[1*256 + i];
        H[2*256 + i] = histo[2*256 + i];
    }
}

__global__ void gpu_histo_accumulate( unsigned int* out, const unsigned int* in, int nBlocks )
{
    // Each thread sums one shade of the r, g, b histograms
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < 3 * 256)
    {
        unsigned int sum = 0;
        for(int j = 0; j < nBlocks; j++)
        {
            sum += in[i + (3*256) * j];
        }            
        out[i] = sum;
    }
}

const std::string input_filename   = "NZ.jpg";
const std::string output_filename1 = "cpu_out.jpg";
const std::string output_filename2 = "gpu_out1.jpg";
const std::string output_filename3 = "gpu_out2.jpg";
constexpr int block_size = 16;

int main()
{
    cudaError_t err = cudaSuccess;
    
    // Using the implicitely selected first cuda device:
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if(err != cudaSuccess){ std::cout << "Error getting device properties for device 0: " << cudaGetErrorString(err) << "\n"; return -1; }
    std::cout << "Selected device name: " << prop.name << "\n";
    
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

    const int nBlocks = std::ceil(h*1.0f / block_size);

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

    // Allocate GPU memory for input:
    color* input_image = nullptr;
    err = cudaMalloc( (void**)&input_image, w*h*sizeof(color) );
    if(err != cudaSuccess ){ std::cout << "Error allocating CUDA memory for input image: " << cudaGetErrorString(err) << "\n"; return -1; }

    err = cudaMemcpy( input_image, data0, w*h*sizeof(color), cudaMemcpyHostToDevice );
    if(err != cudaSuccess){ std::cout << "Error copying memory to device: " << cudaGetErrorString(err) << "\n"; return -1; }

    // Allocate GPU memory for partial histograms:
    unsigned int* partials = nullptr;
    err = cudaMalloc( (void**)&partials, nBlocks * 3 * 256 * sizeof(unsigned int) );
    if(err != cudaSuccess){ std::cout << "Error allocating CUDA memory for partials: " << cudaGetErrorString(err) << "\n"; return -1; }

    // Allocate memory for final histograms:
    unsigned int* output = nullptr;
    err = cudaMalloc( (void**)&output, 3 * 256 * sizeof(unsigned int) );
    if(err != cudaSuccess){ std::cout << "Error allocating CUDA memory for output: " << cudaGetErrorString(err) << "\n"; return -1; }

    // Initialize GPU events:
    std::array<cudaEvent_t, 4> evts;
    for(auto& e : evts)
    {
        err = cudaEventCreate(&e);
        if(err != cudaSuccess){ std::cout << "Error creating CUDA event: " << cudaGetErrorString(err) << "\n"; return -1; }
    }

    // GPU version using global atomics:
    float dt1 = 0.0f;
    {
        // Zero partial results:
        err = cudaMemset(partials, 0, nBlocks * 3 * 256 * sizeof(unsigned int) );
        if( err != cudaSuccess){ std::cout << "Error setting partials to zero: " << cudaGetErrorString(err) << "\n"; return -1; }
        // Zero final results:
        err = cudaMemset(output, 0, 3 * 256 * sizeof(unsigned int) );
        if( err != cudaSuccess){ std::cout << "Error setting partials to zero: " << cudaGetErrorString(err) << "\n"; return -1; }
        // Wait for the memsets to finish:
        err = cudaStreamSynchronize(0);
        if( err != cudaSuccess){ std::cout << "Error in cudaStreamSynchronize (0): " << cudaGetErrorString(err) << "\n"; return -1; }
        
        // First kernel of global histograms:
        {
            dim3 dimGrid( 1, nBlocks );
            dim3 dimBlock( block_size, block_size );
            err = cudaEventRecord(evts[0]);
            if(err != cudaSuccess){ std::cout << "Error recording event 0: " << cudaGetErrorString(err) << "\n"; return -1; }

            gpu_histo_global_atomics<<<dimGrid, dimBlock>>>(partials, (uchar4*)input_image, w, h);

            err = cudaEventRecord(evts[1]);
            if(err != cudaSuccess){ std::cout << "Error recording event 1: " << cudaGetErrorString(err) << "\n"; return -1; }
            
            err = cudaGetLastError();
            if(err != cudaSuccess){ std::cout << "CUDA error in kernel call 'gpu_histo_global_atomics': " << cudaGetErrorString(err) << "\n"; return -1; }
        }

        // Second kernel: accumulate partial results:
        {
            dim3 dimGrid( 1 );
            dim3 dimBlock( 3*256 );
            err = cudaEventRecord(evts[2]);
            if(err != cudaSuccess){ std::cout << "Error recording event 2: " << cudaGetErrorString(err) << "\n"; return -1; }
            
            gpu_histo_accumulate<<<dimGrid, dimBlock>>>(output, partials, nBlocks);
            
            err = cudaEventRecord(evts[3]);
            if(err != cudaSuccess){ std::cout << "Error recording event 3: " << cudaGetErrorString(err) << "\n"; return -1; }

            err = cudaGetLastError();
            if(err != cudaSuccess){ std::cout << "CUDA error in kernel call 'gpu_histo_accumulate': " << cudaGetErrorString(err) << "\n"; return -1; }
        }

        // Transfer resulting histogram to host:
        std::vector<unsigned int> tmp(3*256);
        err = cudaMemcpy( tmp.data(), output, 3*256*sizeof(unsigned int), cudaMemcpyDeviceToHost );
        if(err != cudaSuccess){ std::cout << "Error copying memory to host (1): " << cudaGetErrorString(err) << "\n"; return -1; }
        gpu1.fromLinearMemory(tmp);
        
        // Calculate kernel execution time:
        err = cudaEventSynchronize(evts[3]);
        if(err != cudaSuccess){ std::cout << "CUDA error in cudaEventSynchronize (1): " << cudaGetErrorString(err) << "\n"; return -1; }

        float dt = 0.0f; // milliseconds
        err = cudaEventElapsedTime(&dt, evts[0], evts[1]);
        if(err != cudaSuccess){ std::cout << "CUDA error in cudaEventElapsedTime (1): " << cudaGetErrorString(err) << "\n"; return -1; }
        dt1 = dt;
        err = cudaEventElapsedTime(&dt, evts[2], evts[3]); 
        if(err != cudaSuccess){ std::cout << "CUDA error in cudaEventElapsedTime (2): " << cudaGetErrorString(err) << "\n"; return -1; }
        dt1 += dt;
    }

    // GPU version using shared atomics:
    float dt2 = 0.0f;
    {
        // Zero partial results:
        err = cudaMemset(partials, 0, nBlocks * 3 * 256 * sizeof(unsigned int) );
        if( err != cudaSuccess){ std::cout << "Error setting partials to zero: " << cudaGetErrorString(err) << "\n"; return -1; }
        // Zero final results:
        err = cudaMemset(output, 0, 3 * 256 * sizeof(unsigned int) );
        if( err != cudaSuccess){ std::cout << "Error setting partials to zero: " << cudaGetErrorString(err) << "\n"; return -1; }
        // Wait for the memsets to finish:
        err = cudaStreamSynchronize(0);
        if( err != cudaSuccess){ std::cout << "Error in cudaStreamSynchronize (1): " << cudaGetErrorString(err) << "\n"; return -1; }
      
        // First kernel of global histograms:
        {
            dim3 dimGrid( 1, nBlocks );
            dim3 dimBlock( block_size, block_size );
            
            err = cudaEventRecord(evts[0]);
            if(err != cudaSuccess){ std::cout << "Error recording event 0: " << cudaGetErrorString(err) << "\n"; return -1; }
            
            gpu_histo_shared_atomics<<<dimGrid, dimBlock>>>(partials, (uchar4*)input_image, w, h);
            
            err = cudaEventRecord(evts[1]);
            if(err != cudaSuccess){ std::cout << "Error recording event 1: " << cudaGetErrorString(err) << "\n"; return -1; }
            
            err = cudaGetLastError();
            if(err != cudaSuccess){ std::cout << "CUDA error in kernel call 'gpu_histo_shared_atomics': " << cudaGetErrorString(err) << "\n"; return -1; }
        }

        // Second kernel: accumulate partial results:
        {
            dim3 dimGrid( 1 );
            dim3 dimBlock( 3*256 );
            err = cudaEventRecord(evts[2]);
            if(err != cudaSuccess){ std::cout << "Error recording event 2: " << cudaGetErrorString(err) << "\n"; return -1; }
            
            gpu_histo_accumulate<<<dimGrid, dimBlock>>>(output, partials, nBlocks);
            
            err = cudaEventRecord(evts[3]);
            if(err != cudaSuccess){ std::cout << "Error recording event 3: " << cudaGetErrorString(err) << "\n"; return -1; }

            err = cudaGetLastError();
            if(err != cudaSuccess){ std::cout << "CUDA error in kernel call 'gpu_histo_accumulate' (2): " << cudaGetErrorString(err) << "\n"; return -1; }
        }

        // Transfer resulting histogram to host:
        std::vector<unsigned int> tmp(3*256);
        err = cudaMemcpy( tmp.data(), output, 3*256*sizeof(unsigned int), cudaMemcpyDeviceToHost );
        if(err != cudaSuccess){ std::cout << "Error copying memory to host (2): " << cudaGetErrorString(err) << "\n"; return -1; }
        gpu2.fromLinearMemory(tmp);
        
        // Calculate kernel execution time:
        err = cudaEventSynchronize(evts[3]);
        if(err != cudaSuccess){ std::cout << "CUDA error in cudaEventSynchronize (2): " << cudaGetErrorString(err) << "\n"; return -1; }

        float dt = 0.0f; // milliseconds
        err = cudaEventElapsedTime(&dt, evts[0], evts[1]);
        if(err != cudaSuccess){ std::cout << "CUDA error in cudaEventElapsedTime (3): " << cudaGetErrorString(err) << "\n"; return -1; }
        dt2 = dt;
        err = cudaEventElapsedTime(&dt, evts[2], evts[3]); 
        if(err != cudaSuccess){ std::cout << "CUDA error in cudaEventElapsedTime (4): " << cudaGetErrorString(err) << "\n"; return -1; }
        dt2 += dt;
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
	for(auto& e : evts)
    {
        err = cudaEventDestroy(e);
        if(err != cudaSuccess){ std::cout << "Error destroying CUDA event: " << cudaGetErrorString(err) << "\n"; return -1; }
    }

    err = cudaFree( output );
    if( err != cudaSuccess){ std::cout << "Error freeing allocation 'output': " << cudaGetErrorString(err) << "\n"; return -1; }

    err = cudaFree( partials );
    if( err != cudaSuccess){ std::cout << "Error freeing allocation 'partials': " << cudaGetErrorString(err) << "\n"; return -1; }

    err = cudaFree( input_image );
    if( err != cudaSuccess){ std::cout << "Error freeing allocation 'input_image': " << cudaGetErrorString(err) << "\n"; return -1; }

    stbi_image_free(data0);

	return 0;
}